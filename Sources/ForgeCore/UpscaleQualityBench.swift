//
//  UpscaleQualityBench.swift — ForgeCore
//
//  V10's second half: **how much quality does each upscale tier actually buy over free?**
//
//  🚨 **The point of this file is the Lanczos arm.** V10's first half compared two *learned* tiers to
//  each other and found the quality tier costs 43.7 GB and 29 s against the fast tier's 0.5 GB and 0.9 s.
//  Neither number means anything without a floor: cross-cutting rule **#43c** — *never choose a backend
//  from a regime that cannot discriminate, and carry a trivial baseline.* Lanczos at ~0 GB and
//  sub-millisecond IS that baseline. If a learned tier cannot beat windowed sinc on a content class,
//  it does not belong in that path at any price.
//
//  🔑 **Lanczos cannot add information, and that is exactly why it is the right control.** At ×4, fifteen
//  of every sixteen output pixels are new, and a windowed sinc adds **no frequency content above the
//  source Nyquist** — three quarters of the output band stays empty. A learned model fills that band with
//  *plausible* content. So this bench does not measure "which is sharper"; it measures **how much of the
//  reference each method actually recovers**, which is the only question a full-reference metric answers.
//
//  ⚠️ **Both degradation regimes are run, and that is not thoroughness for its own sake.** No choice of
//  downsampler is neutral: matched Lanczos-down/Lanczos-up flatters Lanczos, while bicubic-down is
//  *in-distribution* for Real-ESRGAN (its training pipeline randomises bicubic/bilinear/area resize).
//  Reporting one regime would hand the win to whichever arm it favoured. Two regimes make the bias
//  visible instead of load-bearing.
//
//  🔑 **The reference is a LOSSLESS CROP, never a resize.** A 2048² centre crop of a ~4K master contains
//  real sensor/render pixels; a resized reference would bake one resampler's artefacts into the ground
//  truth and then score every method against them.
//
//  ⚠️ **If Lanczos ever becomes a shipping tier it must MOVE.** It needs no MLX, so it belongs in a
//  category-A Kit, not in the engine-adapter package. It lives here only because this is a diagnostics
//  surface that already links MediaMeasure and the engine.
//

import Accelerate
import CoreGraphics
import Foundation
import ImageIO   // CGImageSource lives here, not in CoreGraphics
import MediaMeasure
import Metal
import MetalPerformanceShaders

/// One (image × regime × method) cell.
public struct UpscaleQualityRow: Sendable {
    public let imageName: String
    /// `photo` or `graphic` — the corpus's own split, and the axis this bench exists to separate.
    public let contentClass: String
    /// Which downsampler produced the low-res input.
    public let regime: String
    public let method: String
    /// SSIMULACRA2 vs the lossless reference crop. Higher is better; ~90 is "visually fine", negative is
    /// possible and means severely degraded.
    public let score: Double
    public let seconds: Double
}

public enum UpscaleQualityBench {

    public enum BenchError: Error, CustomStringConvertible {
        case noMetalDevice
        case imageLoadFailed(String)
        case tooSmall(String, Int, Int)
        case resampleFailed(String)

        public var description: String {
            switch self {
            case .noMetalDevice: return "no Metal device — the Lanczos arm needs one"
            case .imageLoadFailed(let n): return "could not load \(n)"
            case .tooSmall(let n, let w, let h):
                return "\(n) is \(w)×\(h) — smaller than the 2048² reference crop"
            case .resampleFailed(let why): return "resample failed: \(why)"
            }
        }
    }

    /// Reference side, and the ×4 operating point P16's claims are stated at.
    public static let referenceSide = 2048
    public static let lowSide = 512

    /// 🔑 **Dump every intermediate to disk when set.** A full-reference score is a single number, and a
    /// single number cannot tell "this method is worse" apart from "this method returned garbage" — a
    /// content-INDEPENDENT score is the signature of the latter. Set `FORGE_BENCH_DUMP` to a directory and
    /// the reference, the low-res input and every upscale land there as PNGs, which is the only way to
    /// tell those two apart. (The Anima lesson: an eyeball gate with no reference beside it passes
    /// garbage; a metric with no image beside it *reports* garbage as a result.)
    static var dumpDirectory: URL? {
        ProcessInfo.processInfo.environment["FORGE_BENCH_DUMP"].map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
    }
    /// Cap the sweep for a fast diagnostic pass.
    static var imageLimit: Int? {
        ProcessInfo.processInfo.environment["FORGE_BENCH_LIMIT"].flatMap(Int.init)
    }
    static var regimes: [String] {
        ProcessInfo.processInfo.environment["FORGE_BENCH_REGIME"].map { [$0] } ?? ["bicubic", "lanczos"]
    }

    /// Per-channel mean and standard deviation. **A near-zero σ means a flat image** — the cheap,
    /// unambiguous test for "the model returned nothing" that a score cannot give you.
    public static func statistics(_ image: CGImage) -> (mean: [Double], sigma: [Double])? {
        let w = image.width, h = image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: space,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        var mean = [Double](repeating: 0, count: 3), sq = [Double](repeating: 0, count: 3)
        let n = Double(w * h)
        for i in stride(from: 0, to: w * h * 4, by: 4) {
            for c in 0..<3 {
                let v = Double(bytes[i + c])
                mean[c] += v; sq[c] += v * v
            }
        }
        for c in 0..<3 { mean[c] /= n; sq[c] = max(0, sq[c] / n - mean[c] * mean[c]) }
        return (mean, sq.map { $0.squareRoot() })
    }

    static func dump(_ image: CGImage, as name: String) {
        guard let directory = dumpDirectory else { return }
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent(name + ".png")
        guard let destination = CGImageDestinationCreateWithURL(url as CFURL, "public.png" as CFString, 1, nil)
        else { return }
        CGImageDestinationAddImage(destination, image, nil)
        CGImageDestinationFinalize(destination)
    }

    // MARK: - Entry point

    /// Score every method on every image in both regimes.
    ///
    /// - Parameter learnedArms: the engine-backed tiers to include. Pass `[]` to run the classical arms
    ///   alone (fast, and enough to establish the floor).
    public static func run(imageURLs: [URL],
                           service: ForgeEngineService?,
                           learnedArms: [ForgeModel],
                           progress: (@Sendable (String) -> Void)? = nil) async -> [UpscaleQualityRow] {
        var rows: [UpscaleQualityRow] = []

        for url in (imageLimit.map { Array(imageURLs.prefix($0)) } ?? imageURLs) {
            let name = url.deletingPathExtension().lastPathComponent
            // The corpus encodes its split in the filename; honour it rather than re-deriving it.
            let contentClass = name.contains("_graphic") ? "graphic"
                             : name.contains("_photo") ? "photo" : "unknown"
            guard let master = loadCGImage(url) else {
                progress?("  \(name): SKIP — could not load")
                continue
            }
            guard let reference = centreCrop(master, side: referenceSide) else {
                progress?("  \(name): SKIP — \(master.width)×\(master.height) < \(referenceSide)²")
                continue
            }

            dump(reference, as: "\(name)__reference")
            for regime in regimes {
                guard let low = downsample(reference, to: lowSide, regime: regime) else {
                    progress?("  \(name)/\(regime): SKIP — downsample failed")
                    continue
                }
                dump(low, as: "\(name)__\(regime)__low")

                // --- classical arms: the floor, and a second floor below it ---
                for (method, upscaler) in classicalArms() {
                    let start = Date()
                    guard let up = upscaler(low, referenceSide) else {
                        progress?("  \(name)/\(regime)/\(method): SKIP — upscale failed")
                        continue
                    }
                    let seconds = Date().timeIntervalSince(start)
                    dump(up, as: "\(name)__\(regime)__\(method)")
                    if let stats = statistics(up) {
                        progress?(String(format: "      %@ stats mean %.1f/%.1f/%.1f sigma %.1f/%.1f/%.1f",
                                         method, stats.mean[0], stats.mean[1], stats.mean[2],
                                         stats.sigma[0], stats.sigma[1], stats.sigma[2]))
                    }
                    if let score = try? MediaMeasure.ssimulacra2(reference: reference, distorted: up) {
                        rows.append(UpscaleQualityRow(imageName: name, contentClass: contentClass,
                                                      regime: regime, method: method,
                                                      score: score, seconds: seconds))
                        progress?(String(format: "  %@/%@/%@  %.2f  (%.3fs)", name, regime, method, score, seconds))
                    }
                }

                // --- learned arms, through the engine, exactly as the product would ---
                for model in learnedArms {
                    guard let service else { continue }
                    do {
                        let m = try await service.measureUpscale(low, model: model, scale: 4)
                        dump(m.image, as: "\(name)__\(regime)__\(model.modelName)")
                        if let stats = statistics(m.image) {
                            progress?(String(format: "      %@ stats mean %.1f/%.1f/%.1f sigma %.1f/%.1f/%.1f",
                                             model.modelName, stats.mean[0], stats.mean[1], stats.mean[2],
                                             stats.sigma[0], stats.sigma[1], stats.sigma[2]))
                        }
                        if let score = try? MediaMeasure.ssimulacra2(reference: reference, distorted: m.image) {
                            rows.append(UpscaleQualityRow(imageName: name, contentClass: contentClass,
                                                          regime: regime, method: model.modelName,
                                                          score: score, seconds: m.runSeconds))
                            progress?(String(format: "  %@/%@/%@  %.2f  (%.2fs)",
                                             name, regime, model.modelName, score, m.runSeconds))
                        }
                    } catch {
                        progress?("  \(name)/\(regime)/\(model.modelName): FAILED — \(error)")
                    }
                }
            }
        }
        return rows
    }

    /// The classical arms. **Bicubic sits below Lanczos on purpose** — if the two are indistinguishable on
    /// a content class, the sample cannot discriminate resamplers at all, and a learned tier's margin over
    /// *either* is not yet evidence of anything (#43c).
    static func classicalArms() -> [(String, @Sendable (CGImage, Int) -> CGImage?)] {
        [("lanczos-mps", { img, side in try? lanczosMPS(img, targetSide: side) }),
         ("bicubic-cg",   { img, side in resampleCG(img, side: side, quality: .high) })]
    }

    // MARK: - Lanczos via MPS

    /// `MPSImageLanczosScale` — the same API family V7 cleared (MPSImage carries **zero** deprecation
    /// attributes; headers byte-identical 26.5 → 27.0). ⚠️ V7's one live caveat applies: this encodes to a
    /// classic `MTLCommandBuffer`, so a Metal 4 pipeline needs a separate queue plus `MTLEvent` sync.
    public static func lanczosMPS(_ image: CGImage, targetSide: Int) throws -> CGImage {
        guard let device = MTLCreateSystemDefaultDevice(),
              let queue = device.makeCommandQueue() else { throw BenchError.noMetalDevice }

        let source = try texture(from: image, device: device)
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: targetSide, height: targetSide, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let destination = device.makeTexture(descriptor: descriptor),
              let commands = queue.makeCommandBuffer() else {
            throw BenchError.resampleFailed("could not allocate the destination texture")
        }

        let scaler = MPSImageLanczosScale(device: device)
        scaler.encode(commandBuffer: commands, sourceTexture: source, destinationTexture: destination)
        commands.commit()
        commands.waitUntilCompleted()
        return try cgImage(from: destination)
    }

    // MARK: - Pixel plumbing

    static func loadCGImage(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    /// A lossless centre crop — the reference must not be resampled.
    static func centreCrop(_ image: CGImage, side: Int) -> CGImage? {
        guard image.width >= side, image.height >= side else { return nil }
        let rect = CGRect(x: (image.width - side) / 2, y: (image.height - side) / 2,
                          width: side, height: side)
        return image.cropping(to: rect)
    }

    static func downsample(_ image: CGImage, to side: Int, regime: String) -> CGImage? {
        regime == "lanczos" ? (try? lanczosMPS(image, targetSide: side))
                            : resampleCG(image, side: side, quality: .high)
    }

    /// CoreGraphics `.high` interpolation — bicubic-class. Used for the bicubic arm *and* the bicubic
    /// degradation regime, which is deliberate: the classic SR protocol degrades bicubically.
    static func resampleCG(_ image: CGImage, side: Int, quality: CGInterpolationQuality) -> CGImage? {
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: nil, width: side, height: side, bitsPerComponent: 8,
                                      bytesPerRow: 0, space: space,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return nil }
        context.interpolationQuality = quality
        context.draw(image, in: CGRect(x: 0, y: 0, width: side, height: side))
        return context.makeImage()
    }

    static func texture(from image: CGImage, device: MTLDevice) throws -> MTLTexture {
        let width = image.width, height = image.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let context = CGContext(data: &bytes, width: width, height: height, bitsPerComponent: 8,
                                      bytesPerRow: width * 4, space: space,
                                      bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
        else { throw BenchError.resampleFailed("could not rasterize the source") }
        context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))

        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .rgba8Unorm, width: width, height: height, mipmapped: false)
        descriptor.usage = [.shaderRead, .shaderWrite]
        guard let texture = device.makeTexture(descriptor: descriptor) else {
            throw BenchError.resampleFailed("could not allocate the source texture")
        }
        texture.replace(region: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0,
                        withBytes: bytes, bytesPerRow: width * 4)
        return texture
    }

    static func cgImage(from texture: MTLTexture) throws -> CGImage {
        let width = texture.width, height = texture.height
        var bytes = [UInt8](repeating: 0, count: width * height * 4)
        texture.getBytes(&bytes, bytesPerRow: width * 4,
                         from: MTLRegionMake2D(0, 0, width, height), mipmapLevel: 0)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let provider = CGDataProvider(data: Data(bytes) as CFData),
              let image = CGImage(width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32,
                                  bytesPerRow: width * 4, space: space,
                                  bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue),
                                  provider: provider, decode: nil, shouldInterpolate: false,
                                  intent: .defaultIntent)
        else { throw BenchError.resampleFailed("could not build a CGImage from the texture") }
        return image
    }

    // MARK: - Reporting

    /// Per-image rows first, then per-class means. **Never the grand mean alone** — the whole hypothesis
    /// is that photo and graphic behave differently, so an average across them would hide the finding.
    public static func report(_ rows: [UpscaleQualityRow]) -> [String] {
        guard !rows.isEmpty else { return ["V10-Q — no rows produced"] }
        var lines = ["V10-Q — SSIMULACRA2 vs a lossless 2048² reference, 512→2048 (×4)"]

        let methods = Array(Set(rows.map(\.method))).sorted()
        for regime in ["bicubic", "lanczos"] {
            let inRegime = rows.filter { $0.regime == regime }
            guard !inRegime.isEmpty else { continue }
            lines.append("")
            lines.append("  ── degradation: \(regime) ──")
            lines.append("    " + "image".padding(toLength: 22, withPad: " ", startingAt: 0)
                         + methods.map { $0.padding(toLength: 14, withPad: " ", startingAt: 0) }.joined())
            for name in Array(Set(inRegime.map(\.imageName))).sorted() {
                var row = "    " + name.padding(toLength: 22, withPad: " ", startingAt: 0)
                for method in methods {
                    let cell = inRegime.first { $0.imageName == name && $0.method == method }
                    row += (cell.map { String(format: "%.2f", $0.score) } ?? "—")
                        .padding(toLength: 14, withPad: " ", startingAt: 0)
                }
                lines.append(row)
            }
            for contentClass in ["photo", "graphic"] {
                let subset = inRegime.filter { $0.contentClass == contentClass }
                guard !subset.isEmpty else { continue }
                var row = "    " + "MEAN \(contentClass)".padding(toLength: 22, withPad: " ", startingAt: 0)
                for method in methods {
                    let scores = subset.filter { $0.method == method }.map(\.score)
                    row += (scores.isEmpty ? "—"
                            : String(format: "%.2f", scores.reduce(0, +) / Double(scores.count)))
                        .padding(toLength: 14, withPad: " ", startingAt: 0)
                }
                lines.append(row)
            }
        }
        return lines
    }
}
