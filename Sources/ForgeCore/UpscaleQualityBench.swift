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
//  ⚠️ **Every degradation regime is run, and that is not thoroughness for its own sake.** No choice of
//  downsampler is neutral: matched Lanczos-down/Lanczos-up flatters Lanczos, while bicubic-down is
//  *in-distribution* for Real-ESRGAN (its training pipeline randomises bicubic/bilinear/area resize).
//  Reporting one regime would hand the win to whichever arm it favoured. Multiple regimes make the bias
//  visible instead of load-bearing.
//
//  🚨 **`jpeg` is the regime that can invert the finding, and it is why it was added.** `bicubic` and
//  `lanczos` are *clean* downsamples: the low-res input contains no artefact, so there is **nothing to
//  undo**, and a method that adds no information cannot be wrong about information it did not add.
//  That regime flatters interpolation by construction, and every V10 number before this one was taken
//  in it. A JPEG-crushed source is the opposite case — blocking, ringing and chroma damage are *real
//  degradation with a correct answer* — and it is also what signage actually arrives as. If a learned
//  tier is ever going to beat a windowed sinc on a full-reference metric, this is where.
//
//  🚨 **TWO ORACLES, REPORTED SIDE BY SIDE, NEVER AVERAGED.** `CLAUDE.md`: *SigLIP2 NR-IQA gates whether
//  to restore (no reference); SSIMULACRA2 gates how hard to compress (full-reference).* V10 ran the
//  full-reference oracle alone and got an ordering that the detail axis exactly inverted — which is not
//  a metric malfunctioning but a fidelity metric doing its job on a *generative* method. Both oracles
//  plus the detail axis are recorded per cell here. **The disagreement is the result; collapsing it into
//  one number would delete the finding.**
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

/// One (image × regime × method) cell — **three numbers, deliberately not one.**
public struct UpscaleQualityRow: Sendable {
    public let imageName: String
    /// `photo` or `graphic` — the corpus's own split, and the axis this bench exists to separate.
    public let contentClass: String
    /// Which degradation produced the low-res input: `bicubic`, `lanczos` or `jpeg`.
    public let regime: String
    public let method: String
    /// **Full-reference.** SSIMULACRA2 vs the lossless reference crop. Higher is better; ~90 is
    /// "visually fine", negative is possible and means severely degraded. ⚠️ Structurally biased
    /// *against* invented detail — see this file's header.
    public let score: Double
    /// **No-reference.** SigLIP2 NR-IQA, 0–1, "restoration-pays" semantics: low where enhancement would
    /// help, high on clean content. Filled by the second pass; nil if the oracle was unavailable.
    /// ⚠️ Blind to content identity and gameable by over-sharpening — never read alone either.
    public var nrScore: Double?
    /// **The detail axis.** Mean gradient magnitude on luma. Not a quality score at all — an amount of
    /// high-frequency content — and it is here because it is the axis on which V10's two defensible
    /// orderings disagreed. Compare against `UpscaleQualityAnchor.gradient` for the reference.
    public let gradient: Double
    public let seconds: Double
}

/// A control cell that is **not a method**: the reference itself and the degraded input.
///
/// 🔑 **Without these the no-reference numbers are uninterpretable.** An NR score has no natural zero —
/// 0.62 means nothing until you know what the untouched master and the crushed input score on the same
/// scale. The reference is the ceiling; the low-res input is the floor the upscalers start from. A
/// method scoring *above* the reference is the signature of over-sharpening, which is exactly the way
/// NR-IQA is known to be gameable, and it is only visible with the anchor in the table.
public struct UpscaleQualityAnchor: Sendable {
    public let imageName: String
    public let contentClass: String
    /// `reference` (regime-independent, recorded once as `—`) or the regime whose low-res input this is.
    public let regime: String
    /// `reference` or `low-input`.
    public let kind: String
    public var nrScore: Double?
    public let gradient: Double
    /// Pixel side — the low-res input is 512², the reference 2048². ⚠️ Gradient magnitude is
    /// resolution-dependent, so the low-input gradient is **not** comparable to a 2048² row.
    public let side: Int
}

/// Everything one sweep produced.
public struct UpscaleQualityResult: Sendable {
    public var rows: [UpscaleQualityRow]
    public var anchors: [UpscaleQualityAnchor]
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
    /// ⚠️ Order matters for the receipt, not for the result: `jpeg` runs last so that a sweep killed
    /// part-way still carries both clean regimes complete rather than three regimes half-done.
    public static let allRegimes = ["bicubic", "lanczos", "jpeg"]
    static var regimes: [String] {
        ProcessInfo.processInfo.environment["FORGE_BENCH_REGIME"].map { [$0] } ?? allRegimes
    }
    /// JPEG quality for the `jpeg` regime. **0.3 is chosen to be visibly damaged, not subtly so** — the
    /// question is whether a learned tier can *undo* an artefact, and a quality level where the artefact
    /// is near-invisible cannot discriminate (#43c: never choose from a regime that cannot discriminate).
    static var jpegQuality: Double {
        ProcessInfo.processInfo.environment["FORGE_BENCH_JPEG_Q"].flatMap(Double.init) ?? 0.3
    }

    // MARK: - Resolution-envelope diagnosis

    /// Is SeedVR2's stills path broken by **output resolution** or by **scale factor**?
    ///
    /// 🔑 **Why this is the right question.** `SeedVR2ImageRefiner` already does the thing it looked like
    /// it was missing: CoreImage Lanczos pre-upscale → one-step diffusion refine at 1:1 → LAB colour
    /// transfer. That is the *same* recipe as the video path. The one difference is that the video path
    /// refines in `tileSize`² tiles via `MLXTileProcessor(scale: 1)` while the image path does a single
    /// monolithic pass — and the refiner's own header admits it: *"V1 is single-pass (no tiling):
    /// activation is resolution-driven… Tiling for very large images is a future lever."*
    ///
    /// SeedVR2 uses **windowed attention** (`window: [4,3,3]`) over a latent that is 1/8 the pixel size.
    /// A 2048² single pass is a 256²-token latent — far outside anything the model was trained on, which
    /// would show up as global softness and misregistration rather than as noise. **So the two cells that
    /// separate the hypotheses are (256→1024, ×4) and (512→1024, ×2): same output resolution, different
    /// scale factor. If both are fine and 2048 is broken, it is the envelope, and the fix is tiling —
    /// machinery the package already ships.**
    public static func diagnoseResolutionEnvelope(imageURL: URL,
                                                  service: ForgeEngineService,
                                                  progress: @Sendable (String) -> Void) async {
        guard let master = loadCGImage(imageURL) else {
            progress("  could not load \(imageURL.lastPathComponent)"); return
        }
        // (referenceSide, lowSide, scale) — the reference is always a LOSSLESS crop at that side.
        let cells = [(512, 128, 4), (1024, 256, 4), (1024, 512, 2), (2048, 512, 4)]
        for (refSide, lowSide, scale) in cells {
            guard let reference = centreCrop(master, side: refSide) else {
                progress("  \(refSide)² crop unavailable"); continue
            }
            guard let low = resampleCG(reference, side: lowSide, quality: .high) else { continue }
            let tag = "\(lowSide)to\(refSide)_x\(scale)"

            if let up = resampleCG(low, side: refSide, quality: .high),
               let score = try? MediaMeasure.ssimulacra2(reference: reference, distorted: up) {
                progress(String(format: "  %@  bicubic     %8.2f", tag.padding(toLength: 16, withPad: " ", startingAt: 0), score))
            }
            do {
                let m = try await service.measureUpscale(low, model: .upscaleQuality, scale: scale)
                dump(m.image, as: "envelope__\(tag)__SeedVR2")
                dump(reference, as: "envelope__\(tag)__reference")
                let score = (try? MediaMeasure.ssimulacra2(reference: reference, distorted: m.image)) ?? .nan
                let stats = statistics(m.image)
                // ⚠️ `%@`, never `%s`. `String(format:)` expects a C string for `%s` and prints garbage
                // bytes for a Swift String — the first envelope run emitted "ÿ±ÊÍ" for every label.
                progress(String(format: "  %@  SeedVR2-3B  %8.2f   out %d×%d  %.1fs  sigma %.1f",
                                tag.padding(toLength: 16, withPad: " ", startingAt: 0),
                                score, m.outputWidth, m.outputHeight, m.runSeconds,
                                stats?.sigma[1] ?? -1))
            } catch {
                progress("  \(tag) SeedVR2 FAILED — \(error)")
            }
        }
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

    /// Score every method on every image in every regime, with **both** oracles.
    ///
    /// 🔑 **Two passes, and the split is a memory decision, not a style one.** Pass 1 runs the
    /// transforms and the full-reference oracle; pass 2 runs the no-reference oracle over what pass 1
    /// dumped. Interleaving them would drive SeedVR2 (≈8.4 GB attributed) and SigLIP2 (≈1.2 GB
    /// resident) in and out of the engine's admission window once per cell — and SeedVR2's warm load is
    /// ~10 s against a ~34 s run, so alternating would inflate the sweep by roughly a third to measure
    /// nothing extra. Each model loads once.
    ///
    /// ⚠️ **Pass 2 reads the dumped PNGs, not the in-memory images, and that is deliberate.** PNG is
    /// lossless so the pixels are identical, but scoring the file means the no-reference number is
    /// attached to *the artifact a human can open and look at* — which is the whole method rule V10
    /// exists to enforce (a content-independent score is a defect signature, and only the dumped image
    /// can tell you which it was). Without `FORGE_BENCH_DUMP` there is no pass 2, and the receipt says so.
    ///
    /// - Parameter learnedArms: the engine-backed tiers to include. Pass `[]` to run the classical arms
    ///   alone (fast, and enough to establish the floor).
    public static func run(imageURLs: [URL],
                           service: ForgeEngineService?,
                           learnedArms: [ForgeModel],
                           progress: (@Sendable (String) -> Void)? = nil) async -> UpscaleQualityResult {
        var rows: [UpscaleQualityRow] = []
        var anchors: [UpscaleQualityAnchor] = []

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
            let referenceGradient = meanGradient(reference)
            anchors.append(UpscaleQualityAnchor(imageName: name, contentClass: contentClass,
                                                regime: "—", kind: "reference",
                                                nrScore: nil, gradient: referenceGradient,
                                                side: referenceSide))
            progress?(String(format: "  %@  REFERENCE gradient %.3f", name, referenceGradient))

            for regime in regimes {
                guard let low = downsample(reference, to: lowSide, regime: regime) else {
                    progress?("  \(name)/\(regime): SKIP — downsample failed")
                    continue
                }
                dump(low, as: "\(name)__\(regime)__low")
                anchors.append(UpscaleQualityAnchor(imageName: name, contentClass: contentClass,
                                                    regime: regime, kind: "low-input",
                                                    nrScore: nil, gradient: meanGradient(low),
                                                    side: lowSide))

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
                    let gradient = meanGradient(up)
                    if let score = try? MediaMeasure.ssimulacra2(reference: reference, distorted: up) {
                        rows.append(UpscaleQualityRow(imageName: name, contentClass: contentClass,
                                                      regime: regime, method: method,
                                                      score: score, nrScore: nil,
                                                      gradient: gradient, seconds: seconds))
                        progress?(String(format: "  %@/%@/%@  FR %.2f  grad %.3f  (%.3fs)",
                                         name, regime, method, score, gradient, seconds))
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
                        let gradient = meanGradient(m.image)
                        if let score = try? MediaMeasure.ssimulacra2(reference: reference, distorted: m.image) {
                            rows.append(UpscaleQualityRow(imageName: name, contentClass: contentClass,
                                                          regime: regime, method: model.modelName,
                                                          score: score, nrScore: nil,
                                                          gradient: gradient, seconds: m.runSeconds))
                            progress?(String(format: "  %@/%@/%@  FR %.2f  grad %.3f  (%.2fs)",
                                             name, regime, model.modelName, score, gradient, m.runSeconds))
                        }
                    } catch {
                        progress?("  \(name)/\(regime)/\(model.modelName): FAILED — \(error)")
                    }
                }
            }
        }

        var result = UpscaleQualityResult(rows: rows, anchors: anchors)
        await scoreNoReference(&result, service: service, progress: progress)
        return result
    }

    // MARK: - Pass 2 · the no-reference oracle

    /// Fill `nrScore` on every row and anchor by re-reading what pass 1 dumped.
    ///
    /// 🚨 **The anchors are scored first and on purpose.** An NR score has no natural zero: 0.62 is
    /// neither good nor bad until the untouched master and the degraded input have been put on the same
    /// scale. Scoring them first means that if the run is killed part-way there is still a usable frame
    /// of reference for whatever rows did complete.
    static func scoreNoReference(_ result: inout UpscaleQualityResult,
                                 service: ForgeEngineService?,
                                 progress: (@Sendable (String) -> Void)?) async {
        guard let service else { return }
        guard let directory = dumpDirectory else {
            progress?("  V10-Q NR pass SKIPPED — FORGE_BENCH_DUMP unset, so there are no artifacts to score")
            return
        }
        if await service.packageID(for: .qualityScore) == nil {
            progress?("  V10-Q NR pass SKIPPED — SigLIP2-IQA did not register: "
                      + (await service.outcome(for: .qualityScore)?.failure ?? "no failure recorded"))
            return
        }
        progress?("  ── pass 2: SigLIP2 NR-IQA (no-reference, 0–1, high = clean) ──")

        func score(_ file: String) async -> Double? {
            let url = directory.appendingPathComponent(file + ".png")
            guard let image = loadCGImage(url) else {
                progress?("      NR: missing artifact \(file).png")
                return nil
            }
            do { return try await service.scoreNoReferenceQuality(image) }
            catch {
                progress?("      NR FAILED on \(file) — \(error)")
                return nil
            }
        }

        for i in result.anchors.indices {
            let a = result.anchors[i]
            let file = a.kind == "reference" ? "\(a.imageName)__reference"
                                             : "\(a.imageName)__\(a.regime)__low"
            if let nr = await score(file) {
                result.anchors[i].nrScore = nr
                progress?(String(format: "  %@/%@/%@  NR %.4f", a.imageName, a.regime, a.kind, nr))
            }
        }
        for i in result.rows.indices {
            let r = result.rows[i]
            if let nr = await score("\(r.imageName)__\(r.regime)__\(r.method)") {
                result.rows[i].nrScore = nr
                progress?(String(format: "  %@/%@/%@  NR %.4f", r.imageName, r.regime, r.method, nr))
            }
        }
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

    /// Produce the low-res input for a regime.
    ///
    /// 🔑 **`jpeg` is bicubic-down *then* a lossy round trip**, in that order. Compressing the 4K master
    /// first and then downsampling would average most of the artefact away — the downsample is a
    /// low-pass filter, and it would quietly restore the very thing the regime exists to test.
    static func downsample(_ image: CGImage, to side: Int, regime: String) -> CGImage? {
        switch regime {
        case "lanczos": return try? lanczosMPS(image, targetSide: side)
        case "jpeg":
            guard let small = resampleCG(image, side: side, quality: .high) else { return nil }
            return jpegRoundTrip(small, quality: jpegQuality)
        default: return resampleCG(image, side: side, quality: .high)
        }
    }

    /// Encode to JPEG at `quality` and decode back — a real codec round trip, so the damage is real
    /// blocking/ringing/chroma-subsampling and not a simulation of it.
    static func jpegRoundTrip(_ image: CGImage, quality: Double) -> CGImage? {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, "public.jpeg" as CFString, 1, nil)
        else { return nil }
        CGImageDestinationAddImage(dest, image,
                                   [kCGImageDestinationLossyCompressionQuality: quality] as CFDictionary)
        guard CGImageDestinationFinalize(dest),
              let source = CGImageSourceCreateWithData(data as CFData, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    /// Mean gradient magnitude on luma — **an amount of high-frequency content, not a quality score.**
    ///
    /// Central differences, Rec.709 luma, interior pixels only. It exists because V10's two defensible
    /// orderings disagreed on exactly this axis: SSIMULACRA2 ranked Lanczos first and SeedVR2 last while
    /// gradient ranked them in the opposite order, and neither is wrong — one measures *agreement with
    /// the reference*, the other measures *how much detail is present at all*. ⚠️ Resolution-dependent:
    /// only compare figures computed at the same pixel side.
    public static func meanGradient(_ image: CGImage) -> Double {
        let w = image.width, h = image.height
        guard w > 2, h > 2 else { return .nan }
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        guard let space = CGColorSpace(name: CGColorSpace.sRGB),
              let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: w * 4, space: space,
                                  bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return .nan }
        ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))

        var luma = [Double](repeating: 0, count: w * h)
        for i in 0..<(w * h) {
            let p = i * 4
            luma[i] = 0.2126 * Double(bytes[p]) + 0.7152 * Double(bytes[p + 1]) + 0.0722 * Double(bytes[p + 2])
        }
        var total = 0.0
        for y in 1..<(h - 1) {
            for x in 1..<(w - 1) {
                let i = y * w + x
                let dx = luma[i + 1] - luma[i - 1]
                let dy = luma[i + w] - luma[i - w]
                total += (dx * dx + dy * dy).squareRoot() * 0.5
            }
        }
        return total / Double((w - 2) * (h - 2))
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

    /// Per-image rows first, then per-class means, **for each oracle separately**.
    ///
    /// 🚨 **Never the grand mean alone** — the whole hypothesis is that photo and graphic behave
    /// differently, so an average across them would hide the finding. 🚨 **And never one oracle alone:**
    /// the full-reference and no-reference tables are printed as three separate blocks per regime
    /// (fidelity, no-reference, detail) precisely so a reader has to see where they disagree. Any future
    /// edit that fuses them into a single "quality" column is deleting the result, not tidying it.
    public static func report(_ result: UpscaleQualityResult) -> [String] {
        guard !result.rows.isEmpty else { return ["V10-Q — no rows produced"] }
        var lines = ["V10-Q — two oracles + the detail axis, 512→2048 (×4) vs a lossless 2048² reference",
                     "        FR = SSIMULACRA2 (full-reference, higher better, ~90 = visually fine)",
                     "        NR = SigLIP2 NR-IQA (no-reference, 0–1, high = clean / restoration does not pay)",
                     "        grad = mean gradient magnitude (detail PRESENT, not detail CORRECT)"]

        let methods = Array(Set(result.rows.map(\.method))).sorted()
        let width = 14

        func table(_ title: String,
                   _ inRegime: [UpscaleQualityRow],
                   _ format: @Sendable (UpscaleQualityRow) -> String?) {
            lines.append("")
            lines.append("    \(title)")
            lines.append("      " + "image".padding(toLength: 22, withPad: " ", startingAt: 0)
                         + methods.map { $0.padding(toLength: width, withPad: " ", startingAt: 0) }.joined())
            for name in Array(Set(inRegime.map(\.imageName))).sorted() {
                var row = "      " + name.padding(toLength: 22, withPad: " ", startingAt: 0)
                for method in methods {
                    let cell = inRegime.first { $0.imageName == name && $0.method == method }
                    row += (cell.flatMap(format) ?? "—").padding(toLength: width, withPad: " ", startingAt: 0)
                }
                lines.append(row)
            }
            for contentClass in ["photo", "graphic"] {
                let subset = inRegime.filter { $0.contentClass == contentClass }
                guard !subset.isEmpty else { continue }
                var row = "      " + "MEAN \(contentClass)".padding(toLength: 22, withPad: " ", startingAt: 0)
                for method in methods {
                    let values = subset.filter { $0.method == method }.compactMap { cell -> Double? in
                        guard let text = format(cell) else { return nil }
                        return Double(text)
                    }
                    row += (values.isEmpty ? "—"
                            : String(format: "%.4f", values.reduce(0, +) / Double(values.count)))
                        .padding(toLength: width, withPad: " ", startingAt: 0)
                }
                lines.append(row)
            }
        }

        for regime in allRegimes {
            let inRegime = result.rows.filter { $0.regime == regime }
            guard !inRegime.isEmpty else { continue }
            lines.append("")
            lines.append("  ══ degradation: \(regime)"
                         + (regime == "jpeg" ? String(format: " (q=%.2f — real artefact, a correct answer EXISTS)", jpegQuality)
                                             : " (clean downsample — nothing to undo)") + " ══")

            table("── FULL-REFERENCE · SSIMULACRA2 ──", inRegime) { String(format: "%.2f", $0.score) }
            table("── NO-REFERENCE · SigLIP2 NR-IQA ──", inRegime) { $0.nrScore.map { String(format: "%.4f", $0) } }
            table("── DETAIL · mean gradient magnitude ──", inRegime) { String(format: "%.4f", $0.gradient) }

            // The anchors belong under their regime: they are what the columns above are relative to.
            let refAnchors = result.anchors.filter { $0.kind == "reference" }
            let lowAnchors = result.anchors.filter { $0.kind == "low-input" && $0.regime == regime }
            if !refAnchors.isEmpty || !lowAnchors.isEmpty {
                lines.append("")
                lines.append("    ── ANCHORS (not methods — the ceiling and the floor) ──")
                lines.append("      " + "image".padding(toLength: 22, withPad: " ", startingAt: 0)
                             + "ref NR".padding(toLength: width, withPad: " ", startingAt: 0)
                             + "ref grad".padding(toLength: width, withPad: " ", startingAt: 0)
                             + "low NR".padding(toLength: width, withPad: " ", startingAt: 0)
                             + "low grad(512²)".padding(toLength: width + 2, withPad: " ", startingAt: 0))
                for name in Array(Set(inRegime.map(\.imageName))).sorted() {
                    let reference = refAnchors.first { $0.imageName == name }
                    let low = lowAnchors.first { $0.imageName == name }
                    lines.append("      " + name.padding(toLength: 22, withPad: " ", startingAt: 0)
                        + (reference?.nrScore.map { String(format: "%.4f", $0) } ?? "—").padding(toLength: width, withPad: " ", startingAt: 0)
                        + (reference.map { String(format: "%.4f", $0.gradient) } ?? "—").padding(toLength: width, withPad: " ", startingAt: 0)
                        + (low?.nrScore.map { String(format: "%.4f", $0) } ?? "—").padding(toLength: width, withPad: " ", startingAt: 0)
                        + (low.map { String(format: "%.4f", $0.gradient) } ?? "—"))
                }
            }
        }

        // ⚠️ Stated, not left to the reader: whether the two oracles picked the same winner. This is the
        // one derived number in the receipt, and it is a COUNT OF DISAGREEMENTS, never a blend.
        lines.append("")
        lines.append("  ══ do the two oracles agree on a winner? ══")
        var agree = 0, disagree = 0, undecidable = 0
        for regime in allRegimes {
            for contentClass in ["photo", "graphic"] {
                let subset = result.rows.filter { $0.regime == regime && $0.contentClass == contentClass }
                guard !subset.isEmpty else { continue }
                func best(_ value: @Sendable (UpscaleQualityRow) -> Double?) -> String? {
                    var totals: [String: (sum: Double, n: Int)] = [:]
                    for row in subset {
                        guard let v = value(row) else { continue }
                        let t = totals[row.method] ?? (0, 0)
                        totals[row.method] = (t.sum + v, t.n + 1)
                    }
                    return totals.max { ($0.value.sum / Double($0.value.n)) < ($1.value.sum / Double($1.value.n)) }?.key
                }
                let fr = best { $0.score }
                let nr = best { $0.nrScore }
                let detail = best { $0.gradient }
                let verdict: String
                if let fr, let nr { verdict = fr == nr ? "AGREE" : "DISAGREE"; if fr == nr { agree += 1 } else { disagree += 1 } }
                else { verdict = "n/a (NR pass did not run)"; undecidable += 1 }
                lines.append("    \(regime)/\(contentClass): FR→\(fr ?? "—")  NR→\(nr ?? "—")  detail→\(detail ?? "—")   \(verdict)")
            }
        }
        lines.append("    totals: \(agree) agree · \(disagree) disagree · \(undecidable) undecidable")
        lines.append("    ⚠️ A disagreement is a RESULT, not an error to resolve. FR asks 'is this the same")
        lines.append("       image?', NR asks 'does this look like a good image?' — on super-resolution those")
        lines.append("       are different questions and the honest answer reports both.")
        return lines
    }
}
