//
//  UpscaleFidelityLiveTests.swift
//  ForgeCoreTests
//
//  [VAL] — RealPLKSR (`.upscaleFidelity`, AB-T-0186 Step 1) through the REAL engine on corpus stills: register →
//  prepare → run via `measureUpscale`, the path the product ships. Env-gated (needs the GPU and the corpus), so it
//  never runs in CI:
//
//      FORGE_CORPUS=/Volumes/Satechi/Development/training-resources/Corpus [FORGE_FIDELITY_E2E_OUT=<dir>] \
//        [FORGE_FIDELITY_E2E_IMAGES=<a.png>:<b.png>] [FORGE_FIDELITY_E2E_ARMS=upscaleFidelity,upscale] \
//        swift test --build-system swiftbuild --filter UpscaleFidelityLiveTests
//
//  Default inputs: `derived/320/stills/RBC_photo.png` (320×569) and `derived/1080/stills/RBC_photo.png` (1080×1920 —
//  exactly the 1920×1080-pixel whole-frame ceiling the package's declared footprint was taken at). Per image it
//  asserts the output is 4× both ways with `appliedScale` 4, and that the pixels are sane: each channel's mean within
//  4 levels of the input's (a channel swap or a colour cast fails here), luma spread within 0.5–2× of the input's (a
//  flat or blown-out frame fails), and the output box-downsampled back onto the input grid agrees with the input at
//  ≥ 25 dB luma PSNR — a fidelity tier's own promise, and a generous floor for garbage, not a quality score.
//
//  The fast tier (`.upscale` — Real-ESRGAN in the AB-R-0365 runs, NERVE since 2026-09-28) runs the same inputs
//  afterwards as the cost reference. `FORGE_FIDELITY_E2E_ARMS=upscaleHEART,upscaleFidelity` runs HEART, the opt-in
//  fidelity variant (registered 2026-09-29), through the same path — its 34 MB fp16 file materializes from
//  mlx-community/HEART-fp16 into the attached store on the first prepare. Load and run are timed separately (see
//  `UpscaleDiagnostics`); `phys_footprint` is sampled after each run, and its peak is the PROCESS'S LIFETIME peak
//  (monotonic), which is why the fidelity arm runs first. Outputs and a JSON receipt land in the out dir (default:
//  the temp dir) — corpus media, never committed.
//

import Testing
import CoreGraphics
import Darwin
import Foundation
import ImageIO
import UniformTypeIdentifiers
@testable import ForgeCore

@MainActor
@Suite struct UpscaleFidelityLiveTests {
    nonisolated static let env = ProcessInfo.processInfo.environment

    @Test(.enabled(if: ProcessInfo.processInfo.environment["FORGE_CORPUS"] != nil
                     || ProcessInfo.processInfo.environment["FORGE_FIDELITY_E2E_IMAGES"] != nil))
    func fidelityTierRunsEndToEnd() async throws {
        let inputs: [URL] = Self.env["FORGE_FIDELITY_E2E_IMAGES"].map {
            $0.split(separator: ":").map { URL(fileURLWithPath: String($0)) }
        } ?? {
            let corpus = URL(fileURLWithPath: Self.env["FORGE_CORPUS"]!)
            return ["derived/320/stills/RBC_photo.png", "derived/1080/stills/RBC_photo.png"]
                .map { corpus.appendingPathComponent($0) }
        }()
        let arms: [ForgeModel] = (Self.env["FORGE_FIDELITY_E2E_ARMS"] ?? "upscaleFidelity,upscale")
            .split(separator: ",").compactMap { ForgeModel(rawValue: String($0)) }
        try #require(!arms.isEmpty && arms.allSatisfy { $0.capability == .imageUpscale },
                     "FORGE_FIDELITY_E2E_ARMS must name imageUpscale tiers")
        let outDir = URL(fileURLWithPath: Self.env["FORGE_FIDELITY_E2E_OUT"]
                         ?? FileManager.default.temporaryDirectory.appendingPathComponent("fidelity-e2e").path)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        let service = ForgeEngineService()
        // A model store so a WeightSourcing tier (HEART) can be materialized by the engine — the same seam the guarded
        // live test uses: FORGE_FIDELITY_E2E_STORE, else the guarded test's default; vendored tiers ignore it.
        let store = Self.env["FORGE_FIDELITY_E2E_STORE"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appending(path: "Library/Application Support/Forge/Models")
        if FileManager.default.fileExists(atPath: store.path) { await service.attachModelStore(root: store) }
        await service.registerAll()
        for arm in arms {
            try #require(service.packageID(for: arm) != nil,
                         "\(arm.modelName) must register: \(service.outcome(for: arm)?.failure ?? "no failure recorded")")
        }
        print("e2e: arms \(arms.map { "\($0.modelName) [\(service.packageID(for: $0)!.rawValue)]" })")
        // Whether the engine's MLX buffer-pool cap is in force in THIS process — the variable that separates a
        // package's working set from a retained pool when two processes disagree about the same run (AB-T-0019).
        print("e2e: MLX pool cap applied by the engine: \(service.engine.appliedGPUCacheLimitBytes.map { String(format: "%.2f GB", Double($0) / 1e9) } ?? "none")")

        var receipts: [[String: Any]] = []
        for arm in arms {
            for url in inputs {
                let name = url.deletingPathExtension().lastPathComponent
                let image = try #require(Self.loadImage(url), "could not decode \(url.path)")
                let m = try await service.measureUpscale(image, model: arm, scale: 4)
                let foot = Self.physFootprint()
                let pool = service.engine.gpuPoolSnapshot()

                #expect(m.outputWidth == image.width * 4 && m.outputHeight == image.height * 4,
                        "\(arm.modelName) \(name): \(m.outputWidth)×\(m.outputHeight) for ×4 of \(image.width)×\(image.height)")
                #expect(m.appliedScale == 4, "\(arm.modelName) \(name): appliedScale \(m.appliedScale)")

                let inStats = try #require(Self.stats(image))
                let outStats = try #require(Self.stats(m.image))
                let consistency = try #require(Self.downsampleConsistency(output: m.image, input: image, factor: 4))
                for c in 0..<3 {
                    #expect(abs(outStats.channelMeans[c] - inStats.channelMeans[c]) <= 4,
                            "\(arm.modelName) \(name): channel \(c) mean \(outStats.channelMeans[c]) vs input \(inStats.channelMeans[c])")
                }
                let spread = outStats.lumaStd / max(inStats.lumaStd, 1e-6)
                #expect(spread >= 0.5 && spread <= 2.0, "\(arm.modelName) \(name): luma spread ratio \(spread)")
                #expect(consistency >= 25, "\(arm.modelName) \(name): downsample consistency \(consistency) dB")

                let mpx = Double(m.outputWidth * m.outputHeight) / 1e6
                let tag = "\(name)_\(image.width)x\(image.height)__\(arm.rawValue)"   // the corpus sets share file names
                try Self.writePNG(m.image, to: outDir.appendingPathComponent("\(tag).png"))
                receipts.append([
                    "image": name, "arm": arm.rawValue, "model": arm.modelName,
                    "packageID": service.packageID(for: arm)!.rawValue,
                    "inputWidth": image.width, "inputHeight": image.height,
                    "outputWidth": m.outputWidth, "outputHeight": m.outputHeight, "appliedScale": m.appliedScale,
                    "loadSeconds": m.loadSeconds, "runSeconds": m.runSeconds,
                    "msPerOutputMpx": m.runSeconds * 1000 / mpx,
                    "inputLumaMean": inStats.lumaMean, "outputLumaMean": outStats.lumaMean,
                    "inputLumaStd": inStats.lumaStd, "outputLumaStd": outStats.lumaStd,
                    "inputChannelMeans": inStats.channelMeans, "outputChannelMeans": outStats.channelMeans,
                    "downsampleConsistencyDB": consistency,
                    "physFootprintBytes": foot?.current ?? 0, "physFootprintLifetimePeakBytes": foot?.peak ?? 0,
                    "mlxPoolActiveBytes": pool?.activeBytes ?? 0, "mlxPoolCacheBytes": pool?.cacheBytes ?? 0,
                    "mlxPoolPeakBytes": pool?.peakBytes ?? 0, "mlxPoolCapBytes": pool?.cacheLimitBytes ?? 0,
                ])
                print(String(format: "e2e: %@ %@ %d×%d → %d×%d ×%d  load %.2f s  run %.2f s (%.0f ms/Mpx out)  luma %.1f→%.1f σ %.1f→%.1f  consistency %.1f dB  phys %.2f GB (lifetime peak %.2f GB)",
                             arm.modelName, name, image.width, image.height, m.outputWidth, m.outputHeight, m.appliedScale,
                             m.loadSeconds, m.runSeconds, m.runSeconds * 1000 / mpx,
                             inStats.lumaMean, outStats.lumaMean, inStats.lumaStd, outStats.lumaStd, consistency,
                             Double(foot?.current ?? 0) / 1e9, Double(foot?.peak ?? 0) / 1e9))
                print(String(format: "e2e:   MLX pool after the run: active %.2f GB · cache %.2f GB · peak %.2f GB · cap %.2f GB",
                             Double(pool?.activeBytes ?? 0) / 1e9, Double(pool?.cacheBytes ?? 0) / 1e9,
                             Double(pool?.peakBytes ?? 0) / 1e9, Double(pool?.cacheLimitBytes ?? 0) / 1e9))
            }
        }
        let json = try JSONSerialization.data(withJSONObject: receipts, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: outDir.appendingPathComponent("receipt.json"))
        print("e2e: \(receipts.count) runs → \(outDir.path)")
    }

    // MARK: - helpers

    struct PixelStats { let channelMeans: [Double]; let lumaMean: Double; let lumaStd: Double }

    /// Draws into 8-bit sRGB RGBX so every input decodes onto one pixel grid, whatever its source layout.
    static func rgb(_ image: CGImage) -> (bytes: [UInt8], width: Int, height: Int)? {
        let w = image.width, h = image.height
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = bytes.withUnsafeMutableBytes { raw -> Bool in
            guard let ctx = CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                      space: CGColorSpace(name: CGColorSpace.sRGB)!,
                                      bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue) else { return false }
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        return drawn ? (bytes, w, h) : nil
    }

    /// Rec. 709 luma of gamma-encoded 8-bit values.
    static func luma(_ p: UnsafeBufferPointer<UInt8>, _ i: Int) -> Double {
        0.2126 * Double(p[i]) + 0.7152 * Double(p[i + 1]) + 0.0722 * Double(p[i + 2])
    }

    static func stats(_ image: CGImage) -> PixelStats? {
        guard let (bytes, w, h) = rgb(image) else { return nil }
        let n = Double(w * h)
        var sums = [0.0, 0.0, 0.0], ySum = 0.0, ySq = 0.0
        bytes.withUnsafeBufferPointer { p in
            for i in stride(from: 0, to: p.count, by: 4) {
                sums[0] += Double(p[i]); sums[1] += Double(p[i + 1]); sums[2] += Double(p[i + 2])
                let y = luma(p, i)
                ySum += y; ySq += y * y
            }
        }
        let mean = ySum / n
        return PixelStats(channelMeans: sums.map { $0 / n }, lumaMean: mean, lumaStd: (max(ySq / n - mean * mean, 0)).squareRoot())
    }

    /// Luma PSNR between the input and the output box-averaged back onto the input grid.
    static func downsampleConsistency(output: CGImage, input: CGImage, factor: Int) -> Double? {
        guard let (o, ow, _) = rgb(output), let (i, iw, ih) = rgb(input), ow == iw * factor else { return nil }
        var se = 0.0
        o.withUnsafeBufferPointer { op in
            i.withUnsafeBufferPointer { ip in
                for y in 0..<ih {
                    for x in 0..<iw {
                        var acc = 0.0
                        for dy in 0..<factor {
                            let row = (y * factor + dy) * ow
                            for dx in 0..<factor { acc += luma(op, (row + x * factor + dx) * 4) }
                        }
                        let d = acc / Double(factor * factor) - luma(ip, (y * iw + x) * 4)
                        se += d * d
                    }
                }
            }
        }
        let mse = se / Double(iw * ih)
        return mse == 0 ? .infinity : 10 * log10(255 * 255 / mse)
    }

    /// The process's `phys_footprint` now, and its lifetime peak — the governor's admission basis, sampled here in the
    /// test process rather than in-app (the in-app number carries the app's own overhead on top).
    static func physFootprint() -> (current: UInt64, peak: UInt64)? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return (info.phys_footprint, UInt64(info.ledger_phys_footprint_peak))
    }

    static func loadImage(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    static func writePNG(_ image: CGImage, to url: URL) throws {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }
}
