//
//  UpscaleFastLiveTests.swift
//  ForgeCoreTests
//
//  [VAL] — NERVE (`.upscale`, the fast tier since AB-T-0186 Step 3) through the REAL engine on corpus stills: register
//  → prepare → run via `measureUpscale`, the path the product ships — at ×4 AND at ×2, which NERVE runs natively (the
//  `.fidelity` variant's ×2 route is the native 2x_NERVE_OTF_gan checkpoint). Env-gated (needs the GPU and the corpus),
//  so it never runs in CI:
//
//      FORGE_CORPUS=/Volumes/Satechi/Development/training-resources/Corpus [FORGE_FAST_E2E_OUT=<dir>] \
//        [FORGE_FAST_E2E_IMAGES=<a.png>:<b.png>] [FORGE_FAST_E2E_SCALES=4,2] [FORGE_FAST_E2E_ARMS=upscale] \
//        swift test --build-system swiftbuild --filter UpscaleFastLiveTests
//
//  Default inputs: `derived/320/stills/RBC_photo.png` (320×569) and `derived/1080/stills/RBC_photo.png` (1080×1920 —
//  on the package's 1920×1080-pixel whole-frame ceiling). Per image × scale it asserts the output is exactly `scale`×
//  both ways with `appliedScale == scale`, and that the pixels are sane — the same checks as `UpscaleFidelityLiveTests`
//  (whose helpers it reuses): channel means within 4 levels of the input's, luma spread within 0.5–2×, and the output
//  box-downsampled back onto the input grid within ≥ 25 dB luma PSNR of the input. Load and run are timed separately;
//  `phys_footprint` is sampled after each run (its peak is the process's LIFETIME peak). Outputs and a JSON receipt
//  land in the out dir (default: the temp dir) — corpus media, never committed.
//

import Testing
import CoreGraphics
import Foundation
@testable import ForgeCore

@MainActor
@Suite struct UpscaleFastLiveTests {
    nonisolated static let env = ProcessInfo.processInfo.environment

    @Test(.enabled(if: ProcessInfo.processInfo.environment["FORGE_CORPUS"] != nil
                     || ProcessInfo.processInfo.environment["FORGE_FAST_E2E_IMAGES"] != nil))
    func fastTierRunsEndToEnd() async throws {
        typealias H = UpscaleFidelityLiveTests
        let inputs: [URL] = Self.env["FORGE_FAST_E2E_IMAGES"].map {
            $0.split(separator: ":").map { URL(fileURLWithPath: String($0)) }
        } ?? {
            let corpus = URL(fileURLWithPath: Self.env["FORGE_CORPUS"]!)
            return ["derived/320/stills/RBC_photo.png", "derived/1080/stills/RBC_photo.png"]
                .map { corpus.appendingPathComponent($0) }
        }()
        let scales = (Self.env["FORGE_FAST_E2E_SCALES"] ?? "4,2").split(separator: ",").compactMap { Int($0) }
        let arms: [ForgeModel] = (Self.env["FORGE_FAST_E2E_ARMS"] ?? "upscale")
            .split(separator: ",").compactMap { ForgeModel(rawValue: String($0)) }
        try #require(!arms.isEmpty && arms.allSatisfy { $0.capability == .imageUpscale },
                     "FORGE_FAST_E2E_ARMS must name imageUpscale tiers")
        let outDir = URL(fileURLWithPath: Self.env["FORGE_FAST_E2E_OUT"]
                         ?? FileManager.default.temporaryDirectory.appendingPathComponent("fast-e2e").path)
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        let service = ForgeEngineService()
        await service.registerAll()
        for arm in arms {
            try #require(service.packageID(for: arm) != nil,
                         "\(arm.modelName) must register: \(service.outcome(for: arm)?.failure ?? "no failure recorded")")
        }
        print("e2e: arms \(arms.map { "\($0.modelName) [\(service.packageID(for: $0)!.rawValue)]" }) · scales \(scales)")

        var receipts: [[String: Any]] = []
        for arm in arms {
            for scale in scales {
                for url in inputs {
                    let name = url.deletingPathExtension().lastPathComponent
                    let image = try #require(H.loadImage(url), "could not decode \(url.path)")
                    let m = try await service.measureUpscale(image, model: arm, scale: scale)
                    let foot = H.physFootprint()

                    #expect(m.outputWidth == image.width * scale && m.outputHeight == image.height * scale,
                            "\(arm.modelName) \(name): \(m.outputWidth)×\(m.outputHeight) for ×\(scale) of \(image.width)×\(image.height)")
                    #expect(m.appliedScale == scale, "\(arm.modelName) \(name): appliedScale \(m.appliedScale) for ×\(scale)")

                    let inStats = try #require(H.stats(image))
                    let outStats = try #require(H.stats(m.image))
                    let consistency = try #require(H.downsampleConsistency(output: m.image, input: image, factor: scale))
                    for c in 0..<3 {
                        #expect(abs(outStats.channelMeans[c] - inStats.channelMeans[c]) <= 4,
                                "\(arm.modelName) \(name) ×\(scale): channel \(c) mean \(outStats.channelMeans[c]) vs input \(inStats.channelMeans[c])")
                    }
                    let spread = outStats.lumaStd / max(inStats.lumaStd, 1e-6)
                    #expect(spread >= 0.5 && spread <= 2.0, "\(arm.modelName) \(name) ×\(scale): luma spread ratio \(spread)")
                    #expect(consistency >= 25, "\(arm.modelName) \(name) ×\(scale): downsample consistency \(consistency) dB")

                    let mpx = Double(m.outputWidth * m.outputHeight) / 1e6
                    let tag = "\(name)_\(image.width)x\(image.height)__\(arm.rawValue)_x\(scale)"
                    try H.writePNG(m.image, to: outDir.appendingPathComponent("\(tag).png"))
                    receipts.append([
                        "image": name, "arm": arm.rawValue, "model": arm.modelName,
                        "packageID": service.packageID(for: arm)!.rawValue, "requestedScale": scale,
                        "inputWidth": image.width, "inputHeight": image.height,
                        "outputWidth": m.outputWidth, "outputHeight": m.outputHeight, "appliedScale": m.appliedScale,
                        "loadSeconds": m.loadSeconds, "runSeconds": m.runSeconds,
                        "msPerOutputMpx": m.runSeconds * 1000 / mpx,
                        "inputLumaMean": inStats.lumaMean, "outputLumaMean": outStats.lumaMean,
                        "inputLumaStd": inStats.lumaStd, "outputLumaStd": outStats.lumaStd,
                        "downsampleConsistencyDB": consistency,
                        "physFootprintBytes": foot?.current ?? 0, "physFootprintLifetimePeakBytes": foot?.peak ?? 0,
                    ])
                    print(String(format: "e2e: %@ %@ %d×%d → %d×%d ×%d  load %.2f s  run %.2f s (%.0f ms/Mpx out)  luma %.1f→%.1f σ %.1f→%.1f  consistency %.1f dB  phys %.2f GB (lifetime peak %.2f GB)",
                                 arm.modelName, name, image.width, image.height, m.outputWidth, m.outputHeight, m.appliedScale,
                                 m.loadSeconds, m.runSeconds, m.runSeconds * 1000 / mpx,
                                 inStats.lumaMean, outStats.lumaMean, inStats.lumaStd, outStats.lumaStd, consistency,
                                 Double(foot?.current ?? 0) / 1e9, Double(foot?.peak ?? 0) / 1e9))
                }
            }
        }
        let json = try JSONSerialization.data(withJSONObject: receipts, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: outDir.appendingPathComponent("receipt.json"))
        print("e2e: \(receipts.count) runs → \(outDir.path)")
    }
}
