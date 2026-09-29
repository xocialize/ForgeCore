//
//  UpscaleTierLiveTests.swift
//  ForgeCoreTests
//
//  [VAL] — Optimize's Fast/Best control (AB-T-0187) through the REAL engine: the Kit's `optimize` with
//  `EngineImageEnhancer`, NAFNet restore → NERVE (Fast) or RealPLKSR (Best), on a corpus still, at ×4 and ×2. Per arm it
//  asserts the delivered file is the asked scale both ways and that the receipt — struct and NDJSON — names the tier and
//  model that ran. Env-gated (GPU + corpus), so it never runs in CI:
//
//      FORGE_CORPUS=/Volumes/Satechi/Development/training-resources/Corpus [FORGE_TIER_E2E_IMAGE=<still>] \
//        swift test --build-system swiftbuild --filter UpscaleTierLiveTests
//
//  Default input: `derived/320/stills/RBC_photo.png` (320×569), small enough that the four arms take seconds. Wall times
//  are printed, not asserted: this is the wiring check; the in-app run is where the timings are receipted.
//

import Testing
import CoreGraphics
import Foundation
import ImageIO
import ForgeOptimizerKit
@testable import ForgeCore

@MainActor
@Suite struct UpscaleTierLiveTests {

    @Test(.enabled(if: ProcessInfo.processInfo.environment["FORGE_CORPUS"] != nil
                     || ProcessInfo.processInfo.environment["FORGE_TIER_E2E_IMAGE"] != nil))
    func fastAndBestRunTheirOwnModelsAndSayWhich() async throws {
        let env = ProcessInfo.processInfo.environment
        let src = env["FORGE_TIER_E2E_IMAGE"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: env["FORGE_CORPUS"]!).appendingPathComponent("derived/320/stills/RBC_photo.png")
        let source = try Self.pixelSize(src)

        let service = ForgeEngineService()
        await service.registerAll()
        let enhancer = EngineImageEnhancer(service: service)
        for tier in UpscaleTier.allCases {
            let verdict = await enhancer.availability(of: tier)
            try #require(verdict.isAvailable, "\(tier) must be available on the validating Mac: \(verdict)")
        }

        let out = FileManager.default.temporaryDirectory.appendingPathComponent("tier-e2e-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: out) }
        let forge = ForgeOptimizer(enhancer: enhancer)
        for (tier, model) in [(UpscaleTier.best, "RealPLKSR"), (.fast, "NERVE")] {
            for (factor, scale) in [(UpscaleFactor.x4, 4), (.x2, 2)] {
                let arm = out.appendingPathComponent("\(tier.rawValue)-x\(scale)")
                var result: OptimizeResult?
                for await r in try forge.optimize(.url(src), to: .directory(arm),
                                                  Options(enhance: .on, upscale: factor, upscaleTier: tier)) {
                    result = r
                }
                let r = try #require(result)
                guard case .file(let delivered) = r.output else {
                    Issue.record("\(tier) ×\(scale): expected a delivered file, got \(r.status)"); continue
                }
                let size = try Self.pixelSize(delivered)
                #expect(size.w == source.w * scale && size.h == source.h * scale,
                        "\(tier) ×\(scale): \(size.w)×\(size.h) from \(source.w)×\(source.h)")
                #expect(r.recipe.upscaled == scale)
                #expect(r.recipe.upscaleTier == tier)
                #expect(r.recipe.upscaleModel == model)
                #expect(enhancer.lastAppliedScale == scale)
                let line = try #require(ReceiptJSON.line(ReceiptJSON.result(r)))
                #expect(line.contains("\"upscale_model\":\"\(model)\""), "\(line)")
                #expect(line.contains("\"upscale_tier\":\"\(tier.rawValue)\""), "\(line)")
                print("tier-e2e: \(tier.rawValue) ×\(scale) → \(size.w)×\(size.h) · \(r.recipe) · \(String(format: "%.2f", r.elapsed)) s")
            }
        }
    }

    static func pixelSize(_ url: URL) throws -> (w: Int, h: Int) {
        let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
        let props = try #require(CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any])
        return (try #require(props[kCGImagePropertyPixelWidth] as? Int), try #require(props[kCGImagePropertyPixelHeight] as? Int))
    }
}
