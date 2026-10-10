//
//  ChainedUpscaleLiveTests.swift
//  ForgeCoreTests
//
//  [VAL] — ×6 and ×8 through the REAL engine (AB-D-0119): the Kit's `optimize` with `EngineImageEnhancer`, Best at
//  ×6 (RealPLKSR ×4 → NERVE ×2 → Lanczos) and Fast at ×8 (NERVE ×4 → NERVE ×2) on a corpus still. Asserts the delivered
//  size is the asked factor exactly, and that the receipt — struct and NDJSON — names both models and the fit.
//  Env-gated (GPU + corpus), never in CI:
//
//      FORGE_CORPUS=/Volumes/Satechi/Development/training-resources/Corpus [FORGE_TIER_E2E_IMAGE=<still>] \
//        swift test --build-system swiftbuild --filter ChainedUpscaleLiveTests
//

import Testing
import CoreGraphics
import Foundation
import ImageIO
import ForgeOptimizerKit
@testable import ForgeCore

@MainActor
@Suite struct ChainedUpscaleLiveTests {
    @Test(.enabled(if: ProcessInfo.processInfo.environment["FORGE_CORPUS"] != nil
                     || ProcessInfo.processInfo.environment["FORGE_TIER_E2E_IMAGE"] != nil))
    func sixAndEightRunTheChainAndSayWhich() async throws {
        let env = ProcessInfo.processInfo.environment
        let src = env["FORGE_TIER_E2E_IMAGE"].map { URL(fileURLWithPath: $0) }
            ?? URL(fileURLWithPath: env["FORGE_CORPUS"]!).appendingPathComponent("derived/320/stills/RBC_photo.png")
        let source = try UpscaleTierLiveTests.pixelSize(src)
        let service = ForgeEngineService()
        await service.registerAll()
        let enhancer = EngineImageEnhancer(service: service)
        for tier in [UpscaleTier.best, .fast] {
            let verdict = await enhancer.availability(of: tier)
            try #require(verdict.isAvailable, "\(tier) must be available on the validating Mac: \(verdict)")
        }
        let out = FileManager.default.temporaryDirectory.appendingPathComponent("chain-e2e-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: out) }
        let forge = ForgeOptimizer(enhancer: enhancer)
        let arms: [(UpscaleTier, UpscaleFactor, Int, String)] = [
            (.best, .x6, 6, "RealPLKSR ×4 + NERVE ×2 → ×6 (Lanczos)"),
            (.fast, .x8, 8, "NERVE ×4 + NERVE ×2"),
        ]
        for (tier, factor, scale, name) in arms {
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
            let size = try UpscaleTierLiveTests.pixelSize(delivered)
            #expect(size.w == source.w * scale && size.h == source.h * scale,
                    "\(tier) ×\(scale): \(size.w)×\(size.h) from \(source.w)×\(source.h)")
            #expect(r.recipe.upscaled == scale)
            #expect(r.recipe.upscaleRequested == nil, "a met request is quiet: \(String(describing: r.recipe.upscaleRequested))")
            #expect(r.recipe.upscaleTier == tier)
            #expect(r.recipe.upscaleModel == name, "\(String(describing: r.recipe.upscaleModel))")
            #expect(enhancer.lastAppliedScale == scale)
            let line = try #require(ReceiptJSON.line(ReceiptJSON.result(r)))
            #expect(line.contains("\"upscaled\":\(scale)"), "\(line)")
            #expect(line.contains("NERVE ×2"), "\(line)")
            print("chain-e2e: \(tier.rawValue) ×\(scale) → \(size.w)×\(size.h) · \(r.recipe) · \(String(format: "%.2f", r.elapsed)) s")
        }
    }
}
