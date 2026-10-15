import Testing
import CoreGraphics
import Foundation
import ForgeOptimizerKit
import Anime4KMetal
@testable import ForgeCore

/// AB-T-0294 / AB-D-0146 — the Instant tier: Anime4K's upscale-only VL on Metal shaders, outside the engine. It needs no
/// registration, so these run on a `ForgeEngineService` that registered nothing: if Instant reached the engine, they
/// would fail as "not registered".
@MainActor
@Suite struct InstantTierTests {

    @Test func instantHasNoEngineRegistration() {
        #expect(EngineImageEnhancer.model(for: .instant) == nil)
        for model in ForgeModel.allCases { #expect(EngineImageEnhancer.tier(for: model) != .instant) }
    }

    @Test func availableWithoutRegistrationWhereMetalRuns() async throws {
        try #require(Anime4KUpscaler.isAvailable, "needs a Metal device")
        let verdict = await EngineImageEnhancer(service: ForgeEngineService()).availability(of: .instant)
        #expect(verdict.isAvailable, "\(verdict)")
        #expect(verdict.model == "Anime4K VL")
    }

    /// ×2, ×4, ×6 and ×8 land on the exact size, report the Instant tier, and name the fit when one finished the pixels.
    @Test func upscaleReportsTheTierAndLandsOnTheExactSize() async throws {
        try #require(Anime4KUpscaler.isAvailable, "needs a Metal device")
        let enhancer = EngineImageEnhancer(service: ForgeEngineService())
        let image = try Self.drawing(width: 48, height: 36)
        for (factor, scale, name) in [(UpscaleFactor.x2, 2, "Anime4K VL"), (.x4, 4, "Anime4K VL"),
                                      (.x6, 6, "Anime4K VL ×8 → ×6 (Lanczos)"), (.x8, 8, "Anime4K VL")] {
            let out = try await enhancer.upscaleReporting(image, factor: factor, tier: .instant)
            #expect(out.image.width == 48 * scale && out.image.height == 36 * scale, "\(factor)")
            #expect(out.upscaleTier == .instant)
            #expect(out.upscaleModel == name, "\(factor)")
            #expect(enhancer.lastAppliedScale == scale)
        }
    }

    /// A 16-bit still stays 16-bit through the Instant tier.
    @Test func deepStillKeepsItsDepth() async throws {
        try #require(Anime4KUpscaler.isAvailable, "needs a Metal device")
        let out = try await EngineImageEnhancer(service: ForgeEngineService())
            .upscaleReporting(try Self.drawing(width: 40, height: 30, bits: 16), factor: .x2, tier: .instant)
        #expect(out.image.bitsPerComponent == 16)
    }

    @Test func planDoublesAndFitsAsAB_D_0119Does() {
        let small = ChainedUpscalePlan.Size(width: 1920, height: 1080)
        #expect(InstantUpscalePlan.make(requested: 2, source: small) == .init(requested: 2, doublings: 1, fit: nil, fitsUp: false))
        #expect(InstantUpscalePlan.make(requested: 4, source: small).doublings == 2)
        #expect(InstantUpscalePlan.make(requested: 8, source: small) == .init(requested: 8, doublings: 3, fit: nil, fitsUp: false))
        let six = InstantUpscalePlan.make(requested: 6, source: small)
        #expect(six.doublings == 3)
        #expect(six.fit == .init(width: 11520, height: 6480))
        #expect(!six.fitsUp)
        #expect(six.modelName("Anime4K VL") == "Anime4K VL ×8 → ×6 (Lanczos)")
        // A ×8 intermediate past the cap: ×4, then enlarged to ×6, and the name says so.
        let big = InstantUpscalePlan.make(requested: 6, source: .init(width: 3000, height: 3000))
        #expect(big.doublings == 2)
        #expect(big.fitsUp)
        #expect(big.modelName("Anime4K VL") == "Anime4K VL ×4 → ×6 (Lanczos, enlarged)")
    }

    static func drawing(width: Int, height: Int, bits: Int = 8) throws -> CGImage {
        let info = CGImageAlphaInfo.noneSkipLast.rawValue | (bits == 16 ? CGBitmapInfo.byteOrder16Little.rawValue : 0)
        let ctx = try #require(CGContext(data: nil, width: width, height: height, bitsPerComponent: bits, bytesPerRow: 0,
                                         space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: info))
        ctx.setFillColor(CGColor(srgbRed: 1, green: 0.95, blue: 0.85, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        ctx.setStrokeColor(CGColor(srgbRed: 0.1, green: 0.1, blue: 0.2, alpha: 1))
        ctx.setLineWidth(2)
        ctx.strokeEllipse(in: CGRect(x: 6, y: 5, width: width - 12, height: height - 10))
        return try #require(ctx.makeImage())
    }
}
