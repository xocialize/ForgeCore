import Testing
import CoreGraphics
import Foundation
import ImageIO
import ForgeOptimizerKit
import MLXServeCore
@testable import ForgeCore

/// AB-T-0195 — the engine half of a `.quality` conform. `EngineImageEnhancer.upscaleReporting` runs the tier's one
/// registration with no NAFNet restore (a conform is resize glue), refuses exactly as `enhanceReporting` does, and
/// reports the registration that ran. The offline cases need no weights or GPU: every refusal happens before a model
/// loads, which is also what they assert.
@MainActor
@Suite struct ConformQualityRouteTests {

    /// The adapter has an upscale-only path, so the Kit's "no upscale-only path" refusal is gone; an unregistered
    /// tier is refused with the registration's own reason instead.
    @Test func anUnregisteredTierRefusesTheConformWithTheRegistrationReason() async throws {
        let forge = ForgeOptimizer(enhancer: EngineImageEnhancer(service: ForgeEngineService()))
        do {
            let r = try await forge.conform(try Self.gradient(48, 32), to: MediaSpec(size: .fit(maxWidth: 96, maxHeight: 96)),
                                            quality: .quality(.fast))
            Issue.record("expected a refusal, got \(r)")
        } catch ForgeError.upscaleTierUnavailable(let tier, let why) {
            #expect(tier == .fast)
            #expect(why.hasPrefix("NERVE is not registered"), "\(why)")
        }
    }

    /// A Mac whose budget holds only NERVE: a Best conform is refused with the budget reason, and nothing loads, so
    /// Fast cannot have stood in.
    @Test func aBestConformThisMacCannotHoldIsRefusedAndLoadsNothing() async throws {
        let (service, _, _) = try await UpscaleTierControlTests.serviceHoldingOnlyTheFastTier()
        let forge = ForgeOptimizer(enhancer: EngineImageEnhancer(service: service))
        do {
            let r = try await forge.conform(try Self.gradient(48, 32), to: MediaSpec(size: .exact(width: 192, height: 128)),
                                            quality: .quality(.best))
            Issue.record("expected a refusal, got \(r)")
        } catch ForgeError.upscaleTierUnavailable(let tier, let why) {
            #expect(tier == .best)
            #expect(why.hasPrefix("RealPLKSR needs "), "\(why)")
        }
        #expect(await service.engine.residentPackages.isEmpty, "the refusal came before any model loaded")
    }

    /// Called directly, the adapter refuses an unregistered tier and a whole-clip tier with its own errors.
    @Test func upscaleReportingRefusesUnregisteredAndVideoOnlyTiers() async throws {
        let enhancer = EngineImageEnhancer(service: ForgeEngineService())
        let image = try Self.gradient(16, 16)
        do {
            _ = try await enhancer.upscaleReporting(image, factor: .x2, tier: .best)
            Issue.record("expected EnhanceError.unavailable")
        } catch EngineImageEnhancer.EnhanceError.unavailable(let model, _) {
            #expect(model == .upscaleFidelity)
        }
        do {
            _ = try await enhancer.upscaleReporting(image, factor: .x2, tier: .liveAction)
            Issue.record("expected EnhanceError.videoOnlyTier")
        } catch EngineImageEnhancer.EnhanceError.videoOnlyTier(let tier) {
            #expect(tier == .liveAction)
        }
    }

    /// [VAL] through the real engine: a corpus still conformed by each tier at ×2 and ×4. Each arm must name the model
    /// that ran, measure the model's factor, land exactly on the spec, and load only that tier's upscaler — NAFNet
    /// restore never becomes resident. Env-gated (GPU + corpus):
    ///
    ///     FORGE_CORPUS=/Volumes/Satechi/Development/training-resources/Corpus \
    ///       swift test --build-system swiftbuild --filter ConformQualityRouteTests
    @Test(.enabled(if: ProcessInfo.processInfo.environment["FORGE_CORPUS"] != nil))
    func eachTierRunsItsOwnModelAndNeverTheRestore() async throws {
        let corpus = URL(fileURLWithPath: ProcessInfo.processInfo.environment["FORGE_CORPUS"]!)
        let src = corpus.appendingPathComponent("derived/320/stills/RBC_photo.png")
        let image = try #require(CGImageSourceCreateWithURL(src as CFURL, nil)
            .flatMap { CGImageSourceCreateImageAtIndex($0, 0, nil) })

        for (tier, model) in [(UpscaleTier.fast, "NERVE"), (.best, "RealPLKSR")] {
            let service = ForgeEngineService()
            await service.registerAll()
            let restoreID = try #require(service.packageID(for: .restore))
            let forge = ForgeOptimizer(enhancer: EngineImageEnhancer(service: service))
            // ×2 (fit to 2× the width) and ×4 (exact, 3× wide — the smallest covering factor is 4).
            let arms: [(MediaSpec, Int, Int, Int)] = [
                (MediaSpec(size: .fit(maxWidth: image.width * 2, maxHeight: image.height * 2)), 2,
                 image.width * 2, image.height * 2),
                (MediaSpec(size: .exact(width: image.width * 3, height: image.height * 2)), 4,
                 image.width * 3, image.height * 2),
            ]
            for (spec, factor, w, h) in arms {
                let start = Date()
                let r = try await forge.conform(image, to: spec, quality: .quality(tier))
                print("conform \(tier.rawValue) ×\(factor): \(image.width)×\(image.height) → \(r) in "
                      + String(format: "%.2f s", Date().timeIntervalSince(start)))
                #expect(r.upscaleTier == tier, "\(r)")
                #expect(r.upscaleModel == model, "\(r)")
                #expect(r.modelScale == factor, "\(r)")
                #expect(r.image.width == w && r.image.height == h, "\(r)")
            }
            let resident = await service.engine.residentPackages
            #expect(resident[restoreID] == nil, "a conform must never load the NAFNet restore")
            let tierModel = try #require(EngineImageEnhancer.model(for: tier))
            #expect(resident[try #require(service.packageID(for: tierModel))] != nil)
        }
    }

    static func gradient(_ w: Int, _ h: Int) throws -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = try #require(CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                                         bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let grad = try #require(CGGradient(colorsSpace: cs,
                                           colors: [CGColor(red: 0.1, green: 0.2, blue: 0.85, alpha: 1),
                                                    CGColor(red: 0.95, green: 0.7, blue: 0.1, alpha: 1)] as CFArray,
                                           locations: [0, 1]))
        ctx.drawLinearGradient(grad, start: .zero, end: CGPoint(x: w, y: h), options: [])
        return try #require(ctx.makeImage())
    }
}
