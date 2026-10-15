import Testing
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import ForgeOptimizerKit
import MLXServeCore
import MLXToolKit
@testable import ForgeCore

/// AB-T-0187 — the engine half of Optimize's Fast/Best control. `EngineImageEnhancer` maps `.fast → .upscale` (NERVE)
/// and `.best → .upscaleFidelity` (RealPLKSR) and nothing else, reports the tier and model of the registration that ran,
/// and answers `availability(of:)` from the governor's own arithmetic — so the UI can show Best as unavailable WITH ITS
/// REASON and the Kit can refuse it, never serve it with Fast. No weights or GPU needed: every refusal here happens
/// before a model loads, which is also what the tests assert.
@MainActor
@Suite struct UpscaleTierControlTests {

    @Test func fastRunsNERVEAndBestRunsRealPLKSR() {
        #expect(EngineImageEnhancer.model(for: .fast) == .upscale)
        #expect(EngineImageEnhancer.model(for: .best) == .upscaleFidelity)
        #expect(ForgeModel.upscale.modelName == "NERVE")
        #expect(ForgeModel.upscaleFidelity.modelName == "RealPLKSR")
        for tier in UpscaleTier.allCases {
            guard let model = EngineImageEnhancer.model(for: tier) else { continue }   // Instant has no registration
            #expect(EngineImageEnhancer.tier(for: model) == tier)
        }
    }

    /// The Generative tiers (AB-D-0111): `.generative` runs VOSR2, `.generativeClean` runs Nacre, and each maps back.
    @Test func generativeRunsVOSR2AndGenerativeCleanRunsNacre() {
        #expect(EngineImageEnhancer.model(for: .generative) == .upscaleGenerative)
        #expect(EngineImageEnhancer.model(for: .generativeClean) == .upscaleNacre)
        #expect(EngineImageEnhancer.tier(for: .upscaleGenerative) == .generative)
        #expect(EngineImageEnhancer.tier(for: .upscaleNacre) == .generativeClean)
    }

    /// A composite never ships under one name: the guard's route decides what the receipt calls the pixels.
    @Test func guardedResultsAreNamedByWhatMadeThePixels() {
        #expect(EngineImageEnhancer.guardedModelName(generative: .upscaleNacre, base: .upscale, route: .generative) == "Nacre")
        #expect(EngineImageEnhancer.guardedModelName(generative: .upscaleNacre, base: .upscale, route: .protectWords)
                == "Nacre + NERVE (protected words)")
        #expect(EngineImageEnhancer.guardedModelName(generative: .upscaleGenerative, base: .upscale, route: .keepBase)
                == "NERVE (legible document kept)")
    }

    /// Before registration both Generative tiers are unavailable with the registration reason — nothing stands in.
    @Test func unregisteredGenerativeTiersAreUnavailableWithTheReason() async {
        let enhancer = EngineImageEnhancer(service: ForgeEngineService())
        for tier in [UpscaleTier.generative, .generativeClean] {
            let verdict = await enhancer.availability(of: tier)
            #expect(!verdict.isAvailable)
            #expect(verdict.model == EngineImageEnhancer.model(for: tier)?.modelName)
            #expect(verdict.unavailableReason?.contains("is not registered") == true, "\(verdict)")
        }
    }

    /// HEART (the opt-in variant) and SeedVR2 stay off the control: exposing them is a product call.
    @Test func optInAndGenerativeTiersAreOffTheControl() {
        for model in [ForgeModel.upscaleHEART, .upscaleQuality] {
            #expect(EngineImageEnhancer.tier(for: model) == nil, "\(model.modelName) must not be reachable from Fast/Best")
        }
    }

    /// Before registration nothing is available, and the reason says so — a tier is never "available" by default.
    @Test func unregisteredTiersAreUnavailableWithTheReason() async {
        let enhancer = EngineImageEnhancer(service: ForgeEngineService())
        for tier in UpscaleTier.allCases where !tier.isVideoOnly && tier != .instant {
            let verdict = await enhancer.availability(of: tier)
            #expect(!verdict.isAvailable)
            #expect(verdict.model == EngineImageEnhancer.model(for: tier)?.modelName)
            #expect(verdict.unavailableReason?.contains("is not registered") == true, "\(verdict)")
        }
        // The whole-clip tier is refused as a video tier, registered or not — never "not registered" on a still.
        let live = await enhancer.availability(of: .liveAction)
        #expect(live.unavailableReason == UpscaleTier.liveActionStillReason)
    }

    /// A Mac whose engine budget holds exactly NERVE's declared working set: Fast is available, Best is not, and the
    /// reason carries the numbers. The budget is derived from the declarations, so the verdict depends on neither the
    /// machine running the test nor on today's figures (AB-T-0019 will re-baseline them).
    @Test func bestIsUnavailableWhenItsDeclaredWorkingSetExceedsTheBudget() async throws {
        let (service, fastPeak, bestPeak) = try await Self.serviceHoldingOnlyTheFastTier()
        let enhancer = EngineImageEnhancer(service: service)

        let fast = await enhancer.availability(of: .fast)
        #expect(fast.isAvailable, "\(fast)")
        #expect(fast.model == "NERVE")

        let best = await enhancer.availability(of: .best)
        #expect(!best.isAvailable)
        #expect(best.model == "RealPLKSR")
        #expect(best.unavailableReason == "RealPLKSR needs \(EngineImageEnhancer.gb(bestPeak)); the engine's memory "
                + "budget on this Mac is \(EngineImageEnhancer.gb(fastPeak))")
    }

    /// The Kit's gate over the real enhancer: Optimize asked for Best that this Mac cannot hold fails the item with the
    /// enhancer's reason — and nothing loads, so Fast cannot have stood in.
    @Test func optimizeRefusesBestItCannotRunAndLoadsNothing() async throws {
        let (service, _, _) = try await Self.serviceHoldingOnlyTheFastTier()
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("tier-control-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let src = dir.appendingPathComponent("input.png")
        try Self.writeGradientPNG(to: src, width: 48, height: 32)

        let forge = ForgeOptimizer(enhancer: EngineImageEnhancer(service: service))
        var result: OptimizeResult?
        for await r in try forge.optimize(.url(src), to: .directory(dir.appendingPathComponent("out")),
                                          Options(enhance: .on, upscale: .x4, upscaleTier: .best)) { result = r }
        let r = try #require(result)
        guard case .failed(let why) = r.status else {
            Issue.record("expected a failed item, got \(r.status)"); return
        }
        #expect(why.hasPrefix("upscale tier 'best' unavailable: RealPLKSR needs "), "\(why)")
        #expect(r.recipe.upscaleModel == nil, "no backer ran, so none is named")
        #expect(await service.engine.residentPackages.isEmpty, "the refusal came before any model loaded")
    }

    /// Admission refusals from `engine.run` read as reasons in the user's terms; anything else passes through untouched.
    @Test func governorRefusalsReadAsReasons() {
        #expect(EngineImageEnhancer.governorRefusal(
            EngineError.exceedsMemoryBudget(required: 12_550_000_000, budget: 11_800_000_000))
                == "it needs 12.6 GB and the budget on this Mac is 11.8 GB")
        #expect(EngineImageEnhancer.governorRefusal(
            EngineError.externalTenantsHoldMemory(required: 12_550_000_000, external: 4_000_000_000,
                                                 budget: 11_800_000_000))
                == "it needs 12.6 GB while other work in this app holds 4.0 GB of the 11.8 GB budget")
        #expect(EngineImageEnhancer.governorRefusal(EngineError.noPackage(.imageUpscale)) == nil)
        #expect(EngineImageEnhancer.governorRefusal(CancellationError()) == nil)
        let declined = EngineImageEnhancer.EnhanceError.declined(.upscaleFidelity, "it needs 12.6 GB and the budget on this Mac is 11.8 GB")
        #expect("\(declined)" == "RealPLKSR was declined by the engine's memory governor — it needs 12.6 GB and the "
                + "budget on this Mac is 11.8 GB")
    }

    // MARK: - Fixtures

    /// A registered service whose governor budget is exactly the fast tier's declared working set — read from a first
    /// service, so the fixture follows the declarations wherever re-baselining moves them.
    static func serviceHoldingOnlyTheFastTier() async throws
        -> (service: ForgeEngineService, fastPeak: UInt64, bestPeak: UInt64) {
        let probe = ForgeEngineService(governor: MemoryGovernor(budgetBytes: 1 << 50))
        await probe.registerAll()
        let fastPeak = try #require(await probe.admissibility(for: .upscale)).footprint
        let bestPeak = try #require(await probe.admissibility(for: .upscaleFidelity)).footprint
        try #require(bestPeak > fastPeak, "the fixture needs Best to declare more than Fast")

        let service = ForgeEngineService(governor: MemoryGovernor(budgetBytes: fastPeak))
        await service.registerAll()
        return (service, fastPeak, bestPeak)
    }

    static func writeGradientPNG(to url: URL, width w: Int, height h: Int) throws {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = try #require(CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                                         bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let grad = try #require(CGGradient(colorsSpace: cs,
                                           colors: [CGColor(red: 0.1, green: 0.2, blue: 0.85, alpha: 1),
                                                    CGColor(red: 0.95, green: 0.7, blue: 0.1, alpha: 1)] as CFArray,
                                           locations: [0, 1]))
        ctx.drawLinearGradient(grad, start: .zero, end: CGPoint(x: w, y: h), options: [])
        let image = try #require(ctx.makeImage())
        let dst = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dst, image, nil)
        try #require(CGImageDestinationFinalize(dst))
    }
}
