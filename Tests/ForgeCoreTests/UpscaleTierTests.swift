import Testing
import MLXToolKit
@testable import ForgeCore

/// AB-T-0184/0185: VOSR2 joins Real-ESRGAN and SeedVR2-3B as the THIRD `imageUpscale` registration — the generative
/// stills tier: its own `ForgeModel` case, its own display name, its own explicit PackageID (`vosr2-upscale`).
/// Pinned so the tiers never collapse onto one registration (the Restormer lesson, AB-L-0160).
@Suite struct UpscaleTierTests {
    @Test func generativeTierIsAThirdImageUpscaleRegistration() {
        #expect(ForgeModel.upscaleGenerative.capability == .imageUpscale)
        #expect(ForgeModel.upscaleGenerative.modelName == "VOSR2")
        #expect(ForgeModel.upscaleGenerative.rawValue != ForgeModel.upscaleQuality.rawValue)
        #expect(ForgeModel.allCases.contains(.upscaleGenerative))
    }

    /// Exactly three stills backers behind `imageUpscale`, all reachable from the case list a host iterates.
    @Test func imageUpscaleHasExactlyThreeBackers() {
        let backers = ForgeModel.allCases.filter { $0.capability == .imageUpscale }
        #expect(Set(backers) == Set([.upscale, .upscaleQuality, .upscaleGenerative]))
    }

    /// The guard adapter accepts either generative tier (it is main-actor isolated, like the service it wraps).
    @Test @MainActor func guardAdapterTakesEitherGenerativeTier() async {
        let service = ForgeEngineService()   // no registerAll — construction only
        let a = EngineTextGuardedUpscaler(service: service, generative: .upscaleGenerative)
        #expect(a.generative == .upscaleGenerative && a.base == .upscale)
        let b = EngineTextGuardedUpscaler(service: service)
        #expect(b.generative == .upscaleGenerative)   // VOSR2 is the default generative tier since v0.23.0 (§11.2)
        let c = EngineTextGuardedUpscaler(service: service, generative: .upscaleQuality)
        #expect(c.generative == .upscaleQuality)
    }
}
