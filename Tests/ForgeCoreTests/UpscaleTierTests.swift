import Testing
import MLXToolKit
@testable import ForgeCore

/// AB-T-0184/0185: VOSR2 joins Real-ESRGAN and SeedVR2-3B as the THIRD `imageUpscale` registration — the generative
/// stills tier: its own `ForgeModel` case, its own display name, its own explicit PackageID (`vosr2-upscale`).
/// AB-T-0186 Step 1: RealPLKSR (`4x_webphoto_realplksr`) is the FOURTH — the fidelity tier, the "best" beside
/// Real-ESRGAN's "fast", explicit PackageID `realplksr-webphoto-upscale`.
/// Pinned so the tiers never collapse onto one registration (the Restormer lesson, AB-L-0160).
@Suite struct UpscaleTierTests {
    @Test func generativeTierIsAThirdImageUpscaleRegistration() {
        #expect(ForgeModel.upscaleGenerative.capability == .imageUpscale)
        #expect(ForgeModel.upscaleGenerative.modelName == "VOSR2")
        #expect(ForgeModel.upscaleGenerative.rawValue != ForgeModel.upscaleQuality.rawValue)
        #expect(ForgeModel.allCases.contains(.upscaleGenerative))
    }

    @Test func fidelityTierIsAFourthImageUpscaleRegistration() {
        #expect(ForgeModel.upscaleFidelity.capability == .imageUpscale)
        #expect(ForgeModel.upscaleFidelity.modelName == "RealPLKSR")
        let siblings: [ForgeModel] = [.upscale, .upscaleQuality, .upscaleGenerative]
        #expect(!siblings.map(\.rawValue).contains(ForgeModel.upscaleFidelity.rawValue))
        #expect(!siblings.map(\.modelName).contains(ForgeModel.upscaleFidelity.modelName))
        #expect(ForgeModel.allCases.contains(.upscaleFidelity))
    }

    /// Exactly four stills backers behind `imageUpscale`, all reachable from the case list a host iterates.
    @Test func imageUpscaleHasExactlyFourBackers() {
        let backers = ForgeModel.allCases.filter { $0.capability == .imageUpscale }
        #expect(Set(backers) == Set([.upscale, .upscaleFidelity, .upscaleQuality, .upscaleGenerative]))
    }

    /// Registration runs the license and device gates only — the weights are vendored and load on the first `run` —
    /// so this stays offline. RealPLKSR must clear `.permissiveOnly` + `.blocking` (Apache-2.0 weights and port) and
    /// land on its explicit id, which is the id a UI binds and a receipt names.
    @Test @MainActor func fidelityTierRegistersUnderItsExplicitID() async {
        let service = ForgeEngineService()
        await service.registerAll()
        #expect(service.outcome(for: .upscaleFidelity)?.failure == nil,
                "RealPLKSR did not register: \(service.outcome(for: .upscaleFidelity)?.failure ?? "")")
        #expect(service.packageID(for: .upscaleFidelity)?.rawValue == "realplksr-webphoto-upscale")
    }

    /// Optimize lists the fidelity tier beside Real-ESRGAN; availability stays keyed on `.upscale`, the tier the
    /// enhancer actually runs until the Kit's seam carries a fast-vs-best choice.
    @Test @MainActor func optimizeListsTheFidelityTier() {
        let caps = ForgeCapabilities(service: ForgeEngineService())   // no registerAll — construction only
        let optimize = caps.qualityTierAvailability.first { $0.capability == "Optimize" }
        #expect(optimize?.models == [.restore, .upscale, .upscaleFidelity])
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
