import Testing
import MLXToolKit
import MLXNERVE
import MLXHEART
import ForgeOptimizerKit
import MLXNacre
@testable import ForgeCore

/// AB-T-0184/0185: VOSR2 joins Real-ESRGAN and SeedVR2-3B as the THIRD `imageUpscale` registration — the generative
/// stills tier: its own `ForgeModel` case, its own display name, its own explicit PackageID (`vosr2-upscale`).
/// AB-T-0186 Step 1: RealPLKSR (`4x_webphoto_realplksr`) is the FOURTH — the fidelity tier, the "best" beside
/// Real-ESRGAN's "fast", explicit PackageID `realplksr-webphoto-upscale`.
/// AB-T-0186 Step 3: NERVE REPLACES Real-ESRGAN as `.upscale`, the fast tier — explicit PackageID `nerve-upscale`.
/// AB-T-0189 / AB-A-0128: HEART is the FIFTH — the OPT-IN fidelity variant behind RealPLKSR (the crossover AB-R-0379 gave
/// RealPLKSR DISTS at 12/12 rungs; the operator chose the opt-in slot), explicit PackageID `heart-upscale`, reachable but
/// on no default route. AB-A-0133: Nacre v1 is the SIXTH — the OPT-IN provenance-clean generative variant beside VOSR2,
/// explicit PackageID `nacre-upscale`, reachable but on no default route. Pinned so the tiers never collapse onto one
/// registration (the Restormer lesson, AB-L-0160).
@Suite struct UpscaleTierTests {
    @Test func fastTierIsNERVE() {
        #expect(ForgeModel.upscale.capability == .imageUpscale)
        #expect(ForgeModel.upscale.modelName == "NERVE")
        let siblings: [ForgeModel] = [.upscaleFidelity, .upscaleQuality, .upscaleGenerative, .upscaleHEART, .upscaleNacre]
        #expect(!siblings.map(\.modelName).contains(ForgeModel.upscale.modelName))
    }

    /// Registration is offline (vendored weights load on the first `run`): NERVE must clear `.permissiveOnly` +
    /// `.blocking` (Apache-2.0 weights and port) and land on its explicit id, the one a UI binds and a receipt names.
    @Test @MainActor func fastTierRegistersUnderItsExplicitID() async {
        let service = ForgeEngineService()
        await service.registerAll()
        #expect(service.outcome(for: .upscale)?.failure == nil,
                "NERVE did not register: \(service.outcome(for: .upscale)?.failure ?? "")")
        #expect(service.packageID(for: .upscale)?.rawValue == "nerve-upscale")
    }

    /// ForgeCore registers the package defaults; these are the defaults it relies on — `.fidelity` (kept after the
    /// crossover AB-R-0379, whose DISTS pick `.sharp` failed its own PSNR guard — operator decision AB-D-0107) on the
    /// fp16 lane, with a native ×2 route so HD → 4K never builds 8K.
    @Test func fastTierDefaultsAreThePlansChoice() {
        let c = NERVEConfiguration()
        #expect(c.variant == .fidelity)
        #expect(c.precision == .fp16)
        #expect(c.variant.checkpoint2x != nil)
    }

    @Test func generativeTierIsAThirdImageUpscaleRegistration() {
        #expect(ForgeModel.upscaleGenerative.capability == .imageUpscale)
        #expect(ForgeModel.upscaleGenerative.modelName == "VOSR2")
        #expect(ForgeModel.upscaleGenerative.rawValue != ForgeModel.upscaleQuality.rawValue)
        #expect(ForgeModel.allCases.contains(.upscaleGenerative))
    }

    @Test func fidelityTierIsAFourthImageUpscaleRegistration() {
        #expect(ForgeModel.upscaleFidelity.capability == .imageUpscale)
        #expect(ForgeModel.upscaleFidelity.modelName == "RealPLKSR")
        let siblings: [ForgeModel] = [.upscale, .upscaleQuality, .upscaleGenerative, .upscaleHEART, .upscaleNacre]
        #expect(!siblings.map(\.rawValue).contains(ForgeModel.upscaleFidelity.rawValue))
        #expect(!siblings.map(\.modelName).contains(ForgeModel.upscaleFidelity.modelName))
        #expect(ForgeModel.allCases.contains(.upscaleFidelity))
    }

    /// Exactly seven stills backers behind `imageUpscale`, all reachable from the case list a host iterates.
    @Test func imageUpscaleHasExactlySevenBackers() {
        let backers = ForgeModel.allCases.filter { $0.capability == .imageUpscale }
        #expect(Set(backers) == Set([.upscale, .upscaleFidelity, .upscaleQuality, .upscaleGenerative, .upscaleHEART,
                                     .upscaleNacre, .upscaleInstant]))
    }

    /// HEART is the opt-in fidelity VARIANT: its own case, name and id, behind `imageUpscale` — and on NO default route:
    /// RealPLKSR stays `.upscaleFidelity` and Optimize's "best".
    @Test func heartIsTheOptInFidelityVariant() {
        #expect(ForgeModel.upscaleHEART.capability == .imageUpscale)
        #expect(ForgeModel.upscaleHEART.modelName == "HEART")
        let siblings: [ForgeModel] = [.upscale, .upscaleFidelity, .upscaleQuality, .upscaleGenerative, .upscaleNacre]
        #expect(!siblings.map(\.rawValue).contains(ForgeModel.upscaleHEART.rawValue))
        #expect(!siblings.map(\.modelName).contains(ForgeModel.upscaleHEART.modelName))
        #expect(ForgeModel.upscaleFidelity.modelName == "RealPLKSR")
    }

    /// Registration is offline (the engine materializes the 34 MB file on first prepare, not here): HEART must clear
    /// `.permissiveOnly` + `.blocking` (Apache-2.0 weights and port) and land on its explicit id.
    @Test @MainActor func heartRegistersUnderItsExplicitID() async {
        let service = ForgeEngineService()
        await service.registerAll()
        #expect(service.outcome(for: .upscaleHEART)?.failure == nil,
                "HEART did not register: \(service.outcome(for: .upscaleHEART)?.failure ?? "")")
        #expect(service.packageID(for: .upscaleHEART)?.rawValue == "heart-upscale")
        #expect(service.packageID(for: .upscaleHEART) != service.packageID(for: .upscaleFidelity))
    }

    /// ForgeCore registers HEART's package defaults; these are the defaults it relies on — `.fidelity` on the fp16 lane
    /// (mlx-heart-swift PORTING-SPEC H3), the shipping repo per lane, and the H5 geometry left to the core.
    @Test func heartDefaultsAreThePackages() {
        let c = HEARTConfiguration()
        #expect(c.variant == .fidelity)
        #expect(c.effectiveQuant == .fp16)
        #expect(c.repo == HEARTConfiguration.fp16Repo)
        #expect(c.wholeFrameMaxPixels == nil && c.inputTileSize == nil && c.tileOverlap == nil)
    }

    /// Opt-in means no default route lists it: Optimize's tier list is unchanged by the fifth backer.
    @Test @MainActor func heartIsOnNoDefaultRoute() {
        let caps = ForgeCapabilities(service: ForgeEngineService())   // no registerAll — construction only
        for tier in caps.qualityTierAvailability {
            #expect(!tier.models.contains(.upscaleHEART), "\(tier.capability) lists the opt-in variant")
        }
    }

    /// Nacre is the opt-in provenance-clean GENERATIVE variant: its own case, name and id, behind `imageUpscale` — and
    /// VOSR2 stays `.upscaleGenerative`, the guard's default generative tier.
    @Test func nacreIsTheOptInGenerativeVariant() {
        #expect(ForgeModel.upscaleNacre.capability == .imageUpscale)
        #expect(ForgeModel.upscaleNacre.modelName == "Nacre")
        let siblings: [ForgeModel] = [.upscale, .upscaleFidelity, .upscaleQuality, .upscaleGenerative, .upscaleHEART]
        #expect(!siblings.map(\.rawValue).contains(ForgeModel.upscaleNacre.rawValue))
        #expect(!siblings.map(\.modelName).contains(ForgeModel.upscaleNacre.modelName))
        #expect(ForgeModel.upscaleGenerative.modelName == "VOSR2")
    }

    /// Registration is offline (the engine materializes the 348 MB fp16 lane on first prepare, not here): Nacre must
    /// clear `.permissiveOnly` + `.blocking` (Apache-2.0 weights and port) and land on its explicit id.
    @Test @MainActor func nacreRegistersUnderItsExplicitID() async {
        let service = ForgeEngineService()
        await service.registerAll()
        #expect(service.outcome(for: .upscaleNacre)?.failure == nil,
                "Nacre did not register: \(service.outcome(for: .upscaleNacre)?.failure ?? "")")
        #expect(service.packageID(for: .upscaleNacre)?.rawValue == "nacre-upscale")
        #expect(service.packageID(for: .upscaleNacre) != service.packageID(for: .upscaleGenerative))
    }

    /// ForgeCore registers Nacre's package defaults; these are the defaults it relies on — the fp16 lane, the 128/32
    /// tiling the in-app footprint was declared at (AB-R-0399), a fixed seed, and the shipping repo.
    @Test func nacreDefaultsAreThePackages() {
        let c = NacreConfiguration()
        #expect(c.effectiveQuant == .fp16)
        #expect(c.tileSize == nil && c.tileOverlap == nil)   // nil = the package's measured 128/32
        #expect(NacreConfiguration.repo == "xocialize/nacre-v1-mlx")
        #expect(c.peakActivationBytesHint == NacreConfiguration.declaredActivationBytes(quant: .fp16, tile: 128))
    }

    /// Nacre is reachable as a user-chosen tier (`.generativeClean`, AB-D-0111) and never a default route: the
    /// default tier stays `.fast` (NERVE), and only Optimize lists it.
    @Test @MainActor func nacreIsAUserChosenTierNeverADefault() {
        #expect(Options().upscaleTier == .fast)
        #expect(EngineImageEnhancer.model(for: .fast) == .upscale)
        let caps = ForgeCapabilities(service: ForgeEngineService())   // no registerAll — construction only
        for tier in caps.qualityTierAvailability where tier.capability != "Optimize" {
            #expect(!tier.models.contains(.upscaleNacre), "\(tier.capability) lists the generative variant")
        }
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
        // FlashVSR joined 2026-10-04 as the Live action video tier — listed, availability per clip (`videoUpscaler`).
        // VOSR2 and Nacre joined 2026-10-07 as the Generative stills tiers (AB-D-0111) — availability per tier.
        #expect(optimize?.models == [.restore, .upscale, .upscaleFidelity, .upscaleGenerative, .upscaleNacre,
                                     .upscaleVideoLiveAction])
    }

    /// The guard adapter accepts any generative tier (it is main-actor isolated, like the service it wraps).
    @Test @MainActor func guardAdapterTakesEitherGenerativeTier() async {
        let service = ForgeEngineService()   // no registerAll — construction only
        let a = EngineTextGuardedUpscaler(service: service, generative: .upscaleGenerative)
        #expect(a.generative == .upscaleGenerative && a.base == .upscale)
        let b = EngineTextGuardedUpscaler(service: service)
        #expect(b.generative == .upscaleGenerative)   // VOSR2 is the default generative tier since v0.23.0 (§11.2)
        let c = EngineTextGuardedUpscaler(service: service, generative: .upscaleQuality)
        #expect(c.generative == .upscaleQuality)
        let d = EngineTextGuardedUpscaler(service: service, generative: .upscaleNacre)   // the provenance-clean choice
        #expect(d.generative == .upscaleNacre && d.base == .upscale)
    }

    /// The guard's default policy is the one each pair EARNED: Nacre word protection on (generative best in none of four
    /// measured cells), everything else MaskKit's all-opt-in default — and an explicit policy always wins.
    @Test @MainActor func guardDefaultsToEachPairsMeasuredPolicy() {
        let service = ForgeEngineService()   // no registerAll — construction only
        let nacre = EngineTextGuardedUpscaler(service: service, generative: .upscaleNacre)
        #expect(nacre.policy.wordProtection && !nacre.policy.documentGate)
        for tier in [ForgeModel.upscaleGenerative, .upscaleQuality] {
            let p = EngineTextGuardedUpscaler(service: service, generative: tier).policy
            #expect(!p.wordProtection && !p.documentGate, "\(tier.modelName) has earned no opt-in route")
        }
        let raw = EngineTextGuardedUpscaler(service: service, policy: .default, generative: .upscaleNacre)
        #expect(!raw.policy.wordProtection && !raw.policy.documentGate)
    }
}
