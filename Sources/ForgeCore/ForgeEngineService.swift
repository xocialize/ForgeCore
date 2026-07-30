//
//  ForgeEngineService.swift
//  ForgeCore
//
//  The single owner of the MLX engine, the model store, and every package registration.
//
//  The app this was extracted from spread this across six adapters: each lazily called `useModelStore`
//  and then registered its own package on first use. That made the engine-wide store a race between
//  adapters (whoever ran first won), hid registration failures inside the first inference, and meant a
//  license refusal surfaced as "the Colorize button doesn't work" rather than as a fact about the model.
//
//  Here registration is one eager, ordered pass: every package's outcome is recorded, and the adapters
//  are handed a `PackageID` that is already known-good.
//

import Foundation
import Observation
import OSLog
import MLXServeCore
import MLXToolKit
import MLXNAFNet
import MLXRealESRGAN
import MLXSeedVR2
import MLXBiRefNet
import MLXSigLIP2IQA
import MLXDDColor
import MLXInpaint
import MLXSEARAFT
import MLXEdgeTAM

let log = Logger(subsystem: "com.xocialize.ForgeCore", category: "ForgeCore")

/// The nine model packages Forge registers. One case per registration, *not* per capability —
/// Optimize alone uses two (restore + upscale), and one flow registration feeds four consumers.
///
/// 🔑 **`upscale` and `upscaleQuality` are two registrations behind ONE capability**, which is the
/// engine's multi-package-per-`PackageID` pattern and the first place Forge uses it: Real-ESRGAN is the
/// fast tier, SeedVR2-3B the quality tier, and the planner picks. Adding SeedVR2 does **not** displace
/// Real-ESRGAN — `MODEL-CATALOG.md`'s standing rule is that when SeedVR2 isn't viable (memory, chip,
/// file) `videoUpscale` routes to per-frame Real-ESRGAN: *same tool, lighter backer.*
///
/// 🚨 **`qualityScore` is not a transform — it is the second oracle, and it is here because its absence
/// was a measurable defect.** `CLAUDE.md`'s standing rule is *two* oracles: SigLIP2 NR-IQA gates
/// **whether to restore** (no reference needed on a degraded source), SSIMULACRA2 gates **how hard to
/// compress** (full-reference). ForgeCore linked only the second, so every comparison it could express
/// was a fidelity comparison — and V10 paid for that: a full-reference metric penalises a *generative*
/// upscaler for inventing plausible-but-different detail more than it penalises an interpolator for
/// inventing none, which ranked Lanczos above both learned tiers while the detail ordering was exactly
/// inverted. **The two oracles disagree by construction. Report both; never average them.**
public enum ForgeModel: String, CaseIterable, Sendable {
    case restore        // NAFNet
    case upscale        // Real-ESRGAN — the fast tier
    case upscaleQuality // SeedVR2-3B (int8) — the quality tier, same capability
    case matte          // BiRefNet
    case colorize       // DDColor
    case inpaint        // LaMa / MI-GAN
    case flow           // SEA-RAFT
    case promptMask     // EdgeTAM
    case qualityScore   // SigLIP2 NR-IQA — the no-reference oracle, not a transform

    public var modelName: String {
        switch self {
        case .restore:    return "NAFNet"
        case .upscale:    return "Real-ESRGAN"
        case .upscaleQuality: return "SeedVR2-3B"
        case .matte:      return "BiRefNet"
        case .colorize:   return "DDColor"
        case .inpaint:    return "LaMa / MI-GAN"
        case .flow:       return "SEA-RAFT"
        case .promptMask: return "EdgeTAM"
        case .qualityScore: return "SigLIP2-IQA"
        }
    }

    /// The engine capability this registration backs — use it to bind a progress view
    /// (`MLXEngineUI.ModelStateView(monitor:capability:)`) in a consuming app.
    public var capability: Capability {
        switch self {
        case .restore:    return .imageRestore
        case .upscale:    return .imageUpscale
        // ⚠️ Deliberately `.imageUpscale`, not `.videoUpscale`. The package declares BOTH contracts, and
        // the still path is the one Forge's upscale surface consumes today. Its video contract stays
        // available for a clip-level seam (`CLAUDE.md` V4c) that no app wires yet.
        case .upscaleQuality: return .imageUpscale
        case .matte:      return .matting
        case .colorize:   return .imageColorize
        case .inpaint:    return .imageInpaint
        case .flow:       return .opticalFlow
        case .promptMask: return .promptSegment
        case .qualityScore: return .imageQualityScore
        }
    }
}

/// What happened when a package was registered. A failure here is a fact worth showing, not an error to
/// retry: under `.permissiveOnly` + `.blocking` a `licenseRejected` means the model's weights are not
/// shippable, and `ineligible` means this Mac cannot run it. Neither improves by trying again.
public struct RegistrationOutcome: Sendable {
    public let model: ForgeModel
    public let packageID: PackageID?
    public let failure: String?

    public var isRegistered: Bool { packageID != nil }
}

@MainActor
@Observable
public final class ForgeEngineService {

    /// The engine.
    ///
    /// ⚠️ **`licenseEnforcement: .blocking` is load-bearing and must not be dropped.** As of contract
    /// 1.28.0 (engine v0.37.0) C7/C8 are *declaration* requirements, not refusals: the default
    /// `.advisory` **registers** a package with non-commercial weights and merely records a
    /// `LicenseAdvisory`. Forge is a commercial product, so it refuses instead.
    ///
    /// Nothing catches a regression here at compile time — the parameter is defaulted, so deleting it
    /// still builds and silently loosens every consuming app. Constructing the engine in exactly one
    /// place, in the shared package, is the mitigation. Runtime tell that enforcement has slipped: a
    /// `[License]` log line per advisory.
    public let engine: MLXServeEngine

    /// Registration result per model, in `ForgeModel.allCases` order once `registerAll()` has run.
    public private(set) var outcomes: [RegistrationOutcome] = []

    /// The store root actually handed to the engine — for display, and for diagnosing "it downloaded
    /// into the wrong place".
    public private(set) var attachedStoreRoot: URL?

    private var packageIDs: [ForgeModel: PackageID] = [:]

    /// Which NAFNet checkpoint backs `imageRestore`. **Default `.signage`, by measurement.**
    ///
    /// ⚠️ **The variant names mislead.** `.signage` is not signage-specific — it is a width-24
    /// checkpoint that behaves as a *conservative general* restore. `.siddWidth64` trains on SIDD's
    /// five smartphone sensors and is badly out of domain on anything else.
    ///
    /// Measured (BRIDGE-064) as SSIMULACRA2 of restored-vs-original — how far the restore step moves
    /// already-clean content, where a low score is damage rather than repair:
    ///
    /// | corpus | `.siddWidth64` | `.signage` |
    /// |---|---|---|
    /// | 10 signage stills | mean 64.07 / worst 56.81 | mean 82.97 / worst 80.14 |
    /// | 4 real photographs (Canon·Sony·Nikon·Ricoh RAW) | **mean 42.78 / worst 23.30** | mean 83.16 / worst 70.04 |
    ///
    /// `.siddWidth64` lands every signage still below Forge's own **aggressive** floor (70) and reaches
    /// **23.3** on a photograph — gross departure from an untouched source. `.signage` scores the same
    /// on both corpora (82.97 / 83.16), which is what rules out "it just suits signage".
    ///
    /// Independently corroborated: NIND measured `.siddWidth64` at **−1.48 dB at ISO 1600 — worse than
    /// not denoising** on DSLR content. Different corpus, different metric, same verdict.
    public let restoreVariant: NAFNetVariant

    /// - Parameters:
    ///   - policy: license gate. Leave at `.permissiveOnly` for a shipping product; a research build
    ///     may widen it.
    ///   - enforcement: leave `.blocking`. See the note on `engine`.
    ///   - restoreVariant: see `restoreVariant`. The default changed to `.signage` on evidence.
    public init(policy: LicensePolicy = .permissiveOnly,
                enforcement: LicenseEnforcement = .blocking,
                restoreVariant: NAFNetVariant = .signage) {
        self.engine = MLXServeEngine(policy: policy, licenseEnforcement: enforcement)
        self.restoreVariant = restoreVariant
    }

    // MARK: - Model store

    /// Point the engine at the models folder.
    ///
    /// **Must precede every `register()`** — the engine stamps the store root onto a `ModelStorable`
    /// configuration at registration time, so a package registered before this call downloads into the
    /// default cache instead of the chosen folder.
    ///
    /// `root` is a plain URL by design: owning the folder pick and its security-scoped bookmark is the
    /// app's job (`MLXEngineUI.ModelStorageModel` does it), which keeps this package headless.
    ///
    /// - Returns: false when no folder is available. Recoverable, not fatal — a consumer can run, let
    ///   the user choose a folder, then call this and `registerAll()` again.
    @discardableResult
    public func attachModelStore(root: URL?) async -> Bool {
        await engine.useModelStore(ModelStore(root: root))
        attachedStoreRoot = root
        if let root {
            log.notice("Model store attached: \(root.path, privacy: .public)")
            return true
        }
        log.error("No models folder — weights would land in the default cache")
        return false
    }

    // MARK: - Registration

    /// Register every package, recording each outcome. Registration is cheap: it runs the license
    /// and device-eligibility gates only; weights load lazily on the first `run`.
    public func registerAll() async {
        var results: [RegistrationOutcome] = []
        for model in ForgeModel.allCases {
            results.append(await register(model))
        }
        outcomes = results

        let failed = results.filter { !$0.isRegistered }
        if failed.isEmpty {
            log.notice("All \(results.count, privacy: .public) model packages registered")
        } else {
            for outcome in failed {
                log.error("\(outcome.model.modelName, privacy: .public) did not register: \(outcome.failure ?? "unknown", privacy: .public)")
            }
        }

        if let applied = await engine.appliedGPUCacheLimitBytes {
            log.notice("GPU buffer-pool cap applied by the engine: \(applied, privacy: .public) bytes")
        } else {
            log.notice("GPU buffer-pool cap not applied (unmanaged policy, or MLX's Metal device is unavailable here)")
        }
    }

    private func register(_ model: ForgeModel) async -> RegistrationOutcome {
        do {
            let id: PackageID
            switch model {
            case .restore:
                id = try await engine.register(NAFNetRestorePackage.registration,
                                               configuration: NAFNetConfiguration(variant: restoreVariant))
            case .upscale:
                id = try await engine.register(RealESRGANUpscalePackage.registration,
                                               configuration: RealESRGANConfiguration(variant: .general))
            case .upscaleQuality:
                // int8 is the validated default and **near-lossless** — the port measures it at ~50.3 dB
                // / cos 0.9999 against fp16, at ~4.7 GB vs ~7.5 GB. (⚠️ An earlier version of this comment
                // said fp16 "collapses to ~22.7 dB". That was a misreading: **int4** is the one that
                // degrades to 22.7 dB, and it is deliberately not offered. The choice is right; the stated
                // reason was wrong, and it mattered — it made quantization a suspect when V10 found the
                // stills path failing, which cost a diagnostic detour.)
                // ⚠️ **Tile geometry is env-overridable for diagnostics only** (`FORGE_SEEDVR2_TILE`,
                // `FORGE_SEEDVR2_OVERLAP`, `FORGE_SEEDVR2_WHOLEFRAME`). V10 established that the stills
                // path must tile above ~1 MP output, and then that the *residual* error is concentrated
                // in FLAT regions — 4.23 MAE vs 1.74 for an in-envelope single pass, while edge error is
                // unchanged. That is the overlap blend: neighbouring tiles reconstruct slightly different
                // DC levels from different border context, and feathering two disagreeing levels leaves a
                // ramp across every seam. 🔑 The two levers are **bigger tiles** (fewer seams, and V10
                // showed 1024² is still in-regime so 256 was needlessly small) and **more overlap** (32/256
                // = 12.5%, where tiled diffusion normally wants 25–50%). Sweeping beats guessing — three
                // hypotheses were already wrong here.
                let env = ProcessInfo.processInfo.environment
                let tile = env["FORGE_SEEDVR2_TILE"].flatMap(Int.init) ?? 256
                let overlap = env["FORGE_SEEDVR2_OVERLAP"].flatMap(Int.init) ?? 32
                let wholeFrame = env["FORGE_SEEDVR2_WHOLEFRAME"].flatMap(Int.init) ?? (1024 * 1024)
                id = try await engine.register(SeedVR2UpscalePackage.registration,
                                               configuration: SeedVR2Configuration(
                                                   quant: .int8,
                                                   tileSize: tile,
                                                   tileOverlap: overlap,
                                                   imageWholeFramePixels: wholeFrame))
            case .matte:
                id = try await engine.register(BiRefNetPackage.registration,
                                               configuration: BiRefNetConfiguration())
            case .colorize:
                id = try await engine.register(DDColorPackage.registration,
                                               configuration: DDColorConfiguration())
            case .inpaint:
                id = try await engine.register(InpaintPackage.registration,
                                               configuration: InpaintConfiguration())
            case .flow:
                id = try await engine.register(SEARAFTOpticalFlowPackage.registration,
                                               configuration: SEARAFTConfiguration(variant: .spring, iters: 4))
            case .promptMask:
                id = try await engine.register(EdgeTAMPackage.registration,
                                               configuration: EdgeTAMConfiguration())
            case .qualityScore:
                // `maxPatches` is a coverage/cost knob, not a quality knob: the scorer mean-pools up to
                // N native-scale 224² patches. ⚠️ **The default 8 is too few for a 2048² frame** — 8
                // patches of 224² cover 401 k of 4.19 M pixels, under a tenth of the image, and which
                // tenth is then a sampling artefact rather than a property of the upscaler. 32 covers
                // ~38% and keeps the run sub-second. Stated here because a no-reference score that
                // silently looked at 10% of the picture would be exactly the kind of plausible wrong
                // number V10 exists to catch.
                let patches = ProcessInfo.processInfo.environment["FORGE_IQA_PATCHES"].flatMap(Int.init) ?? 32
                id = try await engine.register(SigLIP2IQAPackage.registration,
                                               configuration: SigLIP2IQAConfiguration(maxPatches: patches))
            }
            packageIDs[model] = id
            return RegistrationOutcome(model: model, packageID: id, failure: nil)
        } catch {
            return RegistrationOutcome(model: model, packageID: nil, failure: Self.describe(error))
        }
    }

    /// Name *why* a registration failed, in the user's terms. `EngineError`'s payloads carry the detail
    /// (which license layer, which device dimension); dropping it turns an actionable fact into
    /// "something went wrong".
    private static func describe(_ error: Error) -> String {
        guard let engineError = error as? EngineError else { return String(describing: error) }
        switch engineError {
        case .licenseRejected(let gate):
            return "license not admitted under the active policy — \(gate)"
        case .ineligible(let eligibility):
            return "this Mac cannot run it — \(eligibility)"
        default:
            return String(describing: engineError)
        }
    }

    // MARK: - Lookup

    /// The registered package backing `model`, or nil if it failed to register. Adapters read this per
    /// call; nil means the capability is unavailable and should be presented as such rather than failing
    /// at the moment the user clicks.
    public func packageID(for model: ForgeModel) -> PackageID? { packageIDs[model] }

    public func outcome(for model: ForgeModel) -> RegistrationOutcome? {
        outcomes.first { $0.model == model }
    }
}
