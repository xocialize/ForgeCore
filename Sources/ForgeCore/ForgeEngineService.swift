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
import MLXFFTformer
import MLXRestormer
import MLXDRUNet
import MLXSCUNet
import MLXCIDNet
import MLXGFPGAN
import MLXRestoreFormer
import MLXBOPBTL
import MLXRealESRGAN
import MLXSeedVR2
import MLXBiRefNet
import MLXSigLIP2IQA
import MLXDDColor
import MLXInpaint
import MLXSEARAFT
import MLXEdgeTAM
import MLXRIFE
import MLXVJEPA2

let log = Logger(subsystem: "com.xocialize.ForgeCore", category: "ForgeCore")

/// 📄 **Machine requirements for every capability registered here: `REQUIREMENTS.md` at the package
/// root** — footprints, backends, OS floors and the measured operating points, each citing a receipt.
/// ⚠️ It opens with a live defect worth knowing before you size a machine: Real-ESRGAN's declared
/// footprint (6.26 GB) is the *tiled* figure, while its default path is whole-frame and measures
/// **21.24 GB at 1080p**. The governor admits against the declaration, not against reality.
///
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
    case promptMaskQuality // SAM 2.1-S — the Quick Select QUALITY tier, same capability (AB-A-0105 / AB-T-0173)
    case qualityScore   // SigLIP2 NR-IQA — the no-reference oracle, not a transform

    // ── The frame-restore wave (BRIDGE-066), face pair (064-face) and relight (BRIDGE-067) ──
    // Registered 2026-07-31. Until then ForgeCore registered nine models while the fleet held
    // sixteen — every case below was a shipped, parity-gated package invisible to every app.
    // ⚠️ NONE of these change a default: NAFNet `.signage` remains the restore chain's backer
    // until BRIDGE-069 measures the head-to-head. Registration makes them REACHABLE, not chosen.
    case deblurMotion      // FFTformer (fp32 — bf16 measured 30 dB of dtype error; ships fp32). Registered, NOT a default route: −0.02 dB on real blur (C2, AB-T-0180)
    case restormerMotion   // Restormer .motionDeblur — the `motionDeblur` route (C2: +1.29 dB on real blur)
    case restormerDefocus  // Restormer .defocusDeblur — the only shippable defocus deblur (C4)
    case restormerDenoise  // Restormer .realDenoise
    case denoiseStrength   // DRUNet — THE strength dial (σ as a continuous input; contract 1.30.0)
    case denoiseBlind      // SCUNet .realPSNR — V4's strongest blind denoiser (GAN variant deliberately unregistered: no-op at ISO1600, never a default)
    case relight           // HVI-CIDNet — first `imageRelight` backer; luma gate is IN-PACKAGE, ForgeCore must not second-guess it
    case faceRestore       // GFPGAN v1.4
    case faceRestoreVQ     // RestoreFormer++ — the VQ-dictionary sibling
    case defectDetect      // BOPBTL scratch detector — first `defectDetect` backer (contract 1.31.0); composes with `inpaint`

    // ── frameInterpolate (AB-T-0038) ──
    // Registered 2026-08-14. RIFE 4.25 was ported and parity-locked 2026-06-18 (cosine 1.0 vs the
    // Python reference) and shipped at v0.8.0 with tiling + the CAN gate — but no registration
    // existed anywhere, so no app could reach it: the same shipped-but-invisible gap the
    // frame-restore wave closed for the image fleet. App-facing seam: `interpolateVideo(_:factor:)`
    // (`Adapters/EngineFrameInterpolator.swift`).
    case frameInterpolate  // RIFE 4.25

    // ── contentClassify (AB-T-0045) ──
    // Registered 2026-08-15 as the §6.3 planner hint's feature extractor. ⚠️ Like `qualityScore`
    // this is an **assessment, not a transform** — it changes no pixels. What it produces that
    // matters is the pooled **[1024] router embedding**; its `ssv2_<index>` tags are ids from the
    // SSv2 hand-object action vocabulary, which ships no label names and is semantically wrong for
    // signage. Never map a tag to a content class — `EngineContentHintProvider` reads the embedding.
    case contentClassify   // V-JEPA2 ViT-L

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
        case .promptMaskQuality: return "SAM 2.1"
        case .qualityScore: return "SigLIP2-IQA"
        case .deblurMotion:     return "FFTformer"
        case .restormerMotion:  return "Restormer (motion)"
        case .restormerDefocus: return "Restormer (defocus)"
        case .restormerDenoise: return "Restormer (denoise)"
        case .denoiseStrength:  return "DRUNet"
        case .denoiseBlind:     return "SCUNet"
        case .relight:          return "HVI-CIDNet"
        case .faceRestore:      return "GFPGAN"
        case .faceRestoreVQ:    return "RestoreFormer++"
        case .defectDetect:     return "BOPBTL"
        case .frameInterpolate: return "RIFE"
        case .contentClassify:  return "V-JEPA2"
        }
    }

    /// The engine capability this registration backs — use it to bind a progress view
    /// (`MLXEngineUI.ModelStateView(monitor:capability:)`) in a consuming app.
    public var capability: Capability {
        switch self {
        case .restore:    return .imageRestore
        case .upscale:    return .imageUpscale
        // ⚠️ This registration backs BOTH `.imageUpscale` and `.videoUpscale` — one package, one model
        // load, two declared contracts. `capability` reports `.imageUpscale` because its only job is to
        // bind a progress view to the download/prepare of that single model; it is not a routing
        // decision. ✅ The video contract IS wired as of 2026-07-31 — see `EngineVideoUpscaler` and
        // `videoTemporalTier` (the earlier "no app wires yet" note is retired).
        case .upscaleQuality: return .imageUpscale
        case .matte:      return .matting
        case .colorize:   return .imageColorize
        case .inpaint:    return .imageInpaint
        case .flow:       return .opticalFlow
        case .promptMask: return .promptSegment
        case .promptMaskQuality: return .promptSegment
        case .qualityScore: return .imageQualityScore
        case .deblurMotion, .restormerMotion, .restormerDefocus, .restormerDenoise,
             .denoiseStrength, .denoiseBlind, .faceRestore, .faceRestoreVQ:
            return .imageRestore
        case .relight: return .imageRelight
        case .defectDetect: return .defectDetect
        case .frameInterpolate: return .frameInterpolate
        case .contentClassify: return .contentClassify
        }
    }
}

/// The one user-facing dial for the video quality tier: how many frames are processed jointly.
///
/// 🔑 **Counter-intuitively, more frames is CHEAPER per frame as well as more coherent** — measured
/// 3.27 s/frame at T=9 against 4.10–4.68 s at T=1 through the full shipping path (`GAP-PROGRAM.md`
/// V12-D). Coherence is a discount here, not a trade, so `.default` is the right default.
///
/// 🚨 **This dial does NOT control memory. Output geometry does.** SeedVR2's streaming banks are
/// allocated **per spatial tile position** (~0.82 GiB each, roughly constant in T), so a larger output
/// frame costs more regardless of the tier — and lowering the tier will NOT rescue a frame that is too
/// big. Measured: one 256² output tile ≈ 11.93 GiB at `.default`; nine tile positions (512² output)
/// ≈ 18.0 GiB. **A 16 GB machine runs `.default` at preview-sized outputs, not at real content**;
/// larger-memory machines run it on real content today.
///
/// ✅ **Bank offload/eviction SHIPPED 2026-07-31** (`mlx-seedvr2-swift` v0.9.0) and is what extends this
/// downmarket: set `ForgeEngineService.bankResidencyBudgetBytes` and banks over the budget round-trip
/// through scratch **bit-identically** — measured 42.02 GiB → 12.69 GiB (3.31×) at 40 tile positions
/// for **≈+7% wall clock (0–19% across repeated arms; remeasured 2026-08-03,
/// `probes/v12b3g_bank_granule_io.out` — the originally published +48% was session drift in a single
/// unrepeated arm)**. ⚠️ Off by default, because the real price is write volume (~840 GiB of scratch
/// for a 10 s clip) — it is how a machine runs a job it otherwise could not, not a blanket optimisation.
///
/// ⚠️ The tier is a **ceiling, not a guarantee**: the driver clamps the effective window down its
/// measured ladder using the engine-stamped `availableBudgetBytes` and the clip's geometry.
public enum VideoTemporalTier: Int, Sendable, CaseIterable {
    /// Per-frame, no temporal joining — the pre-v0.8.0 behaviour, bit-identical to it.
    case off = 1
    /// Steady-state chunks carry only one latent frame; measured the weakest of the three T>1 rungs.
    case conservative = 5
    /// The recommended tier. Best measured coherence-per-cost.
    case `default` = 9
    /// Highest coherence, highest bank pressure.
    case max = 13

    /// The `temporalWindow` value handed to `SeedVR2Configuration`.
    public var temporalWindow: Int { rawValue }
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

/// An app-injected registration ForgeCore runs beside its own — the seam that lets a HIGHER-floored
/// backend plug into a macOS-26 ForgeCore without raising anyone's floor.
///
/// 🔑 **The concrete case this exists for: `CoreAIRealESRGAN`** (macOS 27+, ANE). SPM refuses a
/// 26-floored package depending on a 27-floored one, so ForgeCore cannot link it — but an app whose
/// own deployment target is 27+ can, and injects it here. Below 27 the app simply doesn't, and the
/// MLX backer keeps serving the capability: the OS split is the design.
///
/// The closure performs the actual `engine.register(...)` call because configuration types are
/// per-package; ForgeCore stays generic. Outcomes surface in `externalOutcomes` with the same
/// honesty rules as built-ins: a failure is a fact to show, not an error to retry.
public struct ExternalRegistration: Sendable {
    public let name: String
    public let capability: Capability
    public let register: @Sendable (MLXServeEngine) async throws -> PackageID

    public init(name: String, capability: Capability,
                register: @escaping @Sendable (MLXServeEngine) async throws -> PackageID) {
        self.name = name
        self.capability = capability
        self.register = register
    }
}

/// Outcome of one external registration — parallel to `RegistrationOutcome`, keyed by name because
/// externals are not `ForgeModel` cases (that enum is ForgeCore's own registry of record).
public struct ExternalRegistrationOutcome: Sendable {
    public let name: String
    public let capability: Capability
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

    /// How many frames SeedVR2's video path processes jointly — the quality/cost dial for
    /// `videoUpscale`. Higher = more temporally coherent AND (measured) cheaper per frame, but the
    /// memory cost is driven by OUTPUT GEOMETRY, not by this number. See `VideoTemporalTier`.
    public let videoTemporalTier: VideoTemporalTier

    /// Byte budget for RESIDENT per-tile streaming banks on the video path. `nil` (default) keeps
    /// every bank in memory, which is the behaviour every measurement to date describes.
    ///
    /// 🔑 **This is the knob that lets the quality tier exceed ~9 tile positions.** Bank residency is
    /// `tilePositions × ~0.765 GiB` (`GAP-PROGRAM.md` V12-B) — 6.9 GiB at a 512² output but 26.8 GiB at
    /// SD ×2, more than a 16 GB machine has. Set a budget and the excess round-trips through a scratch
    /// directory **bit-identically** — measured 42.02 GiB → 12.69 GiB at a 1 GiB budget for ≈+7% wall
    /// clock (0–19% across repeated arms; `probes/v12b3g_bank_granule_io.out`, superseding the
    /// drift-contaminated +48% of `v12b2_bank_eviction.out`). ⚠️ That budget is the extreme; use
    /// `bankBudgetToFit` and take the largest that fits.
    ///
    /// ⚠️ **Do not set this as a blanket optimisation.** The real price is write volume — 59.7 GiB
    /// measured for 17 frames at 40 tile positions, ~840 GiB for a 10 s clip. It is how a machine runs a job it otherwise could not;
    /// a machine with the headroom should leave it `nil` and write nothing. Use
    /// `videoUpscaleFits(outputTilePositions:budgetBytes:bankResidencyBudgetBytes:)` to decide.
    public let bankResidencyBudgetBytes: Int?

    /// App-injected registrations, run by `registerAll()` after the built-ins. See
    /// `ExternalRegistration` for why this exists (the CoreAI/ANE backend is the first user).
    public let externalRegistrations: [ExternalRegistration]

    /// Outcome per external registration, in injection order, once `registerAll()` has run.
    public private(set) var externalOutcomes: [ExternalRegistrationOutcome] = []

    /// - Parameters:
    ///   - policy: license gate. Leave at `.permissiveOnly` for a shipping product; a research build
    ///     may widen it.
    ///   - enforcement: leave `.blocking`. See the note on `engine`.
    ///   - restoreVariant: see `restoreVariant`. The default changed to `.signage` on evidence.
    ///   - bankResidencyBudgetBytes: see the property. `nil` keeps every bank resident.
    public init(policy: LicensePolicy = .permissiveOnly,
                enforcement: LicenseEnforcement = .blocking,
                restoreVariant: NAFNetVariant = .signage,
                videoTemporalTier: VideoTemporalTier = .default,
                bankResidencyBudgetBytes: Int? = nil,
                externalRegistrations: [ExternalRegistration] = [],
                governor: MemoryGovernor? = nil) {
        // `governor` nil = the engine's device default. A fixed budget is for drills: an app that declares a canvas
        // (ForgeExternalTenant) proves its shrink path under a budget that only fits if the canvas sheds.
        self.engine = MLXServeEngine(policy: policy, licenseEnforcement: enforcement, governor: governor)
        self.restoreVariant = restoreVariant
        self.videoTemporalTier = videoTemporalTier
        self.bankResidencyBudgetBytes = bankResidencyBudgetBytes
            ?? ProcessInfo.processInfo.environment["FORGE_SEEDVR2_BANK_BUDGET"].flatMap(Int.init)
        self.externalRegistrations = externalRegistrations
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

        var external: [ExternalRegistrationOutcome] = []
        for reg in externalRegistrations {
            do {
                let id = try await reg.register(engine)
                external.append(ExternalRegistrationOutcome(
                    name: reg.name, capability: reg.capability, packageID: id, failure: nil))
                log.notice("External registration \(reg.name, privacy: .public) registered")
            } catch {
                external.append(ExternalRegistrationOutcome(
                    name: reg.name, capability: reg.capability, packageID: nil,
                    failure: String(describing: error)))
                log.error("External registration \(reg.name, privacy: .public) failed: \(String(describing: error), privacy: .public)")
            }
        }
        externalOutcomes = external

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
                // ✅ **The A/B these knobs existed for is DONE and the package default moved to 128/8**
                // (`mlx-realesrgan-swift` 0.7.0). t128 is 4.65× faster end to end at 8192² output and is
                // the arm WITHOUT a detectable seam — t64 seams at z +1.50 against a null sweep, t128 at
                // +0.46. `FORGE_RE_TILE` / `FORGE_RE_OVERLAP` stay for future geometry work; leaving both
                // unset now takes the fast, clean path rather than the old 64.
                // ⚠️ `tileOverlap` is ABSOLUTE (it covers the receptive field). Do NOT raise it to keep
                // the old 1:8 ratio — that costs throughput and buys no margin.
                let reEnv = ProcessInfo.processInfo.environment
                id = try await engine.register(RealESRGANUpscalePackage.registration,
                                               configuration: RealESRGANConfiguration(
                                                   variant: .general,
                                                   inputTileSize: reEnv["FORGE_RE_TILE"].flatMap(Int.init),
                                                   tileOverlap: reEnv["FORGE_RE_OVERLAP"].flatMap(Int.init)))
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
                // Halo (context padding): the V10-fix lever that changes each tile's CONTEXT rather
                // than its blend — must be a multiple of 8 keeping tile+2·halo /16. Default 0
                // pending the halo sweep's verdict.
                let halo = env["FORGE_SEEDVR2_HALO"].flatMap(Int.init) ?? 0
                let wholeFrame = env["FORGE_SEEDVR2_WHOLEFRAME"].flatMap(Int.init) ?? (1024 * 1024)
                id = try await engine.register(SeedVR2UpscalePackage.registration,
                                               configuration: SeedVR2Configuration(
                                                   quant: .int8,
                                                   temporalWindow: videoTemporalTier.temporalWindow,
                                                   tileSize: tile,
                                                   tileOverlap: overlap,
                                                   tileHalo: halo,
                                                   imageWholeFramePixels: wholeFrame,
                                                   bankResidencyBudgetBytes: bankResidencyBudgetBytes))
            case .matte:
                id = try await engine.register(BiRefNetPackage.registration,
                                               configuration: BiRefNetConfiguration())
            case .deblurMotion:
                id = try await engine.register(FFTformerRestorePackage.registration,
                                               configuration: FFTformerConfiguration())
            // Restormer: ONE package, THREE registrations — the variant is a Configuration knob, so
            // each behaviour needs its own PackageID to be individually reachable/preparable
            // (BRIDGE-066 asked that the mechanism be recorded: it is per-registration config,
            // exactly the multi-package-per-capability pattern `upscale`/`upscaleQuality` set).
            // 🔴 The explicit `id:` IS that mechanism. Without it the engine keys all three by the
            // manifest's surface name ("restormer-restore"), and a re-registration under an existing id
            // REPLACES it — so from 2026-07-31 until AB-T-0180 the last one (`.realDenoise`) answered
            // for motion and defocus too, under their names. `upscale`/`upscaleQuality` never hit this
            // because they are different packages. Pinned by `RegistrationIdentityTests`.
            case .restormerMotion:
                id = try await engine.register(RestormerRestorePackage.registration,
                                               configuration: RestormerConfiguration(variant: .motionDeblur),
                                               id: "restormer-motion-deblur")
            case .restormerDefocus:
                id = try await engine.register(RestormerRestorePackage.registration,
                                               configuration: RestormerConfiguration(variant: .defocusDeblur),
                                               id: "restormer-defocus-deblur")
            case .restormerDenoise:
                id = try await engine.register(RestormerRestorePackage.registration,
                                               configuration: RestormerConfiguration(variant: .realDenoise),
                                               id: "restormer-real-denoise")
            case .denoiseStrength:
                id = try await engine.register(DRUNetRestorePackage.registration,
                                               configuration: DRUNetConfiguration())
            case .denoiseBlind:
                // .realPSNR only — V4 measured the GAN variant 0.9–1.5 dB behind and a no-op at
                // ISO1600. It stays selectable via configuration if a perceptual surface ever wants
                // it; it does not get a registration until something measures a use for one.
                id = try await engine.register(SCUNetRestorePackage.registration,
                                               configuration: SCUNetConfiguration(variant: .realPSNR))
            case .relight:
                // Package defaults on purpose (BRIDGE-067: "take the package's default variant; do
                // not choose one in ForgeCore"). The luma gate lives in-package — a correctly-dark
                // input coming back unmodified is the CONTRACT, not a failure.
                id = try await engine.register(CIDNetRelightPackage.registration,
                                               configuration: CIDNetConfiguration())
            case .faceRestore:
                id = try await engine.register(GFPGANRestorePackage.registration,
                                               configuration: GFPGANConfiguration())
            case .faceRestoreVQ:
                id = try await engine.register(RestoreFormerRestorePackage.registration,
                                               configuration: RestoreFormerConfiguration())
            case .defectDetect:
                // The detection half of automatic damage repair — its mask feeds the registered
                // `.inpaint`. ⚠️ Corpus validation on real scratched photographs is deliberately
                // deferred to the app/restore-scaffolding stage (P7 row); upstream synthetic
                // behaviour: 100% recall / 65.7% precision / zero false pixels on a clean control.
                id = try await engine.register(BOPBTLDetectPackage.registration,
                                               configuration: BOPBTLConfiguration())
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
            case .promptMaskQuality:
                // SAM 2.1-Small (mlx-edgetam-swift 0.6.0, AB-T-0173): the Quick Select QUALITY tier — a belly
                // click selects the whole fox (IoU 0.968 vs EdgeTAM 0.008) and a click on one gate square selects
                // only that square. Image only: EdgeTAM stays the fast default and the only `trackObject`. Same
                // request/response, same `softMatte` mode, so `EnginePromptMaskProvider` serves both through
                // `model:`. Declared split 0.25 GB resident + 2.6 GB activation (measured phys 2.35 GB at 1024²).
                // Explicit id: two packages, two PackageIDs — never keyed by a manifest surface name that a
                // sibling might share (the Restormer lesson, AB-L-0160).
                id = try await engine.register(SAM21Package.registration,
                                               configuration: SAM21Configuration(),
                                               id: "sam21-prompt-segment")
            case .frameInterpolate:
                // Package defaults on purpose (the BRIDGE-067 discipline): factor 2, and the tiling
                // geometry (whole-frame ≤ 1080p, tile 1024 / overlap 128 above) is the package's own
                // measured 4K-admissible configuration, not ForgeCore's to second-guess.
                // ⚠️ The output re-encode is frame-stream-native's SOFTWARE-HEVC default, and that
                // default is load-bearing: hardware VideoToolbox HEVC right after heavy Metal compute
                // stalls — and on quarantined builds (AB-B-0002, 26A5406e) panics the machine on
                // teardown. No Forge surface may set FRAMESTREAM_ENCODE=hardware.
                id = try await engine.register(RIFEInterpolatePackage.registration,
                                               configuration: RIFEConfiguration())
            case .contentClassify:
                // Package defaults on purpose (the BRIDGE-067 discipline): the repo, the 16-frame
                // uniform sampling and the 256² preprocessing are the port's parity-locked pipeline,
                // not ForgeCore's to second-guess. `topK` is left at 5 even though the planner reads
                // only the embedding — the tags cost nothing and travel into receipts verbatim as
                // provenance for the calibration rows.
                id = try await engine.register(VJEPA2ContentClassifyPackage.registration,
                                               configuration: VJEPA2Configuration())
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

    /// PackageID of an app-injected registration, by its injection name.
    public func packageID(forExternal name: String) -> PackageID? {
        externalOutcomes.first(where: { $0.name == name })?.packageID
    }

    public func outcome(for model: ForgeModel) -> RegistrationOutcome? {
        outcomes.first { $0.model == model }
    }
}
