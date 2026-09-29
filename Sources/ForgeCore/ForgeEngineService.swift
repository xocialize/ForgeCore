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
import MLXNERVE
import MLXSeedVR2
import MLXVOSR
import MLXRealPLKSR
import MLXHEART
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
/// ⚠️ It opens with a defect CLASS worth knowing before you size a machine: a `QuantFootprint` is a scalar
/// per quant, so a package whose default path scales with input can under-declare it. The recorded instance
/// is Real-ESRGAN — declared 6.26 GB (the *tiled* figure) while its default whole-frame path measured
/// **21.24 GB at 1080p**; the governor admits against the declaration, not against reality. Real-ESRGAN left
/// the registered set on 2026-09-28 (NERVE, its replacement as `.upscale`, declares the max over a measured
/// envelope that includes its whole-frame path); the class still applies to every input-dependent package.
///
/// Every model package Forge registers. One case per registration, *not* per capability —
/// Optimize alone uses two (restore + upscale), and one flow registration feeds four consumers.
///
/// 🔑 **`upscale`, `upscaleFidelity`, `upscaleQuality`, `upscaleGenerative` and `upscaleHEART` are five registrations
/// behind ONE capability**, which is the engine's multi-package-per-`PackageID` pattern and the first place Forge uses
/// it: NERVE is the fast tier (2026-09-28, replacing Real-ESRGAN), RealPLKSR the fidelity tier (2026-09-28), SeedVR2-3B
/// the quality tier, VOSR2 the generative stills tier (2026-09-26), HEART the opt-in fidelity variant (2026-09-29,
/// reachable but never a default route), and the planner picks. Adding SeedVR2 does **not**
/// displace the fast tier — `MODEL-CATALOG.md`'s standing rule is that when SeedVR2 isn't viable (memory, chip,
/// file) `videoUpscale` routes to the per-frame fast tier: *same tool, lighter backer.*
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
    // ── upscale (AB-T-0186 Step 3) ──
    // Since 2026-09-28: NERVE (Phips/NERVE, `mlx-nerve-swift`) — the provenance-clean FAST tier, replacing Real-ESRGAN,
    // whose DF2K-trained weights were the worst-provenanced tier Forge shipped (AB-L-0084). 1.8 M-param pure conv,
    // Apache-2.0 code AND weights trained only on the CC0 LUCID corpus (platform-declared CC0, AB-D-0106), vendored —
    // no download. Parity ≥ 138 dB vs the upstream PyTorch code and the author's ONNX; fp16 lane by default; ~21 ms per
    // output Mpx on M5 Max (1.35–1.40× Real-ESRGAN, 0.16× RealPLKSR); native ×4 AND ×2, so HD → 4K never builds the 8K
    // intermediate (AB-B-0013). On the Forge bench (NERVE-EVAL §5) `.fidelity` is the best full-reference arm on damaged
    // photo, where Real-ESRGAN sat below bicubic. Real-ESRGAN's package stays available; ForgeCore registered only its
    // `.general`, so no anime path is lost.
    case upscale        // NERVE — the fast tier
    case upscaleQuality // SeedVR2-3B (int8) — the quality tier, same capability
    // ── upscaleGenerative (AB-T-0184 / AB-T-0185) ──
    // Registered 2026-09-26. VOSR2 (CSWRY/VOSR, CVPR 2026): a 1.4B one-step generative super-resolution DiT
    // conditioned on DINOv2-L, ported to Swift/MLX (`mlx-vosr-swift`, e2e 115.5 dB vs the PyTorch oracle);
    // the THIRD `imageUpscale` backer — the generative STILLS tier. Gated in VOSR-EVAL §10 (G1–G5 PASS,
    // AB-R-0348): the first learned tier to beat the classical floor on SSIMULACRA2 in the JPEG regime, ×1
    // restoration through ×4, 0.3 s per 512² tile. ⚠️ Generative: it REWRITES letters in already-legible
    // small text (AB-L-0172) — reach it through `EngineTextGuardedUpscaler`, whose routes stay opt-in per
    // measured tier pair; nothing routes to it by default. Weights: mlx-community/VOSR2-fp16 (int8 under budget).
    case upscaleGenerative // VOSR2 (fp16; int8 below 7 GB) — the generative stills tier, same capability
    // ── upscaleFidelity (AB-T-0186 Step 1) ──
    // Registered 2026-09-28. RealPLKSR `4x_webphoto_realplksr` (`mlx-realplksr-swift`): the first model whose
    // weights are OURS end to end — trained by xocialize on the provenance-clean Commons-SR corpus, Apache-2.0,
    // vendored in the package (no download). The FOURTH `imageUpscale` backer. It beat Real-ESRGAN at all 12
    // degradation rungs including clean (AB-R-0285) and is the best non-generative arm on every damaged cell of the
    // Forge bench (VOSR-EVAL §10), so it is routed by COST, not by degradation: the honest user control is
    // "fast vs best" (fast = Real-ESRGAN then, NERVE since 2026-09-28). ~111–125 ms per Mpx of output on M5 Max, ~8×
    // Real-ESRGAN (AB-R-0177), ~6× NERVE (mlx-nerve-swift PORTING-SPEC N6).
    // Named `upscaleFidelity`, not `upscaleWebPhoto`: AB-R-0285 measured it as a general quality tier, not a
    // web-photo one, and `upscaleQuality` already means SeedVR2. "Fidelity" is what separates it from the two
    // diffusion tiers: one deterministic feed-forward pass, gated on full-reference fidelity (PSNR/DISTS) — its GAN
    // losses add texture, but nothing is sampled. ⚠️ Not a general-superiority claim — the hold-out was Commons
    // content (AB-R-0285 ⚠️3); a third-party clean set is the missing measurement.
    case upscaleFidelity   // RealPLKSR (fp32, vendored) — the fidelity tier, "best" beside NERVE's "fast"
    // ── upscaleHEART (AB-T-0189 / AB-T-0190 / AB-A-0128) ──
    // Registered 2026-09-29 as the OPT-IN fidelity variant behind RealPLKSR — reachable (`measureUpscale`, a host's own
    // routing), never a default route. HEART (Phips/HEART, `mlx-heart-swift`): Philip Hofmann's 16.7 M-param HAT-iLN
    // window-attention SR, Apache-2.0 code and weights trained only on the CC0 LUCID corpus (AB-D-0106); the FIFTH
    // `imageUpscale` backer. On the Forge bench `.fidelity` won every damaged cell against every arm (AB-R-0370); on the
    // twelve-rung Commons ladder (AB-R-0379) RealPLKSR wins DISTS at 12/12 rungs while HEART wins PSNR at 12/12
    // (+0.42 dB) and ties SSIMULACRA2 — DISTS is in RealPLKSR's training loss (AB-L-0186), so the plan's rule judged on
    // the incumbent's objective; the operator chose the opt-in slot rather than a default. Cost: 416–450 ms per output
    // Mpx on M5 Max, ≈ 4× RealPLKSR (mlx-heart-swift PORTING-SPEC H4). Port parity 127.5 dB vs the upstream PyTorch
    // code; fp16 lane 67.6 dB vs that oracle and SSIMULACRA2-indistinguishable from fp32 (H3). ⚠️ i-LN normalises over
    // the whole tensor, so its tiled output is NOT bit-equivalent to whole-frame (H5): the package defaults run whole
    // frames only up to 512² of input and 512/32 tiles above (fp16 peak 3.2 GB through 1080p → 8K).
    case upscaleHEART      // HEART .fidelity (fp16, mlx-community/HEART-fp16) — the opt-in fidelity VARIANT, not a default
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
        case .upscale:    return "NERVE"
        case .upscaleQuality: return "SeedVR2-3B"
        case .upscaleGenerative: return "VOSR2"
        case .upscaleFidelity: return "RealPLKSR"
        case .upscaleHEART: return "HEART"
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
        case .upscaleGenerative: return .imageUpscale
        case .upscaleFidelity: return .imageUpscale
        case .upscaleHEART: return .imageUpscale
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

    /// The configuration each model was registered with — kept so `admissibility(for:)` evaluates exactly
    /// the variant the engine will load, not a re-constructed default that could drift from it.
    private var configurations: [ForgeModel: any PackageConfiguration] = [:]

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
                id = try await registerRecording(model, NAFNetRestorePackage.registration,
                                                        configuration: NAFNetConfiguration(variant: restoreVariant))
            case .upscale:
                // NERVE (mlx-nerve-swift ≥ 0.1.0) — the fast tier since 2026-09-28, replacing Real-ESRGAN (AB-T-0186
                // Step 3). Package defaults on purpose (the BRIDGE-067 discipline): variant `.fidelity` — KEPT after
                // the crossover (AB-R-0379): its DISTS rule pointed at `.sharp`, but the pre-registered guard fired
                // (`.sharp` is below bicubic PSNR at 12/12 rungs; PSNR and SSIMULACRA2 prefer `.fidelity`), and the
                // operator kept `.fidelity` (AB-A-0128 → AB-D-0107) —, the fp16 lane (70.5+ dB vs fp32, faster, half the peak), whole-frame
                // up to 1920×1080 INPUT pixels, and above that 512 px tiles with a 56 px frame-cropped halo that
                // reproduce the whole-frame output (NERVE's receptive field is 51 px; plain overlap tiles read 62.9 dB
                // with the error in the seams). Native ×4 and ×2 (`.fidelity` ×2 → the native 2x_NERVE_OTF_gan
                // checkpoint), ×1/×3 post-downsampled; `appliedScale` says what ran, which `EngineImageEnhancer` records.
                // Declared split footprint is MEASURED over inputs up to 3840×2160 at ×4, the whole-frame path included
                // (fp16 256 MB + 5.9 GB) — not Real-ESRGAN's tiled-only figure (REQUIREMENTS.md). Explicit id: without
                // one the engine keys by the surface name, and a second NERVE registration would silently replace this
                // one (AB-L-0160).
                id = try await registerRecording(model, NERVEUpscalePackage.registration,
                                                        configuration: NERVEConfiguration(),
                                                        id: "nerve-upscale")
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
                id = try await registerRecording(model, SeedVR2UpscalePackage.registration,
                                                        configuration: SeedVR2Configuration(
                                                            quant: .int8,
                                                            temporalWindow: videoTemporalTier.temporalWindow,
                                                            tileSize: tile,
                                                            tileOverlap: overlap,
                                                            tileHalo: halo,
                                                            imageWholeFramePixels: wholeFrame,
                                                            bankResidencyBudgetBytes: bankResidencyBudgetBytes))
            case .upscaleGenerative:
                // VOSR2 (mlx-vosr-swift 0.1.0): fp16 is the default lane (49.3 dB vs the fp32 oracle); the package
                // itself drops to its int8 DiT (46.7 dB) when the governor stamps less than 7 GB of headroom
                // (BudgetAware), and tiles the VAE above 2048² of output. `defaultScale` 2 mirrors SeedVR2 so a
                // request without a scale means the same thing on both generative tiers. The engine materializes
                // mlx-community/VOSR2-fp16 (3.35 GB) into its store on first prepare. `FORGE_VOSR2_QUANT=int8|fp32`
                // is a diagnostics lever only. Explicit id (the Restormer lesson, AB-L-0160).
                let vosrQuant: Quant = {
                    switch ProcessInfo.processInfo.environment["FORGE_VOSR2_QUANT"] {
                    case "int8": return .int8
                    case "fp32": return .fp32
                    default: return .fp16
                    }
                }()
                id = try await registerRecording(model, VOSRUpscalePackage.registration,
                                                        configuration: VOSRConfiguration(quant: vosrQuant, defaultScale: 2),
                                                        id: "vosr2-upscale")
            case .upscaleFidelity:
                // RealPLKSR (mlx-realplksr-swift 0.1.0). Package defaults on purpose (the BRIDGE-067 discipline): the
                // one vendored checkpoint (`.webphoto`, 29.6 MB — `needsDownload` reads false, nothing materializes),
                // whole-frame up to 1920×1080 INPUT pixels, 256/32 tiles above. Native ×4; a ×2 request is
                // post-downsampled and `appliedScale` says so. ⚠️ The declared footprint was taken AT that whole-frame
                // ceiling — PROVISIONAL 1.05 GB floor + 11.5 GB activation at 1080p ×4, derived from a 4.86 GB MLX
                // peak, not measured in-app (REQUIREMENTS.md) — so admission charges the 1080p worst case for every
                // input. Explicit id, equal to the manifest's surface name today: without one the engine keys by that
                // name, so a second checkpoint registered from this package would silently replace this one — and only
                // a live run shows it (AB-L-0160).
                id = try await registerRecording(model, RealPLKSRUpscalePackage.registration,
                                                        configuration: RealPLKSRConfiguration(),
                                                        id: "realplksr-webphoto-upscale")
            case .upscaleHEART:
                // HEART (mlx-heart-swift 0.1.0). Package defaults on purpose (the BRIDGE-067 discipline): variant
                // `.fidelity` (heart_4x_otf_v2), the fp16 lane (every reduction held at fp32 inside; the package's
                // BudgetAware drops a requested fp32 to fp16 under 14 GB), whole-frame up to 512² of INPUT pixels and
                // 512/32 tiles above — measured, not inherited, because i-LN makes tiles differ from the whole frame.
                // Native ×4; a ×2 request is post-downsampled (measured better than the native heart_2x checkpoint on
                // damaged input) and `appliedScale` says so. The engine materializes only this variant's file
                // (`WeightSourcing`, one role per variant × lane: mlx-community/HEART-fp16/heart_4x_otf_v2_fp16.safetensors
                // + config.json, 34 MB) into its store on first prepare. Declared footprint fp16 1.06 GB + 7.6 GB on the
                // in-app basis, PROVISIONAL (derived through the tile-driver sibling's measured ratios from a 3.18 GB
                // MLX peak; REQUIREMENTS.md). Explicit id (AB-L-0160).
                id = try await registerRecording(model, HEARTUpscalePackage.registration,
                                                        configuration: HEARTConfiguration(),
                                                        id: "heart-upscale")
            case .matte:
                id = try await registerRecording(model, BiRefNetPackage.registration,
                                                        configuration: BiRefNetConfiguration())
            case .deblurMotion:
                id = try await registerRecording(model, FFTformerRestorePackage.registration,
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
                id = try await registerRecording(model, RestormerRestorePackage.registration,
                                                        configuration: RestormerConfiguration(variant: .motionDeblur),
                                                        id: "restormer-motion-deblur")
            case .restormerDefocus:
                id = try await registerRecording(model, RestormerRestorePackage.registration,
                                                        configuration: RestormerConfiguration(variant: .defocusDeblur),
                                                        id: "restormer-defocus-deblur")
            case .restormerDenoise:
                id = try await registerRecording(model, RestormerRestorePackage.registration,
                                                        configuration: RestormerConfiguration(variant: .realDenoise),
                                                        id: "restormer-real-denoise")
            case .denoiseStrength:
                id = try await registerRecording(model, DRUNetRestorePackage.registration,
                                                        configuration: DRUNetConfiguration())
            case .denoiseBlind:
                // .realPSNR only — V4 measured the GAN variant 0.9–1.5 dB behind and a no-op at
                // ISO1600. It stays selectable via configuration if a perceptual surface ever wants
                // it; it does not get a registration until something measures a use for one.
                id = try await registerRecording(model, SCUNetRestorePackage.registration,
                                                        configuration: SCUNetConfiguration(variant: .realPSNR))
            case .relight:
                // Package defaults on purpose (BRIDGE-067: "take the package's default variant; do
                // not choose one in ForgeCore"). The luma gate lives in-package — a correctly-dark
                // input coming back unmodified is the CONTRACT, not a failure.
                id = try await registerRecording(model, CIDNetRelightPackage.registration,
                                                        configuration: CIDNetConfiguration())
            case .faceRestore:
                id = try await registerRecording(model, GFPGANRestorePackage.registration,
                                                        configuration: GFPGANConfiguration())
            case .faceRestoreVQ:
                id = try await registerRecording(model, RestoreFormerRestorePackage.registration,
                                                        configuration: RestoreFormerConfiguration())
            case .defectDetect:
                // The detection half of automatic damage repair — its mask feeds the registered
                // `.inpaint`. ⚠️ Corpus validation on real scratched photographs is deliberately
                // deferred to the app/restore-scaffolding stage (P7 row); upstream synthetic
                // behaviour: 100% recall / 65.7% precision / zero false pixels on a clean control.
                id = try await registerRecording(model, BOPBTLDetectPackage.registration,
                                                        configuration: BOPBTLConfiguration())
            case .colorize:
                id = try await registerRecording(model, DDColorPackage.registration,
                                                        configuration: DDColorConfiguration())
            case .inpaint:
                id = try await registerRecording(model, InpaintPackage.registration,
                                                        configuration: InpaintConfiguration())
            case .flow:
                id = try await registerRecording(model, SEARAFTOpticalFlowPackage.registration,
                                                        configuration: SEARAFTConfiguration(variant: .spring, iters: 4))
            case .promptMask:
                id = try await registerRecording(model, EdgeTAMPackage.registration,
                                                        configuration: EdgeTAMConfiguration())
            case .promptMaskQuality:
                // SAM 2.1-Small (mlx-edgetam-swift 0.6.0, AB-T-0173): the Quick Select QUALITY tier — a belly
                // click selects the whole fox (IoU 0.968 vs EdgeTAM 0.008) and a click on one gate square selects
                // only that square. Image only: EdgeTAM stays the fast default and the only `trackObject`. Same
                // request/response, same `softMatte` mode, so `EnginePromptMaskProvider` serves both through
                // `model:`. Declared split 0.25 GB resident + 2.6 GB activation (measured phys 2.35 GB at 1024²).
                // Explicit id: two packages, two PackageIDs — never keyed by a manifest surface name that a
                // sibling might share (the Restormer lesson, AB-L-0160).
                id = try await registerRecording(model, SAM21Package.registration,
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
                id = try await registerRecording(model, RIFEInterpolatePackage.registration,
                                                        configuration: RIFEConfiguration())
            case .contentClassify:
                // Package defaults on purpose (the BRIDGE-067 discipline): the repo, the 16-frame
                // uniform sampling and the 256² preprocessing are the port's parity-locked pipeline,
                // not ForgeCore's to second-guess. `topK` is left at 5 even though the planner reads
                // only the embedding — the tags cost nothing and travel into receipts verbatim as
                // provenance for the calibration rows.
                id = try await registerRecording(model, VJEPA2ContentClassifyPackage.registration,
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
                id = try await registerRecording(model, SigLIP2IQAPackage.registration,
                                                        configuration: SigLIP2IQAConfiguration(maxPatches: patches))
            }
            packageIDs[model] = id
            return RegistrationOutcome(model: model, packageID: id, failure: nil)
        } catch {
            return RegistrationOutcome(model: model, packageID: nil, failure: Self.describe(error))
        }
    }

    /// `engine.register`, remembering the configuration for `admissibility(for:)`.
    private func registerRecording(_ model: ForgeModel, _ registration: PackageRegistration,
                                   configuration: any PackageConfiguration,
                                   id: PackageID? = nil) async throws -> PackageID {
        let packageID = try await engine.register(registration, configuration: configuration, id: id)
        configurations[model] = configuration
        return packageID
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

    /// The engine's admission verdict for `model` exactly as registered — device eligibility, and whether
    /// its declared working set can fit the memory budget at all (`fitsBudget`: after evicting every idle
    /// resident) — computed without loading anything. nil when the model did not register.
    ///
    /// ⚠️ It is the governor's own arithmetic over the package's DECLARED footprint, so it is exactly as
    /// honest as that declaration: a provisional figure above the real peak refuses a run that would have
    /// fit. That is why declarations get re-baselined in-app (AB-T-0019).
    public func admissibility(for model: ForgeModel) async -> Admissibility? {
        guard let id = packageIDs[model], let configuration = configurations[model],
              let manifest = await engine.manifest(for: id) else { return nil }
        return await engine.admissibility(for: manifest.requirements, configuration: configuration)
    }

    /// The engine's memory budget on this Mac — what `admissibility(for:)` measures against.
    public var memoryBudgetBytes: UInt64 {
        get async { await engine.memory.budgetBytes }
    }
}
