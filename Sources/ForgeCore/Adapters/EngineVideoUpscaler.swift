//
//  EngineVideoUpscaler.swift — ForgeCore
//
//  The `videoUpscale` capability, wired 2026-07-31. This is the app-facing seam for Forge's video
//  quality tier; before this, `videoUpscale` had no binding anywhere and no app could reach it.
//
//  🔑 **One registration, two contracts.** `SeedVR2UpscalePackage` declares both `ImageUpscaleContract`
//  and `VideoUpscaleContract`, so this adapter resolves the SAME `PackageID` the still path uses
//  (`.upscaleQuality`) and simply sends a `VideoUpscaleRequest` to it. There is no second registration
//  and no second model load — which also means a prepared image tier makes the video tier warm.
//
//  ⚠️ **This is a whole-clip call, not a frame seam.** The package decodes, drives the temporal
//  chunking driver, and re-encodes internally (AVFoundation via `frame-stream-native`; no FFmpeg).
//  Cost scales with clip length: measured ~3.3 s per output frame at `.default` on an M5 Max, so a
//  10 s 24 fps clip is roughly 13 minutes. **A UI must treat this as a long job with progress and
//  cancellation, never as a request that might return quickly.** Cancellation is honoured at the
//  package's CAN seams (per chunk).
//
//  🔑 **Since 2026-10-04 there are two video tiers, and the route between them is DECLARED.** FlashVSR
//  (`.upscaleVideoLiveAction`) is the live-action generative tier: one-step, streaming, much faster than SeedVR2 —
//  and it renders anime, cartoons and graphics as photographic texture. Forge has no validated detector for that
//  distinction (the mechanical classifier is off by measurement, the V-JEPA2 hint ships inert), so the caller says
//  what the clip is: `upscaleVideo(_:scale:content:)` sends `.liveAction` to FlashVSR and everything else to the
//  SeedVR2 route `upscaleVideo(_:scale:)` has always taken. A tier that cannot run fails the call with its reason;
//  it is never swapped for the other (the AB-T-0187 rule the stills tiers follow).
//

import CoreGraphics
import Foundation
import MLXFlashVSR
import MLXServeCore
import MLXToolKit

/// What the caller DECLARES a clip to be — the only input the live-action route trusts.
public enum VideoContent: String, Sendable, CaseIterable {
    /// Camera footage of the real world — the content FlashVSR is for.
    case liveAction
    /// Anime, cartoons, motion graphics, screen recordings, UI, text overlays — never FlashVSR.
    case drawn
    /// Not declared. Routes like `.drawn`: the generative live-action tier is reached by declaration, never by default.
    case unspecified
}

/// What one video upscale produced, plus the facts a receipt should carry.
public struct VideoUpscaleOutcome: Sendable {
    /// The tier that ran — `.upscaleQuality` (SeedVR2) or `.upscaleVideoLiveAction` (FlashVSR). A receipt names this.
    public let model: ForgeModel
    public let video: Video
    /// What the package actually applied. ⚠️ May differ from the request — report this, not the ask.
    public let appliedScale: Int
    public let seconds: Double

    /// Wall-clock per output frame, when the container reported a frame rate and duration. The number
    /// a UI needs to estimate a job, and the one that makes the length of this call obvious.
    public var secondsPerFrame: Double? {
        guard let fps = video.frameRate, let duration = video.durationSeconds,
              fps > 0, duration > 0 else { return nil }
        return seconds / (fps * duration)
    }
}

extension ForgeEngineService {

    /// Upscale a whole clip through the SeedVR2 temporal path.
    ///
    /// 🚨 **Check `videoUpscaleFits(outputTilePositions:budgetBytes:bankResidencyBudgetBytes:)` before
    /// offering this on a small-memory machine.** The temporal path's memory is driven by OUTPUT
    /// GEOMETRY — streaming banks are allocated per spatial tile position, ~0.765 GiB each and constant
    /// in the temporal window — so a big frame is expensive at every tier and **lowering
    /// `videoTemporalTier` will not rescue it** (`GAP-PROGRAM.md` V12-D/V12-B). What *does* rescue it is
    /// `bankResidencyBudgetBytes`; `bankBudgetToFit` turns a refusal into the budget that would work.
    ///
    /// - Parameter scale: `nil` uses the package's default (2). Only 2 and 4 are accepted.
    public func upscaleVideo(_ video: Video, scale: Int? = nil) async throws -> VideoUpscaleOutcome {
        try await upscaleVideo(video, scale: scale, on: .upscaleQuality)
    }

    /// Which registered tier serves a clip of declared `content`: FlashVSR for `.liveAction`, SeedVR2 for everything
    /// else (the route that existed before the live-action tier). `nonisolated`: a UI can label its control before
    /// touching the main actor.
    nonisolated public static func videoUpscaleRoute(for content: VideoContent) -> ForgeModel {
        content == .liveAction ? .upscaleVideoLiveAction : .upscaleQuality
    }

    /// Upscale a whole clip on the tier its DECLARED content routes to (`videoUpscaleRoute(for:)`).
    ///
    /// 🚨 **The declaration is the routing decision — ask the user rather than guess.** Declaring a cartoon `.liveAction`
    /// sends it to a generative model that turns flat colour and line art into photographic texture. If the routed tier
    /// is unavailable (did not register, or the engine refuses the clip — e.g. FlashVSR above its measured 1920×1152
    /// output ceiling), this throws with that reason; it never falls back to the other tier.
    ///
    /// FlashVSR facts a UI needs: ×4 native and ×2; the first output arrives after 25 source frames, then 8 at a time;
    /// every source frame comes back, with its own timestamp; memory scales with OUTPUT PIXELS, not clip length (check
    /// `liveActionVideoUpscaleFits`).
    public func upscaleVideo(_ video: Video, scale: Int? = nil, content: VideoContent) async throws -> VideoUpscaleOutcome {
        try await upscaleVideo(video, scale: scale, on: Self.videoUpscaleRoute(for: content))
    }

    private func upscaleVideo(_ video: Video, scale: Int?, on model: ForgeModel) async throws -> VideoUpscaleOutcome {
        guard let packageID = packageID(for: model) else {
            throw DiagnosticsError.notRegistered(model, outcome(for: model)?.failure)
        }
        // ⚠️ `prepare` is keyed by capability with the package as disambiguator. Here the capability is
        // `.videoUpscale` even though `ForgeModel.upscaleQuality.capability` reports `.imageUpscale` —
        // that property binds a progress view to one model load; it is not the routing key.
        _ = try await engine.prepare(.videoUpscale, package: packageID)

        let started = Date()
        let response = try await engine.run(VideoUpscaleRequest(video: video, scale: scale),
                                            package: packageID)
        let seconds = Date().timeIntervalSince(started)
        guard let upscale = response as? VideoUpscaleResponse else {
            throw DiagnosticsError.unexpectedResponse(model)
        }
        return VideoUpscaleOutcome(model: model,
                                   video: upscale.video,
                                   appliedScale: upscale.appliedScale,
                                   seconds: seconds)
    }

    /// Whether the live-action tier (FlashVSR) can be expected to run a `sourceWidth × sourceHeight` clip at `scale`
    /// within `budgetBytes`, computed from the PACKAGE'S OWN declaration so this gate and the engine's admission cannot
    /// drift apart: resident + max(representative activation, `ActivationScaling` at the padded output pixels).
    ///
    /// - Returns: `withinCeiling` — whether the output is inside the measured envelope (1920×1152 for the bf16 lane); the
    ///   engine refuses a run beyond it before loading anything, so a UI should say so rather than offer the control.
    ///   `fits` — within the ceiling AND the predicted bytes fit the budget. `predictedBytes` — what admission reserves.
    ///
    /// Basis: `phys_footprint` measured through the real MLXServeEngine in a CLI process (bf16: 9.2 / 19.1 / 33.7 GB at
    /// 640×384 / 1280×768 / 1920×1152 out) ×1.2 headroom — the in-app re-baseline is open (REQUIREMENTS.md).
    nonisolated public static func liveActionVideoUpscaleFits(
        sourceWidth: Int, sourceHeight: Int, scale: Int = 4, budgetBytes: UInt64
    ) -> (fits: Bool, withinCeiling: Bool, predictedBytes: UInt64) {
        let quant = FlashVSRConfiguration().quant
        guard let fp = FlashVSRUpscalePackage.manifest.requirements.footprints.first(where: { $0.quant == quant }) else {
            return (false, false, .max)
        }
        let pixels = Double(liveActionOutputPixels(sourceWidth: sourceWidth, sourceHeight: sourceHeight, scale: scale))
        var activation = fp.peakActivationBytes
        var within = true
        if let scaling = fp.activationScaling {
            activation = max(activation, scaling.projectedBytes(at: pixels))
            within = scaling.covers(pixels)
        }
        let predicted = fp.residentBytes + activation
        return (within && predicted <= budgetBytes, within, predicted)
    }

    /// The pixels FlashVSR actually computes per output frame: the ×`scale` frame padded up to multiples of 128 — the
    /// unit of the package's `ActivationScaling` (its `WorkloadDeclaring` computes the same from the clip's `tkhd`).
    nonisolated public static func liveActionOutputPixels(sourceWidth: Int, sourceHeight: Int, scale: Int = 4) -> Int {
        let pad = { (n: Int) in (n * scale + 127) / 128 * 128 }
        return pad(sourceWidth) * pad(sourceHeight)
    }

    /// Whether this machine can be expected to run the video tier at a given output geometry.
    ///
    /// 🔑 **The model that makes this predictable**, measured in `GAP-PROGRAM.md` V12-D: base residency
    /// plus **~0.82 GiB per spatial tile position**, and the per-tile term dominates as frames grow. The
    /// prediction matched a 9-tile measurement (18.0 GiB) within 0.3 GiB, which is why this is offered
    /// as a gate rather than a guess.
    ///
    /// 🚨 **"Per tile position" is a unit of convenience, not a mechanism — the underlying cost is per
    /// OUTPUT PIXEL, and that closes off the obvious optimisation.** The bank is `CausalConv3d`'s
    /// retained tail, `x[..., (T - memoryFrames)...]`, which keeps the tile's FULL spatial extent — so
    /// bank bytes scale with tile *area*, and `positions × area` is just the output area. Restated
    /// honestly: **≈ 13.4 KB of retained tail per output pixel**, invariant to how the frame is cut up.
    ///
    /// ⚠️ **Therefore neither knob a reader will reach for first actually works**, and both are now
    /// CI-enforced by `VAEStreamingBankSizeTests` in `mlx-seedvr2-swift` rather than left as prose:
    /// - **Lowering `videoTemporalTier` does not help** — the tails are constant in T by construction
    ///   (`memoryFrames = kernel − stride`), which V12-S also measured at 784 MiB across the ladder.
    /// - **Changing the tile size does not help either** — fewer, larger tiles hold proportionally
    ///   larger tails. ⚠️ Note this is the *opposite* of the Real-ESRGAN t128 result; different
    ///   mechanism, do not carry the conclusion across packages.
    ///
    /// ✅ **What DOES work is bank eviction, and it shipped** (`mlx-seedvr2-swift` v0.9.0). Un-evicted,
    /// the requirement is `positions × 0.765 GiB`: 512² → 6.9 GiB, SD ×2 → 26.8 GiB, **HD→4K → 170 tile
    /// positions = 130.0 GiB of banks alone**. Set `ForgeEngineService.bankResidencyBudgetBytes` and
    /// everything over the budget round-trips through scratch **bit-identically**, so this gate's
    /// prediction drops to `base + min(extra banks, budget)` — which is what the
    /// `bankResidencyBudgetBytes` parameter here models, and `bankBudgetToFit` inverts.
    ///
    /// 🔑 **Measured end to end, twice** (`probes/v12b2_bank_eviction.out` 2026-07-31;
    /// `probes/v12b3g_bank_granule_io.out` 2026-08-03 with repeated arms in both orders + cooldowns):
    /// **42.02 GiB → 12.69 GiB (3.31×)** at a 1 GiB budget, output **bit-identical** (max |Δ| = 0 over
    /// 18 frames), for **≈+7% wall clock (0–19% across arms)**. The memory model held in both sessions —
    /// the remeasure landed within 0.6% at 1 GiB AND at the `bankBudgetToFit` budget (13.50 vs 13.57
    /// predicted GiB) — which is why this gate is offered as a prediction rather than a guess.
    ///
    /// 🚨 **Cost-figure history, kept honest:** 2.7× → ~6–13% → +48% → ≈+7%. The first two divided
    /// measurements taken under different conditions; the +48% measured the composite but ran the
    /// treatment arm ONCE, and it absorbed session drift (the remeasure's four evicted arms all run
    /// 20–28% below it, with baselines matching). A subsequent A/B also falsified the
    /// "serialization-bound" decomposition — the bank round trip runs at disk rate on both the
    /// safetensors and contiguous-granule transports. *Measure the composite, and repeat the arms.*
    ///
    /// ⚠️ **1 GiB is the extreme budget** — 39 of 40 banks spilling. Use `bankBudgetToFit` and take the
    /// largest budget that fits. ⚠️ The durable price is write volume: 59.7 GiB for 17 frames
    /// (machine-reported by `lastVideoBankStatistics`), ~840 GiB for a 10 s clip. ⚠️ Eviction only moves
    /// `phys_footprint` because the engine caps the MLX buffer cache. ⚠️ Cost band measured on a 128 GB
    /// host; the 16 GB-arm composite is the open receipt.
    ///
    /// - Parameter outputTilePositions: tile positions the output frame will occupy —
    ///   `ceil(outW / step) * ceil(outH / step)` for the configured tile size and overlap
    ///   (default 256/32 → step 224). A 256² output is 1; a 512² output is 9; SD ×2 is ~15.
    /// - Parameter budgetBytes: the memory the host is willing to commit. Pass the engine's own budget
    ///   if you have it; otherwise a fraction of physical memory the app is comfortable occupying.
    /// - Returns: whether it fits, and the predicted peak — so a UI can explain a refusal with a number
    ///   instead of greying a control out silently.
    ///
    /// 🔑 `nonisolated`: this is arithmetic over measured constants, and a host may well want to gate
    /// from a background context before touching the main actor.
    nonisolated public static func videoUpscaleFits(
        outputTilePositions: Int,
        budgetBytes: UInt64,
        bankResidencyBudgetBytes: Int? = nil
    ) -> (fits: Bool, predictedBytes: UInt64) {
        // Measured single-tile peak at the default tier, and the per-extra-tile bank term.
        let singleTileBytes: Double = 11.93 * 1024 * 1024 * 1024
        let perTileBankBytes: Double = 0.82 * 1024 * 1024 * 1024
        let extra = Double(max(0, outputTilePositions - 1))

        // With eviction on, resident banks are capped: the excess spills instead of accumulating.
        // ⚠️ The cap applies to the EXTRA banks only — the active tile's own state is inside
        // `singleTileBytes`, so a budget of 0 is still a valid, fully-spilling configuration.
        let extraBankBytes: Double
        if let cap = bankResidencyBudgetBytes {
            extraBankBytes = min(extra * perTileBankBytes, Double(max(0, cap)))
        } else {
            extraBankBytes = extra * perTileBankBytes
        }
        let predicted = UInt64(singleTileBytes + extraBankBytes)
        return (predicted <= budgetBytes, predicted)
    }

    /// The bank residency budget that would make a given geometry fit — the inverse of the gate, so a
    /// host can turn "this does not fit" into "run it with eviction at N bytes" rather than a refusal.
    ///
    /// Returns `nil` if the geometry cannot fit at any budget (the base residency alone exceeds
    /// `budgetBytes`), which is an honest "no" rather than a budget that would still OOM.
    ///
    /// ⚠️ **A returned budget is not free.** Every byte of bank over it round-trips through scratch once
    /// per chunk — ≈+7% wall clock (0–19% across repeated arms, `probes/v12b3g_bank_granule_io.out`),
    /// and write volume in the hundreds of GiB for a real clip. Offer it where the alternative is not
    /// running at all.
    nonisolated public static func bankBudgetToFit(outputTilePositions: Int,
                                       budgetBytes: UInt64) -> Int? {
        let singleTileBytes: Double = 11.93 * 1024 * 1024 * 1024
        guard Double(budgetBytes) > singleTileBytes else { return nil }
        let perTileBankBytes: Double = 0.82 * 1024 * 1024 * 1024
        let extra = Double(max(0, outputTilePositions - 1))
        let headroom = Double(budgetBytes) - singleTileBytes
        return Int(min(extra * perTileBankBytes, headroom))
    }
}
