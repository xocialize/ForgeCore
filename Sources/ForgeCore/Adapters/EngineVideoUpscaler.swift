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

import CoreGraphics
import Foundation
import MLXServeCore
import MLXToolKit

/// What one video upscale produced, plus the facts a receipt should carry.
public struct VideoUpscaleOutcome: Sendable {
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
        guard let packageID = packageID(for: .upscaleQuality) else {
            throw DiagnosticsError.notRegistered(.upscaleQuality, outcome(for: .upscaleQuality)?.failure)
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
            throw DiagnosticsError.unexpectedResponse(.upscaleQuality)
        }
        return VideoUpscaleOutcome(video: upscale.video,
                                   appliedScale: upscale.appliedScale,
                                   seconds: seconds)
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
    /// 🔑 **Measured end to end** (`probes/v12b2_bank_eviction.out`, 17 frames at 40 tile positions,
    /// arm A bracketed either side of arm B): **42.02 GiB → 12.69 GiB (3.31×)** for **+48%** wall clock
    /// at a 1 GiB budget, with output **bit-identical** (max |Δ| = 0 over 18 frames). The model above
    /// extrapolated to 40 positions within 4.5% un-evicted and 2% evicted, which is why this gate is
    /// offered as a prediction rather than a guess.
    ///
    /// 🚨 **That +48% replaces two earlier published estimates, 2.7× and ~6–13%, both wrong for the
    /// same reason** — each divided one measurement by another taken under different conditions rather
    /// than measuring the composite. The round trip is *serialization*-bound (57 tensors per bank
    /// through safetensors), ~7× below raw disk bandwidth; a contiguous bank format is the headroom.
    ///
    /// ⚠️ **1 GiB is the extreme budget** — 39 of 40 banks spilling. Use `bankBudgetToFit` and take the
    /// largest budget that fits; overhead falls as fewer banks spill. ⚠️ The durable price is write
    /// volume: 59.7 GiB for 17 frames, ~840 GiB for a 10 s clip. ⚠️ And eviction only moves
    /// `phys_footprint` because the engine caps the MLX buffer cache.
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
    /// per chunk — ~6–13% wall clock, and write volume in the hundreds of GiB for a real clip. Offer it
    /// where the alternative is not running at all.
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
