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
    /// 🚨 **Check `canRunVideoUpscale(outputTilePositions:)` before offering this on a small-memory
    /// machine.** The temporal path's memory is driven by OUTPUT GEOMETRY — streaming banks are
    /// allocated per spatial tile position, ~0.82 GiB each and roughly constant in the temporal window —
    /// so a big frame is expensive at every tier and **lowering `videoTemporalTier` will not rescue it**
    /// (`GAP-PROGRAM.md` V12-D). The driver clamps what it can, but the honest gate is up front.
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
    /// - Parameter outputTilePositions: tile positions the output frame will occupy —
    ///   `ceil(outW / step) * ceil(outH / step)` for the configured tile size and overlap
    ///   (default 256/32 → step 224). A 256² output is 1; a 512² output is 9; SD ×2 is ~15.
    /// - Parameter budgetBytes: the memory the host is willing to commit. Pass the engine's own budget
    ///   if you have it; otherwise a fraction of physical memory the app is comfortable occupying.
    /// - Returns: whether it fits, and the predicted peak — so a UI can explain a refusal with a number
    ///   instead of greying a control out silently.
    public static func videoUpscaleFits(outputTilePositions: Int,
                                        budgetBytes: UInt64) -> (fits: Bool, predictedBytes: UInt64) {
        // Measured single-tile peak at the default tier, and the per-extra-tile bank term.
        let singleTileBytes: Double = 11.93 * 1024 * 1024 * 1024
        let perTileBankBytes: Double = 0.82 * 1024 * 1024 * 1024
        let extra = Double(max(0, outputTilePositions - 1))
        let predicted = UInt64(singleTileBytes + extra * perTileBankBytes)
        return (predicted <= budgetBytes, predicted)
    }
}
