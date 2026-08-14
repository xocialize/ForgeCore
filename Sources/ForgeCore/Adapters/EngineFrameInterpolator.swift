//
//  EngineFrameInterpolator.swift — ForgeCore
//
//  The `frameInterpolate` capability, wired 2026-08-14 (AB-T-0038). This is the app-facing seam for
//  frame-rate up-conversion; before this, RIFE 4.25 was ported and parity-locked (2026-06-18) but had
//  no registration and no binding anywhere — no app could reach it.
//
//  ⚠️ **This is a whole-clip call, not a frame seam.** The package decodes, interpolates pair by pair,
//  and re-encodes internally (AVFoundation via `frame-stream-native`; no FFmpeg). Frames stream
//  decode → interpolate → encode with per-source-frame cancellation (the package's CAN seams), so
//  memory stays bounded — but wall clock scales with clip length, and a UI must treat this as a long
//  job with progress and cancellation, never as a request that might return quickly.
//
//  ⚠️ **The output encode is frame-stream-native's SOFTWARE-HEVC default, and that default is
//  load-bearing.** Hardware VideoToolbox HEVC right after heavy Metal compute stalls
//  (`isReadyForMoreMediaData` never recovers) — and on quarantined builds (AB-B-0002, 26A5406e) the
//  teardown panics the machine. No Forge surface may opt back into hardware (`FRAMESTREAM_ENCODE`).
//

import Foundation
import MLXServeCore
import MLXToolKit

/// What one frame interpolation produced, plus the facts a receipt should carry.
public struct FrameInterpolateOutcome: Sendable {
    public let video: Video
    /// The multiplier the package actually applied. ⚠️ May differ from the request — report this,
    /// not the ask.
    public let appliedFactor: Int
    public let seconds: Double

    /// Wall-clock per *output* frame, when the container reported a frame rate and duration — the
    /// number a UI needs to estimate a job.
    public var secondsPerOutputFrame: Double? {
        guard let fps = video.frameRate, let duration = video.durationSeconds,
              fps > 0, duration > 0 else { return nil }
        return seconds / (fps * duration)
    }
}

extension ForgeEngineService {

    /// Interpolate a whole clip to `factor`× its frame rate through the RIFE path.
    ///
    /// Factor 2 (the package default) synthesizes one midpoint per adjacent pair at t=0.5; factor N
    /// synthesizes N−1 at t=k/N. Output is re-encoded at factor× the source fps (HEVC,
    /// BT.709-tagged). ≤ 1080p inputs run the whole-frame path; above it the package's tiler bounds
    /// memory (tile 1024 / overlap 128 — its measured 4K-admissible geometry).
    ///
    /// - Parameter factor: `nil` uses the package's default (2).
    public func interpolateVideo(_ video: Video, factor: Int? = nil) async throws -> FrameInterpolateOutcome {
        guard let packageID = packageID(for: .frameInterpolate) else {
            throw DiagnosticsError.notRegistered(.frameInterpolate, outcome(for: .frameInterpolate)?.failure)
        }
        _ = try await engine.prepare(.frameInterpolate, package: packageID)

        let started = Date()
        let response = try await engine.run(FrameInterpolateRequest(video: video, factor: factor),
                                            package: packageID)
        let seconds = Date().timeIntervalSince(started)
        guard let interpolated = response as? FrameInterpolateResponse else {
            throw DiagnosticsError.unexpectedResponse(.frameInterpolate)
        }
        return FrameInterpolateOutcome(video: interpolated.video,
                                       appliedFactor: interpolated.appliedFactor,
                                       seconds: seconds)
    }
}
