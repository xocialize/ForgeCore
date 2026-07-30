//
//  UpscaleDiagnostics.swift — ForgeCore
//
//  A timed, single-image upscale for benchmarking a registered tier.
//
//  🔑 **Why this is in ForgeCore and not in the app that asked for it.** The measurement needs
//  `ImageUpscaleRequest` / `ImageUpscaleResponse` (MLXToolKit) and `PackageID` (MLXServeCore). An app that only
//  links `ForgeCore` cannot name them, and making every consuming app link the engine to time a call
//  would push the MLX seam up into category C, which is the exact drift the category-B split exists to
//  prevent. So the engine-touching half lives here and the app keeps the parts that are genuinely its
//  own: the probe, the process-footprint sampler and the receipt.
//
//  ⚠️ **Load and run are timed separately, and that is load-bearing.** A combined figure hides which
//  tier is slow — the quality tier pays a large one-time weight load that the fast tier does not — and
//  conflating them also lets whichever model happened to be prepared first look faster. Any caller
//  comparing tiers needs the split, so it is in the return type rather than left to the caller.
//
//  ⚠️ This measures the *engine* path, not a raw model call: license gate, admission and the store all
//  participate. That is deliberate — it is the path the product actually ships — but it means these
//  numbers are not comparable to a bare package smoke test.
//

import CoreGraphics
import Foundation
import MLXServeCore
import MLXToolKit   // the request/response contracts live here, not in MLXServeCore

/// What one timed upscale cost. **`peakGPUBytes` is deliberately absent:** MLX's own peak under-reads the
/// engine's admission basis by ~2.5–2.9×, so a memory question has to be answered with the process
/// working set, which only the host app can sample.
public struct UpscaleMeasurement: Sendable {
    /// `prepare(_:package:)` — weight load, one-time per process.
    public let loadSeconds: Double
    /// `run(_:package:)` — the per-image cost, and the only half a "faster at 4×" claim is about.
    public let runSeconds: Double
    public let outputWidth: Int
    public let outputHeight: Int
    /// What the model actually did. ⚠️ May differ from the requested scale — a fixed-factor upscaler
    /// applies its own, and reporting the request instead of the result is how a receipt starts lying.
    public let appliedScale: Int
}

extension ForgeEngineService {

    /// Run one image through a registered upscale tier, timing load and run separately.
    ///
    /// - Parameters:
    ///   - model: which registration to exercise — `.upscale` (Real-ESRGAN, fast) or `.upscaleQuality`
    ///     (SeedVR2-3B, quality). Both back `.imageUpscale`.
    /// - Throws: `DiagnosticsError.notRegistered` when the tier failed to register, carrying the
    ///   registration failure text rather than a bare nil — a license rejection and a missing weights
    ///   folder are different problems and the caller should be able to tell them apart.
    public func measureUpscale(_ image: CGImage,
                               model: ForgeModel,
                               scale: Int) async throws -> UpscaleMeasurement {
        guard let packageID = packageID(for: model) else {
            throw DiagnosticsError.notRegistered(model, outcome(for: model)?.failure)
        }
        let raw = try EngineImageCoding.rawImage(from: image)

        // ⚠️ `prepare` is keyed by CAPABILITY with the package as a disambiguator — which matters here
        // precisely because both tiers back `.imageUpscale`. Passing the package is not optional
        // politeness: omit it and the engine resolves whichever registration it likes, so a run labelled
        // "SeedVR2" could silently measure Real-ESRGAN. That would be an invisible wrong answer, which is
        // the failure mode this whole spike exists to avoid.
        let loadStart = Date()
        _ = try await engine.prepare(model.capability, package: packageID)
        let loadSeconds = Date().timeIntervalSince(loadStart)

        let runStart = Date()
        let response = try await engine.run(ImageUpscaleRequest(image: raw, scale: scale),
                                            package: packageID)
        let runSeconds = Date().timeIntervalSince(runStart)

        guard let upscale = response as? ImageUpscaleResponse else {
            throw DiagnosticsError.unexpectedResponse(model)
        }
        let out = try EngineImageCoding.cgImage(from: upscale.image)
        return UpscaleMeasurement(loadSeconds: loadSeconds,
                                  runSeconds: runSeconds,
                                  outputWidth: out.width,
                                  outputHeight: out.height,
                                  appliedScale: upscale.appliedScale)
    }

    public enum DiagnosticsError: Error, CustomStringConvertible {
        case notRegistered(ForgeModel, String?)
        case unexpectedResponse(ForgeModel)

        public var description: String {
            switch self {
            case .notRegistered(let model, let why):
                return "\(model.modelName) is not registered — \(why ?? "no failure recorded")"
            case .unexpectedResponse(let model):
                return "\(model.modelName) returned an unexpected response type"
            }
        }
    }
}
