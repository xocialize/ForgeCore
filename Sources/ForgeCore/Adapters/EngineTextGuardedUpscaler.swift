//
//  EngineTextGuardedUpscaler.swift
//  ForgeCore
//
//  MaskKit's `TextLegibilityGuard` in front of a GENERATIVE stills upscale tier — `.upscaleQuality` (SeedVR2-3B)
//  or `.upscaleGenerative` (VOSR2, registered 2026-09-26; AB-T-0184/0185) — chosen per instance by `generative:`.
//
//  Why (VOSR-EVAL §10–§11, 2026-09-26): a generative upscaler fixes more wrong words than it breaks, but turns
//  ~10 % of the words the input already reads with certainty into plausible wrong words ("Mummie" → "Manunie").
//  For signage that is the failure that matters. Against VOSR2 at ×1 the image-level document gate was the
//  policy that removed it without making anything else worse.
//
//  🚨 Measured END TO END through THIS adapter (`GuardedUpscaleLiveTests`, three ScreenSR pages, Vision word-F1
//  vs the HR page; input reads 0.937 / 0.846 / 0.116):
//    SeedVR2-3B ×2 (2026-09-26 §11.1)   generative 0.937 / 0.913 / 0.651   keepBase 0.800 / 0.633 / 0.426
//    VOSR2 ×2       (2026-09-26 §11.2)   generative 0.952 / 0.928 / 0.635   keepBase 0.800 / 0.633 / 0.426
//    VOSR2 ×4                            generative 0.920 / 0.926 / 0.625   keepBase 0.778 / 0.000 / 0.356
//    SeedVR2-3B ×4                       generative 0.902 / 0.862 / 0.623   (same base)
//  The generative route read best on every page for BOTH tiers at BOTH scales, and the base tier (Real-ESRGAN)
//  read worse than the un-restored input on the dense pages — it breaks and merges small strokes, and at ×4 it
//  wiped page 090 to 0.000. The word composite sat between the two. So for these tier pairs the gate would have
//  made text worse; both routes are OPT-IN in the policy, and a tier pair earns a route only by being measured
//  through this adapter (the live test is the harness). The ×1 letter-rewrite hazard (AB-L-0172) is real but was
//  measured against the un-restored input as the base, which this adapter's base tier is not — ×1 stays unmeasured.
//  VOSR2 is the default generative tier since v0.23.0: it reads at least as well and runs ~2× faster at ×4.
//
//  Three routes, decided on the INPUT before any model runs:
//    .generative    — the default → the generative tier alone;
//    .keepBase      — opt-in (`documentGate`) → a legible document page → the base tier alone;
//    .protectWords  — opt-in (`wordProtection`) → both tiers, composited under the glyph-proportional word mask.
//

import Foundation
import CoreGraphics
import OSLog
import MaskKit
import MLXServeCore
import MLXToolKit

@MainActor
public final class EngineTextGuardedUpscaler {

    public struct Output: Sendable {
        public let image: CGImage
        public let verdict: TextLegibilityGuard.Verdict
        public let route: TextLegibilityGuard.Verdict.Recommendation
        /// The protect mask at output size when the route was `.protectWords`, for a UI overlay or a receipt.
        public let mask: CGImage?
        public let generativeSeconds: Double
        public let baseSeconds: Double
        public let assessSeconds: Double
    }

    public enum GuardedUpscaleError: Error, CustomStringConvertible {
        case notRegistered(ForgeModel, String?)
        case unexpectedResponse(ForgeModel)
        case sizeMismatch(base: (Int, Int), generative: (Int, Int))
        public var description: String {
            switch self {
            case .notRegistered(let m, let why): return "\(m.modelName) is not registered — \(why ?? "no failure recorded")"
            case .unexpectedResponse(let m): return "\(m.modelName) returned an unexpected response type"
            case .sizeMismatch(let b, let g): return "base \(b.0)×\(b.1) and generative \(g.0)×\(g.1) outputs differ in size"
            }
        }
    }

    private let service: ForgeEngineService
    /// The guard's policy — MaskKit's defaults route `.generative` for everything; `documentGate` and
    /// `wordProtection` are opt-in per measured tier pair.
    public var policy: TextLegibilityGuard.Policy
    /// The generative tier: `.upscaleGenerative` (VOSR2, the default since v0.23.0 — measured through this adapter
    /// at ×2 and ×4, VOSR-EVAL §11.2) or `.upscaleQuality` (SeedVR2-3B). The routes a tier pair earns are decided by
    /// `GuardedUpscaleLiveTests` + the OCR check, never assumed from another pair.
    public var generative: ForgeModel
    /// The non-generative tier the protected words come from — the fast tier (`.upscale`: NERVE since 2026-09-28,
    /// Real-ESRGAN in the 2026-09-26 route measurements above): feed-forward, deterministic — "conservative
    /// non-generative restoration inside the mask" (the N7 rule). The routes were earned with Real-ESRGAN as the base;
    /// re-run `GuardedUpscaleLiveTests` before trusting an opt-in route with the new base.
    public var base: ForgeModel

    public init(service: ForgeEngineService,
                policy: TextLegibilityGuard.Policy = .default,
                generative: ForgeModel = .upscaleGenerative,
                base: ForgeModel = .upscale) {
        precondition(generative.capability == .imageUpscale && base.capability == .imageUpscale)
        self.service = service
        self.policy = policy
        self.generative = generative
        self.base = base
    }

    /// Assess the input, then run the tier(s) the verdict calls for and composite.
    public func upscale(_ image: CGImage, scale: Int) async throws -> Output {
        let t0 = Date()
        let verdict = try await Self.assess(image, policy: policy)
        let assessSeconds = Date().timeIntervalSince(t0)
        log.notice("text guard: \(verdict.regions.count) lines read, \(verdict.protected.count) certain words, route \(verdict.recommendation.rawValue, privacy: .public) (\(String(format: "%.2f", assessSeconds))s)")

        switch verdict.recommendation {
        case .generative:
            let (out, s) = try await run(generative, image: image, scale: scale)
            return Output(image: out, verdict: verdict, route: .generative, mask: nil,
                          generativeSeconds: s, baseSeconds: 0, assessSeconds: assessSeconds)
        case .keepBase:
            let (out, s) = try await run(base, image: image, scale: scale)
            return Output(image: out, verdict: verdict, route: .keepBase, mask: nil,
                          generativeSeconds: 0, baseSeconds: s, assessSeconds: assessSeconds)
        case .protectWords:
            let (gen, gs) = try await run(generative, image: image, scale: scale)
            let (bas, bs) = try await run(base, image: image, scale: scale)
            guard gen.width == bas.width, gen.height == bas.height else {
                throw GuardedUpscaleError.sizeMismatch(base: (bas.width, bas.height), generative: (gen.width, gen.height))
            }
            let mask = try TextLegibilityGuard.mask(for: verdict, width: gen.width, height: gen.height, policy: policy)
            let out = try TextLegibilityGuard.composite(base: bas, generative: gen, mask: mask)
            return Output(image: out, verdict: verdict, route: .protectWords, mask: mask,
                          generativeSeconds: gs, baseSeconds: bs, assessSeconds: assessSeconds)
        }
    }

    /// Vision is synchronous and a tiled frame takes real time — never on the main actor.
    nonisolated static func assess(_ image: CGImage, policy: TextLegibilityGuard.Policy) async throws -> TextLegibilityGuard.Verdict {
        try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                continuation.resume(with: Swift.Result { try TextLegibilityGuard.assess(image, policy: policy) })
            }
        }
    }

    private func run(_ model: ForgeModel, image: CGImage, scale: Int) async throws -> (CGImage, Double) {
        guard let packageID = service.packageID(for: model) else {
            throw GuardedUpscaleError.notRegistered(model, service.outcome(for: model)?.failure)
        }
        let raw = try EngineImageCoding.rawImage(from: image)
        // prepare keyed by capability + package: both tiers back .imageUpscale (see UpscaleDiagnostics).
        _ = try await service.engine.prepare(model.capability, package: packageID)
        let start = Date()
        let response = try await service.engine.run(ImageUpscaleRequest(image: raw, scale: scale), package: packageID)
        guard let up = response as? ImageUpscaleResponse else { throw GuardedUpscaleError.unexpectedResponse(model) }
        return (try EngineImageCoding.cgImage(from: up.image), Date().timeIntervalSince(start))
    }
}
