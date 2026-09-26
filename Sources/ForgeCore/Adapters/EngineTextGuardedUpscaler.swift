//
//  EngineTextGuardedUpscaler.swift
//  ForgeCore
//
//  MaskKit's `TextLegibilityGuard` in front of the GENERATIVE stills upscale tier (SeedVR2-3B today, the VOSR2
//  registration when it lands — AB-T-0184).
//
//  Why (VOSR-EVAL §10–§11, 2026-09-26): a generative upscaler fixes more wrong words than it breaks, but turns
//  ~10 % of the words the input already reads with certainty into plausible wrong words ("Mummie" → "Manunie").
//  For signage that is the failure that matters. The measured policy that removes it without making anything
//  else worse is the image-level DOCUMENT GATE: a page that already reads as legible text goes to the
//  non-generative tier whole (corruption 59 → 3 on such pages; the gate never fired on 1,200 illegible inputs).
//  Word-level compositing is available as an opt-in (`policy.wordProtection`) and measured net-negative off
//  dense legible pages — a mixed-fidelity line is a new degradation for the recognizer.
//
//  Three routes, decided on the INPUT before any model runs:
//    .keepBase      — a legible document page → the base tier alone (no patchwork, no 3B model spun up);
//    .generative    — the default → the generative tier alone;
//    .protectWords  — opt-in → both tiers, composited under the glyph-proportional word mask.
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
    /// The guard's policy — MaskKit's measured defaults: document gate on, word protection OFF.
    public var policy: TextLegibilityGuard.Policy
    /// The generative tier — SeedVR2-3B today; the VOSR2 registration when it lands (AB-T-0184).
    public var generative: ForgeModel
    /// The non-generative tier the protected words come from. Real-ESRGAN: feed-forward, deterministic —
    /// "conservative non-generative restoration inside the mask" (the N7 rule).
    public var base: ForgeModel

    public init(service: ForgeEngineService,
                policy: TextLegibilityGuard.Policy = .default,
                generative: ForgeModel = .upscaleQuality,
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
