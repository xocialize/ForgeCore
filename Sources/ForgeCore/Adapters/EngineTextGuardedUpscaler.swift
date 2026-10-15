//
//  EngineTextGuardedUpscaler.swift
//  ForgeCore
//
//  MaskKit's `TextLegibilityGuard` in front of a GENERATIVE stills upscale tier — `.upscaleQuality` (SeedVR2-3B),
//  `.upscaleGenerative` (VOSR2, registered 2026-09-26; AB-T-0184/0185) or `.upscaleNacre` (Nacre v1, the provenance-clean
//  variant, registered 2026-10-05; AB-A-0133) — chosen per instance by `generative:`.
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
//  — those with Real-ESRGAN as the base. Re-measured 2026-09-29 with NERVE as the base (§11.3, VOSR2 generative):
//    ×2   keepBase 0.894 / 0.810 / 0.597   protectWords 0.898 / 0.823 / 0.629   generative 0.952 / 0.928 / 0.635
//    ×4   keepBase 0.876 / 0.754 / 0.488   protectWords 0.876 / 0.765 / 0.635   generative 0.920 / 0.926 / 0.625
//  The generative route read best on every page for BOTH generative tiers at BOTH scales with Real-ESRGAN as the
//  base, and in 5 of 6 cells with NERVE (the sixth: the composite +0.010 on the sparse page at ×4, and −0.006 on
//  the same page at ×2 — noise). NERVE is a much better base than Real-ESRGAN, which broke and merged small strokes
//  and at ×4 wiped page 090 to 0.000 — but NERVE still reads BELOW the un-restored input on both dense pages, so the
//  document gate would still route legible pages to a worse result. So for every tier pair measured, both routes
//  stay OPT-IN in the policy, and a tier pair earns a route only by being measured through this adapter (the live
//  test is the harness). The ×1 letter-rewrite hazard (AB-L-0172) is real but was measured against the un-restored
//  input as the base, which this adapter's base tier is not — ×1 stays unmeasured.
//  VOSR2 is the default generative tier since v0.23.0: it reads at least as well and runs ~2× faster at ×4.
//
//  🔑 **Nacre is the first pair to EARN an opt-in route** (2026-10-05, AB-A-0133; NERVE base; pages 010 / 095 — page 090
//  is not in the local fixtures; input reads 0.937 / 0.116):
//    ×2   generative 0.820 / 0.551   keepBase 0.894 / 0.597   protectWords 0.894 / 0.572
//    ×4   generative 0.816 / 0.521   keepBase 0.876 / 0.488   protectWords 0.876 / 0.571
//  The generative route is best in NONE of the four cells. Nacre rewrites words the input already reads ("houses" →
//  "bouses", "she" → "lhe") and lifts the paper white (p95 225 vs the HR page's 214), where VOSR2 holds both.
//  protectWords never loses to generative (+0.021 … +0.074) and has the best worst case (0.571); the document gate
//  adds a win on 095 ×2 but loses 095 ×4 (0.488). So `measuredPolicy(for: .upscaleNacre)` turns word protection ON
//  and leaves the document gate off, and that is the policy a Nacre guard gets unless the caller passes one. The
//  base routes reproduced the 2026-09-29 NERVE receipts exactly (0.894 / 0.597 ×2, 0.876 / 0.488 ×4).
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
    /// `wordProtection` are opt-in per measured tier pair. Defaults to `measuredPolicy(for: generative)` at init.
    /// It does not follow later changes to `generative`.
    public var policy: TextLegibilityGuard.Policy
    /// The generative tier: `.upscaleGenerative` (VOSR2, the default since v0.23.0 — measured through this adapter
    /// at ×2 and ×4, VOSR-EVAL §11.2), `.upscaleQuality` (SeedVR2-3B) or `.upscaleNacre` (Nacre, the provenance-clean
    /// choice). The routes a tier pair earns are decided by `GuardedUpscaleLiveTests` + the OCR check, never assumed
    /// from another pair.
    public var generative: ForgeModel
    /// The non-generative tier the protected words come from — the fast tier (`.upscale`: NERVE since 2026-09-28,
    /// Real-ESRGAN in the 2026-09-26 route measurements above): feed-forward, deterministic — "conservative
    /// non-generative restoration inside the mask" (the N7 rule). Re-measured with NERVE as the base 2026-09-29
    /// (VOSR-EVAL §11.3): no opt-in route is earned for the VOSR2 pair at ×2 or ×4. A new base tier means re-running
    /// `GuardedUpscaleLiveTests` before trusting any opt-in route with it.
    public var base: ForgeModel
    /// An optional second reader for WORD CONFIRMATION (`policy.wordConfirmation`): the protected words become the
    /// certain Vision words this reader also read (`TextLegibilityGuard.confirm`). `SenseNovaTextReader.resident(in:
    /// packageID:)` backs it with the U1 package when that is loaded; nil, or a reader that fails, leaves Vision's
    /// verdict as it was. It changes WHICH words `.protectWords` freezes, never the route itself.
    public var secondReader: (any TextSecondReader)?

    /// The policy a tier pair has EARNED through this adapter (`GuardedUpscaleLiveTests` + the OCR check, NERVE as the
    /// base). Every pair gets MaskKit's all-opt-in `.default` except the measured exceptions:
    ///   • `.upscaleNacre` — word protection ON, document gate off: the generative route was best in none of the four
    ///     measured cells, and protectWords never lost to it (2026-10-05, header table).
    /// A pair not listed here has earned nothing beyond the default — measure it before adding a case.
    public static func measuredPolicy(for generative: ForgeModel) -> TextLegibilityGuard.Policy {
        switch generative {
        case .upscaleNacre:
            var policy = TextLegibilityGuard.Policy.default
            policy.wordProtection = true
            return policy
        default:
            return .default
        }
    }

    /// - Parameter policy: `nil` (the default) takes `measuredPolicy(for: generative)`; pass `.default` to force
    ///   MaskKit's all-opt-in policy (the raw generative route) for any tier.
    public init(service: ForgeEngineService,
                policy: TextLegibilityGuard.Policy? = nil,
                generative: ForgeModel = .upscaleGenerative,
                base: ForgeModel = .upscale,
                secondReader: (any TextSecondReader)? = nil) {
        precondition(generative.capability == .imageUpscale && base.capability == .imageUpscale)
        self.service = service
        self.policy = policy ?? Self.measuredPolicy(for: generative)
        self.generative = generative
        self.base = base
        self.secondReader = secondReader
    }

    /// Assess the input, then run the tier(s) the verdict calls for and composite.
    public func upscale(_ image: CGImage, scale: Int) async throws -> Output {
        let t0 = Date()
        var verdict = try await Self.assess(image, policy: policy)
        if policy.wordConfirmation, let reader = secondReader, !verdict.protected.isEmpty {
            verdict = await TextLegibilityGuard.confirm(verdict, image: image, reader: reader, policy: policy)
        }
        let assessSeconds = Date().timeIntervalSince(t0)
        let confirmed = verdict.confirmed.map { " (\($0.words) of \($0.ofCertain) confirmed by the second reader)" } ?? ""
        log.notice("text guard: \(verdict.regions.count) lines read, \(verdict.protected.count) certain words\(confirmed, privacy: .public), route \(verdict.recommendation.rawValue, privacy: .public) (\(String(format: "%.2f", assessSeconds))s)")

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
        // Each tier gets its own depth (AB-T-0233): a 16-bit image goes to a tier that keeps 16 bits (VOSR2 ≥ 0.3.0, NERVE ≥
        // 0.2.0) as `.rawRGBA16`, and to any other (Nacre) at 8 bits. The composite then keeps 16 bits when either tier did.
        let raw = try EngineImageCoding.rawImage(from: image, keepingDepth: await service.keepsSixteenBits(model))
        // prepare keyed by capability + package: both tiers back .imageUpscale (see UpscaleDiagnostics).
        _ = try await service.engine.prepare(model.capability, package: packageID)
        let start = Date()
        let response = try await service.engine.run(ImageUpscaleRequest(image: raw, scale: scale), package: packageID)
        guard let up = response as? ImageUpscaleResponse else { throw GuardedUpscaleError.unexpectedResponse(model) }
        return (try EngineImageCoding.cgImage(from: up.image), Date().timeIntervalSince(start))
    }
}
