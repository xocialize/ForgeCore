//
//  EngineImageEnhancer.swift
//  ForgeCore
//
//  ForgeOptimizerKit's `ImageEnhancer` seam, engine-backed: NAFNet restore → the upscale tier the caller chose —
//  NERVE for `.fast` (Real-ESRGAN until 2026-09-28), RealPLKSR for `.best` (AB-T-0187).
//  Both wrappers mirror `.rawBGRA8`, so the whole chain stays raw — no PNG hop between stages.
//

import Foundation
import CoreGraphics
import OSLog
import ForgeOptimizerKit
import MLXServeCore
import MLXToolKit

@MainActor
public final class EngineImageEnhancer: ImageEnhancer {

    private let service: ForgeEngineService

    /// The scale the upscale tier actually applied on the most recent enhance, or nil if no upscale ran.
    ///
    /// History: Real-ESRGAN's `general` was a fixed-4× model, and asking it for 2× used to yield 4× pixels while the
    /// recipe recorded "2×" — a product cannot offer an option that silently does something else.
    /// `mlx-realesrgan-swift` 0.5.0 fixed the model side by post-downsampling to the requested scale; NERVE runs ×2
    /// natively and post-downsamples only ×1/×3, and RealPLKSR post-downsamples a ×2 request from its native ×4. This
    /// records what came back so the receipt describes the output rather than the request.
    public private(set) var lastAppliedScale: Int?

    public init(service: ForgeEngineService) {
        self.service = service
    }

    /// The one registration each tier runs, and the only one: `.fast` → NERVE (`.upscale`), `.best` → RealPLKSR
    /// (`.upscaleFidelity`). The routing axis is cost — RealPLKSR beat the previous fast tier at all 12 degradation
    /// rungs (AB-R-0285) at ~5× NERVE's wall time (0.75 s vs 3.7 s at 1080p ×4, M5 Max) — so no classifier picks it.
    /// HEART (`.upscaleHEART`), VOSR2 (`.upscaleGenerative`) and Nacre (`.upscaleNacre`) are deliberately off this map:
    /// exposing them is a product call, and all three stay reachable through `measureUpscale` and a host's own routing.
    ///
    /// `.liveAction` maps to FlashVSR (`.upscaleVideoLiveAction`) so the receipt mapping covers every tier, but this
    /// enhancer never runs it: it is a whole-clip video tier served by `EngineVideoUpscaler`, and a still asked for it
    /// is refused (by the Kit first, and by `availability(of:)` / `enhanceReporting` here too).
    public static func model(for tier: UpscaleTier) -> ForgeModel {
        switch tier {
        case .fast: return .upscale
        case .best: return .upscaleFidelity
        case .liveAction: return .upscaleVideoLiveAction
        }
    }

    /// The inverse of `model(for:)` — the tier a registration IS, which is what the receipt reports back.
    public static func tier(for model: ForgeModel) -> UpscaleTier? {
        UpscaleTier.allCases.first { Self.model(for: $0) == model }
    }

    public func enhance(_ image: CGImage, options: Options) async throws -> CGImage {
        try await enhanceReporting(image, options: options).image
    }

    public func enhanceReporting(_ image: CGImage, options: Options) async throws -> EnhanceOutcome {
        let engine = service.engine
        var current = try EngineImageCoding.rawImage(from: image)

        guard let restoreID = service.packageID(for: .restore) else {
            throw EnhanceError.unavailable(.restore, service.outcome(for: .restore)?.failure)
        }
        let restoreStart = Date()
        let restoreResponse = try await engine.run(ImageRestoreRequest(image: current), package: restoreID)
        guard let restored = (restoreResponse as? ImageRestoreResponse)?.image else {
            throw EnhanceError.unexpectedResponse(.imageRestore)
        }
        current = restored
        log.notice("Enhance: restore (NAFNet) \(image.width, privacy: .public)×\(image.height, privacy: .public) in \(Date().timeIntervalSince(restoreStart), format: .fixed(precision: 2), privacy: .public) s")

        guard let requestedScale = options.upscale.requestedScale else {
            lastAppliedScale = nil
            return EnhanceOutcome(image: try EngineImageCoding.cgImage(from: current))
        }
        return try await upscale(current, scale: requestedScale, tier: options.upscaleTier,
                                 source: (image.width, image.height), label: "Enhance")
    }

    /// The upscale stage alone — no NAFNet restore. ForgeOptimizerKit's `conform(_:to:quality: .quality(tier))`
    /// calls this (Kit ≥ 0.21.0, AB-T-0195): a conform is resize glue between pipeline stages, so a restore inside it
    /// would change the content the next stage receives. Same tier map, refusals and report as `enhanceReporting`.
    public func upscaleReporting(_ image: CGImage, factor: UpscaleFactor, tier: UpscaleTier) async throws -> EnhanceOutcome {
        guard let requestedScale = factor.requestedScale else {
            lastAppliedScale = nil
            return EnhanceOutcome(image: image)
        }
        return try await upscale(try EngineImageCoding.rawImage(from: image), scale: requestedScale, tier: tier,
                                 source: (image.width, image.height), label: "Conform")
    }

    /// Run `tier`'s one registration on `current` at `scale`, and report the registration that ran.
    private func upscale(_ current: Image, scale requestedScale: Int, tier: UpscaleTier,
                         source: (width: Int, height: Int), label: String) async throws -> EnhanceOutcome {
        // The tier's own backer or nothing. A tier that cannot run throws with its reason; the other tier never
        // stands in for it, because a "Best" that ran Fast would ship one model's pixels under another's name.
        guard !tier.isVideoOnly else {
            throw EnhanceError.videoOnlyTier(tier)
        }
        let model = Self.model(for: tier)
        guard let upscaleID = service.packageID(for: model) else {
            throw EnhanceError.unavailable(model, service.outcome(for: model)?.failure)
        }
        let upscaleStart = Date()
        let response: any CapabilityResponse
        do {
            response = try await service.engine.run(ImageUpscaleRequest(image: current, scale: requestedScale),
                                                    package: upscaleID)
        } catch {
            if let why = Self.governorRefusal(error) { throw EnhanceError.declined(model, why) }
            throw error
        }
        guard let upscaleResponse = response as? ImageUpscaleResponse else {
            throw EnhanceError.unexpectedResponse(.imageUpscale)
        }
        let out = try EngineImageCoding.cgImage(from: upscaleResponse.image)
        lastAppliedScale = upscaleResponse.appliedScale
        if upscaleResponse.appliedScale != requestedScale {
            log.notice("Upscale requested \(requestedScale)x, model applied \(upscaleResponse.appliedScale)x — recording the applied factor")
        }
        log.notice("\(label, privacy: .public): upscale (\(model.modelName, privacy: .public), \(tier.rawValue, privacy: .public)) \(source.width, privacy: .public)×\(source.height, privacy: .public) → \(out.width, privacy: .public)×\(out.height, privacy: .public) in \(Date().timeIntervalSince(upscaleStart), format: .fixed(precision: 2), privacy: .public) s")

        // Reported from the registration that ran, not from `options` (AB-T-0013's rule, one field over).
        return EnhanceOutcome(image: out, upscaleTier: Self.tier(for: model), upscaleModel: model.modelName)
    }

    /// Can `tier` run on this Mac, and if not, why — the governor's own arithmetic over the model's declared
    /// footprint, evaluated without loading anything. The Kit asks this before every upscale and fails the item on
    /// a refusal; ForgeOptimizerUI asks it to show a tier as unavailable with this reason.
    ///
    /// Two refusals are decided here: the tier's model did not register (license, device, eligibility — the
    /// registration failure text says which), or its declared working set exceeds the engine's whole budget, which
    /// no amount of evicting can fix. A run that the governor still declines at admission (another app tenant
    /// holding memory, a workload over its declared ceiling) fails with `EnhanceError.declined` and its reason.
    public func availability(of tier: UpscaleTier) async -> UpscaleTierAvailability {
        let model = Self.model(for: tier)
        guard !tier.isVideoOnly else {
            return .unavailable(tier, model: model.modelName, reason: UpscaleTier.liveActionStillReason)
        }
        guard service.packageID(for: model) != nil else {
            let why = service.outcome(for: model)?.failure ?? "registration has not run"
            return .unavailable(tier, model: model.modelName, reason: "\(model.modelName) is not registered — \(why)")
        }
        guard let verdict = await service.admissibility(for: model) else {
            return .available(tier, model: model.modelName)
        }
        if !verdict.eligibility.isEligible {
            return .unavailable(tier, model: model.modelName,
                                reason: "this Mac cannot run \(model.modelName) — \(verdict.eligibility)")
        }
        if !verdict.fitsBudget {
            let budget = await service.memoryBudgetBytes
            return .unavailable(tier, model: model.modelName,
                                reason: "\(model.modelName) needs \(Self.gb(verdict.footprint)); the engine's memory budget on this Mac is \(Self.gb(budget))")
        }
        return .available(tier, model: model.modelName)
    }

    public enum EnhanceError: Error, CustomStringConvertible {
        /// The model's registration failed (or has not run); carries the registration failure text.
        case unavailable(ForgeModel, String?)
        /// The engine's memory governor refused the run at admission; carries its reason in the user's terms.
        case declined(ForgeModel, String)
        case unexpectedResponse(Capability)
        /// A whole-clip video tier (`UpscaleTier.isVideoOnly`) asked of the per-frame enhancer.
        case videoOnlyTier(UpscaleTier)

        public var description: String {
            switch self {
            case .unavailable(let model, let why):
                return "\(model.modelName) is not registered — \(why ?? "registration has not run")"
            case .declined(let model, let why):
                return "\(model.modelName) was declined by the engine's memory governor — \(why)"
            case .unexpectedResponse(let capability):
                return "unexpected response from \(capability)"
            case .videoOnlyTier:
                return UpscaleTier.liveActionStillReason
            }
        }
    }

    /// A memory-governor refusal from `engine.run`, in the user's terms; nil for any other error.
    static func governorRefusal(_ error: Error) -> String? {
        guard let refusal = error as? EngineError else { return nil }
        switch refusal {
        case .exceedsMemoryBudget(let required, let budget):
            return "it needs \(gb(required)) and the budget on this Mac is \(gb(budget))"
        case .workloadExceedsMemoryBudget(_, _, _, let required, let budget):
            return "this input needs \(gb(required)) and the budget on this Mac is \(gb(budget))"
        case .externalTenantsHoldMemory(let required, let external, let budget):
            return "it needs \(gb(required)) while other work in this app holds \(gb(external)) of the \(gb(budget)) budget"
        case .workloadExceedsDeclaredCeiling(_, let axis, let requested, let ceiling):
            return "this input (\(requested) \(axis)) is past the \(ceiling) its memory declaration was measured at"
        default:
            return nil
        }
    }

    static func gb(_ bytes: UInt64) -> String { String(format: "%.1f GB", Double(bytes) / 1e9) }
}

extension UpscaleFactor {
    /// Integer scale for the engine request; nil means "skip the upscale stage".
    var requestedScale: Int? {
        switch self {
        case .none: return nil
        case .x2:   return 2
        case .x4:   return 4
        }
    }
}
