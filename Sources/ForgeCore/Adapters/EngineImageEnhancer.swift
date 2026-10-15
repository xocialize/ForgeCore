//
//  EngineImageEnhancer.swift
//  ForgeCore
//
//  ForgeOptimizerKit's `ImageEnhancer` seam, engine-backed: NAFNet restore → the upscale tier the caller chose —
//  NERVE for `.fast` (Real-ESRGAN until 2026-09-28), RealPLKSR for `.best` (AB-T-0187), and for `.instant` Anime4K's
//  upscale-only VL on Metal shaders, outside the engine (AB-D-0146).
//  Both wrappers mirror `.rawBGRA8`, so the whole chain stays raw — no PNG hop between stages.
//  16 bits (AB-A-0158): each hop sends a deep image as `.rawRGBA16` when that hop's registration keeps 16 bits
//  (`ForgeEngineService.keepsSixteenBits` — NAFNet ≥ 0.6.0, RealPLKSR, NERVE and HEART ≥ 0.2.0), and `.rawBGRA8` otherwise, so a
//  16-bit source keeps its depth through every model that can carry it and loses it only at one that cannot. The restore
//  hop's 16-bit answer goes on to the upscale hop as it is only when that tier keeps 16 bits too (`image(_:forHopKeepingDepth:)`).
//

import Foundation
import CoreGraphics
import OSLog
import ForgeOptimizerKit
import Anime4KMetal
import MaskKit
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

    /// The Instant tier's backer, created on first use. It holds compiled Metal pipelines only; its textures are released
    /// after every call.
    private var instant: Anime4KUpscaler?

    public init(service: ForgeEngineService) {
        self.service = service
    }

    /// The one registration each tier runs, and the only one: `.fast` → NERVE (`.upscale`), `.best` → RealPLKSR
    /// (`.upscaleFidelity`). The routing axis is cost — RealPLKSR beat the previous fast tier at all 12 degradation
    /// rungs (AB-R-0285) at ~5× NERVE's wall time (0.75 s vs 3.7 s at 1080p ×4, M5 Max) — so no classifier picks it.
    /// The generative stills tiers (AB-D-0111, 0.33.0): `.generative` → VOSR2 (`.upscaleGenerative`), `.generativeClean` →
    /// Nacre (`.upscaleNacre`), each run behind `EngineTextGuardedUpscaler` with the policy its pair earned
    /// (`measuredPolicy(for:)`). HEART (`.upscaleHEART`) stays off this map: its opt-in slot is reachable through
    /// `measureUpscale` and a host's own routing (AB-A-0128).
    ///
    /// `.liveAction` maps to FlashVSR (`.upscaleVideoLiveAction`) so the receipt mapping covers every tier, but this
    /// enhancer never runs it: it is a whole-clip video tier served by `EngineVideoUpscaler`, and a still asked for it
    /// is refused (by the Kit first, and by `availability(of:)` / `enhanceReporting` here too).
    ///
    /// `.instant` has no registration, so it maps to nil (0.40.0). It is the efficiency tier (AB-D-0146): Anime4K's
    /// upscale-only VL network as Metal compute shaders (`anime4k-metal`), run directly on the GPU with no engine, no
    /// admission, no download and no memory tier. It reports the model `instantModelName`.
    public static func model(for tier: UpscaleTier) -> ForgeModel? {
        switch tier {
        case .fast: return .upscale
        case .best: return .upscaleFidelity
        case .liveAction: return .upscaleVideoLiveAction
        case .generative: return .upscaleGenerative
        case .generativeClean: return .upscaleNacre
        case .instant: return nil
        }
    }

    /// What the receipt calls the Instant tier's pixels.
    public static let instantModelName = "Anime4K VL"

    /// The inverse of `model(for:)` — the tier a registration IS, which is what the receipt reports back.
    public static func tier(for model: ForgeModel) -> UpscaleTier? {
        UpscaleTier.allCases.first { Self.model(for: $0) == model }
    }

    public func enhance(_ image: CGImage, options: Options) async throws -> CGImage {
        try await enhanceReporting(image, options: options).image
    }

    public func enhanceReporting(_ image: CGImage, options: Options) async throws -> EnhanceOutcome {
        let engine = service.engine
        var current = try EngineImageCoding.rawImage(from: image, keepingDepth: await service.keepsSixteenBits(.restore))

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
        if tier == .instant {
            return try await instantUpscale(image, scale: requestedScale, label: "Conform")
        }
        guard let model = Self.model(for: tier) else { throw EnhanceError.notAnEngineTier(tier) }
        // The tier's own registration decides the depth it is sent (AB-A-0158): a deep source stays 16-bit only into a
        // model that declares it.
        let keep = await service.keepsSixteenBits(model)
        return try await upscale(try EngineImageCoding.rawImage(from: image, keepingDepth: keep), scale: requestedScale,
                                 tier: tier, source: (image.width, image.height), label: "Conform")
    }

    /// Run `tier`'s one registration on `current` at `scale`, and report the registration that ran. A scale past
    /// the tiers' native ×4 (×6, ×8) is served as a chain — see `ChainedUpscalePlan` (AB-D-0119).
    private func upscale(_ current: Image, scale requestedScale: Int, tier: UpscaleTier,
                         source: (width: Int, height: Int), label: String) async throws -> EnhanceOutcome {
        // The tier's own backer or nothing. A tier that cannot run throws with its reason; the other tier never
        // stands in for it, because a "Best" that ran Fast would ship one model's pixels under another's name.
        guard !tier.isVideoOnly else {
            throw EnhanceError.videoOnlyTier(tier)
        }
        if tier == .instant {
            return try await instantUpscale(try EngineImageCoding.cgImage(from: current), scale: requestedScale,
                                            label: label)
        }
        guard let model = Self.model(for: tier) else { throw EnhanceError.notAnEngineTier(tier) }
        guard let upscaleID = service.packageID(for: model) else {
            throw EnhanceError.unavailable(model, service.outcome(for: model)?.failure)
        }
        // NAFNet ≥ 0.6.0 hands on `.rawRGBA16` for a deep source; a bare tier that can't take it gets it at 8 bits (the
        // engine refuses `.rawRGBA16` on a surface that doesn't declare it). The guarded tiers decode it themselves.
        let current = tier.isStillOnly ? current
            : try Self.image(current, forHopKeepingDepth: await service.keepsSixteenBits(model))
        let plan = ChainedUpscalePlan.make(requested: requestedScale,
                                           source: .init(width: source.width, height: source.height))
        if plan.isChained {
            return try await chainedUpscale(current, plan: plan, tier: tier, model: model, upscaleID: upscaleID,
                                            source: source, label: label)
        }
        // A generative stills tier runs behind the text-legibility guard, never bare (AB-D-0111).
        if tier.isStillOnly {
            return try await guardedUpscale(current, scale: requestedScale, tier: tier, model: model,
                                            source: source, label: label)
        }
        return try await singlePass(current, scale: requestedScale, tier: tier, model: model, upscaleID: upscaleID,
                                    source: source, label: label)
    }

    /// `image` as the next hop should receive it: as it is, or a `.rawRGBA16` image at 8 bits (`.rawBGRA8`) for a hop that
    /// doesn't keep 16 bits.
    static func image(_ image: Image, forHopKeepingDepth keep: Bool) throws -> Image {
        guard image.format == .rawRGBA16, !keep else { return image }
        return try EngineImageCoding.rawImage(from: try EngineImageCoding.cgImage(from: image), keepingDepth: false)
    }

    /// One bare pass of `model` at `scale` — the non-generative tiers' whole story.
    private func singlePass(_ current: Image, scale requestedScale: Int, tier: UpscaleTier, model: ForgeModel,
                            upscaleID: PackageID, source: (width: Int, height: Int), label: String) async throws -> EnhanceOutcome {
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

    /// ×6 / ×8 (AB-D-0119): pass 1 is the chosen tier at its native ×4 exactly as a ×4 request would run it (guarded
    /// for a generative tier); pass 2 is the fast fidelity tier (`.upscale`, NERVE) at ×2 on those pixels; a ×6 is
    /// then fitted down from ×8 with Lanczos. The receipt names every model whose pixels shipped and the fit, and
    /// `lastAppliedScale` is measured from the pixels. The second pass needs NERVE registered; if it is not, the
    /// item fails with that reason rather than shipping a ×4 under a ×6 name.
    private func chainedUpscale(_ current: Image, plan: ChainedUpscalePlan, tier: UpscaleTier, model: ForgeModel,
                                upscaleID: PackageID, source: (width: Int, height: Int), label: String) async throws -> EnhanceOutcome {
        let start = Date()
        let first: EnhanceOutcome
        if tier.isStillOnly {
            first = try await guardedUpscale(current, scale: plan.primaryScale, tier: tier, model: model,
                                             source: source, label: "\(label) pass 1")
        } else {
            first = try await singlePass(current, scale: plan.primaryScale, tier: tier, model: model,
                                         upscaleID: upscaleID, source: source, label: "\(label) pass 1")
        }
        var image = first.image
        if let secondary = plan.secondaryScale {
            guard let fastID = service.packageID(for: .upscale) else {
                throw EnhanceError.unavailable(.upscale, service.outcome(for: .upscale)?.failure)
            }
            let passStart = Date()
            let response: any CapabilityResponse
            do {
                let keep = await service.keepsSixteenBits(.upscale)
                response = try await service.engine.run(
                    ImageUpscaleRequest(image: try EngineImageCoding.rawImage(from: image, keepingDepth: keep),
                                        scale: secondary), package: fastID)
            } catch {
                if let why = Self.governorRefusal(error) { throw EnhanceError.declined(.upscale, why) }
                throw error
            }
            guard let second = response as? ImageUpscaleResponse else {
                throw EnhanceError.unexpectedResponse(.imageUpscale)
            }
            image = try EngineImageCoding.cgImage(from: second.image)
            if second.appliedScale != secondary {
                log.notice("Chained pass 2 requested \(secondary)x, \(ForgeModel.upscale.modelName, privacy: .public) applied \(second.appliedScale)x")
            }
            log.notice("\(label, privacy: .public) pass 2: upscale (\(ForgeModel.upscale.modelName, privacy: .public)) → \(image.width, privacy: .public)×\(image.height, privacy: .public) in \(Date().timeIntervalSince(passStart), format: .fixed(precision: 2), privacy: .public) s")
        }
        if let fit = plan.fit {
            let fitStart = Date()
            image = try ImageResampling.lanczos(image, toWidth: fit.width, height: fit.height)
            log.notice("\(label, privacy: .public) fit: Lanczos → \(fit.width, privacy: .public)×\(fit.height, privacy: .public) in \(Date().timeIntervalSince(fitStart), format: .fixed(precision: 2), privacy: .public) s")
        }
        lastAppliedScale = Int((Double(image.width) / Double(source.width)).rounded())
        let name = plan.modelName(primaryName: first.upscaleModel ?? model.modelName,
                                  secondaryName: ForgeModel.upscale.modelName)
        log.notice("\(label, privacy: .public): chained upscale ×\(plan.requested, privacy: .public) (\(name, privacy: .public), \(tier.rawValue, privacy: .public)) \(source.width, privacy: .public)×\(source.height, privacy: .public) → \(image.width, privacy: .public)×\(image.height, privacy: .public) in \(Date().timeIntervalSince(start), format: .fixed(precision: 2), privacy: .public) s")
        return EnhanceOutcome(image: image, upscaleTier: first.upscaleTier, upscaleModel: name,
                              upscaleRoute: first.upscaleRoute)
    }

    /// The Instant tier (AB-D-0146): Anime4K VL doubles the image on the GPU, off the main actor, keeping its depth and
    /// alpha. ×2, ×4 and ×8 are one, two and three doublings; ×6 is ×8 fitted down with Lanczos, or ×4 fitted up when the
    /// ×8 intermediate would pass `ChainedUpscalePlan.intermediatePixelCap` (AB-D-0119's rule, named in the receipt).
    /// Unlike the engine tiers' chains, every pass is the same tier, so Instant needs no model at any factor.
    private func instantUpscale(_ image: CGImage, scale requestedScale: Int, label: String) async throws -> EnhanceOutcome {
        guard Anime4KUpscaler.isAvailable else {
            throw EnhanceError.instantUnavailable(Anime4KUpscaler.unavailableReason ?? "no Metal device")
        }
        let upscaler: Anime4KUpscaler
        if let existing = instant { upscaler = existing } else { upscaler = try Anime4KUpscaler(); instant = upscaler }
        let plan = InstantUpscalePlan.make(requested: requestedScale,
                                           source: .init(width: image.width, height: image.height))
        let start = Date()
        var out = try await Task.detached(priority: .userInitiated) {
            try upscaler.upscale(image, doublings: plan.doublings)
        }.value
        if let fit = plan.fit {
            out = try ImageResampling.lanczos(out, toWidth: fit.width, height: fit.height)
        }
        lastAppliedScale = Int((Double(out.width) / Double(image.width)).rounded())
        let name = plan.modelName(Self.instantModelName)
        log.notice("\(label, privacy: .public): instant upscale (\(name, privacy: .public)) \(image.width, privacy: .public)×\(image.height, privacy: .public) → \(out.width, privacy: .public)×\(out.height, privacy: .public) in \(Date().timeIntervalSince(start), format: .fixed(precision: 3), privacy: .public) s")
        return EnhanceOutcome(image: out, upscaleTier: .instant, upscaleModel: name)
    }

    /// A generative tier through `EngineTextGuardedUpscaler`, with the policy its pair EARNED (`measuredPolicy(for:)`:
    /// VOSR2 routes generative; Nacre protects legible words with the fast tier's pixels, AB-R-0400). Reports the tier,
    /// the model or models whose pixels shipped, and the guard's route — so a composite never ships under one name.
    private func guardedUpscale(_ current: Image, scale: Int, tier: UpscaleTier, model: ForgeModel,
                                source: (width: Int, height: Int), label: String) async throws -> EnhanceOutcome {
        let guarded = EngineTextGuardedUpscaler(service: service, generative: model)
        if guarded.policy.wordProtection || guarded.policy.documentGate, service.packageID(for: guarded.base) == nil {
            throw EnhanceError.unavailable(guarded.base, service.outcome(for: guarded.base)?.failure)
        }
        let start = Date()
        let out: EngineTextGuardedUpscaler.Output
        do {
            out = try await guarded.upscale(try EngineImageCoding.cgImage(from: current), scale: scale)
        } catch {
            if let why = Self.governorRefusal(error) { throw EnhanceError.declined(model, why) }
            throw error
        }
        lastAppliedScale = Int((Double(out.image.width) / Double(source.width)).rounded())
        let name = Self.guardedModelName(generative: model, base: guarded.base, route: out.route)
        log.notice("\(label, privacy: .public): guarded upscale (\(name, privacy: .public), \(tier.rawValue, privacy: .public), route \(out.route.rawValue, privacy: .public)) \(source.width, privacy: .public)×\(source.height, privacy: .public) → \(out.image.width, privacy: .public)×\(out.image.height, privacy: .public) in \(Date().timeIntervalSince(start), format: .fixed(precision: 2), privacy: .public) s")
        return EnhanceOutcome(image: out.image, upscaleTier: Self.tier(for: model), upscaleModel: name,
                              upscaleRoute: out.route.rawValue)
    }

    /// The name a guarded result ships under: the generative model alone, or both models when the guard composited
    /// the fast tier's pixels in (protected words), or the fast tier alone when it kept a legible document page.
    static func guardedModelName(generative: ForgeModel, base: ForgeModel,
                                 route: TextLegibilityGuard.Verdict.Recommendation) -> String {
        switch route {
        case .generative: return generative.modelName
        case .protectWords: return "\(generative.modelName) + \(base.modelName) (protected words)"
        case .keepBase: return "\(base.modelName) (legible document kept)"
        }
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
        if tier == .instant {
            // No registration and no governor: the only requirement is a Metal GPU.
            return Anime4KUpscaler.isAvailable
                ? .available(.instant, model: Self.instantModelName)
                : .unavailable(.instant, model: Self.instantModelName,
                               reason: Anime4KUpscaler.unavailableReason ?? "no Metal device")
        }
        guard let model = Self.model(for: tier) else {
            return .unavailable(tier, reason: EnhanceError.notAnEngineTier(tier).description)
        }
        guard !tier.isVideoOnly else {
            return .unavailable(tier, model: model.modelName, reason: UpscaleTier.liveActionStillReason)
        }
        guard service.packageID(for: model) != nil else {
            let why = service.outcome(for: model)?.failure ?? "registration has not run"
            return .unavailable(tier, model: model.modelName, reason: "\(model.modelName) is not registered — \(why)")
        }
        // A guarded generative tier whose earned policy can route words to the fast tier needs that tier too.
        if tier.isStillOnly {
            let policy = EngineTextGuardedUpscaler.measuredPolicy(for: model)
            if policy.wordProtection || policy.documentGate, service.packageID(for: .upscale) == nil {
                let why = service.outcome(for: .upscale)?.failure ?? "registration has not run"
                return .unavailable(tier, model: model.modelName,
                                    reason: "the text guard's fast tier (\(ForgeModel.upscale.modelName)) is not registered — \(why)")
            }
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
        /// The Instant tier cannot run here (no Metal device); carries the reason.
        case instantUnavailable(String)
        /// A tier with no engine registration reached an engine-only path (a defect, never a user error).
        case notAnEngineTier(UpscaleTier)

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
            case .instantUnavailable(let why):
                return "the Instant tier cannot run — \(why)"
            case .notAnEngineTier(let tier):
                return "the \(tier.rawValue) tier has no engine registration"
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
    /// Integer scale for the engine request; nil means "skip the upscale stage". ×6 and ×8 exceed every tier's
    /// native ×4 and are served as a chain (`ChainedUpscalePlan`).
    var requestedScale: Int? {
        switch self {
        case .none: return nil
        case .x2:   return 2
        case .x4:   return 4
        case .x6:   return 6
        case .x8:   return 8
        }
    }
}
