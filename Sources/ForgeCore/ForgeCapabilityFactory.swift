//
//  ForgeCapabilityFactory.swift
//  ForgeCore
//
//  The seam between the engine adapters and whatever drives them.
//
//  The factory hands out the *providers*, grouped the way the four capabilities consume them, rather
//  than constructing the `ForgeCapability` classes — those live in the `*UI` packages (ForgeOptimizerUI,
//  ExtractUI, ColorizeUI, EraseUI), which sit above this seam. A consumer that wants them adds those
//  package dependencies and injects these same provider instances; nothing here changes.
//
//  The sharing is deliberate and load-bearing: exactly **one** flow provider and **one** prompt-mask
//  provider exist, and they are handed to every capability that needs them. Two instances would mean two
//  registrations of the same model and twice the resident weights.
//

import Foundation

/// Every engine-backed provider Forge offers, plus which of them actually registered.
@MainActor
public struct ForgeCapabilities {

    /// Optimize — NAFNet restore → NERVE (`.fast`) or RealPLKSR (`.best`) upscale (`ForgeOptimizerKit.ImageEnhancer`).
    /// Ask it `availability(of:)` to know whether a tier can run here, and why not.
    public let enhancer: EngineImageEnhancer
    /// Extract quality tier — BiRefNet (`ExtractKit.MatteProvider`).
    public let matte: EngineMatteProvider
    /// Revive — DDColor (`ColorizeKit.ColorizeProvider`).
    public let colorize: EngineColorizeProvider
    /// Erase quality tiers — LaMa / MI-GAN (`EraseKit.InpaintProvider`).
    public let inpaint: EngineInpaintProvider
    /// Shared across Extract video matte, Colorize temporal consistency, upscale stabilization, and
    /// Erase mask propagation — one SEA-RAFT registration, four consumers.
    public let flow: EngineFlowProvider
    /// Shared by Extract and Erase — one EdgeTAM registration, two capabilities.
    public let promptMask: EnginePromptMaskProvider
    /// Quick Select **quality** tier — SAM 2.1-S (`.promptMaskQuality`), image only. The same `PromptMaskProvider`
    /// seam over a second `promptSegment` registration; a host offers it as "Best" beside EdgeTAM's "Fast", or
    /// uses it for off-subject clicks where EdgeTAM's single-click recall is weak (AB-A-0105).
    public let promptMaskQuality: EnginePromptMaskProvider
    /// Optimize's §6.3 planner hint — V-JEPA2 (`ForgeOptimizerKit.ContentHintProvider`).
    ///
    /// 🚨 **Injecting this today costs and buys nothing yet, by design.** The adapter ships inert
    /// (confidence clamped below the Kit's 0.9 gate — see `EngineContentHintProvider`), so a hint
    /// never raises a floor and never saves a search, while each classification still pays a
    /// ~1.2 GB V-JEPA2 load plus a full copy of the input. The one reason to inject it now is to
    /// **collect calibration rows** (pass `observe:` and pair each embedding with that item's
    /// optimize receipt — AB-T-0045 step 4). An app that just wants faster optimize should leave
    /// `hintProvider` nil: the Kit's behavioral class ratchet is the fallback and is free.
    public let contentHint: EngineContentHintProvider

    /// The four V1 capabilities and whether the models they need are registered. A capability with an
    /// unavailable Tier-2 model is not necessarily dead — Extract and Erase still have their Tier-0/1
    /// paths — so this reports the *quality tier*, not the capability.
    public var qualityTierAvailability: [(capability: String, models: [ForgeModel], available: Bool)] {
        [
            // `.upscaleFidelity` (RealPLKSR) is Optimize's "best" upscale tier (`Options.upscaleTier`, AB-T-0187), but
            // availability stays keyed on the fast tier: Optimize runs without Best, and whether Best can run HERE is a
            // per-tier answer with a reason — `enhancer.availability(of: .best)` — not a capability-wide boolean.
            ("Optimize", [.restore, .upscale, .upscaleFidelity], isRegistered(.restore) && isRegistered(.upscale)),
            ("Extract",  [.matte, .promptMask, .promptMaskQuality, .flow], isRegistered(.matte)),
            ("Colorize", [.colorize, .flow], isRegistered(.colorize)),
            ("Erase",    [.inpaint, .promptMask, .promptMaskQuality, .flow], isRegistered(.inpaint)),
        ]
    }

    private let service: ForgeEngineService

    private func isRegistered(_ model: ForgeModel) -> Bool {
        service.packageID(for: model) != nil
    }

    public init(service: ForgeEngineService) {
        self.service = service
        self.enhancer   = EngineImageEnhancer(service: service)
        self.matte      = EngineMatteProvider(service: service)
        self.colorize   = EngineColorizeProvider(service: service)
        self.inpaint    = EngineInpaintProvider(service: service)
        self.flow       = EngineFlowProvider(service: service)
        self.promptMask = EnginePromptMaskProvider(service: service)
        self.promptMaskQuality = EnginePromptMaskProvider(service: service, model: .promptMaskQuality)
        self.contentHint = EngineContentHintProvider(service: service)
    }
}
