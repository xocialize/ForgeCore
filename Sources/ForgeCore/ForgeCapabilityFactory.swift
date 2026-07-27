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

    /// Optimize — NAFNet restore → Real-ESRGAN upscale (`ForgeOptimizerKit.ImageEnhancer`).
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

    /// The four V1 capabilities and whether the models they need are registered. A capability with an
    /// unavailable Tier-2 model is not necessarily dead — Extract and Erase still have their Tier-0/1
    /// paths — so this reports the *quality tier*, not the capability.
    public var qualityTierAvailability: [(capability: String, models: [ForgeModel], available: Bool)] {
        [
            ("Optimize", [.restore, .upscale], isRegistered(.restore) && isRegistered(.upscale)),
            ("Extract",  [.matte, .promptMask, .flow], isRegistered(.matte)),
            ("Colorize", [.colorize, .flow], isRegistered(.colorize)),
            ("Erase",    [.inpaint, .promptMask, .flow], isRegistered(.inpaint)),
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
    }
}
