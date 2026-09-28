//
//  EngineImageEnhancer.swift
//  ForgeCore
//
//  ForgeOptimizerKit's `ImageEnhancer` seam, engine-backed: NAFNet restore → NERVE upscale (the fast tier; Real-ESRGAN
//  until 2026-09-28).
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

    /// The scale the fast tier actually applied on the most recent enhance, or nil if no upscale ran.
    ///
    /// History: Real-ESRGAN's `general` was a fixed-4× model, and asking it for 2× used to yield 4× pixels while the
    /// recipe recorded "2×" — a product cannot offer an option that silently does something else.
    /// `mlx-realesrgan-swift` 0.5.0 fixed the model side by post-downsampling to the requested scale; NERVE runs ×2
    /// natively and post-downsamples only ×1/×3. This records what came back so the receipt describes the output
    /// rather than the request.
    public private(set) var lastAppliedScale: Int?

    public init(service: ForgeEngineService) {
        self.service = service
    }

    public func enhance(_ image: CGImage, options: Options) async throws -> CGImage {
        let engine = service.engine
        var current = try EngineImageCoding.rawImage(from: image)

        guard let restoreID = service.packageID(for: .restore) else {
            throw EnhanceError.unavailable(.restore)
        }
        let restoreResponse = try await engine.run(ImageRestoreRequest(image: current), package: restoreID)
        guard let restored = (restoreResponse as? ImageRestoreResponse)?.image else {
            throw EnhanceError.unexpectedResponse(.imageRestore)
        }
        current = restored

        if let requestedScale = options.upscale.requestedScale {
            guard let upscaleID = service.packageID(for: .upscale) else {
                throw EnhanceError.unavailable(.upscale)
            }
            let response = try await engine.run(ImageUpscaleRequest(image: current, scale: requestedScale),
                                                package: upscaleID)
            guard let upscaleResponse = response as? ImageUpscaleResponse else {
                throw EnhanceError.unexpectedResponse(.imageUpscale)
            }
            current = upscaleResponse.image
            lastAppliedScale = upscaleResponse.appliedScale
            if upscaleResponse.appliedScale != requestedScale {
                log.notice("Upscale requested \(requestedScale)x, model applied \(upscaleResponse.appliedScale)x — recording the applied factor")
            }
        } else {
            lastAppliedScale = nil
        }

        return try EngineImageCoding.cgImage(from: current)
    }

    public enum EnhanceError: Error {
        case unavailable(ForgeModel)
        case unexpectedResponse(Capability)
    }
}

private extension UpscaleFactor {
    /// Integer scale for the engine request; nil means "skip the upscale stage".
    var requestedScale: Int? {
        switch self {
        case .none: return nil
        case .x2:   return 2
        case .x4:   return 4
        }
    }
}
