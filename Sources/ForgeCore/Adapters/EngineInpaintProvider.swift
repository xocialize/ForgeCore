//
//  EngineInpaintProvider.swift
//  ForgeCore
//
//  EraseKit's `InpaintProvider` seam, engine-backed by LaMa (quality) / MI-GAN (fast).
//
//  The one two-input seam in Forge: (image, mask) → filled image, where the mask is grayscale and
//  **white = remove, black = keep**. EraseKit's Tier-0 `ClassicalInpainter` still handles small holes
//  with no provider injected; this upgrades quality on large or structured ones.
//

import Foundation
import CoreGraphics
import EraseKit
import MLXServeCore
import MLXToolKit

@MainActor
public final class EngineInpaintProvider: InpaintProvider {

    /// Provider identity. **Not** the model for a given run — see `engineName(for:)`.
    public nonisolated let name = "LaMa / MI-GAN"

    /// The model that actually runs each tier. `.fast` is MI-GAN, `.best` is LaMa; reporting "LaMa"
    /// for both is the receipt bug BRIDGE-060 exists to fix.
    public nonisolated func engineName(for quality: InpaintQuality) -> String {
        quality == .fast ? "MI-GAN" : "LaMa"
    }

    private let service: ForgeEngineService

    public init(service: ForgeEngineService) {
        self.service = service
    }

    public func inpaint(_ image: CGImage, mask: CGImage, quality: InpaintQuality) async throws -> CGImage {
        guard let packageID = service.packageID(for: .inpaint) else { throw InpaintError.unavailable }
        let request = InpaintRequest(image: try EngineImageCoding.pngImage(from: image),
                                     mask: try EngineImageCoding.pngImage(from: mask),
                                     mode: quality == .fast ? InpaintContract.fast : InpaintContract.best)
        let response = try await service.engine.run(request, package: packageID)
        guard let output = (response as? InpaintResponse)?.image else { throw InpaintError.unexpectedResponse }
        return try EngineImageCoding.cgImage(from: output)
    }

    public enum InpaintError: Error { case unavailable, unexpectedResponse }
}
