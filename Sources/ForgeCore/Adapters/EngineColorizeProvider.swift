//
//  EngineColorizeProvider.swift
//  ForgeCore
//
//  ColorizeKit's `ColorizeProvider` seam, engine-backed by DDColor. Unlike Extract there is no OS
//  colorization API, so this capability has no Tier-0/1 fallback — without a provider, Colorize is
//  simply unavailable.
//

import Foundation
import CoreGraphics
import ColorizeKit
import MLXServeCore
import MLXToolKit

@MainActor
public final class EngineColorizeProvider: ColorizeProvider {

    public nonisolated let name = "DDColor"

    private let service: ForgeEngineService

    public init(service: ForgeEngineService) {
        self.service = service
    }

    public func colorize(_ image: CGImage, variant: ColorizeVariant) async throws -> CGImage {
        guard let packageID = service.packageID(for: .colorize) else { throw ColorizeError.unavailable }
        let request = ColorizeRequest(image: try EngineImageCoding.pngImage(from: image),
                                      mode: Self.mode(for: variant))
        let response = try await service.engine.run(request, package: packageID)
        guard let output = (response as? ColorizeResponse)?.image else { throw ColorizeError.unexpectedResponse }
        return try EngineImageCoding.cgImage(from: output)
    }

    private static func mode(for variant: ColorizeVariant) -> Mode {
        switch variant {
        case .fast:     return ColorizeContract.fast
        case .best:     return ColorizeContract.best
        case .artistic: return ColorizeContract.artistic
        }
    }

    public enum ColorizeError: Error { case unavailable, unexpectedResponse }
}
