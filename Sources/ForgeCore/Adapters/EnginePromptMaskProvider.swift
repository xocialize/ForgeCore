//
//  EnginePromptMaskProvider.swift
//  ForgeCore
//
//  MaskKit's `PromptMaskProvider` seam, engine-backed by EdgeTAM `promptSegment` — the shared
//  click/box-to-select mask source for Extract (subject to cut out) and Erase (object to remove).
//  One registration, both capabilities.
//
//  `MaskPrompt` points and box are already in source-image pixels, which is exactly what
//  `PromptSegmentRequest` expects, so the mapping is a straight value translation.
//

import Foundation
import CoreGraphics
import MaskKit
import MLXServeCore
import MLXToolKit

@MainActor
public final class EnginePromptMaskProvider: PromptMaskProvider {

    public nonisolated let name = "EdgeTAM"

    private let service: ForgeEngineService

    public init(service: ForgeEngineService) {
        self.service = service
    }

    public func mask(_ image: CGImage, prompt: MaskPrompt) async throws -> CGImage {
        guard let packageID = service.packageID(for: .promptMask) else { throw PromptError.unavailable }
        let request = PromptSegmentRequest(
            image: try EngineImageCoding.pngImage(from: image),
            points: prompt.points.map { [Float($0.x), Float($0.y)] },
            pointLabels: prompt.points.map { $0.include ? 1 : 0 },
            box: prompt.box.map { [Float($0.minX), Float($0.minY), Float($0.maxX), Float($0.maxY)] }
        )
        let response = try await service.engine.run(request, package: packageID)
        guard let segment = response as? PromptSegmentResponse else { throw PromptError.unexpectedResponse }
        return try EngineImageCoding.cgImage(fromEncoded: segment.matte.data)
    }

    public enum PromptError: Error { case unavailable, unexpectedResponse }
}
