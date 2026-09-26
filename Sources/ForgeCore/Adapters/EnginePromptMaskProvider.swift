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
import MLXEdgeTAM
import MLXServeCore
import MLXToolKit

@MainActor
public final class EnginePromptMaskProvider: PromptMaskProvider {

    /// Which `promptSegment` registration answers: `.promptMask` (EdgeTAM — fast, and the only video tracker)
    /// or `.promptMaskQuality` (SAM 2.1-S — the quality tier: a single click lands first time on flat-shaded
    /// subjects, AB-T-0173). Same request, same response, same `softMatte` mode constant.
    public nonisolated let model: ForgeModel
    public nonisolated let name: String

    private let service: ForgeEngineService
    /// Anti-aliased edges (EdgeTAM ≥ 0.5.0 `softMatte`): a ~1–2 px coverage ramp whose 0.5 level is exactly the binary
    /// edge, 0/1 everywhere else. Off by default — the binary matte is what existing callers were validated against.
    /// Not a sigmoid of the logits: that measured as a ~20 px ramp with background haze (AB-T-0171).
    public var softEdges = false

    public init(service: ForgeEngineService, model: ForgeModel = .promptMask) {
        precondition(model.capability == .promptSegment, "EnginePromptMaskProvider needs a promptSegment registration")
        self.service = service
        self.model = model
        self.name = model.modelName
    }

    public func mask(_ image: CGImage, prompt: MaskPrompt) async throws -> CGImage {
        guard let packageID = service.packageID(for: model) else { throw PromptError.unavailable }
        let request = PromptSegmentRequest(
            image: try EngineImageCoding.pngImage(from: image),
            points: prompt.points.map { [Float($0.x), Float($0.y)] },
            pointLabels: prompt.points.map { $0.include ? 1 : 0 },
            box: prompt.box.map { [Float($0.minX), Float($0.minY), Float($0.maxX), Float($0.maxY)] },
            mode: softEdges ? EdgeTAMPackage.softMatte : nil   // == SAM21Package.softMatte
        )
        let response = try await service.engine.run(request, package: packageID)
        guard let segment = response as? PromptSegmentResponse else { throw PromptError.unexpectedResponse }
        return try EngineImageCoding.cgImage(fromEncoded: segment.matte.data)
    }

    public enum PromptError: Error { case unavailable, unexpectedResponse }
}
