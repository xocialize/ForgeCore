//
//  EngineFlowProvider.swift
//  ForgeCore
//
//  Dense optical flow (SEA-RAFT), engine-backed.
//
//  Conforms to **both** `ExtractKit.FlowProvider` and `ForgeOptimizerKit.VideoFlowProvider`: the two Kits
//  declare the identical `flow(_:_:)` signature, so one SEA-RAFT registration feeds four consumers —
//  Extract video matte, Colorize temporal consistency, upscale stabilization, and Erase mask
//  propagation. Reach for this before registering a second copy of any model.
//

import Foundation
import CoreGraphics
import ExtractKit
import ForgeOptimizerKit
import MediaMeasure
import MLXServeCore
import MLXToolKit

@MainActor
public final class EngineFlowProvider: FlowProvider, VideoFlowProvider {

    public nonisolated let name = "SEA-RAFT"

    private let service: ForgeEngineService

    public init(service: ForgeEngineService) {
        self.service = service
    }

    /// Dense flow `a → b`: pixel (x, y) in `a` moved to (x + u, y + v) in `b`.
    public func flow(_ a: CGImage, _ b: CGImage) async throws -> DenseFlow {
        guard let packageID = service.packageID(for: .flow) else { throw FlowError.unavailable }
        let request = OpticalFlowRequest(image0: try EngineImageCoding.pngImage(from: a),
                                         image1: try EngineImageCoding.pngImage(from: b))
        let response = try await service.engine.run(request, package: packageID)
        guard let field = (response as? OpticalFlowResponse)?.flow else { throw FlowError.unexpectedResponse }
        // FlowField.uv and DenseFlow.uv share the exact layout: row-major W*H*2, interleaved u,v.
        return DenseFlow(width: field.width, height: field.height, uv: field.uv)
    }

    public enum FlowError: Error { case unavailable, unexpectedResponse }
}
