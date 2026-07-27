//
//  EngineMatteProvider.swift
//  ForgeCore
//
//  ExtractKit's `MatteProvider` seam, engine-backed by BiRefNet — Extract's quality tier.
//  This is the template every other adapter in this folder mirrors.
//
//  PNG at the boundary: `mlx-birefnet-swift` does not mirror `.rawBGRA8` (BRIDGE-041), and `Matte` is
//  PNG-only by contract in either case.
//

import Foundation
import CoreGraphics
import OSLog
import ExtractKit
import MLXServeCore
import MLXToolKit
import MLXBiRefNet

@MainActor
public final class EngineMatteProvider: MatteProvider {

    public nonisolated let name = "BiRefNet"

    private let service: ForgeEngineService

    public init(service: ForgeEngineService) {
        self.service = service
    }

    /// The quality that **actually ran**, once `matte` has returned.
    ///
    /// `.best` silently degrading to `.fast` on a device that cannot hold the 2048² working set is a
    /// downgrade the user asked against and would never learn about: the call succeeds, a matte comes
    /// back, and the only trace was a log line no UI reads. R1's row 15 requires the fallback be
    /// **visible, not silent** — and `MatteProvider.matte` returns a bare `CGImage`, so there is no
    /// channel in the protocol to carry it. Reported here, and shown by the harness.
    public private(set) var lastAppliedQuality: ExtractQuality?
    /// True when the last run was downgraded from what was requested.
    public private(set) var lastRunDidFallBack = false

    public func matte(_ image: CGImage, quality: ExtractQuality) async throws -> CGImage {
        guard let packageID = service.packageID(for: .matte) else { throw MatteError.unavailable }
        let input = try EngineImageCoding.pngImage(from: image)
        do {
            let out = try await run(input, package: packageID, best: quality == .best)
            lastAppliedQuality = quality
            lastRunDidFallBack = false
            return out
        } catch BiRefNetPackage.BiRefNetError.insufficientMemoryForBest {
            // best runs at 2048² (~18 GB working set). Rather than fail, drop to fast so the user still
            // gets a matte — the package declares its footprint at the fast envelope and guards best
            // at runtime precisely so the host can make this choice.
            log.notice("BiRefNet best does not fit this device's working set — falling back to fast")
            let out = try await run(input, package: packageID, best: false)
            lastAppliedQuality = .fast
            lastRunDidFallBack = true
            return out
        }
    }

    private func run(_ image: Image, package: PackageID, best: Bool) async throws -> CGImage {
        let request = MattingRequest(image: image,
                                     preferredKind: .softAlpha,
                                     mode: best ? MattingContract.best : MattingContract.fast)
        let response = try await service.engine.run(request, package: package)
        guard let matte = (response as? MattingResponse)?.matte else { throw MatteError.unexpectedResponse }
        return try EngineImageCoding.cgImage(fromEncoded: matte.data)
    }

    public enum MatteError: Error { case unavailable, unexpectedResponse }
}
