//
//  VideoUpscaleLiveActionLiveTests.swift — ForgeCore
//
//  [VAL] — FlashVSR (`.upscaleVideoLiveAction`, registered 2026-10-04) through the REAL engine via the path the product
//  ships: `registerAll` → `attachModelStore` → `upscaleVideo(_:scale:content: .liveAction)`. Env-gated (needs the GPU,
//  a live-action clip and — unless the store already holds it — a 3.5 GB download), so it never runs in CI:
//
//      FORGE_FLASHVSR_E2E_VIDEO=<live-action .mov|.mp4> [FORGE_FLASHVSR_E2E_STORE=<models root>] \
//        [FORGE_FLASHVSR_E2E_OUT=<dir>] swift test --build-system swiftbuild --filter VideoUpscaleLiveActionLiveTests
//
//  Asserts: the route took FlashVSR (the outcome names it), `appliedScale` 4, the output is exactly 4× both ways, and
//  EVERY source frame came back (FlashVSR's stream emits one output per input — upstream's own bookkeeping drops up to
//  8 at the tail). The output lands in the out dir (default: the temp dir) — clip media, never committed.
//

import AVFoundation
import Foundation
import MLXToolKit
import Testing

@testable import ForgeCore

@MainActor
@Suite struct VideoUpscaleLiveActionLiveTests {
    nonisolated static let env = ProcessInfo.processInfo.environment
    nonisolated static var clip: URL? { env["FORGE_FLASHVSR_E2E_VIDEO"].map { URL(fileURLWithPath: $0) } }

    /// (frame count, width, height) of a clip's first video track, by decoding it.
    static func probe(_ url: URL) async throws -> (frames: Int, width: Int, height: Int) {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw CocoaError(.fileReadCorruptFile) }
        let size = try await track.load(.naturalSize)
        let reader = try AVAssetReader(asset: asset)
        let out = AVAssetReaderTrackOutput(track: track, outputSettings: [
            kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        reader.add(out)
        reader.startReading()
        var n = 0
        while out.copyNextSampleBuffer() != nil { n += 1 }
        return (n, Int(size.width), Int(size.height))
    }

    @Test(.enabled(if: clip != nil, "set FORGE_FLASHVSR_E2E_VIDEO to a live-action clip"))
    func liveActionClipRunsOnFlashVSRAndKeepsEveryFrame() async throws {
        let clip = try #require(Self.clip)
        let service = ForgeEngineService()
        if let store = Self.env["FORGE_FLASHVSR_E2E_STORE"].map({ URL(fileURLWithPath: $0) }) {
            await service.attachModelStore(root: store)
        }
        await service.registerAll()
        #expect(service.outcome(for: .upscaleVideoLiveAction)?.failure == nil)

        let source = try await Self.probe(clip)
        let format: Video.Format = clip.pathExtension.lowercased() == "mov" ? .mov : .mp4
        let outcome = try await service.upscaleVideo(Video(format: format, data: try Data(contentsOf: clip)),
                                                     scale: 4, content: .liveAction)
        #expect(outcome.model == .upscaleVideoLiveAction)
        #expect(outcome.appliedScale == 4)

        let outDir = Self.env["FORGE_FLASHVSR_E2E_OUT"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.temporaryDirectory
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)
        let outURL = outDir.appendingPathComponent("flashvsr_live_\(clip.deletingPathExtension().lastPathComponent).mp4")
        try outcome.video.data.write(to: outURL)
        let result = try await Self.probe(outURL)
        #expect(result.width == source.width * 4 && result.height == source.height * 4,
                "\(result.width)×\(result.height) from \(source.width)×\(source.height)")
        #expect(result.frames == source.frames, "\(result.frames) frames out for \(source.frames) in")
        print(String(format: "[VAL] FlashVSR live action: %d frames %d×%d → %d×%d in %.1f s (%.2f s/frame) → %@",
                     result.frames, source.width, source.height, result.width, result.height, outcome.seconds,
                     outcome.secondsPerFrame ?? .nan, outURL.path))
    }
}
