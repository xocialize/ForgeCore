//
//  WebOptimizeWiringTests.swift — ForgeCore
//
//  The cross-package wiring for the web-deliverable verb (ForgeOptimizerKit 0.5.0's `webOptimize`).
//  ForgeCore does not wrap the verb — it supplies the providers a host injects into `ForgeOptimizer` —
//  so "properly wired" means exactly two things, both provable without weights:
//
//    1. the resolved Kit carries the verb (a floor regression un-compiles this file), and
//    2. ForgeCore's engine-backed providers satisfy `ForgeOptimizer`'s constructor seams and the
//       verb runs through that wiring end-to-end.
//
//  Enhance stays `.off`, so the engine adapters are constructed but never invoked — the same
//  laziness a host relies on when models aren't downloaded yet.
//

import CoreGraphics
import Foundation
import ForgeOptimizerKit
import ImageIO
import Testing
import UniformTypeIdentifiers

@testable import ForgeCore

@Suite("webOptimize wiring")
@MainActor
struct WebOptimizeWiringTests {

    /// JPEG in → PNG out through `ForgeCapabilities`' own provider instances.
    @Test func providersDriveWebOptimizeEndToEnd() async throws {
        let service = ForgeEngineService()
        let caps = ForgeCapabilities(service: service)
        let forge = ForgeOptimizer(enhancer: caps.enhancer, flowProvider: caps.flow)

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("forgecore-web-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: tmp, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tmp) }
        let src = tmp.appendingPathComponent("input.jpg")
        try Self.writeJPEG(Self.makeTexturedImage(96, 96), to: src)

        var results: [OptimizeResult] = []
        for await r in try forge.webOptimize(.url(src), to: .directory(tmp.appendingPathComponent("o"))) {
            results.append(r)
        }
        #expect(results.count == 1)
        let r = try #require(results.first)
        guard case .optimized = r.status else {
            Issue.record("expected .optimized, got \(r.status)")
            return
        }
        #expect(r.outputType == .png)
        #expect(r.recipe.codec == "PNG")
        // The measured round-trip rode the receipt through this wiring — that is the assertion.
        // The strict ≥99.5 losslessness claim lives in media-bridge's suite against a clean sRGB
        // source; a JPEG-decoded reference pays a colorspace re-tag rounding of a few tenths
        // (this fixture measures ≈99.26), which is precisely why the score is measured, not asserted.
        let score = try #require(r.after.qualityScore)
        #expect(score > 99, "the PNG round-trip must measure ~lossless, got \(score)")
        guard case .file(let out) = r.output else {
            Issue.record("expected .file output")
            return
        }
        #expect(out.pathExtension == "png")
        #expect(FileManager.default.fileExists(atPath: out.path))
    }

    // MARK: - Fixtures

    /// Gradient + deterministic texture (the Kit's own fixture recipe) — detailed enough that the
    /// JPEG is a genuinely lossy source, so the measured score is a real round-trip claim.
    private static func makeTexturedImage(_ w: Int, _ h: Int) -> CGImage {
        var bytes = [UInt8](repeating: 0, count: w * h * 4)
        for y in 0..<h { for x in 0..<w {
            let noise = ((x * 131 + y * 57) % 64) - 32
            let i = (y * w + x) * 4
            bytes[i] = UInt8(clamping: x * 2 + noise)
            bytes[i + 1] = UInt8(clamping: y * 2 + noise)
            bytes[i + 2] = UInt8(clamping: 128 + noise)
            bytes[i + 3] = 255
        } }
        let cs = CGColorSpace(name: CGColorSpace.sRGB)!
        let ctx = CGContext(data: &bytes, width: w, height: h, bitsPerComponent: 8,
                            bytesPerRow: w * 4, space: cs,
                            bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
        return ctx.makeImage()!
    }

    private static func writeJPEG(_ image: CGImage, to url: URL) throws {
        struct FixtureError: Error {}
        guard let dst = CGImageDestinationCreateWithURL(
            url as CFURL, UTType.jpeg.identifier as CFString, 1, nil) else { throw FixtureError() }
        CGImageDestinationAddImage(dst, image, [
            kCGImageDestinationLossyCompressionQuality: 0.6] as CFDictionary)
        guard CGImageDestinationFinalize(dst) else { throw FixtureError() }
    }
}
