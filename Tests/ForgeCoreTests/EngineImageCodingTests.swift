//
//  EngineImageCodingTests.swift
//  ForgeCoreTests
//
//  CoreGraphics-only, by design: these must pass without a GPU, without weights, and without the
//  metallib. Everything here is the boundary arithmetic — the part that can be wrong silently.
//

import Testing
import CoreGraphics
import Foundation
@testable import ForgeCore
import MLXToolKit

@Suite("EngineImageCoding")
struct EngineImageCodingTests {

    /// The failure this helper exists to prevent is a BGRA/RGBA channel swap, which produces images that
    /// look plausible — red and blue exchanged — rather than obviously broken. Only a byte-exact
    /// assertion catches it; "the image decoded" does not.
    ///
    /// Colours are opaque and fully saturated so premultiplication is the identity and the bytes must
    /// survive unchanged.
    @Test("rawBGRA8 round-trip is byte-exact")
    func rawRoundTripIsByteExact() throws {
        let source: [UInt8] = [
            0, 0, 255, 255,      255, 0, 0, 255,      // red,   blue
            0, 255, 0, 255,      255, 255, 255, 255,  // green, white
        ]
        let start = Image.rawBGRA8(data: Data(source), width: 2, height: 2, bytesPerRow: 8)

        let cg = try EngineImageCoding.cgImage(from: start)
        #expect(cg.width == 2)
        #expect(cg.height == 2)

        let end = try EngineImageCoding.rawImage(from: cg)
        #expect(Array(end.data) == source)
        #expect(end.format == .rawBGRA8)
        #expect(end.width == 2)
        #expect(end.height == 2)
        #expect(end.bytesPerRow == 8)
    }

    /// The same assertion the app runs at bring-up, so a regression fails CI rather than waiting to be
    /// noticed at launch.
    @Test("self-check agrees with the test")
    func selfCheckPasses() {
        #expect(throws: Never.self) { try EngineImageCoding.selfCheckRawRoundTrip().get() }
    }

    /// `.rawBGRA8` requires width and height by contract. Decoding one without them must be a named
    /// error, not a crash or a silently mis-sized image.
    @Test("rawBGRA8 without dimensions is a named error")
    func rawWithoutDimensionsThrows() {
        let malformed = Image(format: .rawBGRA8, data: Data(repeating: 0, count: 16))
        #expect(throws: EngineImageCoding.CodingError.missingDimensions) {
            _ = try EngineImageCoding.cgImage(from: malformed)
        }
    }

    /// A wrapper that ignores a raw request and answers in PNG must still work — that fallback is what
    /// lets the five wrappers which have not adopted rawBGRA8 keep running.
    @Test("PNG encode survives a round trip through the decoder")
    func pngRoundTripPreservesSize() throws {
        let context = CGContext(data: nil, width: 4, height: 3, bitsPerComponent: 8, bytesPerRow: 16,
                                space: EngineImageCoding.bgraColorSpace,
                                bitmapInfo: EngineImageCoding.bgraBitmapInfo)
        let original = try #require(context.flatMap { ctx -> CGImage? in
            ctx.setFillColor(red: 0.2, green: 0.6, blue: 0.9, alpha: 1)
            ctx.fill(CGRect(x: 0, y: 0, width: 4, height: 3))
            return ctx.makeImage()
        })

        let encoded = try EngineImageCoding.pngImage(from: original)
        #expect(encoded.format == .png)

        let decoded = try EngineImageCoding.cgImage(from: encoded)
        #expect(decoded.width == 4)
        #expect(decoded.height == 3)
    }

    /// Every model must map to a distinct capability-and-name pair; a duplicate would mean two
    /// registrations silently competing for one capability.
    @Test("model table is consistent")
    func modelTableIsConsistent() {
        let models = ForgeModel.allCases
        #expect(models.count == 7)
        #expect(Set(models.map(\.modelName)).count == models.count)
        #expect(Set(models.map(\.capability)).count == models.count)
    }
}
