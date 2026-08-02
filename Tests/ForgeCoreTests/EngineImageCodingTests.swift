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

    /// Every registration must have a distinct NAME. Capabilities are deliberately NOT unique.
    ///
    /// ⚠️ **This test used to assert one capability per model, and that assertion was wrong** — it
    /// encoded the opposite of the architecture. The engine supports **multiple packages per
    /// capability, selected by `PackageID`**, and Forge relies on it: `.upscale` (Real-ESRGAN, fast)
    /// and `.upscaleQuality` (SeedVR2-3B, quality) both back `.imageUpscale` on purpose. The old
    /// assertion also hard-coded a count of 7 that fell out of date when those two rows landed, so it
    /// had been failing rather than protecting anything.
    ///
    /// What actually needs protecting is that no two registrations share a *name* — that would make
    /// `RegistrationOutcome` ambiguous in a UI — and that every case really is registerable.
    @Test("model table is consistent")
    func modelTableIsConsistent() {
        let models = ForgeModel.allCases
        // ⚠️ Not a hard-coded total — that assertion has now gone stale TWICE (at 7, then at 9,
        // which failed the moment BRIDGE-066/067 registered the frame-restore wave). The doc
        // comment above already named the lesson; this line finally applies it. A floor still
        // catches accidental case deletion without re-failing on every legitimate addition.
        #expect(models.count >= 19)
        #expect(Set(models.map(\.modelName)).count == models.count)
        #expect(Set(models.map(\.rawValue)).count == models.count)
        // Capability sharing is intentional and load-bearing: assert the tier pair specifically, so a
        // future refactor that accidentally splits them fails here rather than silently changing routing.
        #expect(ForgeModel.upscale.capability == .imageUpscale)
        #expect(ForgeModel.upscaleQuality.capability == .imageUpscale)
        // The BRIDGE-066 wave: eight imageRestore backers total (NAFNet + wave + faces), one relight.
        let restoreBackers = models.filter { $0.capability == .imageRestore }
        #expect(restoreBackers.count == 9)
        #expect(ForgeModel.relight.capability == .imageRelight)
        #expect(ForgeModel.defectDetect.capability == .defectDetect)
        // Restormer is ONE package, THREE registrations — distinct cases, distinct display names.
        let restormers: [ForgeModel] = [.restormerMotion, .restormerDefocus, .restormerDenoise]
        #expect(Set(restormers.map(\.modelName)).count == 3)
    }
}
