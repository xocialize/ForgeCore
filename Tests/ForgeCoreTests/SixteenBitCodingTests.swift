//
//  SixteenBitCodingTests.swift
//  ForgeCoreTests
//
//  The 16-bit model boundary at ForgeCore's seams (AB-A-0158, MLXToolKit contract 1.52.0): EngineImageCoding's
//  `.rawRGBA16` path and the chain's Lanczos fit. CoreGraphics / CoreImage only — no GPU, no weights, no metallib.
//

import Testing
import CoreGraphics
import Foundation
@testable import ForgeCore
import MLXToolKit

@Suite("16-bit coding")
struct SixteenBitCodingTests {

    /// An opaque 16-bit sRGB image whose red channel steps ~0.1 of an 8-bit step per pixel — the smooth-sky case.
    static func deepImage(width w: Int, height h: Int) throws -> CGImage {
        var words = [UInt16](repeating: UInt16.max, count: w * h * 4)
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4
                words[i] = UInt16(20_000 + x * 26); words[i + 1] = UInt16(9_000 + y * 11)
                words[i + 2] = UInt16((x * 977 + y * 131) % 65_536)
            }
        }
        return try #require(words.withUnsafeMutableBytes { raw in
            CGContext(data: raw.baseAddress, width: w, height: h, bitsPerComponent: 16, bytesPerRow: w * 8,
                      space: EngineImageCoding.bgraColorSpace,
                      bitmapInfo: EngineImageCoding.rgba16BitmapInfo)?.makeImage()
        })
    }

    static func eightBitImage(width w: Int, height h: Int) throws -> CGImage {
        let ctx = try #require(CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                         space: EngineImageCoding.bgraColorSpace,
                                         bitmapInfo: EngineImageCoding.bgraBitmapInfo))
        for x in 0..<w {
            ctx.setFillColor(red: CGFloat(x) / CGFloat(w), green: 0.4, blue: 0.7, alpha: 1)
            ctx.fill(CGRect(x: x, y: 0, width: 1, height: h))
        }
        return try #require(ctx.makeImage())
    }

    static func redLevels(_ image: Image) -> Set<UInt16> {
        let words = image.data.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
        return Set(stride(from: 0, to: words.count, by: 4).map { words[$0] })
    }

    @Test("a deep image round-trips through .rawRGBA16 exactly")
    func deepRoundTripIsExact() throws {
        let deep = try Self.deepImage(width: 33, height: 7)
        let raw = try EngineImageCoding.rawImage(from: deep, keepingDepth: true)
        #expect(raw.format == .rawRGBA16)
        #expect(raw.width == 33 && raw.height == 7 && raw.bytesPerRow == 33 * 8)
        let back = try EngineImageCoding.cgImage(from: raw)
        #expect(back.bitsPerComponent == 16)
        #expect(try EngineImageCoding.rawImage(from: back, keepingDepth: true).data == raw.data)
    }

    /// Opting in changes nothing for an 8-bit image, and not opting in changes nothing for a deep one: both are the
    /// `.rawBGRA8` this always produced.
    @Test("keepingDepth only ever changes a deep image")
    func eightBitPathIsUnchanged() throws {
        let eight = try Self.eightBitImage(width: 16, height: 4)
        let plain = try EngineImageCoding.rawImage(from: eight)
        let opted = try EngineImageCoding.rawImage(from: eight, keepingDepth: true)
        #expect(plain.format == .rawBGRA8)
        #expect(opted == plain)

        let deep = try Self.deepImage(width: 16, height: 4)
        #expect(try EngineImageCoding.rawImage(from: deep).format == .rawBGRA8)
        #expect(try EngineImageCoding.rawImage(from: deep, keepingDepth: false).format == .rawBGRA8)
    }

    @Test(".rawRGBA16 without dimensions is a named error")
    func rawRGBA16WithoutDimensionsThrows() {
        #expect(throws: EngineImageCoding.CodingError.missingDimensions) {
            _ = try EngineImageCoding.cgImage(from: Image(format: .rawRGBA16, data: Data(count: 64)))
        }
    }

    /// Which tiers keep 16 bits is read from the registered manifests, not a table: RealPLKSR (≥ 0.2.0) declares
    /// `.rawRGBA16`; NERVE and NAFNet do not yet; an unregistered model never does. No model loads.
    @MainActor @Test("keepsSixteenBits reads the registered surface's declaration")
    func sixteenBitTiersComeFromTheManifests() async {
        #expect(await ForgeEngineService().keepsSixteenBits(.upscaleFidelity) == false)   // not registered
        let service = ForgeEngineService()
        await service.registerAll()
        #expect(await service.keepsSixteenBits(.upscaleFidelity))
        #expect(await service.keepsSixteenBits(.upscale) == false)
        #expect(await service.keepsSixteenBits(.restore) == false)
        #expect(await service.engine.residentPackages.isEmpty)
    }

    /// The chain's fit (`ChainedUpscale.swift`): a deep image used to come out of `createCGImage` at 8 bits.
    @Test("the chain's Lanczos fit keeps 16 bits for a deep image, and 8 for an 8-bit one")
    func lanczosKeepsDepth() throws {
        let deep = try Self.deepImage(width: 400, height: 16)
        let fitted = try ImageResampling.lanczos(deep, toWidth: 300, height: 12)
        #expect(fitted.width == 300 && fitted.height == 12)
        #expect(fitted.bitsPerComponent == 16)
        #expect(Self.redLevels(try EngineImageCoding.rawImage(from: fitted, keepingDepth: true)).count > 256)

        let eight = try ImageResampling.lanczos(try Self.eightBitImage(width: 64, height: 8), toWidth: 48, height: 6)
        #expect(eight.bitsPerComponent == 8)
    }
}
