//
//  EngineImageCoding.swift
//  ForgeCore
//
//  The ONE CGImage ⇄ MLXToolKit.Image conversion in this app (BRIDGE-034). The old app target carried
//  six private copies of this pair — one per adapter — and only `EngineImageEnhancer` had been migrated
//  off the PNG round-trip, so five adapters PNG-encoded and decoded every image they handed the engine.
//
//  Two encodings, chosen per wrapper:
//
//    • `.rawBGRA8` (contract 1.9.0) — raw interleaved BGRA8, no compression at the model boundary. Only
//      usable when the wrapper *mirrors* it (raw in ⇒ raw out). Measured on the warm leg: NAFNet −74%,
//      Real-ESRGAN −20%.
//    • `.png` — the compatible path, for wrappers that have not adopted rawBGRA8.
//
//  As of the pinned graph, `mlx-nafnet-swift` and `mlx-realesrgan-swift` mirror rawBGRA8; BiRefNet,
//  DDColor, LaMa, EdgeTAM and SEA-RAFT do not (see AGENT_BRIDGE.md BRIDGE-041). Callers state which
//  they need — this type does not guess, because handing raw bytes to a PNG-only wrapper fails as a
//  decode error deep inside the package rather than here.
//
//  16 bits (contract 1.52.0, AB-A-0158): `.rawRGBA16` carries a deep image to a surface that declares it
//  (`ToolDescriptor.rawImageFormats`; `ForgeEngineService.keepsSixteenBits`). The caller opts in per call
//  (`rawImage(from:keepingDepth:)`), because the engine refuses `.rawRGBA16` on a surface that does not
//  declare it; an 8-bit image is `.rawBGRA8` either way, byte for byte. Decoding is by format, so
//  `cgImage(from:)` returns a 16-bit image whenever the package answered in 16 bits.
//

import Foundation
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import MLXToolKit

public enum EngineImageCoding {

    public enum CodingError: Error {
        case encodeFailed
        case decodeFailed
        /// A `.rawBGRA8` image arrived without the width/height the contract requires.
        case missingDimensions
    }

    // MARK: - Bitmap layout

    /// 32BGRA, premultiplied-first + little-endian — the byte order `CVPixelBuffer` uses, so the future
    /// Metal path shares this layout rather than converting again.
    public static let bgraBitmapInfo =
        CGImageAlphaInfo.premultipliedFirst.rawValue | CGBitmapInfo.byteOrder32Little.rawValue
    public static let bgraColorSpace = CGColorSpace(name: CGColorSpace.sRGB) ?? CGColorSpaceCreateDeviceRGB()

    /// RGBA, 16 bits per component, little-endian, premultiplied — `.rawRGBA16` as CoreGraphics draws it.
    public static let rgba16BitmapInfo =
        CGImageAlphaInfo.premultipliedLast.rawValue | CGBitmapInfo.byteOrder16Little.rawValue

    /// Whether `cg` carries more than 8 bits per component — the images the 16-bit boundary exists for.
    public static func isDeep(_ cg: CGImage) -> Bool { cg.bitsPerComponent > 8 }

    // MARK: - CGImage → Image

    /// Render into raw pixels and hand them over directly. `keepingDepth`: a deep `cg` becomes `.rawRGBA16` —
    /// pass it only for a surface that keeps 16 bits (`ForgeEngineService.keepsSixteenBits`). An 8-bit `cg`, or
    /// `keepingDepth: false`, gives the tightly-packed 32BGRA `.rawBGRA8` this always produced.
    public static func rawImage(from cg: CGImage, keepingDepth: Bool = false) throws -> Image {
        if keepingDepth, isDeep(cg) { return try rawRGBA16(from: cg) }
        let w = cg.width, h = cg.height
        let bytesPerRow = w * 4
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8,
                                  bytesPerRow: bytesPerRow, space: bgraColorSpace,
                                  bitmapInfo: bgraBitmapInfo),
              let base = ctx.data else { throw CodingError.encodeFailed }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return Image.rawBGRA8(data: Data(bytes: base, count: bytesPerRow * h),
                              width: w, height: h, bytesPerRow: bytesPerRow)
    }

    /// Render into a tightly-packed RGBA16 buffer (sRGB, `rgba16BitmapInfo`) as `.rawRGBA16`.
    public static func rawRGBA16(from cg: CGImage) throws -> Image {
        let w = cg.width, h = cg.height
        let bytesPerRow = w * 8
        guard let ctx = CGContext(data: nil, width: w, height: h, bitsPerComponent: 16,
                                  bytesPerRow: bytesPerRow, space: bgraColorSpace,
                                  bitmapInfo: rgba16BitmapInfo),
              let base = ctx.data else { throw CodingError.encodeFailed }
        ctx.draw(cg, in: CGRect(x: 0, y: 0, width: w, height: h))
        return Image.rawRGBA16(data: Data(bytes: base, count: bytesPerRow * h),
                               width: w, height: h, bytesPerRow: bytesPerRow)
    }

    /// PNG-encode, for wrappers that have not adopted rawBGRA8.
    public static func pngImage(from cg: CGImage) throws -> Image {
        let data = NSMutableData()
        guard let dest = CGImageDestinationCreateWithData(data, UTType.png.identifier as CFString, 1, nil)
        else { throw CodingError.encodeFailed }
        CGImageDestinationAddImage(dest, cg, nil)
        guard CGImageDestinationFinalize(dest) else { throw CodingError.encodeFailed }
        return Image(format: .png, data: data as Data, width: cg.width, height: cg.height)
    }

    // MARK: - Image → CGImage

    /// Accepts either encoding: raw pixels go straight into a `CGImage` via `CGDataProvider`; png/jpeg
    /// fall back to `CGImageSource`. A wrapper that ignores a raw request and answers in PNG still works.
    public static func cgImage(from image: Image) throws -> CGImage {
        if image.format == .rawRGBA16 { return try cgImage(fromRGBA16: image) }
        guard image.format == .rawBGRA8 else { return try cgImage(fromEncoded: image.data) }
        guard let w = image.width, let h = image.height else { throw CodingError.missingDimensions }
        let bytesPerRow = image.bytesPerRow ?? (w * 4)
        guard let provider = CGDataProvider(data: image.data as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 8, bitsPerPixel: 32,
                               bytesPerRow: bytesPerRow, space: bgraColorSpace,
                               bitmapInfo: CGBitmapInfo(rawValue: bgraBitmapInfo),
                               provider: provider, decode: nil, shouldInterpolate: false,
                               intent: .defaultIntent) else {
            throw CodingError.decodeFailed
        }
        return cg
    }

    /// A `.rawRGBA16` image as a 16-bit sRGB `CGImage` over its bytes.
    static func cgImage(fromRGBA16 image: Image) throws -> CGImage {
        guard let w = image.width, let h = image.height else { throw CodingError.missingDimensions }
        let bytesPerRow = image.bytesPerRow ?? (w * 8)
        guard image.data.count >= bytesPerRow * h,
              let provider = CGDataProvider(data: image.data as CFData),
              let cg = CGImage(width: w, height: h, bitsPerComponent: 16, bitsPerPixel: 64,
                               bytesPerRow: bytesPerRow, space: bgraColorSpace,
                               bitmapInfo: CGBitmapInfo(rawValue: rgba16BitmapInfo),
                               provider: provider, decode: nil, shouldInterpolate: false,
                               intent: .defaultIntent) else {
            throw CodingError.decodeFailed
        }
        return cg
    }

    /// Decode compressed bytes — mattes are PNG-only by contract, so they come through here.
    public static func cgImage(fromEncoded data: Data) throws -> CGImage {
        guard let src = CGImageSourceCreateWithData(data as CFData, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw CodingError.decodeFailed
        }
        return cg
    }

    // MARK: - Self-check

    /// Byte-exact 2×2 R/G/B/white round-trip through the raw path — the BRIDGE-024 test, kept in the app
    /// so it can run without a test target and without touching MLX (no metallib, no GPU).
    ///
    /// It catches the failure this helper exists to prevent: a BGRA/RGBA channel swap, which produces
    /// images that look plausible (red and blue exchanged) rather than obviously broken.
    public static func selfCheckRawRoundTrip() -> Result<Void, Error> {
        // Opaque, fully-saturated colours: premultiplication is the identity, so the bytes must survive
        // the round trip exactly. B, G, R, A order.
        let source: [UInt8] = [
            0, 0, 255, 255,      255, 0, 0, 255,      // red,  blue
            0, 255, 0, 255,      255, 255, 255, 255,  // green, white
        ]
        do {
            let start = Image.rawBGRA8(data: Data(source), width: 2, height: 2, bytesPerRow: 8)
            let cg = try cgImage(from: start)
            guard cg.width == 2, cg.height == 2 else {
                return .failure(CodingError.decodeFailed)
            }
            let end = try rawImage(from: cg)
            guard Array(end.data) == source else {
                return .failure(SelfCheckFailure.bytesDiffer(expected: source, got: Array(end.data)))
            }
            return .success(())
        } catch {
            return .failure(error)
        }
    }

    public enum SelfCheckFailure: Error, CustomStringConvertible {
        case bytesDiffer(expected: [UInt8], got: [UInt8])

        public var description: String {
            switch self {
            case .bytesDiffer(let expected, let got):
                return "rawBGRA8 round-trip is not byte-exact — expected \(expected), got \(got)"
            }
        }
    }
}
