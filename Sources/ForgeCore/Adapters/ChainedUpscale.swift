//
//  ChainedUpscale.swift
//  ForgeCore
//
//  ×6 and ×8 on a fleet whose every stills tier stops at ×4 (AB-D-0119, 2026-10-10).
//
//  No model in the registry goes past its native ×4 — RealPLKSR, NERVE, HEART, VOSR2, Nacre and SeedVR2 all clamp a
//  larger request to 4 and say so in `appliedScale` (the BRIDGE-029 contract). Topaz Gigapixel serves 6× (GIGAPIXEL-GAP
//  §3, G-1). This is how Forge serves it without a new model:
//
//      pass 1   the tier the user chose, at its native ×4 (guarded, for a generative tier — nothing changes there)
//      pass 2   the fast FIDELITY tier (`.upscale`, NERVE) at its native ×2 on pass 1's pixels → ×8
//      fit      for ×6, a Lanczos resample of the ×8 result down to exactly 6× (CoreImage); ×8 needs none
//
//  Why NERVE and not the same tier twice: the second pass only has to carry pixels that are already model-made, and
//  NERVE is deterministic, ~5× faster than RealPLKSR and ~17–21 ms per output megapixel (AB-R-0365) — the ×8
//  intermediate of a 1080p source is 133 MP, so the second pass is where the cost lives. Why no generative second
//  pass: stacking two generative models compounds invention and would run the text guard twice on text it already
//  judged; the guarded generative pass stays pass 1, and the fidelity pass only enlarges it.
//
//  Why fit DOWN from ×8 rather than UP from ×4: a Lanczos 1.5× enlargement of the ×4 output softens exactly the
//  detail the model paid for; a 0.75× reduction of the ×8 output keeps it. The only time the plan fits up is when the
//  ×8 intermediate would exceed `intermediatePixelCap` (16384² ≈ 268 MP of BGRA8), which no measured envelope covers;
//  the plan says so (`fitsUp`) and the receipt names it.
//
//  The plan is a pure function so it can be tested without an engine; the enhancer runs it.
//

import CoreGraphics
import CoreImage
import Foundation

/// How a requested linear factor is served by tiers that cap at `nativeMax`.
public struct ChainedUpscalePlan: Equatable, Sendable {
    public struct Size: Equatable, Sendable {
        public let width: Int
        public let height: Int
        public init(width: Int, height: Int) { self.width = width; self.height = height }
    }

    /// What the caller asked for.
    public let requested: Int
    /// The factor the chosen tier runs at (≤ `nativeMax`).
    public let primaryScale: Int
    /// A second pass on the fast fidelity tier at this factor, or nil for a single-pass request.
    public let secondaryScale: Int?
    /// The exact output size the chained pixels are resampled to, or nil when the passes land on it exactly.
    public let fit: Size?
    /// True when the fit ENLARGES the model output (the ×8 intermediate was over the cap) — the degraded shape.
    public let fitsUp: Bool

    /// Every stills tier's native ceiling today.
    public static let nativeMax = 4
    /// The largest factor the chain serves: ×4 then ×2.
    public static let chainedMax = 8
    /// The largest intermediate the second pass may produce: 16384² ≈ 268 MP, ≈ 1.07 GB of BGRA8.
    public static let intermediatePixelCap = 16_384 * 16_384

    public var isChained: Bool { secondaryScale != nil || fit != nil }

    /// The output size the plan produces for `source`.
    public func outputSize(for source: Size) -> Size {
        if let fit { return fit }
        let factor = primaryScale * (secondaryScale ?? 1)
        return Size(width: source.width * factor, height: source.height * factor)
    }

    /// - requested: the integer factor asked for; values above `chainedMax` are served as `chainedMax`, values
    ///   at or below `nativeMax` as a single pass (exactly today's behaviour).
    public static func make(requested: Int, source: Size, nativeMax: Int = nativeMax,
                            intermediatePixelCap: Int = intermediatePixelCap) -> ChainedUpscalePlan {
        let asked = min(max(requested, 1), chainedMax)
        guard asked > nativeMax else {
            return ChainedUpscalePlan(requested: asked, primaryScale: asked, secondaryScale: nil, fit: nil, fitsUp: false)
        }
        let target = Size(width: source.width * asked, height: source.height * asked)
        let intermediate = Size(width: source.width * nativeMax * 2, height: source.height * nativeMax * 2)
        if intermediate.width * intermediate.height <= intermediatePixelCap {
            // ×4 then ×2 = ×8; ×8 lands exactly, anything less fits DOWN.
            let fit: Size? = asked == nativeMax * 2 ? nil : target
            return ChainedUpscalePlan(requested: asked, primaryScale: nativeMax, secondaryScale: 2, fit: fit, fitsUp: false)
        }
        // Over the cap: one model pass at ×4, then a resample UP to the target — the honest fallback, named as such.
        return ChainedUpscalePlan(requested: asked, primaryScale: nativeMax, secondaryScale: nil, fit: target, fitsUp: true)
    }

    /// The receipt's name for the pixels: every model that touched them and the fit that finished them.
    /// `primaryName` is what pass 1 shipped under (a guarded composite keeps its own name).
    public func modelName(primaryName: String, secondaryName: String) -> String {
        var name = "\(primaryName) ×\(primaryScale)"
        if let secondaryScale { name += " + \(secondaryName) ×\(secondaryScale)" }
        if fit != nil { name += fitsUp ? " → ×\(requested) (Lanczos, enlarged)" : " → ×\(requested) (Lanczos)" }
        return name
    }
}

/// The one resampler the chain uses: CoreImage's Lanczos scale, sRGB in and out, cropped to the exact size. A deep
/// (16-bit) image comes out 16-bit (AB-A-0158); an 8-bit one renders exactly as it always has.
public enum ImageResampling {
    public enum ResampleError: Error, CustomStringConvertible {
        case badSize(Int, Int)
        case renderFailed
        public var description: String {
            switch self {
            case .badSize(let w, let h): return "cannot resample to \(w)×\(h)"
            case .renderFailed: return "CoreImage could not render the resampled image"
            }
        }
    }

    private static let context: CIContext = {
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        return CIContext(options: [.workingColorSpace: space, .outputColorSpace: space, .cacheIntermediates: false])
    }()

    /// Lanczos-resample `image` to exactly `width`×`height`. Aspect is whatever the caller's sizes say; the chain
    /// only ever asks for the source aspect at an integer factor.
    public static func lanczos(_ image: CGImage, toWidth width: Int, height: Int) throws -> CGImage {
        guard width > 0, height > 0 else { throw ResampleError.badSize(width, height) }
        if image.width == width, image.height == height { return image }
        let input = CIImage(cgImage: image)
        let scale = Double(height) / Double(image.height)
        let aspect = (Double(width) / Double(image.width)) / scale
        let filter = CIFilter(name: "CILanczosScaleTransform")!
        filter.setValue(input, forKey: kCIInputImageKey)
        filter.setValue(scale, forKey: kCIInputScaleKey)
        filter.setValue(aspect, forKey: kCIInputAspectRatioKey)
        guard let output = filter.outputImage else { throw ResampleError.renderFailed }
        // Lanczos pads its extent by the filter support; crop to the exact target so the size is what was asked.
        let rect = CGRect(x: 0, y: 0, width: width, height: height)
        // Without a `format:` CoreImage renders 8 bits, which would undo a 16-bit chain at its last step.
        let rendered = EngineImageCoding.isDeep(image)
            ? context.createCGImage(output.cropped(to: rect), from: rect, format: .RGBA16,
                                    colorSpace: CGColorSpace(name: CGColorSpace.sRGB))
            : context.createCGImage(output.cropped(to: rect), from: rect)
        guard let cg = rendered else { throw ResampleError.renderFailed }
        return cg
    }
}
