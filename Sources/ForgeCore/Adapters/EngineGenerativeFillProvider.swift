//
//  EngineGenerativeFillProvider.swift
//  ForgeCore
//
//  Generative Fill and Generative Expand, engine-backed (AB-A-0166, AB-D-0141).
//
//  `fill(_:mask:prompt:)` is the editor's `GenerativeFillProvider` seam to the letter (AB-A-0166):
//  the mask is grey, white = generate, at the image's size; the result is the whole image at the same size, with the
//  generated region composited through the mask — a soft selection blends, and every pixel outside it is the input's
//  exactly. A host conforms this class to that protocol in one line
//  (`extension EngineGenerativeFillProvider: @retroactive GenerativeFillProvider {}`); ForgeCore does not depend on
//  the editor's AI package, which consumes the net-clean Kits and never ForgeCore.
//
//  Expand is a sibling method on the SAME provider, not a second seam: it is the Image Engine's `image.extend` route —
//  the same model, marker, guard and harmonize, plus its LaMa seam bridge — so one registration and one report serve
//  both, as they did in the bake-off (AB-R-0534's extend arm).
//
//  The route itself lives in `GenerativeFill/` (engine-free; see `GenerativeFillRoute`'s header for what is the
//  bake-off's and what is new). This file only puts the engine behind it: the edit model is `.generativeFill`
//  (whichever model the service's `GenerativeFillTier` names) and the fallback / seam bridge is `.inpaint` (LaMa).
//

import Foundation
import CoreGraphics
import MLXServeCore
import MLXToolKit

/// What one fill or expand ran, for the host's receipt and the layer's provenance (AB-T-0246).
public struct GenerativeFillReport: Sendable, Codable, Equatable {
    public enum Operation: String, Sendable, Codable { case fill, expand }

    public var operation: Operation
    /// The memory tier's label ("16 GB", "24 GB", "32 GB+").
    public var tier: String
    /// The edit model and its exact tier, e.g. "SenseNova-U1.5-8B-MoT 8step-v2-4bit".
    public var model: String
    /// The weight quantization ("int4", "int8", "bf16").
    public var quant: String
    /// The weights' licence (SPDX), read from the package manifest the engine registered.
    public var weightLicense: String
    /// The port's code licence (SPDX), from the same manifest.
    public var portCodeLicense: String
    /// The engine registration that ran.
    public var packageID: String
    /// The prompt as the route read it (trimmed); empty = blend the area away.
    public var prompt: String
    /// The route's account: instruction, window, seed, retries, fallback. nil when the mask selected nothing.
    public var trace: GenerativeFillTrace?
    /// The inpainter, when one ran: LaMa as the guard's fallback, or as an expand's seam bridge.
    public var inpaintModel: String?
    /// Wall time of the whole call, model loads included.
    public var seconds: Double
}

/// An image and the report of the call that made it.
public struct GenerativeFillResult: @unchecked Sendable {
    public let image: CGImage
    public let report: GenerativeFillReport
}

@MainActor
public final class EngineGenerativeFillProvider {

    /// The provider's identity for a receipt: the tier's model when the provider was made. The model of a given call is
    /// in its report — `fillReporting(_:mask:prompt:seed:)`.
    public nonisolated let name: String

    /// The models a fill can need, for a host's download offer and tier gating: the edit model, and LaMa — the guard's
    /// fallback and expand's seam bridge. Offer both before the first fill.
    public nonisolated let requiredModels: [ForgeModel] = [.generativeFill, .inpaint]

    private let service: ForgeEngineService
    private let onReport: (@Sendable (GenerativeFillReport) -> Void)?

    /// - Parameter onReport: called with every call's report — how a host reached through the plain seam (which returns
    ///   only an image) still records model, quant and licence per generated layer.
    public init(service: ForgeEngineService, onReport: (@Sendable (GenerativeFillReport) -> Void)? = nil) {
        self.service = service
        self.onReport = onReport
        let tier = service.registeredGenerativeFillTier ?? service.generativeFillTier
        self.name = tier.map { "Generative Fill — \($0.model.displayName)" } ?? "Generative Fill (no tier on this Mac)"
    }

    public enum GenerativeFillError: Error, CustomStringConvertible {
        /// `.generativeFill` did not register: no tier for this Mac, or the engine refused the model (licence, chip).
        case unavailable(String)
        /// LaMa (`.inpaint`) is not registered, and this call needs it.
        case inpaintUnavailable
        /// The engine answered with something other than an image.
        case unexpectedResponse
        /// A negative padding, or none at all.
        case invalidPadding

        public var description: String {
            switch self {
            case .unavailable(let why): "Generative Fill is unavailable: \(why)"
            case .inpaintUnavailable: "Generative Fill needs LaMa (.inpaint) for its fallback and expand's seam, and LaMa is not registered"
            case .unexpectedResponse: "The model returned no image"
            case .invalidPadding: "Expand needs non-negative padding on at least one side"
            }
        }
    }

    // MARK: - The seam

    /// Fill `mask`'s white area of `image` from `prompt` (empty = blend it away, which also removes an object).
    /// The editor's `GenerativeFillProvider.fill(_:mask:prompt:)`.
    public func fill(_ image: CGImage, mask: CGImage, prompt: String) async throws -> CGImage {
        try await fillReporting(image, mask: mask, prompt: prompt).image
    }

    /// `fill`, with the call's report. `seed` nil = a fresh random seed each call (a re-run gives a new variation); the
    /// route uses `seed`, then `seed + 1` if the guard rejects the first painting.
    public func fillReporting(_ image: CGImage, mask: CGImage, prompt: String, seed: UInt64? = nil) async throws -> GenerativeFillResult {
        try Task.checkCancellation()
        let started = ContinuousClock.now
        let context = try await runContext(needsInpaint: false)
        let (output, trace) = try await Self.fill(image, mask: mask, prompt: prompt, seed: seed ?? Self.randomSeed(),
                                                  route: context.route, paint: context.paint, inpaint: context.inpaint)
        return finish(output, trace: trace, operation: .fill, prompt: prompt, context: context, started: started)
    }

    // MARK: - Expand

    /// Extend `image` by `left` / `top` / `right` / `bottom` pixels with a seamless continuation of the scene (showing
    /// `prompt`, if any). The result is the larger canvas; the original's pixels sit at (`left`, `top`) unchanged.
    public func expand(_ image: CGImage, left: Int, top: Int, right: Int, bottom: Int, prompt: String = "") async throws -> CGImage {
        try await expandReporting(image, left: left, top: top, right: right, bottom: bottom, prompt: prompt).image
    }

    /// `expand`, with the call's report.
    public func expandReporting(_ image: CGImage, left: Int, top: Int, right: Int, bottom: Int, prompt: String = "",
                                seed: UInt64? = nil) async throws -> GenerativeFillResult {
        guard min(left, top, right, bottom) >= 0, left + top + right + bottom > 0 else { throw GenerativeFillError.invalidPadding }
        try Task.checkCancellation()
        let started = ContinuousClock.now
        let context = try await runContext(needsInpaint: true)
        let (output, trace) = try await Self.expand(image, left: left, top: top, right: right, bottom: bottom, prompt: prompt,
                                                    seed: seed ?? Self.randomSeed(), route: context.route,
                                                    paint: context.paint, inpaint: context.inpaint)
        return finish(output, trace: trace, operation: .expand, prompt: prompt, context: context, started: started)
    }

    // MARK: - Engine binding

    private struct RunContext {
        let tier: GenerativeFillTier
        let packageID: PackageID
        let license: LicenseDeclaration?
        let route: GenerativeFillRoute
        let paint: GenerativeFillRoute.Painter
        let inpaint: GenerativeFillRoute.Inpainter
    }

    private func runContext(needsInpaint: Bool) async throws -> RunContext {
        guard let fillID = service.packageID(for: .generativeFill), let tier = service.registeredGenerativeFillTier else {
            throw GenerativeFillError.unavailable(service.outcome(for: .generativeFill)?.failure ?? "not registered — call registerAll()")
        }
        let lamaID = service.packageID(for: .inpaint)
        if needsInpaint, lamaID == nil { throw GenerativeFillError.inpaintUnavailable }
        let engine = service.engine
        let paint: GenerativeFillRoute.Painter = { marked, instruction, seed, width, height in
            // Steps and guidance stay the package's tier defaults — SenseNova's distill tiers 8 steps at cfg 1, Klein's
            // turbo 4 at 1 — which are exactly what the route passed in the bake-off (the catalog's GenerationDefaults).
            let request = IEditRequest(images: [try EngineImageCoding.pngImage(from: marked.cgImage)], prompt: instruction,
                                       width: width, height: height, seed: seed)
            let response = try await engine.run(request, package: fillID)
            guard let output = (response as? IEditResponse)?.image else { throw GenerativeFillError.unexpectedResponse }
            return FillPixels(try EngineImageCoding.cgImage(from: output))
        }
        let inpaint: GenerativeFillRoute.Inpainter = { canvas, hole in
            guard let lamaID else { throw GenerativeFillError.inpaintUnavailable }
            // LaMa, as the route's fallback / seam bridge called it (no mode = LaMa, never MI-GAN).
            let request = InpaintRequest(image: try EngineImageCoding.pngImage(from: canvas.cgImage),
                                         mask: try EngineImageCoding.pngImage(from: hole.cgImage), mode: InpaintContract.best)
            let response = try await engine.run(request, package: lamaID)
            guard let output = (response as? InpaintResponse)?.image else { throw GenerativeFillError.unexpectedResponse }
            return FillPixels(try EngineImageCoding.cgImage(from: output))
        }
        return RunContext(tier: tier, packageID: fillID, license: await engine.manifest(for: fillID)?.license,
                          route: GenerativeFillRoute(maxWindowPixels: tier.maxWindowPixels, multipleOf: tier.model.multipleOf),
                          paint: paint, inpaint: inpaint)
    }

    private func finish(_ image: CGImage, trace: GenerativeFillTrace?, operation: GenerativeFillReport.Operation,
                        prompt: String, context: RunContext, started: ContinuousClock.Instant) -> GenerativeFillResult {
        let elapsed = ContinuousClock.now - started
        let report = GenerativeFillReport(
            operation: operation, tier: context.tier.label, model: context.tier.model.displayName,
            quant: context.tier.model.quant.rawValue,
            weightLicense: context.license?.weightLicense.identifier ?? "unknown",
            portCodeLicense: context.license?.portCodeLicense.identifier ?? "unknown",
            packageID: context.packageID.rawValue, prompt: FillRoute.normalized(prompt: prompt), trace: trace,
            inpaintModel: (trace?.fellBackToInpaint == true || trace?.seamBridged == true) ? "LaMa" : nil,
            seconds: Double(elapsed.components.seconds) + Double(elapsed.components.attoseconds) * 1e-18)
        onReport?(report)
        return GenerativeFillResult(image: image, report: report)
    }

    /// A seed that survives JSON (≤ 2^53 − 1) and leaves room for the retry's `+ 1` — the route's `seed(count: 2)`.
    nonisolated static func randomSeed() -> UInt64 { UInt64.random(in: 0...(9_007_199_254_740_991 - 2)) }

    // MARK: - Pixel work, off the main actor

    /// Decode, run the route, composite. A deep (16-bit) image is composited at 16 bits, so every pixel outside the mask
    /// keeps its full depth; the painting itself is the model's 8 bits.
    @concurrent
    nonisolated static func fill(_ image: CGImage, mask maskImage: CGImage, prompt: String, seed: UInt64,
                                 route: GenerativeFillRoute, paint: GenerativeFillRoute.Painter,
                                 inpaint: GenerativeFillRoute.Inpainter) async throws -> (CGImage, GenerativeFillTrace?) {
        let pixels = FillPixels(image)
        let mask = FillMask(maskImage, width: image.width, height: image.height)
        guard let (patch, trace) = try await route.fill(pixels, mask: mask, prompt: prompt, seed: seed,
                                                        paint: paint, inpaint: inpaint) else {
            return (image, nil)                                         // nothing selected: the image, untouched
        }
        try Task.checkCancellation()
        if EngineImageCoding.isDeep(image) {
            var deep = FillPixels16(image)
            deep.composite(patch)
            return (deep.cgImage, trace)
        }
        return (patch.applied(to: pixels).cgImage, trace)
    }

    @concurrent
    nonisolated static func expand(_ image: CGImage, left: Int, top: Int, right: Int, bottom: Int, prompt: String, seed: UInt64,
                                   route: GenerativeFillRoute, paint: GenerativeFillRoute.Painter,
                                   inpaint: GenerativeFillRoute.Inpainter) async throws -> (CGImage, GenerativeFillTrace) {
        let (canvas, patch, trace) = try await route.expand(FillPixels(image), left: left, top: top, right: right, bottom: bottom,
                                                            prompt: prompt, seed: seed, paint: paint, inpaint: inpaint)
        try Task.checkCancellation()
        if EngineImageCoding.isDeep(image) {
            var deep = FillPixels16(image).padded(left: left, top: top, right: right, bottom: bottom)
            deep.composite(patch)
            return (deep.cgImage, trace)
        }
        return (patch.applied(to: canvas).cgImage, trace)
    }
}

/// 16-bit sRGB premultiplied RGBA (`EngineImageCoding.rgba16BitmapInfo`), for the final composite of a deep image.
struct FillPixels16 {
    let width: Int
    let height: Int
    var rgba: [UInt16]

    init(width: Int, height: Int) {
        self.width = width; self.height = height
        rgba = [UInt16](repeating: 0, count: width * height * 4)
    }

    init(_ image: CGImage) {
        self.init(width: image.width, height: image.height)
        let (w, h) = (width, height)
        rgba.withUnsafeMutableBytes { buffer in
            let ctx = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 16, bytesPerRow: w * 8,
                                space: FillPixels.sRGB, bitmapInfo: EngineImageCoding.rgba16BitmapInfo)!
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
    }

    var cgImage: CGImage {
        var copy = rgba
        let (w, h) = (width, height)
        return copy.withUnsafeMutableBytes { buffer in
            CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 16, bytesPerRow: w * 8,
                      space: FillPixels.sRGB, bitmapInfo: EngineImageCoding.rgba16BitmapInfo)!.makeImage()!
        }
    }

    /// On a larger canvas at (`left`, `top`); the new area is empty — an expand's patch covers it at full strength.
    func padded(left: Int, top: Int, right: Int, bottom: Int) -> FillPixels16 {
        var out = FillPixels16(width: width + left + right, height: height + top + bottom)
        for row in 0..<height {
            let src = row * width * 4, dst = ((row + top) * out.width + left) * 4
            out.rgba.replaceSubrange(dst..<dst + width * 4, with: rgba[src..<src + width * 4])
        }
        return out
    }

    /// `FillRoute.composite` at 16 bits: the patch (8-bit, widened ×257) blended by its mask; black keeps these pixels.
    mutating func composite(_ patch: FillPatch) {
        let box = patch.box
        for y in 0..<box.height {
            for x in 0..<box.width {
                let m = UInt32(patch.mask.values[y * box.width + x])
                guard m > 0 else { continue }
                let p = (y * box.width + x) * 4, d = ((box.y + y) * width + box.x + x) * 4
                for c in 0..<4 {
                    let n = UInt32(patch.pixels.rgba[p + c]) * 257, o = UInt32(rgba[d + c])
                    rgba[d + c] = UInt16((n * m + o * (255 - m) + 127) / 255)
                }
            }
        }
    }
}
