//
//  GenerativeFillRoute.swift
//  ForgeCore
//
//  Generative Fill and Expand as the Image Engine runs them (`RunPlan.inpaint` / `.extend` / `.paint`), plus the one
//  thing the Image Engine never needed: a padded WINDOW around the selection, so a 16 GB Mac can fill a 4K photo
//  through a 768² model call (AB-A-0166, AB-D-0141).
//
//  The model and LaMa are injected as closures. `EngineGenerativeFillProvider` passes engine calls; a test passes a
//  fake, so the whole route — window, marker, guard, retry, LaMa fallback, harmonize, seam bridge — runs on a fixture
//  with no GPU and no weights.
//
//  ⚠️ What is the bake-off's route and what is new, so nobody mistakes one for the other:
//  - **Unchanged** (FillRoute, verbatim): the instruction text, the magenta marker, the guard's thresholds and its one
//    retry with the next seed, the LaMa fallback after two rejections, the harmonize, the composite through the mask,
//    the extend's padded canvas, its per-side guard with the dark check, and its LaMa seam bridge.
//  - **New**: the window. When the whole picture fits the tier's window cap, the window IS the whole picture, and the
//    call is exactly the route's (the bake-off's 1024-class canvases at a 1024² cap — AB-R-0534's "fill" arm). When it
//    doesn't, the window is a square around the selection, chosen by the Image Engine's own measured rule for a square
//    around a region (`Media.square`, the removal gate's, AB-R-0464) with the cap's side in place of 1024 — which on a
//    1024-class picture at a 768² cap is the shape of the bake-off's "fill 768²" arm (a square around the region,
//    rendered at 768²). A selection longer than the picture's short side has no square, and gets the route's own
//    fallback for an over-envelope canvas: the whole picture, scaled into the cap.
//  - **New**: the marked window is handed to the model at the size it renders at, so the model's vision tokens are
//    bounded by the cap too. Every bake-off case already had input size = render size, so the measured regime is
//    unchanged; only an over-cap input (which the bake-off never sent) is now resized before the call, not inside it.
//  - **New**: the model's reference floor (AB-A-0199). SenseNova sizes its edit reference to at least 512² but renders
//    the size it is asked for, so a smaller render comes back zoomed by render/512 about its top-left corner (448²:
//    scale 0.875, −28 px). A window whose render would fall under `minimumRenderPixels` is resampled up until it doesn't,
//    on the grid and inside the cap, and its painting is resized back to the window as any resampled one is. The
//    bake-off's canvases were all above the floor, so its regime is unchanged.
//  - **New**: LaMa (fallback and seam bridge) is bounded to `inpaintMaxPixels` (4 × the window cap by default). Every
//    bake-off canvas was under it, so LaMa ran at its own size there and does here; a larger canvas is scaled into the
//    bound for LaMa and back, and the composite takes only the hole / band from it.
//

import CoreGraphics
import Foundation

/// Where the edit model paints, and the size it renders at.
public struct FillWindow: Sendable, Equatable, Codable {
    /// The window, in the picture's pixels (the padded canvas's for an expand).
    public var box: FillBox
    public var modelWidth: Int
    public var modelHeight: Int

    public init(box: FillBox, modelWidth: Int, modelHeight: Int) {
        self.box = box; self.modelWidth = modelWidth; self.modelHeight = modelHeight
    }

    /// Whether the model renders the window at its own size (no resample in or out).
    public var isOneToOne: Bool { box.width == modelWidth && box.height == modelHeight }
}

/// A painted region, before the composite through the mask.
public struct FillPatch: Sendable {
    /// Where the patch goes, in the output's pixels (for an expand, the whole padded canvas).
    public var box: FillBox
    /// The painting at `box`'s size.
    public var pixels: FillPixels
    /// What the composite blends by, at `box`'s size: the caller's grey mask (fill) or the new area (expand).
    public var mask: FillMask

    /// `base` with this patch composited through its mask at `box` (pixels outside the mask keep `base`'s exactly).
    public func applied(to base: FillPixels) -> FillPixels {
        let local = FillRoute.crop(base, box)
        return FillRoute.paste(FillRoute.composite(result: pixels, over: local, mask: mask), into: base, at: box)
    }
}

/// What one fill or expand did — the route's half of the provider's receipt.
public struct GenerativeFillTrace: Sendable, Equatable, Codable {
    /// The edit instruction sent to the model, verbatim.
    public var instruction: String
    public var window: FillWindow
    /// The seed of the painting that was used; nil when LaMa filled instead.
    public var seed: UInt64?
    /// 0 or 1: the guard rejected the first painting and the next seed was used.
    public var retries: Int
    /// The guard's last rejection ("marker" / "dark"), if any.
    public var rejected: String?
    /// The guard rejected both paintings and LaMa filled the region (the route's fallback).
    public var fellBackToInpaint: Bool
    /// Expand only: LaMa bridged a band on the new side of the seam.
    public var seamBridged: Bool
}

/// The fill / expand route over injected model calls. See the file header for what is the bake-off's and what is new.
public struct GenerativeFillRoute: Sendable {
    /// The edit model: paint the magenta area of `marked` (already at `width` × `height`) as `instruction` says, with
    /// `seed`, rendering at `width` × `height`.
    public typealias Painter = @Sendable (_ marked: FillPixels, _ instruction: String, _ seed: UInt64,
                                          _ width: Int, _ height: Int) async throws -> FillPixels
    /// The inpainter (LaMa): fill `hole`'s white pixels of `canvas` from their surroundings; same size back.
    public typealias Inpainter = @Sendable (_ canvas: FillPixels, _ hole: FillMask) async throws -> FillPixels

    /// The tier's window cap, in pixels (768² at 16 GB, 1024² above — AB-D-0141).
    public var maxWindowPixels: Int
    /// The model's size grid (SenseNova's token is 32 px, Klein's latent patch 16).
    public var multipleOf: Int
    /// The largest canvas LaMa runs on at its own size.
    public var inpaintMaxPixels: Int
    /// The smallest render, in pixels, the model paints in place (SenseNova's 512² reference floor, AB-A-0199); 0 = none.
    /// A smaller window is resampled up to it. The cap wins if the two can't both hold.
    public var minimumRenderPixels: Int

    public init(maxWindowPixels: Int, multipleOf: Int, minimumRenderPixels: Int = 0, inpaintMaxPixels: Int? = nil) {
        precondition(maxWindowPixels > 0 && multipleOf > 0 && minimumRenderPixels >= 0)
        self.maxWindowPixels = maxWindowPixels
        self.multipleOf = multipleOf
        self.minimumRenderPixels = minimumRenderPixels
        self.inpaintMaxPixels = inpaintMaxPixels ?? 4 * maxWindowPixels
    }

    // MARK: - Window

    /// The window a fill of `region` paints in, on a `width` × `height` picture.
    public func window(for region: FillBox, width w: Int, height h: Int) -> FillWindow {
        let m = multipleOf
        let whole = FillBox(x: 0, y: 0, width: w, height: h)
        // The whole picture fits the cap: exactly the route's call.
        if let fitted = FillRoute.fit(width: w, height: h, maxPixels: maxWindowPixels, multipleOf: m) {
            let size = floored(fitted, width: w, height: h)
            return FillWindow(box: whole, modelWidth: size.width, modelHeight: size.height)
        }
        // No square holds the selection: the route's own answer for an over-envelope canvas — all of it, scaled down.
        let short = min(w, h), long = max(region.width, region.height)
        guard long <= short else {
            let size = floored(FillRoute.clamp(width: w, height: h, maxPixels: maxWindowPixels, multipleOf: m), width: w, height: h)
            return FillWindow(box: whole, modelWidth: size.width, modelHeight: size.height)
        }
        // A square around the selection: 2.5× its long side for context, at least the cap's side (or the short side),
        // at most the short side (`Media.square`, with the cap's side for its 1024). A small selection on a large
        // photo gets a 1:1 window, so the fill isn't upscaled into a soft patch.
        let capSide = Int(Double(maxWindowPixels).squareRoot())
        var side = min(short, max(min(capSide, short), Int((Double(long) * 2.5).rounded())))
        let snapped = side / m * m                                      // on the model's grid: no resample at 1:1
        if snapped >= max(long, m) { side = snapped }
        let (cx, cy) = (region.x + region.width / 2, region.y + region.height / 2)
        let box = FillBox(x: min(max(cx - side / 2, 0), w - side), y: min(max(cy - side / 2, 0), h - side),
                          width: side, height: side)
        let size = floored(FillRoute.clamp(width: side, height: side, maxPixels: maxWindowPixels, multipleOf: m),
                           width: side, height: side)
        return FillWindow(box: box, modelWidth: size.width, modelHeight: size.height)
    }

    /// The render size for a whole canvas (an expand): the route's `envelope.fit ?? envelope.clamp` at the cap, lifted to
    /// the model's reference floor.
    public func modelSize(width w: Int, height h: Int) -> (width: Int, height: Int) {
        floored(FillRoute.fit(width: w, height: h, maxPixels: maxWindowPixels, multipleOf: multipleOf)
                    ?? FillRoute.clamp(width: w, height: h, maxPixels: maxWindowPixels, multipleOf: multipleOf),
                width: w, height: h)
    }

    /// `size`, the render for a `w` × `h` window, unless it is under `minimumRenderPixels`: then the window scaled up by
    /// √(floor / area), each side rounded to the grid, and the side furthest under that scale grown a step at a time until
    /// the floor holds (AB-A-0199: 451² → 512², 300² → 512²). A lift that would pass the cap is not made.
    func floored(_ size: (width: Int, height: Int), width w: Int, height h: Int) -> (width: Int, height: Int) {
        guard size.width * size.height < minimumRenderPixels, w > 0, h > 0 else { return size }
        let m = multipleOf
        let scale = (Double(minimumRenderPixels) / Double(w * h)).squareRoot()
        var (rw, rh) = (max(m, Int((Double(w) * scale / Double(m)).rounded()) * m),
                        max(m, Int((Double(h) * scale / Double(m)).rounded()) * m))
        while rw * rh < minimumRenderPixels {
            if Double(rw) / Double(w) <= Double(rh) / Double(h) { rw += m } else { rh += m }
        }
        return rw * rh <= maxWindowPixels ? (rw, rh) : size
    }

    // MARK: - Fill (RunPlan.inpaint's image.fill branch)

    /// Fill `mask`'s area of `image` from `prompt` (empty = blend it away, which is also how this route removes an
    /// object). nil when the mask selects nothing. The patch still has to be composited — `FillPatch.applied(to:)`.
    @concurrent
    public func fill(_ image: FillPixels, mask: FillMask, prompt: String, seed: UInt64,
                     paint: Painter, inpaint: Inpainter) async throws -> (patch: FillPatch, trace: GenerativeFillTrace)? {
        precondition(mask.width == image.width && mask.height == image.height, "the mask must be at the image's size")
        let prompt = FillRoute.normalized(prompt: prompt)
        guard let region = FillRoute.bounds(of: mask) else { return nil }
        let window = window(for: region, width: image.width, height: image.height)
        let canvas = FillRoute.crop(image, window.box), local = FillRoute.crop(mask, window.box)
        var trace = GenerativeFillTrace(instruction: FillRoute.fillInstruction(prompt: prompt), window: window, seed: nil,
                                        retries: 0, rejected: nil, fellBackToInpaint: false, seamBridged: false)
        if let painted = try await paintRegion(canvas, region: local, prompt: prompt, parts: nil, checkDark: false,
                                               seed: seed, paint: paint, trace: &trace) {
            return (FillPatch(box: window.box, pixels: painted, mask: local), trace)
        }
        // The model left its region unpainted twice: LaMa fills it, so the marker never reaches an output.
        try Task.checkCancellation()
        trace.fellBackToInpaint = true
        let filled = try await inpaintBounded(canvas, hole: local.hole, inpaint: inpaint)
        return (FillPatch(box: window.box, pixels: filled, mask: local), trace)
    }

    // MARK: - Expand (RunPlan.extend's generative branch)

    /// Extend `image` by `left`/`top`/`right`/`bottom` pixels, continuing the scene (and showing `prompt`, if any).
    /// The patch covers the whole padded canvas; composite it over `padded(image)` — the original's pixels stay exact.
    @concurrent
    public func expand(_ image: FillPixels, left: Int, top: Int, right: Int, bottom: Int, prompt: String, seed: UInt64,
                       paint: Painter, inpaint: Inpainter) async throws -> (canvas: FillPixels, patch: FillPatch, trace: GenerativeFillTrace) {
        precondition(left >= 0 && top >= 0 && right >= 0 && bottom >= 0 && left + top + right + bottom > 0)
        let prompt = FillRoute.normalized(prompt: prompt)
        let (canvas, mask) = FillRoute.pad(image, left: left, top: top, right: right, bottom: bottom)
        let size = modelSize(width: canvas.width, height: canvas.height)
        let window = FillWindow(box: FillBox(x: 0, y: 0, width: canvas.width, height: canvas.height),
                                modelWidth: size.width, modelHeight: size.height)
        var trace = GenerativeFillTrace(
            instruction: FillRoute.extendInstruction(left: left, top: top, right: right, bottom: bottom, prompt: prompt),
            window: window, seed: nil, retries: 0, rejected: nil, fellBackToInpaint: false, seamBridged: false)
        let sides = FillRoute.sides(width: canvas.width, height: canvas.height, left: left, top: top, right: right, bottom: bottom)
        guard let paintedRegion = try await paintRegion(canvas, region: mask, prompt: prompt, parts: sides, checkDark: true,
                                                        seed: seed, paint: paint, trace: &trace) else {
            try Task.checkCancellation()
            trace.fellBackToInpaint = true
            let filled = try await inpaintBounded(canvas, hole: mask, inpaint: inpaint)
            return (canvas, FillPatch(box: window.box, pixels: filled, mask: mask), trace)
        }
        let painted = FillRoute.composite(result: paintedRegion, over: canvas, mask: mask)
        // The edit model continued its own re-rendering of the picture, so its border can meet the original out of step
        // (a trunk a little to one side). LaMa bridges a thin band on the new side of the seam from both sides' context;
        // the original's pixels stay as they are.
        try Task.checkCancellation()
        let band = FillRoute.seamBand(of: mask, width: FillRoute.seamBandWidth(width: canvas.width, height: canvas.height))
        let bridged = try await inpaintBounded(painted, hole: band, inpaint: inpaint)
        trace.seamBridged = true
        let result = FillRoute.composite(result: bridged, over: painted, mask: band)
        return (canvas, FillPatch(box: window.box, pixels: result, mask: mask), trace)
    }

    // MARK: - Steps

    /// RunPlan.paint: the model paints `region` (shown to it in magenta) as the instruction says; the result comes back
    /// at the canvas's size. A result that keeps the marker or leaves the region dark is retried once with the next
    /// seed, and nil after that — the caller falls back. A used result is harmonized toward the canvas.
    private func paintRegion(_ canvas: FillPixels, region: FillMask, prompt: String, parts: [FillMask]?, checkDark: Bool,
                             seed: UInt64, paint: Painter, trace: inout GenerativeFillTrace) async throws -> FillPixels? {
        let (mw, mh) = (trace.window.modelWidth, trace.window.modelHeight)
        let marked = FillRoute.mark(canvas, region: region).resized(mw, mh)
        let checkable = FillRoute.checkable(prompt: prompt)
        for attempt in 0..<2 {
            try Task.checkCancellation()
            let attemptSeed = seed &+ UInt64(attempt)
            let result = try await paint(marked, trace.instruction, attemptSeed, mw, mh).resized(canvas.width, canvas.height)
            let problem = checkable
                ? FillRoute.paintingProblem(result, canvas: canvas, parts: parts ?? [region], checkDark: checkDark) : nil
            guard let problem else {
                trace.seed = attemptSeed
                trace.retries = attempt
                return FillRoute.harmonize(result, toward: canvas, region: region)
            }
            trace.rejected = problem
        }
        return nil
    }

    /// LaMa at the canvas's own size up to `inpaintMaxPixels`; a larger canvas is scaled into that bound and back.
    private func inpaintBounded(_ canvas: FillPixels, hole: FillMask, inpaint: Inpainter) async throws -> FillPixels {
        guard canvas.width * canvas.height > inpaintMaxPixels else { return try await inpaint(canvas, hole) }
        let size = FillRoute.clamp(width: canvas.width, height: canvas.height, maxPixels: inpaintMaxPixels, multipleOf: 8)
        let small = canvas.resized(size.width, size.height)
        let smallHole = FillMask(hole.cgImage, width: size.width, height: size.height).hole
        return try await inpaint(small, smallHole).resized(canvas.width, canvas.height)
    }
}
