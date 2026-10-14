//
//  FillRoute.swift
//  ForgeCore
//
//  The Image Engine's generative fill route, as engine-free primitives (AB-A-0166).
//
//  🔑 **This is a port, not a re-design, and it has to stay one.** The Generative Fill bake-off (AB-R-0534,
//  pixel-agency-image-engine docs/FILL-BAKEOFF.md) measured SenseNova and Klein *through this route*: the magenta
//  marker, the paint guard (marker or dark → retry with the next seed, then LaMa), the seam harmonize and the composite
//  through the mask. Its verdict holds only for the same route. Every function below is a line-for-line port of
//  pixel-agency-image-engine 0.2.9 (e6fc698) `Media.swift` / `RunPlan.swift` — the same arithmetic, the same constants,
//  the same instruction text — so a change here is a change to what was measured. Source lines are named per function.
//
//  Nothing here imports MLX or the engine: it is CoreGraphics over 8-bit sRGB premultiplied RGBA, so it is testable on
//  a fixture with no GPU, and the Image Engine (which already links ForgeCore) can adopt these instead of keeping its
//  own copy. Names carry a `Fill` prefix so they never shadow the Image Engine's internal `Pixels` / `Mask` / `Box`.
//

import CoreGraphics
import Foundation

/// Pixels the route works in: sRGB, 8 bits per channel, premultiplied RGBA, top row first
/// (pixel-agency-image-engine `Pixels`, Media.swift:9).
public struct FillPixels: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public var rgba: [UInt8]

    public static let sRGB = CGColorSpace(name: CGColorSpace.sRGB)!

    public init(width: Int, height: Int, rgba: [UInt8]? = nil) {
        self.width = width; self.height = height
        self.rgba = rgba ?? [UInt8](repeating: 0, count: width * height * 4)
    }

    /// Draws `image` into a new buffer of `width` × `height` (its own size by default), converting to sRGB.
    public init(_ image: CGImage, width: Int? = nil, height: Int? = nil) {
        let (w, h) = (width ?? image.width, height ?? image.height)
        self.init(width: w, height: h)
        rgba.withUnsafeMutableBytes { buffer in
            let ctx = CGContext(data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                                space: Self.sRGB, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        }
    }

    public var cgImage: CGImage {
        var copy = rgba
        return copy.withUnsafeMutableBytes { buffer in
            CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                      space: Self.sRGB, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!.makeImage()!
        }
    }

    public func resized(_ w: Int, _ h: Int) -> FillPixels { w == width && h == height ? self : FillPixels(cgImage, width: w, height: h) }
}

/// An 8-bit grey mask, top row first. White (255) = change (pixel-agency-image-engine `Mask`, Media.swift:94).
public struct FillMask: Sendable, Equatable {
    public let width: Int
    public let height: Int
    public var values: [UInt8]

    public init(width: Int, height: Int, values: [UInt8]? = nil) {
        self.width = width; self.height = height
        self.values = values ?? [UInt8](repeating: 0, count: width * height)
    }

    /// A mask from any image (its luminance), resampled to `width` × `height`.
    public init(_ image: CGImage, width: Int, height: Int) {
        self.init(width: width, height: height)
        values.withUnsafeMutableBytes { buffer in
            let ctx = CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                                space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!
            ctx.interpolationQuality = .high
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
        }
    }

    /// Every pixel the mask lets change, at full strength: what a model is asked to repaint. The composite still
    /// blends by the mask itself, so black pixels stay untouched and grey ones blend; but a model shown only the mask's
    /// solid core paints the soft fringe back in from what it can still see.
    public var hole: FillMask { FillMask(width: width, height: height, values: values.map { $0 > 0 ? 255 : 0 }) }

    public var cgImage: CGImage {
        var copy = values
        return copy.withUnsafeMutableBytes { buffer in
            CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width,
                      space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)!.makeImage()!
        }
    }
}

/// A pixel rectangle, top row first (pixel-agency-image-engine `Box`, Media.swift:566).
public struct FillBox: Equatable, Hashable, Sendable, Codable {
    public var x: Int, y: Int, width: Int, height: Int

    public init(x: Int, y: Int, width: Int, height: Int) {
        self.x = x; self.y = y; self.width = width; self.height = height
    }

    /// Grown by `margin` on every side, kept inside a `width` × `height` picture.
    public func grown(by margin: Int, within w: Int, _ h: Int) -> FillBox {
        let (x0, y0) = (max(0, x - margin), max(0, y - margin))
        let (x1, y1) = (min(w, x + width + margin), min(h, y + height + margin))
        return FillBox(x: x0, y: y0, width: x1 - x0, height: y1 - y0)
    }
}

/// The route's steps. Each is the Image Engine's, verbatim in behaviour; see the file header.
public enum FillRoute {

    // MARK: - Instructions (RunPlan.swift:288–291, 376–380)

    /// The edit instruction for a fill (`image.fill`): seamless when there is no prompt — which is also how the
    /// bake-off removed objects (AB-R-0534: SenseNova removes through it with no naming model) — else "replace with".
    public static func fillInstruction(prompt: String) -> String {
        prompt.isEmpty
            ? "Fill the bright magenta area so it blends seamlessly into the rest of the picture, as if nothing had been there. Keep everything else unchanged."
            : "Replace the bright magenta area with \(prompt). Match the picture's lighting, perspective and style, and keep everything else unchanged."
    }

    /// The edit instruction for an extend (`image.extend`). Naming the edges matters: told only "around it", edit
    /// models often leave a side blank.
    public static func extendInstruction(left: Int, top: Int, right: Int, bottom: Int, prompt: String) -> String {
        let edges = [("left", left), ("right", right), ("top", top), ("bottom", bottom)].filter { $0.1 > 0 }.map(\.0)
        let named = edges.count > 1 ? edges.dropLast().joined(separator: ", ") + " and " + edges.last! : edges.first ?? "outer"
        return "Replace the bright magenta \(edges.count > 1 ? "areas" : "area") at the \(named) "
            + "\(edges.count > 1 ? "edges" : "edge") with a seamless continuation of the scene"
            + (prompt.isEmpty ? "" : " showing \(prompt)") + ". Keep everything else unchanged."
    }

    /// A prompt that asks for the marker's colour can't be checked for it (RunPlan.swift:421–422).
    public static func checkable(prompt: String) -> Bool {
        !["magenta", "pink", "fuchsia", "purple"].contains { prompt.lowercased().contains($0) }
    }

    /// The prompt as the route reads it: trimmed of surrounding whitespace and newlines (RunPlan.swift:284, 369).
    public static func normalized(prompt: String) -> String {
        prompt.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    // MARK: - Marker and guard (Media.swift:316–357)

    /// `image` with the mask's non-black area painted flat magenta: how an edit model is shown where to paint. Its
    /// instruction then says "replace the bright magenta area with …", the phrasing edit models follow reliably.
    public static let marker: [UInt8] = [255, 0, 255, 255]

    public static func mark(_ image: FillPixels, region: FillMask) -> FillPixels {
        var out = image
        for i in 0..<(image.width * image.height) where region.values[i] > 0 {
            for c in 0..<4 { out.rgba[i * 4 + c] = marker[c] }
        }
        return out
    }

    /// Why an edit model's painting can't be used, or nil, checking each of `parts` (extend: one per new side; fill:
    /// the mask) on its own, so a good side can't hide a bad one. Edit models fail at it in two ways (R1b): they keep
    /// the magenta marker, as is or as a shaded purple surface, or they leave the area dark.
    /// - `marker`: over a quarter of a part is purple-dominant (red and blue both well above green).
    /// - `dark` (`checkDark`, for extend): a part is under half as bright as the picture along its edge.
    public static func paintingProblem(_ result: FillPixels, canvas: FillPixels, parts: [FillMask], checkDark: Bool) -> String? {
        let (w, h) = (canvas.width, canvas.height)
        func luma(_ p: FillPixels, _ i: Int) -> Double {
            0.299 * Double(p.rgba[i * 4]) + 0.587 * Double(p.rgba[i * 4 + 1]) + 0.114 * Double(p.rgba[i * 4 + 2])
        }
        for part in parts {
            var inside = 0, purple = 0, brightness = 0.0
            for i in 0..<(w * h) where part.values[i] > 0 {
                inside += 1
                let (r, g, b) = (Int(result.rgba[i * 4]), Int(result.rgba[i * 4 + 1]), Int(result.rgba[i * 4 + 2]))
                if r - g > 40 && b - g > 40 { purple += 1 }
                brightness += luma(result, i)
            }
            guard inside > 0 else { continue }
            if Double(purple) / Double(inside) > 0.25 { return "marker" }
            guard checkDark else { continue }
            // The picture's own pixels along this part: the band just outside it.
            let outside = FillMask(width: w, height: h, values: part.values.map { $0 > 0 ? 0 : 255 })
            let band = seamBand(of: outside, width: max(8, max(w, h) / 48))
            var edge = 0.0, n = 0
            for i in 0..<(w * h) where band.values[i] > 0 && canvas.rgba[i * 4 + 3] > 0 { edge += luma(canvas, i); n += 1 }
            if n > 0, brightness / Double(inside) < 0.5 * (edge / Double(n)) { return "dark" }
        }
        return nil
    }

    // MARK: - Extend canvas (Media.swift:297–317, 358–373)

    /// The image on a larger canvas: the original at (`left`, `top`), the new area filled with the nearest edge pixels
    /// as a starting point, and a mask that is white exactly over the new area.
    public static func pad(_ image: FillPixels, left: Int, top: Int, right: Int, bottom: Int) -> (FillPixels, FillMask) {
        let w = image.width + left + right, h = image.height + top + bottom
        var canvas = FillPixels(width: w, height: h)
        var mask = FillMask(width: w, height: h)
        for y in 0..<h {
            let sy = min(max(y - top, 0), image.height - 1)
            for x in 0..<w {
                let sx = min(max(x - left, 0), image.width - 1)
                let inside = x >= left && x < left + image.width && y >= top && y < top + image.height
                let s = (sy * image.width + sx) * 4, d = (y * w + x) * 4
                for c in 0..<4 { canvas.rgba[d + c] = image.rgba[s + c] }
                if !inside { mask.values[y * w + x] = 255 }
            }
        }
        return (canvas, mask)
    }

    /// The new sides of a padded canvas, one mask each (left, right, top, bottom; corners go with left and right).
    public static func sides(width w: Int, height h: Int, left: Int, top: Int, right: Int, bottom: Int) -> [FillMask] {
        var masks: [FillMask] = []
        func side(_ inside: (Int, Int) -> Bool) {
            var m = FillMask(width: w, height: h)
            for y in 0..<h { for x in 0..<w where inside(x, y) { m.values[y * w + x] = 255 } }
            masks.append(m)
        }
        if left > 0 { side { x, _ in x < left } }
        if right > 0 { side { x, _ in x >= w - right } }
        if top > 0 { side { x, y in y < top && x >= left && x < w - right } }
        if bottom > 0 { side { x, y in y >= h - bottom && x >= left && x < w - right } }
        return masks
    }

    // MARK: - Harmonize (Media.swift:372–419)

    /// Makes a generative result meet the original without a seam, leaving the original untouched.
    ///
    /// An edit model re-renders the whole picture, so its painted region continues *its* version of the rest, not the
    /// original, which differs a little in tone (SenseNova: 8–11 MAE outside the mask, AB-R-0533). Compositing the
    /// original back then leaves a visible seam. This measures `original − result` outside the region, spreads it
    /// smoothly into the region (normalized convolution on a coarse grid: an average of the nearby differences that
    /// fades with distance), and adds it to the result. Where the model already matches the original, it changes nothing.
    public static func harmonize(_ result: FillPixels, toward original: FillPixels, region: FillMask) -> FillPixels {
        let (w, h) = (original.width, original.height)
        let cell = max(1, max(w, h) / 256)
        let (gw, gh) = ((w + cell - 1) / cell, (h + cell - 1) / cell)
        var diff = [Float](repeating: 0, count: gw * gh * 3), weight = [Float](repeating: 0, count: gw * gh)
        for y in 0..<h {
            for x in 0..<w where region.values[y * w + x] == 0 {
                let g = (y / cell) * gw + x / cell, i = (y * w + x) * 4
                weight[g] += 1
                for c in 0..<3 { diff[g * 3 + c] += Float(original.rgba[i + c]) - Float(result.rgba[i + c]) }
            }
        }
        let sigma = max(2, Float(max(gw, gh)) / 40)                     // ~2.5% of the long side
        blur(&weight, gw, gh, channels: 1, sigma: sigma)
        blur(&diff, gw, gh, channels: 3, sigma: sigma)
        let floor = Float(cell * cell) / 2                              // half a cell's support: fades beyond the seam
        var out = result
        for y in 0..<h {
            for x in 0..<w where region.values[y * w + x] > 0 {
                // Bilinear in the grid, at the pixel's centre.
                let gx = min(Float(gw - 1), max(0, (Float(x) + 0.5) / Float(cell) - 0.5))
                let gy = min(Float(gh - 1), max(0, (Float(y) + 0.5) / Float(cell) - 0.5))
                let (x0, y0) = (Int(gx), Int(gy)), (x1, y1) = (min(gw - 1, Int(gx) + 1), min(gh - 1, Int(gy) + 1))
                let (fx, fy) = (gx - Float(x0), gy - Float(y0))
                func at(_ a: [Float], _ gx: Int, _ gy: Int, _ c: Int, _ n: Int) -> Float { a[(gy * gw + gx) * n + c] }
                func lerp(_ a: [Float], _ c: Int, _ n: Int) -> Float {
                    let top = at(a, x0, y0, c, n) * (1 - fx) + at(a, x1, y0, c, n) * fx
                    let bottom = at(a, x0, y1, c, n) * (1 - fx) + at(a, x1, y1, c, n) * fx
                    return top * (1 - fy) + bottom * fy
                }
                let support = max(lerp(weight, 0, 1), floor)
                let i = (y * w + x) * 4
                for c in 0..<3 {
                    let v = Float(result.rgba[i + c]) + lerp(diff, c, 3) / support
                    out.rgba[i + c] = UInt8(min(Float(result.rgba[i + 3]), max(0, v.rounded())))   // premultiplied
                }
            }
        }
        return out
    }

    /// A separable Gaussian blur in place, over `channels` interleaved planes (Media.swift:421–447).
    public static func blur(_ a: inout [Float], _ w: Int, _ h: Int, channels n: Int, sigma: Float) {
        let r = Int((sigma * 3).rounded(.up))
        let raw = (-r...r).map { exp(-Float($0 * $0) / (2 * sigma * sigma)) }
        let total = raw.reduce(0, +)
        let kernel = raw.map { $0 / total }
        var tmp = a
        for y in 0..<h {
            for x in 0..<w {
                for c in 0..<n {
                    var s: Float = 0
                    for k in -r...r { let xx = min(w - 1, max(0, x + k)); s += a[(y * w + xx) * n + c] * kernel[k + r] }
                    tmp[(y * w + x) * n + c] = s
                }
            }
        }
        for y in 0..<h {
            for x in 0..<w {
                for c in 0..<n {
                    var s: Float = 0
                    for k in -r...r { let yy = min(h - 1, max(0, y + k)); s += tmp[(yy * w + x) * n + c] * kernel[k + r] }
                    a[(y * w + x) * n + c] = s
                }
            }
        }
    }

    /// The part of `region` within `width` pixels of the rest of the picture: the strip on the new side of a seam
    /// (Media.swift:448–467).
    public static func seamBand(of region: FillMask, width band: Int) -> FillMask {
        let (w, h) = (region.width, region.height)
        // Dilate the kept area (region == 0) by `band`, separably, with running counts; keep what lands in the region.
        var rows = [UInt8](repeating: 0, count: w * h)
        for y in 0..<h {
            var prefix = [Int](repeating: 0, count: w + 1)
            for x in 0..<w { prefix[x + 1] = prefix[x] + (region.values[y * w + x] == 0 ? 1 : 0) }
            for x in 0..<w where prefix[min(w, x + band + 1)] - prefix[max(0, x - band)] > 0 { rows[y * w + x] = 1 }
        }
        var out = FillMask(width: w, height: h)
        for x in 0..<w {
            var prefix = [Int](repeating: 0, count: h + 1)
            for y in 0..<h { prefix[y + 1] = prefix[y] + Int(rows[y * w + x]) }
            for y in 0..<h where region.values[y * w + x] > 0 && prefix[min(h, y + band + 1)] - prefix[max(0, y - band)] > 0 {
                out.values[y * w + x] = 255
            }
        }
        return out
    }

    /// The seam band's width for a canvas (RunPlan.swift:391; also the dark check’s, Media.swift:350).
    public static func seamBandWidth(width w: Int, height h: Int) -> Int { max(8, max(w, h) / 48) }

    // MARK: - Composite (Media.swift:255–269)

    /// The model's result over the original where the mask is white, blended where grey, and the original's exact
    /// pixels where black (Image §1.7: untouched pixels stay untouched). This is the feather: a soft selection blends.
    public static func composite(result: FillPixels, over original: FillPixels, mask: FillMask) -> FillPixels {
        let r = result.resized(original.width, original.height)
        var out = original
        for i in 0..<(original.width * original.height) {
            let m = Int(mask.values[i])
            guard m > 0 else { continue }
            for c in 0..<4 {
                let o = Int(original.rgba[i * 4 + c]), n = Int(r.rgba[i * 4 + c])
                out.rgba[i * 4 + c] = UInt8((n * m + o * (255 - m) + 127) / 255)
            }
        }
        return out
    }

    // MARK: - Geometry (Media.swift:482–532)

    /// The mask's non-black pixels' bounding box, or nil when it is all black.
    public static func bounds(of mask: FillMask) -> FillBox? {
        var (x0, y0, x1, y1) = (mask.width, mask.height, -1, -1)
        for y in 0..<mask.height {
            for x in 0..<mask.width where mask.values[y * mask.width + x] > 0 {
                x0 = min(x0, x); x1 = max(x1, x); y0 = min(y0, y); y1 = max(y1, y)
            }
        }
        return x1 < 0 ? nil : FillBox(x: x0, y: y0, width: x1 - x0 + 1, height: y1 - y0 + 1)
    }

    public static func crop(_ image: FillPixels, _ box: FillBox) -> FillPixels {
        var out = FillPixels(width: box.width, height: box.height)
        for row in 0..<box.height {
            let src = ((box.y + row) * image.width + box.x) * 4
            out.rgba.replaceSubrange(row * box.width * 4..<(row + 1) * box.width * 4, with: image.rgba[src..<src + box.width * 4])
        }
        return out
    }

    public static func crop(_ mask: FillMask, _ box: FillBox) -> FillMask {
        var out = FillMask(width: box.width, height: box.height)
        for row in 0..<box.height {
            let src = (box.y + row) * mask.width + box.x
            out.values.replaceSubrange(row * box.width..<(row + 1) * box.width, with: mask.values[src..<src + box.width])
        }
        return out
    }

    /// `image` with `patch` written over it at `box`'s origin.
    public static func paste(_ patch: FillPixels, into image: FillPixels, at box: FillBox) -> FillPixels {
        var out = image
        for row in 0..<patch.height {
            let dst = ((box.y + row) * image.width + box.x) * 4
            out.rgba.replaceSubrange(dst..<dst + patch.width * 4, with: patch.rgba[row * patch.width * 4..<(row + 1) * patch.width * 4])
        }
        return out
    }

    // MARK: - Size envelope (mlx-image-runtime ImageServerAPI/Models.swift:42–62, `SizeEnvelope`)

    /// Floors to the multiple, then checks the area. nil when outside the envelope — the route's `envelope.fit`.
    public static func fit(width: Int, height: Int, maxPixels: Int, multipleOf m: Int) -> (width: Int, height: Int)? {
        let w = (width / m) * m
        let h = (height / m) * m
        guard w >= m, h >= m, w * h <= maxPixels else { return nil }
        return (w, h)
    }

    /// The largest size with this aspect ratio that fits — the route's `envelope.clamp`, used when `fit` fails.
    public static func clamp(width: Int, height: Int, maxPixels: Int, multipleOf m: Int) -> (width: Int, height: Int) {
        if let fitted = fit(width: width, height: height, maxPixels: maxPixels, multipleOf: m) { return fitted }
        let scale = (Double(maxPixels) / Double(max(1, width * height))).squareRoot()
        let w = max(m, (Int(Double(width) * scale) / m) * m)
        let h = max(m, (Int(Double(height) * scale) / m) * m)
        return (w, h)
    }
}
