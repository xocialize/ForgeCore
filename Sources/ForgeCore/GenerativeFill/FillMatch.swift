//
//  FillMatch.swift
//  ForgeCore
//
//  Make a painting match the photo around it in level and grain (AB-T-0425). New beside the route, not part of the
//  bake-off's port (FillRoute.swift stays verbatim).
//
//  Measured on the real SenseNova (32 GB+ tier, 8step-v2-8bit; PixelAgencyEditor's GenerativeGrainProbe on the hand-test's
//  noisy 800 × 600 photo, grain = σ of luma minus its 5 × 5 mean, 9.3 in the photo):
//  - **Grain.** The route's resample to the render size and back keeps about 6.0 of 9.3 even at 0.96–0.99 scale (the
//    interpolation averages neighbouring pixels). The model repaints at about 3–4 at its render size, and that comes back at
//    2.5–3.5 at the window's size. So every generated area reads smooth beside the photo, whatever its size.
//  - **Level.** The model re-renders the whole picture a little off in tone (−9, −5 and +18 luma on three cases). The
//    harmonize corrects that only near the seam and fades it with distance, so an area's middle keeps the drift: an expand's
//    new sky stayed +5 to +10 lighter than the photo beside it.
//
//  What this does:
//  - `levelled`: the drift (original − painting) outside the area, spread smoothly over the whole picture at the area's
//    own scale (normalized convolution, σ = the area's thickness: how far its farthest pixel is from the photo), and added
//    to the painting everywhere. The harmonize
//    then runs as it always did, on what's left near the seam, so the two never add up.
//  - `grained`: the photo's own grain, transplanted. The high-pass (each channel minus its 5 × 5 mean) of flat tiles of
//    the photo around the area is copied, tile by tile, into the area, scaled by √(1 − (painting's / photo's)²) so a
//    painting that already has some grain gets only what it lacks. Copying the photo's residual keeps its spectrum, its
//    colour (monochrome grain stays monochrome) and its JPEG character, which a synthetic noise would not. A photo with no
//    grain gets none; a painting that already matches gets none.
//

import EraseFill
import Foundation

/// What `FillMatch` did to one painting: the route's trace carries it, and a host shows it with the layer's provenance.
public struct FillMatchReport: Sendable, Equatable, Codable {
    /// The level shift added inside the area, its mean per channel (R, G, B), in 8-bit levels: the model's drift taken out.
    public var level: [Double]
    /// The photo's grain around the area: the σ of its luma high-pass, trimmed so an edge doesn't count (`robustSigma`).
    public var grain: Double
    /// The painting's grain inside the area, the same measure, before and after.
    public var paintingGrain: Double
    public var resultGrain: Double

    public init(level: [Double], grain: Double, paintingGrain: Double, resultGrain: Double) {
        self.level = level; self.grain = grain; self.paintingGrain = paintingGrain; self.resultGrain = resultGrain
    }
}

public enum FillMatch {

    // MARK: - Level

    /// `painting` with the model's drift taken out: `original − painting`, measured where `region` is black, spread over the
    /// picture at the area's scale and added to every pixel. Returns the mean shift inside the area per channel.
    ///
    /// Where the model changed what's outside the area (it re-rendered clouds over a clear sky, say), that isn't drift, and
    /// counting it would shift the area by the clouds. So the field is made twice: the second time without the outside
    /// pixels that differ from the first field by more than 2.5 × their typical difference (at least 6 levels). Those pixels
    /// come back as the original's: outside the area the painting is never composited, only measured by the harmonize, which
    /// would otherwise carry the change into the area as a smudge along the seam.
    ///
    /// `shift` false keeps only that last part, for an expand: its new area is new content with no photo around it to match
    /// deep inside, and a field at its thickness (160 px for a 160 px strip) averages drift from rows the seam never sees;
    /// measured on the real model it lifted a new sky 5 luma over the photo's top rows. The harmonize's own correction, which
    /// is local to the seam, is the right one there.
    public static func levelled(_ painting: FillPixels, toward original: FillPixels, region: FillMask,
                                shift: Bool = true) -> (FillPixels, [Double]) {
        let (w, h) = (original.width, original.height)
        guard FillRoute.bounds(of: region) != nil, painting.width == w, painting.height == h else { return (painting, [0, 0, 0]) }
        let cell = max(1, max(w, h) / 128)
        let (gw, gh) = ((w + cell - 1) / cell, (h + cell - 1) / cell)
        func difference(_ i: Int, _ c: Int) -> Float { Float(original.rgba[i * 4 + c]) - Float(painting.rgba[i * 4 + c]) }
        func luma(_ i: Int) -> Float { 0.299 * difference(i, 0) + 0.587 * difference(i, 1) + 0.114 * difference(i, 2) }

        // The drift field over the grid from the outside pixels `counts` admits; nil when none is outside.
        var sigma: Float = 0
        func field(_ counts: (Int) -> Bool) -> [Float]? {
            var diff = [Float](repeating: 0, count: gw * gh * 3), weight = [Float](repeating: 0, count: gw * gh)
            for y in 0..<h {
                for x in 0..<w where region.values[y * w + x] == 0 && counts(y * w + x) {
                    let g = (y / cell) * gw + x / cell, i = y * w + x
                    weight[g] += 1
                    for c in 0..<3 { diff[g * 3 + c] += difference(i, c) }
                }
            }
            guard weight.contains(where: { $0 > 0 }) else { return nil }
            if sigma == 0 {
                // The area's own scale: how far its farthest pixel is from the photo (a 200 × 170 fill: 85 px; an expand's
                // 160 px top strip: 160 px). A wider σ averages drift from parts of the picture the area doesn't touch.
                sigma = max(2, Float(max(16, Self.thickness(weight, gw, gh) * cell)) / Float(cell))
            }
            FillRoute.blur(&weight, gw, gh, channels: 1, sigma: sigma)
            FillRoute.blur(&diff, gw, gh, channels: 3, sigma: sigma)
            // The average drift nearby, wherever there's support; none where there isn't (beyond ~5σ of any counted pixel).
            let peak = weight.max() ?? 0
            var out = [Float](repeating: 0, count: gw * gh * 3)
            for g in 0..<(gw * gh) where weight[g] > peak * 1e-6 {
                for c in 0..<3 { out[g * 3 + c] = diff[g * 3 + c] / weight[g] }
            }
            return out
        }
        // The field at a pixel, bilinear in the grid at its centre.
        func sample(_ f: [Float], _ x: Int, _ y: Int, _ c: Int) -> Float {
            let gx = min(Float(gw - 1), max(0, (Float(x) + 0.5) / Float(cell) - 0.5))
            let gy = min(Float(gh - 1), max(0, (Float(y) + 0.5) / Float(cell) - 0.5))
            let (x0, y0) = (Int(gx), Int(gy)), (x1, y1) = (min(gw - 1, Int(gx) + 1), min(gh - 1, Int(gy) + 1))
            let (fx, fy) = (gx - Float(x0), gy - Float(y0))
            let top = f[(y0 * gw + x0) * 3 + c] * (1 - fx) + f[(y0 * gw + x1) * 3 + c] * fx
            let bottom = f[(y1 * gw + x0) * 3 + c] * (1 - fx) + f[(y1 * gw + x1) * 3 + c] * fx
            return top * (1 - fy) + bottom * fy
        }
        guard let first = field({ _ in true }) else { return (painting, [0, 0, 0]) }    // the area is the whole picture
        // How far each outside pixel's difference is from the first field, in luma, on the difference's 5 × 5 mean over
        // outside pixels (so the photo's grain doesn't widen the limit and hide a soft cloud edge); what the model changed
        // stands out.
        var lumaDiff = [Float](repeating: 0, count: w * h), outside = [Float](repeating: 0, count: w * h)
        for i in 0..<(w * h) where region.values[i] == 0 { lumaDiff[i] = luma(i); outside[i] = 1 }
        let (smoothDiff, smoothCount) = (Self.boxSum(lumaDiff, w, h, radius: 2), Self.boxSum(outside, w, h, radius: 2))
        var residual = [Float](repeating: 0, count: w * h)
        var magnitudes: [Float] = []
        for y in 0..<h {
            for x in 0..<w where region.values[y * w + x] == 0 {
                let i = y * w + x
                let f = 0.299 * sample(first, x, y, 0) + 0.587 * sample(first, x, y, 1) + 0.114 * sample(first, x, y, 2)
                residual[i] = abs(smoothDiff[i] / max(smoothCount[i], 1) - f)
                if (x + y) % 3 == 0 { magnitudes.append(residual[i]) }
            }
        }
        magnitudes.sort()
        let limit = max(6, 2.5 * 1.4826 * magnitudes[magnitudes.count / 2])
        let drift = field({ residual[$0] <= limit }) ?? first

        var out = painting
        var sum = [Double](repeating: 0, count: 3), count = 0.0
        for y in 0..<h {
            for x in 0..<w {
                let i = (y * w + x) * 4, inside = region.values[y * w + x] > 0
                for c in 0..<3 {
                    let shift = shift ? sample(drift, x, y, c) : 0
                    out.rgba[i + c] = UInt8(min(Float(painting.rgba[i + 3]), max(0, (Float(painting.rgba[i + c]) + shift).rounded())))
                    if inside { sum[c] += Double(shift) }
                }
                if !inside && residual[y * w + x] > limit {
                    for c in 0..<4 { out.rgba[i + c] = original.rgba[i + c] }
                }
                if inside { count += 1 }
            }
        }
        return (out, sum.map { count > 0 ? $0 / count : 0 })
    }

    /// The sum of `a` over the (2r + 1)² square around each pixel, clipped at the edges (separable running sums).
    static func boxSum(_ a: [Float], _ w: Int, _ h: Int, radius r: Int) -> [Float] {
        var rows = [Float](repeating: 0, count: w * h), out = rows
        for y in 0..<h {
            var s: Float = 0
            for x in 0..<min(w, r) { s += a[y * w + x] }
            for x in 0..<w {
                if x + r < w { s += a[y * w + x + r] }
                if x - r - 1 >= 0 { s -= a[y * w + x - r - 1] }
                rows[y * w + x] = s
            }
        }
        for x in 0..<w {
            var s: Float = 0
            for y in 0..<min(h, r) { s += rows[y * w + x] }
            for y in 0..<h {
                if y + r < h { s += rows[(y + r) * w + x] }
                if y - r - 1 >= 0 { s -= rows[(y - r - 1) * w + x] }
                out[y * w + x] = s
            }
        }
        return out
    }

    /// The largest distance, in cells, from a cell with no outside pixel (weight 0) to one with some (8-connected steps).
    static func thickness(_ weight: [Float], _ gw: Int, _ gh: Int) -> Int {
        var distance = weight.map { $0 > 0 ? 0 : Int.max }
        var frontier = (0..<(gw * gh)).filter { distance[$0] == 0 }
        var d = 0
        while !frontier.isEmpty {
            var next: [Int] = []
            for g in frontier {
                let (x, y) = (g % gw, g / gw)
                for dy in -1...1 {
                    for dx in -1...1 where (dx != 0 || dy != 0) && x + dx >= 0 && x + dx < gw && y + dy >= 0 && y + dy < gh {
                        let n = (y + dy) * gw + x + dx
                        if distance[n] == Int.max { distance[n] = d + 1; next.append(n) }
                    }
                }
            }
            if !next.isEmpty { d += 1 }
            frontier = next
        }
        return max(1, d)
    }

    // MARK: - Grain

    /// The side of the tiles the grain is copied in.
    static let tile = 16

    /// `painting` with the photo's grain added inside `region` (where it is non-black), from flat tiles of `original` around
    /// it. `seed` picks the tiles, so a fill is reproducible from its seed. Pixels where `region` is black are untouched.
    public static func grained(_ painting: FillPixels, original: FillPixels, region: FillMask, seed: UInt64)
        -> (FillPixels, grain: Double, paintingGrain: Double, resultGrain: Double) {
        let (w, h) = (original.width, original.height)
        guard let bounds = FillRoute.bounds(of: region), painting.width == w, painting.height == h else { return (painting, 0, 0, 0) }
        // Work in the area's box grown by a context ring: the grain is the photo's near the area.
        let ring = max(64, max(bounds.width, bounds.height) / 2)
        let box = bounds.grown(by: ring, within: w, h)
        let photo = HighPass(original, box: box), paint = HighPass(painting, box: box)
        let (bw, bh) = (box.width, box.height)
        func inRegion(_ x: Int, _ y: Int) -> Bool { region.values[(box.y + y) * w + box.x + x] > 0 }

        // The photo's grain: robust σ of its luma high-pass outside the area (2 px clear of it).
        let near = Self.dilated(region, box: box, by: 2)
        var outside: [Float] = [], inside: [Float] = []
        for y in 0..<bh { for x in 0..<bw { if near[y * bw + x] == 0 { outside.append(photo.luma[y * bw + x]) } else if inRegion(x, y) { inside.append(paint.luma[y * bw + x]) } } }
        let grain = robustSigma(outside), paintingGrain = robustSigma(inside)
        guard grain >= 1, paintingGrain < grain * 0.95 else { return (painting, Double(grain), Double(paintingGrain), Double(paintingGrain)) }

        // Source tiles: wholly outside the area and flat, so no edge or texture comes along, on an 8 px stride.
        let t = tile
        var sources: [(x: Int, y: Int, luma: Float)] = []
        for sy in stride(from: 0, through: bh - t, by: 8) {
            for sx in stride(from: 0, through: bw - t, by: 8) {
                var clear = true, s2: Float = 0, mean: Float = 0, peak: Float = 0
                var (low, high) = (Float.greatestFiniteMagnitude, -Float.greatestFiniteMagnitude)
                for y in sy..<(sy + t) where clear {
                    for x in sx..<(sx + t) {
                        let i = y * bw + x
                        if near[i] != 0 { clear = false; break }
                        let v = photo.luma[i]; s2 += v * v; mean += photo.base[i]
                        peak = max(peak, abs(photo.rgb[i * 3]), abs(photo.rgb[i * 3 + 1]), abs(photo.rgb[i * 3 + 2]))
                        low = min(low, photo.base[i]); high = max(high, photo.base[i])
                    }
                }
                // Flat: no rougher than 1.3 × the grain overall; no channel of any pixel past 4 × (an edge between two
                // colours of one brightness, a yellow sun on a blue sky, hides from luma and would come along as a streak);
                // and the level under it steady (within 2 × the grain, at least 8).
                guard clear, (s2 / Float(t * t)).squareRoot() <= grain * 1.3, peak <= grain * 4,
                      high - low <= max(8, 2 * grain) else { continue }
                sources.append((sx, sy, mean / Float(t * t)))
            }
        }
        guard !sources.isEmpty else { return (painting, Double(grain), Double(paintingGrain), Double(paintingGrain)) }

        var rng = SplitMix(seed: seed ^ 0x6772_6169_6e00_0425)
        var out = painting
        for ty in stride(from: 0, to: bh, by: t) {
            for tx in stride(from: 0, to: bw, by: t) {
                let (x1, y1) = (min(bw, tx + t), min(bh, ty + t))
                // The painting's grain here (the tile and 8 px around it, inside the area), and its mean level.
                var local: [Float] = [], level: Float = 0, any = false
                for y in max(0, ty - 8)..<min(bh, y1 + 8) {
                    for x in max(0, tx - 8)..<min(bw, x1 + 8) where inRegion(x, y) {
                        local.append(paint.luma[y * bw + x])
                        if y >= ty && y < y1 && x >= tx && x < x1 { level += paint.base[y * bw + x]; any = true }
                    }
                }
                guard any else { continue }
                let here = local.count >= 64 ? robustSigma(local) : paintingGrain
                guard here < grain else { continue }
                let gain = (1 - (here / grain) * (here / grain)).squareRoot()
                // Of four random flat tiles, the one whose level is nearest the painting's here: grain varies with brightness.
                var count = 0
                for y in ty..<y1 { for x in tx..<x1 where inRegion(x, y) { count += 1 } }
                level /= Float(max(count, 1))
                var pick = sources[Int(rng.next() % UInt64(sources.count))]
                for _ in 0..<3 {
                    let other = sources[Int(rng.next() % UInt64(sources.count))]
                    if abs(other.luma - level) < abs(pick.luma - level) { pick = other }
                }
                for y in ty..<y1 {
                    for x in tx..<x1 where inRegion(x, y) {
                        let s = (pick.y + y - ty) * bw + pick.x + x - tx
                        let i = ((box.y + y) * w + box.x + x) * 4
                        let alpha = Float(out.rgba[i + 3])
                        for c in 0..<3 {
                            let v = Float(out.rgba[i + c]) + gain * photo.rgb[s * 3 + c]
                            out.rgba[i + c] = UInt8(min(alpha, max(0, v.rounded())))
                        }
                    }
                }
            }
        }
        let after = HighPass(out, box: box)
        var result: [Float] = []
        for y in 0..<bh { for x in 0..<bw where inRegion(x, y) && near[y * bw + x] != 0 { result.append(after.luma[y * bw + x]) } }
        return (out, Double(grain), Double(paintingGrain), Double(robustSigma(result)))
    }

    // MARK: - Pieces

    /// Each channel minus its 5 × 5 box mean (edges clamped), within `box`: the grain and the finest detail. Also luma's
    /// high-pass and luma's 5 × 5 mean (the level the grain sits on).
    struct HighPass {
        var rgb: [Float]
        var luma: [Float]
        var base: [Float]

        init(_ p: FillPixels, box: FillBox) {
            let (bw, bh) = (box.width, box.height)
            var planes = [[Float]](repeating: [Float](repeating: 0, count: bw * bh), count: 3)
            for y in 0..<bh {
                for x in 0..<bw {
                    let i = ((box.y + y) * p.width + box.x + x) * 4
                    for c in 0..<3 { planes[c][y * bw + x] = Float(p.rgba[i + c]) }
                }
            }
            func boxMean(_ a: [Float]) -> [Float] {
                var rows = [Float](repeating: 0, count: bw * bh), out = rows
                for y in 0..<bh { for x in 0..<bw { var s: Float = 0; for k in -2...2 { s += a[y * bw + min(bw - 1, max(0, x + k))] }; rows[y * bw + x] = s / 5 } }
                for y in 0..<bh { for x in 0..<bw { var s: Float = 0; for k in -2...2 { s += rows[min(bh - 1, max(0, y + k)) * bw + x] }; out[y * bw + x] = s / 5 } }
                return out
            }
            let means = planes.map(boxMean)
            rgb = [Float](repeating: 0, count: bw * bh * 3)
            luma = [Float](repeating: 0, count: bw * bh)
            base = luma
            for i in 0..<(bw * bh) {
                for c in 0..<3 { rgb[i * 3 + c] = planes[c][i] - means[c][i] }
                luma[i] = 0.299 * rgb[i * 3] + 0.587 * rgb[i * 3 + 1] + 0.114 * rgb[i * 3 + 2]
                base[i] = 0.299 * means[0][i] + 0.587 * means[1][i] + 0.114 * means[2][i]
            }
        }
    }

    /// The σ of `values` within ±3 × 1.4826 × their median absolute value: the grain's σ whatever its distribution (the
    /// median alone reads uniform grain 28 % high), blind to the few large values an edge leaves.
    static func robustSigma(_ values: [Float]) -> Float {
        guard !values.isEmpty else { return 0 }
        // A strided sample is plenty and keeps a 4K expand cheap.
        let stride = max(1, values.count / 65_536)
        let sample = Swift.stride(from: 0, to: values.count, by: stride).map { values[$0] }
        var magnitudes = sample.map(abs)
        magnitudes.sort()
        let limit = 3 * 1.4826 * magnitudes[magnitudes.count / 2]
        guard limit > 0 else { return 0 }
        var (n, s, s2): (Float, Float, Float) = (0, 0, 0)
        for v in sample where abs(v) <= limit { n += 1; s += v; s2 += v * v }
        let mean = s / n
        return max(0, s2 / n - mean * mean).squareRoot()
    }

    /// `region`'s non-black pixels within `box`, grown by `by` px (a square), as 0/1 at the box's size.
    static func dilated(_ region: FillMask, box: FillBox, by r: Int) -> [UInt8] {
        let (bw, bh) = (box.width, box.height)
        var rows = [UInt8](repeating: 0, count: bw * bh), out = rows
        for y in 0..<bh {
            for x in 0..<bw {
                for k in -r...r {
                    let xx = box.x + x + k
                    if xx >= 0, xx < region.width, region.values[(box.y + y) * region.width + xx] > 0 { rows[y * bw + x] = 1; break }
                }
            }
        }
        for y in 0..<bh { for x in 0..<bw { for k in -r...r where y + k >= 0 && y + k < bh && rows[(y + k) * bw + x] != 0 { out[y * bw + x] = 1; break } } }
        return out
    }

    /// A small, seedable generator (SplitMix64), so a fill's grain is reproducible from its seed.
    struct SplitMix {
        var state: UInt64
        init(seed: UInt64) { state = seed }
        mutating func next() -> UInt64 {
            state &+= 0x9E37_79B9_7F4A_7C15
            var z = state
            z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
            z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
            return z ^ (z >> 31)
        }
    }
}
