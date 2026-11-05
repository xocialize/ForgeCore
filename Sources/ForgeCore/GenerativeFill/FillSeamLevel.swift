//
//  FillSeamLevel.swift
//  ForgeCore
//
//  An expand's seam band, levelled to the photo's edge (AB-T-0450). New beside the route, not part of the bake-off's port.
//
//  The expand's LaMa seam band (`FillRoute.seamBand`, 18 px on a 900 × 760 canvas) bridges the model's re-rendering to the
//  photo, but on the real model (SenseNova 8step-v2, both tiers) it comes back with a low-frequency fault the eye reads as
//  the photo's rectangle even on a clear sky (AB-R-0736, AB-R-0739):
//  - a dark line about 5 luma deep in the middle of the band (125 at −6 px against 131 at −12 and 128 at −2), and
//  - the strip beyond the band sits off the photo's edge level: on an empty prompt −4.3 to −7.3 luma across the seam.
//  FillMatch deliberately gives an expand no level shift (its field averages rows the seam never sees), so nothing
//  else in the route takes either out.
//
//  What this does, per new side, on the side's low-frequency level only (each row parallel to the seam smoothed along
//  the seam, σ = `along`, with what departs from the row's own trend — a cloud, a branch — left out of the level):
//  - **The photo's edge, continued:** its first `fitRows` rows fitted with a level and a gradient (a sky darkens upward:
//    0.3 luma a row on the hand-test photo), which a continuation carries on into the strip. Lifting the strip to the
//    edge's level instead, flat, measured as a lighter halo ending 30–50 px out (AB-T-0450's second candidate).
//  - **Band:** the band's level is replaced by a straight run from the photo's continued edge to the level just beyond
//    the band. Whatever the band holds above or below its level (texture, a cloud's body) is kept.
//  - **Ramp** (`reachBands` > 1): the strip's step from the photo's continued edge is spread out beyond the band, fading
//    over the next (`reachBands` − 1) band widths, so the strip meets the photo on its gradient and returns to the
//    painting's own level further out. An offset only: the strip's own content keeps its shape.
//  Both corrections are smooth along the seam, so they move levels, never edges. Both fade out where the step, or the band's
//  departure from a straight run, is past `limit` (in luma; nothing is left at twice it): there the painting holds other
//  content against the photo — a cloud cut at the seam, a tree's crown — not a drift, and levelling it would smear that
//  content toward the photo's edge (measured: a dark smudge beside a cut cloud, up to 122 levels moved, AB-T-0450). The
//  photo's own pixels are never touched.
//

import Foundation

/// What `FillSeamLevel` did to one expand.
public struct FillSeamLevelReport: Sendable, Equatable, Codable {
    /// Per levelled side ("left", "right", "top", "bottom"): the mean luma step between the photo's edge and the strip just
    /// beyond the band (photo − strip), the mean gate (the share of the correction applied along the seam: 1 where the step
    /// is drift, 0 where it is content), and the largest luma change made to any pixel.
    public var sides: [String: [Double]]

    public init(sides: [String: [Double]]) { self.sides = sides }
}

public enum FillSeamLevel {

    public struct Settings: Sendable, Equatable, Codable {
        /// The smoothing along the seam, σ in px: the shortest stretch whose level is corrected on its own.
        public var along: Double
        /// Where the correction ends, in band widths from the photo: 1 = the band only, 3 = the band plus two widths of ramp.
        public var reachBands: Double
        /// The photo's rows its edge is fitted over: a level and a gradient, which the band continues.
        public var fitRows: Int
        /// The steepest gradient (luma per px) the band continues; a steeper fit is scaled down to it.
        public var maxSlope: Double
        /// The largest step or band departure (luma) corrected in full; the correction fades to nothing at twice it.
        public var limit: Double

        public init(along: Double = 24, reachBands: Double = 5, fitRows: Int = 16, maxSlope: Double = 0.5, limit: Double = 12) {
            precondition(along > 0 && reachBands >= 1 && fitRows >= 2 && maxSlope >= 0 && limit >= 0)
            self.along = along; self.reachBands = reachBands; self.fitRows = fitRows; self.maxSlope = maxSlope; self.limit = limit
        }

        /// The band only: its line out, the strip as painted.
        public static let band = Settings(reachBands: 1)
        /// The default: σ 24 px along the seam, the band and a ramp of four more band widths. Measured against σ 12 with two
        /// or four widths of ramp on the hand-test expand (AB-T-0450): the shorter smoothing followed the strip's own blotches,
        /// so the ramp read at 1:1 as a darker wavy band beside cut clouds and a wavy edge 30–50 px out on a clear sky; at 24
        /// neither shows, and the seam's line and step measure the same.
        public static let standard = Settings()
    }

    /// `image` (an expand's padded canvas, painted and bridged) with each new side's seam band levelled to `photo`'s edge.
    /// `band` is the route's seam band width.
    public static func levelled(_ image: FillPixels, photo: FillBox, band: Int,
                                settings: Settings = .standard) -> (FillPixels, FillSeamLevelReport) {
        let (w, h) = (image.width, image.height)
        let (x0, y0, x1, y1) = (photo.x, photo.y, photo.x + photo.width, photo.y + photo.height)
        guard band >= 1, photo.width > 0, photo.height > 0, x0 >= 0, y0 >= 0, x1 <= w, y1 <= h else {
            return (image, FillSeamLevelReport(sides: [:]))
        }
        let plane = (0..<3).map { c in (0..<(w * h)).map { Float(image.rgba[$0 * 4 + c]) } }

        // The four sides: span = along the seam, extent = how far the new strip reaches, index(u, d) = the pixel at u along
        // the seam and d px out (d ≥ 1 new side, d ≤ 0 the photo's row 1 − d).
        let sides: [Side] = [
            Side(name: "left", span: photo.height, extent: x0, depth: photo.width) { u, d in (y0 + u) * w + x0 - d },
            Side(name: "right", span: photo.height, extent: w - x1, depth: photo.width) { u, d in (y0 + u) * w + x1 - 1 + d },
            Side(name: "top", span: photo.width, extent: y0, depth: photo.height) { u, d in (y0 - d) * w + x0 + u },
            Side(name: "bottom", span: photo.width, extent: h - y1, depth: photo.height) { u, d in (y1 - 1 + d) * w + x0 + u },
        ].map { side in
            var s = side
            s.measure(plane, band: band, settings: settings)
            return s
        }
        guard sides.contains(where: { $0.reach > 0 }) else { return (image, FillSeamLevelReport(sides: [:])) }

        var out = image
        var largest = [Double](repeating: 0, count: 4)
        for y in 0..<h {
            for x in 0..<w {
                let dx = max(x0 - x, x - (x1 - 1), 0), dy = max(y0 - y, y - (y1 - 1), 0)
                guard dx > 0 || dy > 0 else { continue }
                // Which side(s) the pixel faces: the one it is straight out from, or in a corner both, blended by angle.
                let horizontal = x < x0 ? 0 : 1, vertical = y < y0 ? 2 : 3
                let (u, v) = (min(max(y - y0, 0), photo.height - 1), min(max(x - x0, 0), photo.width - 1))
                var delta: (Float, Float, Float) = (0, 0, 0)
                let weightH: Float = dy == 0 ? 1 : dx == 0 ? 0 : Float(dx * dx) / Float(dx * dx + dy * dy)
                let weightV: Float = 1 - weightH
                if weightH > 0, let d = sides[horizontal].delta(u, dx) {
                    delta.0 += weightH * d.0; delta.1 += weightH * d.1; delta.2 += weightH * d.2
                }
                if weightV > 0, let d = sides[vertical].delta(v, dy) {
                    delta.0 += weightV * d.0; delta.1 += weightV * d.1; delta.2 += weightV * d.2
                }
                guard delta.0 != 0 || delta.1 != 0 || delta.2 != 0 else { continue }
                let i = (y * w + x) * 4
                let alpha = Float(image.rgba[i + 3])
                // Rounded with a fixed per-pixel dither, so a smooth ramp of a few levels doesn't land on the 8-bit grid as
                // straight one-level contours along the seam (the very shape this takes out).
                var hash = UInt64(y * w + x) &* 0x9E37_79B9_7F4A_7C15
                hash ^= hash >> 29
                let dither = Float(hash & 0xFFFF) / 65536
                func put(_ c: Int, _ d: Float) {
                    out.rgba[i + c] = UInt8(min(alpha, max(0, (Float(image.rgba[i + c]) + d + dither).rounded(.down))))
                }
                put(0, delta.0); put(1, delta.1); put(2, delta.2)
                let changed = Double(abs(0.299 * delta.0 + 0.587 * delta.1 + 0.114 * delta.2))
                let side = dy >= dx ? vertical : horizontal
                largest[side] = max(largest[side], changed)
            }
        }
        var report: [String: [Double]] = [:]
        for (k, s) in sides.enumerated() where s.reach > 0 { report[s.name] = [s.meanStep, s.meanGate, largest[k]] }
        return (out, FillSeamLevelReport(sides: report))
    }

    // MARK: - One side

    struct Side {
        let name: String
        let span: Int
        let extent: Int
        /// How deep the photo is behind this side (its rows the edge is fitted over).
        let depth: Int
        let index: (Int, Int) -> Int
        /// How far out the correction reaches (0: this side isn't levelled).
        var reach = 0
        var band = 0
        /// Per u and channel (u * 3 + c): the photo's edge level and the target level just beyond the band.
        var edge: [Float] = [], beyond: [Float] = []
        /// The step spread over the ramp (photo − strip), per u and channel, already gated.
        var step: [Float] = []
        /// How much of the correction applies at u (1 drift, 0 content), smooth along the seam.
        var gate: [Float] = []
        /// The band's own level, per row d (1…band), u and channel.
        var level: [[Float]] = []
        var meanStep = 0.0, meanGate = 0.0

        init(name: String, span: Int, extent: Int, depth: Int, index: @escaping (Int, Int) -> Int) {
            self.name = name; self.span = span; self.extent = extent; self.depth = depth; self.index = index
        }

        /// Measure the levels. A side whose strip is too thin to have rows beyond the band to anchor to is left alone.
        mutating func measure(_ plane: [[Float]], band b: Int, settings: Settings) {
            let anchorRows = 6
            guard span >= 2, extent >= b + anchorRows, depth >= settings.fitRows else { return }
            band = b
            reach = max(b, min(Int((Double(b) * settings.reachBands).rounded()), extent - anchorRows))
            let kernel = FillSeamLevel.kernel(sigma: settings.along)
            // The mean of a set of rows, per u and channel, then its level along the seam.
            func rows(_ ds: [Int]) -> [Float] {
                var mean = [Float](repeating: 0, count: span * 3)
                for u in 0..<span {
                    for d in ds {
                        let i = index(u, d)
                        for c in 0..<3 { mean[u * 3 + c] += plane[c][i] }
                    }
                }
                let n = Float(ds.count)
                return FillSeamLevel.level(mean.map { $0 / n }, span: span, kernel: kernel)
            }
            // The photo's edge: its first `fitRows` rows (d = 0, −1, …), fitted with a straight line per u and channel — a
            // level at d = 0 and a gradient, which a continuation of the photo carries on into the strip (a sky darkens
            // upward; measured on the hand-test photo, 0.3 luma a row). Its slope is held to `maxSlope` in luma.
            let photoRows = (0..<settings.fitRows).map { rows([-$0]) }
            let meanD = -Float(settings.fitRows - 1) / 2
            let spreadD = (0..<settings.fitRows).map { (-Float($0) - meanD) * (-Float($0) - meanD) }.reduce(0, +)
            edge = [Float](repeating: 0, count: span * 3)
            var slope = [Float](repeating: 0, count: span * 3)
            for u in 0..<span {
                var mean: [Float] = [0, 0, 0], s: [Float] = [0, 0, 0]
                for c in 0..<3 {
                    mean[c] = photoRows.map { $0[u * 3 + c] }.reduce(0, +) / Float(settings.fitRows)
                    for (k, row) in photoRows.enumerated() { s[c] += (-Float(k) - meanD) * (row[u * 3 + c] - mean[c]) }
                    s[c] /= spreadD
                }
                let sl = abs(0.299 * s[0] + 0.587 * s[1] + 0.114 * s[2])
                let scale = sl > Float(settings.maxSlope) ? Float(settings.maxSlope) / sl : 1
                for c in 0..<3 {
                    slope[u * 3 + c] = s[c] * scale
                    edge[u * 3 + c] = mean[c] - slope[u * 3 + c] * meanD
                }
            }
            let after = rows(Array((b + 1)...(b + anchorRows)))
            let anchor = Float(b) + Float(anchorRows + 1) / 2                     // the centre of the rows `after` averages
            level = (1...b).map { rows([$0]) }
            // The step: the photo's edge continued out to the anchor, minus the strip there. Spread over the ramp beyond the
            // band, if there is one.
            step = (0..<(span * 3)).map { edge[$0] + slope[$0] * anchor - after[$0] }
            func luma(_ a: [Float], _ u: Int) -> Float { 0.299 * a[u * 3] + 0.587 * a[u * 3 + 1] + 0.114 * a[u * 3 + 2] }
            // The band's far end (d = band + 1): the strip's level there (its anchor level, carried back along the photo's
            // gradient), plus what of the step the ramp adds at d = band + 1.
            let first = ramp(b + 1)
            beyond = (0..<(span * 3)).map { after[$0] + slope[$0] * (Float(b + 1) - anchor) + step[$0] * first }
            // The gate: where the step, or the band's departure from a straight run between its two sides, is past
            // `limit`, the strip holds other content than the photo's edge there (a cloud the painting put against it, a
            // tree's crown), and levelling it would smear that content toward the photo: the correction fades out, to
            // nothing at twice the limit. A drift — a few levels of tone and a band's line — is well inside it.
            // The same fade where the photo's edge, or the strip just beyond the band, changes along the seam by more than
            // the limit within the smoothing's reach: a horizon or a trunk crossing the seam, whose level the smoothing
            // blends across it (measured: a dark smudge on both side strips where the hand-test photo's hills meet them).
            let limit = Float(settings.limit)
            let reachAlong = max(1, Int(settings.along.rounded()))
            // Judged on the rows' plain level (every pixel counted): the robust level leaves a narrow hill out as an outlier,
            // which is right for the level and wrong for the gate.
            func plain(_ ds: [Int]) -> [Float] {
                var mean = [Float](repeating: 0, count: span)
                for u in 0..<span {
                    for d in ds {
                        let i = index(u, d)
                        mean[u] += 0.299 * plane[0][i] + 0.587 * plane[1][i] + 0.114 * plane[2][i]
                    }
                }
                return FillSeamLevel.smooth(mean.map { $0 / Float(ds.count) }, span: span, kernel: kernel)
            }
            let edgeLuma = plain(Array((1 - settings.fitRows)...0)), afterLuma = plain(Array((b + 1)...(b + anchorRows)))
            func range(_ a: [Float], _ u: Int) -> Float {
                let window = a[max(0, u - reachAlong)...min(span - 1, u + reachAlong)]
                return (window.max() ?? 0) - (window.min() ?? 0)
            }
            var raw = [Float](repeating: 0, count: span)
            for u in 0..<span {
                var worst = max(abs(luma(step, u)), range(edgeLuma, u), range(afterLuma, u))
                for d in 1...b {
                    let t = Float(d) / Float(b + 1)
                    let chord = (0..<3).map { edge[u * 3 + $0] + (beyond[u * 3 + $0] - edge[u * 3 + $0]) * t - level[d - 1][u * 3 + $0] }
                    worst = max(worst, abs(0.299 * chord[0] + 0.587 * chord[1] + 0.114 * chord[2]))
                }
                raw[u] = limit <= 0 ? 0 : min(1, max(0, 2 - worst / limit))
            }
            // Closed over 2σ more on each side before it is smoothed, so the smoothing doesn't carry it back onto the content.
            let erode = 2 * reachAlong
            let eroded = (0..<span).map { raw[max(0, $0 - erode)...min(span - 1, $0 + erode)].min() ?? 0 }
            gate = FillSeamLevel.smooth(eroded, span: span, kernel: kernel)
            var total = 0.0
            for u in 0..<span { total += Double(luma(step, u)) }
            meanStep = total / Double(span)
            meanGate = Double(gate.reduce(0, +)) / Double(span)
        }

        /// The ramp's weight at `d` (1 at the band's edge, 0 from `reach` + 1 on); 0 everywhere for a band-only side.
        func ramp(_ d: Int) -> Float {
            guard reach > band, d > band else { return d > band ? 0 : 1 }
            guard d <= reach else { return 0 }
            return Float(0.5 * (1 + cos(Double.pi * Double(d - band) / Double(reach - band + 1))))
        }

        /// The correction at `u` along the seam, `d` px out; nil past the reach.
        func delta(_ u: Int, _ d: Int) -> (Float, Float, Float)? {
            guard reach > 0, d >= 1, d <= reach else { return nil }
            let i = u * 3, g = gate[u]
            guard g > 0 else { return nil }
            if d <= band {
                // A straight run from the photo's edge to the band's far end, in place of the band's own level.
                let t = Float(d) / Float(band + 1), row = level[d - 1]
                func at(_ c: Int) -> Float { g * (edge[i + c] + (beyond[i + c] - edge[i + c]) * t - row[i + c]) }
                return (at(0), at(1), at(2))
            }
            let r = ramp(d) * g
            return (step[i] * r, step[i + 1] * r, step[i + 2] * r)
        }
    }

    // MARK: - Levels along a seam

    /// `values` smoothed by `kernel` along the seam (normalized where it runs off either end).
    static func smooth(_ values: [Float], span n: Int, kernel: [Float]) -> [Float] {
        let r = kernel.count / 2
        return (0..<n).map { u in
            var (s, wsum): (Float, Float) = (0, 0)
            for k in max(0, u - r)...min(n - 1, u + r) { s += kernel[k - u + r] * values[k]; wsum += kernel[k - u + r] }
            return s / wsum
        }
    }

    static func kernel(sigma: Double) -> [Float] {
        let r = max(1, Int((sigma * 3).rounded(.up)))
        return (-r...r).map { Float(exp(-Double($0 * $0) / (2 * sigma * sigma))) }
    }

    /// A row's level along the seam (interleaved RGB per u): smoothed by `kernel`, twice — the second time without the
    /// pixels that depart from the first by more than 3 × their typical departure (at least 4 luma), so a cloud or a branch
    /// in the row doesn't count toward its level. Where every pixel nearby departs, the first pass's level stands.
    static func level(_ rgb: [Float], span n: Int, kernel: [Float]) -> [Float] {
        let r = kernel.count / 2
        func smooth(_ weights: [Float]) -> ([Float], [Float]) {
            var out = [Float](repeating: 0, count: n * 3), support = [Float](repeating: 0, count: n)
            for u in 0..<n {
                var s: (Float, Float, Float, Float) = (0, 0, 0, 0)
                for k in max(0, u - r)...min(n - 1, u + r) {
                    let wk = kernel[k - u + r] * weights[k]
                    s.0 += wk; s.1 += wk * rgb[k * 3]; s.2 += wk * rgb[k * 3 + 1]; s.3 += wk * rgb[k * 3 + 2]
                }
                support[u] = s.0
                if s.0 > 0 { out[u * 3] = s.1 / s.0; out[u * 3 + 1] = s.2 / s.0; out[u * 3 + 2] = s.3 / s.0 }
            }
            return (out, support)
        }
        let (first, _) = smooth([Float](repeating: 1, count: n))
        let departure = (0..<n).map { u in
            abs(0.299 * (rgb[u * 3] - first[u * 3]) + 0.587 * (rgb[u * 3 + 1] - first[u * 3 + 1])
                + 0.114 * (rgb[u * 3 + 2] - first[u * 3 + 2]))
        }
        let typical = 1.4826 * departure.sorted()[n / 2]
        let limit = max(4, 3 * typical)
        let (second, support) = smooth(departure.map { $0 <= limit ? 1 : 0 })
        let full = kernel.reduce(0, +)
        return (0..<(n * 3)).map { support[$0 / 3] > 0.05 * full ? second[$0] : first[$0] }
    }
}
