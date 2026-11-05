//
//  SeamLevelProbe.swift — AB-T-0450 measurement probe. Kept on the probe branch, not for main.
//
//  The hand-test expand (photo.jpg 800 × 600 → 900 × 760, bottom centre: 50, 160, 50, 0) through the route as the app runs
//  it (FillMatch on, guard, 18 px LaMa seam band), on a memory tier, with the seam band levelled (`FillSeamLevel`) or not:
//  - base:     today (seamLevel nil), live.
//  - live:     the provider's route (`EngineGenerativeFillProvider.route`, seamLevel .standard), live — the A/B pair for
//              the time (same seed, same painting).
//  - off:      base replayed on its own painting (LaMa live): must equal base byte for byte (parity, and LaMa's determinism).
//  - band:     .band (the band's line out, no ramp), replayed.
//  - standard: .standard (σ 24, band + 4 band widths of ramp: the default), replayed.
//  - sigma12:  σ 12 px along the seam and 2 band widths of ramp (the candidate before the default), replayed.
//
//  LEVEL_STORE=<models> LEVEL_IMAGE=<photo> LEVEL_OUT=<dir> LEVEL_GB=24|32 swift test -c release --skip-build \
//      --filter SeamLevelProbe/paint
//  LEVEL_ANALYZE=<dir> LEVEL_IMAGE=<photo> swift test -c release --skip-build --filter SeamLevelProbe/analyze
//

import Testing
import CoreGraphics
import CoreText
import Darwin
import Foundation
import ImageIO
import MLX
import MLXServeCore
import MLXToolKit
import os
import UniformTypeIdentifiers
@testable import ForgeCore

@MainActor
@Suite(.serialized)
struct SeamLevelProbe {
    nonisolated static let env = ProcessInfo.processInfo.environment
    nonisolated static let e = (left: 50, top: 160, right: 50, bottom: 0)
    nonisolated static let prompts: [(tag: String, text: String)] = [("clouds", "more sky with soft clouds"), ("empty", "")]
    nonisolated static let variants: [(name: String, settings: FillSeamLevel.Settings?)] = [
        ("off", nil), ("band", .band), ("standard", .standard),
        ("sigma12", FillSeamLevel.Settings(along: 12, reachBands: 3)),
    ]

    actor Log {
        var paints: [(seconds: Double, raw: FillPixels)] = []
        var inpaints: [Double] = []
        func paint(_ s: Double, _ raw: FillPixels) { paints.append((s, raw)) }
        func inpaint(_ s: Double) { inpaints.append(s) }
        func reset() { paints = []; inpaints = [] }
    }

    @Test(.enabled(if: env["LEVEL_OUT"] != nil && env["LEVEL_STORE"] != nil))
    func paint() async throws {
        let imagePath = try #require(Self.env["LEVEL_IMAGE"])
        let photo = FillPixels(try #require(Self.load(URL(fileURLWithPath: imagePath))))
        let out = URL(fileURLWithPath: Self.env["LEVEL_OUT"]!)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)
        let gb = Int(Self.env["LEVEL_GB"] ?? "32") ?? 32
        let seeds = (Self.env["LEVEL_SEEDS"] ?? "2026,7,4242").split(separator: ",").compactMap { UInt64($0) }
        let tier: GenerativeFillTier = gb == 16 ? .senseNova16GB : gb == 24 ? .senseNova24GB : .senseNova32GB
        // As the editor's ForgeModule: an emulated Mac gets 74 % of its memory as the engine's budget; 32 = this Mac.
        let governor = gb == 32 ? nil : MemoryGovernor(budgetBytes: UInt64(Double(UInt64(gb) << 30) * 0.74))
        let service = ForgeEngineService(governor: governor, generativeFillTier: tier)
        await service.attachModelStore(root: URL(fileURLWithPath: Self.env["LEVEL_STORE"]!))
        await service.registerAll()
        let fillID = try #require(service.packageID(for: .generativeFill))
        let lamaID = try #require(service.packageID(for: .inpaint))
        let engine = service.engine
        print("level-probe: tier \(tier.label) \(tier.model.displayName), budget \(await service.memoryBudgetBytes)")
        let log = Log()

        let live: GenerativeFillRoute.Painter = { marked, instruction, seed, width, height in
            let request = IEditRequest(images: [try EngineImageCoding.pngImage(from: marked.cgImage)], prompt: instruction,
                                       width: width, height: height, seed: seed)
            let started = Date()
            let response = try await engine.run(request, package: fillID)
            let raw = FillPixels(try EngineImageCoding.cgImage(from: try #require((response as? IEditResponse)?.image)))
            await log.paint(Date().timeIntervalSince(started), raw)
            return raw
        }
        func replay(_ raws: [FillPixels]) -> GenerativeFillRoute.Painter {
            let box = OSAllocatedUnfairLock(initialState: 0)
            return { _, _, _, _, _ in let k = box.withLock { v in defer { v += 1 }; return v }; return raws[min(k, raws.count - 1)] }
        }
        let inpaint: GenerativeFillRoute.Inpainter = { canvas, hole in
            let request = InpaintRequest(image: try EngineImageCoding.pngImage(from: canvas.cgImage),
                                         mask: try EngineImageCoding.pngImage(from: hole.cgImage), mode: InpaintContract.best)
            let started = Date()
            let response = try await engine.run(request, package: lamaID)
            await log.inpaint(Date().timeIntervalSince(started))
            return FillPixels(try EngineImageCoding.cgImage(from: try #require((response as? InpaintResponse)?.image)))
        }
        let base = tier.route
        print("level-probe: tier route seamLevel = \(String(describing: base.seamLevel)), matchesSurroundings \(base.matchesSurroundings)")
        func route(_ s: FillSeamLevel.Settings?) -> GenerativeFillRoute { var r = base; r.seamLevel = s; return r }
        let (canvas, _) = FillRoute.pad(photo, left: Self.e.left, top: Self.e.top, right: Self.e.right, bottom: Self.e.bottom)

        struct Run { var seconds: Double; var physPeak: UInt64; var physBefore: UInt64; var mlxPeak: Int }
        func expand(_ r: GenerativeFillRoute, prompt: String, seed: UInt64, paint: GenerativeFillRoute.Painter) async throws
            -> (FillPixels, GenerativeFillTrace, Run) {
            await log.reset()
            GPU.resetPeakMemory()
            let before = Self.phys(), sampler = Sampler(), started = Date()
            let (_, patch, trace) = try await r.expand(photo, left: Self.e.left, top: Self.e.top, right: Self.e.right,
                                                       bottom: Self.e.bottom, prompt: prompt, seed: seed, paint: paint, inpaint: inpaint)
            let s = Date().timeIntervalSince(started)
            return (patch.applied(to: canvas), trace, Run(seconds: s, physPeak: sampler.stop(), physBefore: before, mlxPeak: Memory.peakMemory))
        }

        // Warm-up: loads both models.
        _ = try await expand(route(nil), prompt: "", seed: 1, paint: live)

        let g = { (b: UInt64) in String(format: "%.2f", Double(b) / 1e9) }
        var lines = ["variant\tprompt\tseed\tlive\ttotal_s\tpaint_s\tlama_n\tlama_s\tpost_s\tphys_before_GB\tphys_peak_GB\tmlx_peak_GB\tretries\tidentical_to_base\tlevel_top\tlevel_left\tlevel_right"]
        func record(_ v: String, _ p: String, _ seed: UInt64, live: Bool, _ t: GenerativeFillTrace, _ run: Run, identical: String) async {
            let paints = await log.paints.map(\.seconds), lamas = await log.inpaints
            let paintS = live ? paints.reduce(0, +) : 0
            func side(_ n: String) -> String { t.seamLevel?.sides[n].map { $0.map { String(format: "%.2f", $0) }.joined(separator: "/") } ?? "-" }
            lines.append([v, p, "\(seed)", "\(live)", String(format: "%.3f", run.seconds), String(format: "%.3f", paintS),
                          "\(lamas.count)", String(format: "%.3f", lamas.reduce(0, +)), String(format: "%.3f", run.seconds - paintS),
                          g(run.physBefore), g(run.physPeak), g(UInt64(max(0, run.mlxPeak))), "\(t.retries)", identical,
                          side("top"), side("left"), side("right")].joined(separator: "\t"))
            print(lines.last!)
            try? lines.joined(separator: "\n").write(to: out.appendingPathComponent("runs.tsv"), atomically: true, encoding: .utf8)
        }
        func save(_ p: FillPixels, _ name: String) throws { try Self.writePNG(p.cgImage, to: out.appendingPathComponent(name)) }

        var cpu = ["prompt\tseed\tlevel_ms_runs"]
        for p in Self.prompts {
            for seed in seeds {
                // base, live.
                let (b, bt, br) = try await expand(route(nil), prompt: p.text, seed: seed, paint: live)
                let raws = await log.paints.map(\.raw)
                await record("base", p.tag, seed, live: true, bt, br, identical: "-")
                try save(b, "base-\(p.tag)-\(seed).png")
                for (k, raw) in raws.enumerated() { try save(raw, "raw\(k)-\(p.tag)-\(seed).png") }
                // standard, live: the A/B pair with base.
                let app = EngineGenerativeFillProvider.route(for: tier, matchesSurroundings: true)
                let (l, lt, lr) = try await expand(app, prompt: p.text, seed: seed, paint: live)
                let repainted = await log.paints.map(\.raw)
                await record("live", p.tag, seed, live: true, lt, lr, identical: repainted == raws ? "paint" : "PAINT-DIFFERS")
                try save(l, "live-\(p.tag)-\(seed).png")
                // The variants, replayed on base's painting(s).
                var standard: FillPixels?
                for v in Self.variants {
                    let (img, t, run) = try await expand(route(v.settings), prompt: p.text, seed: seed, paint: replay(raws))
                    await record(v.name, p.tag, seed, live: false, t, run, identical: img == b ? "yes" : "no")
                    try save(img, "\(v.name)-\(p.tag)-\(seed).png")
                    if v.name == "standard" { standard = img }
                }
                if let standard { print("level-probe: live == replayed standard: \(standard == l)") }
                // The correction's own CPU time, on this canvas (the route calls it once per expand).
                let band = FillRoute.seamBandWidth(width: canvas.width, height: canvas.height)
                let photoBox = FillBox(x: Self.e.left, y: Self.e.top, width: photo.width, height: photo.height)
                var ms: [Double] = []
                for _ in 0..<7 {
                    let t0 = Date()
                    _ = FillSeamLevel.levelled(b, photo: photoBox, band: band, settings: .standard)
                    ms.append(Date().timeIntervalSince(t0) * 1000)
                }
                cpu.append("\(p.tag)\t\(seed)\t" + ms.map { String(format: "%.1f", $0) }.joined(separator: ","))
                try cpu.joined(separator: "\n").write(to: out.appendingPathComponent("cpu.tsv"), atomically: true, encoding: .utf8)
            }
        }
        _ = service
    }

    // MARK: - Analyze

    nonisolated static let analyzed = ["base", "live", "off", "band", "standard", "sigma12"]

    @Test(.enabled(if: env["LEVEL_ANALYZE"] != nil))
    func analyze() throws {
        let dir = URL(fileURLWithPath: Self.env["LEVEL_ANALYZE"]!)
        let imagePath = try #require(Self.env["LEVEL_IMAGE"])
        let photo = FillPixels(try #require(Self.load(URL(fileURLWithPath: imagePath))))
        let (canvas, mask) = FillRoute.pad(photo, left: Self.e.left, top: Self.e.top, right: Self.e.right, bottom: Self.e.bottom)
        let (w, h) = (canvas.width, canvas.height)
        let o = (x0: Self.e.left, y0: Self.e.top, x1: w - Self.e.right, y1: h - Self.e.bottom)
        let photoSky = Self.photoSkyRows(canvas, o)
        let names = try FileManager.default.contentsOfDirectory(atPath: dir.path).filter { n in
            Self.analyzed.contains { n.hasPrefix("\($0)-") } && n.hasSuffix(".png") }.sorted()
        let header = ["case", "variant", "prompt", "seed", "top_step", "top_gstep", "natural_step", "top_line", "top_line_at", "strip_line", "strip_line_at",
                      "photo_line", "prof_m12", "prof_m6", "prof_m2", "prof_p6", "L_step", "L_line", "R_step", "R_line",
                      "top_cut_px", "seams_cut", "cloud_pct", "extrap", "straight_seam", "straight_far", "straight_far_at",
                      "grain_new", "grain_photo", "changed_px", "max_change"]
        var tsv = [header.joined(separator: "\t")]
        let ds = Array(-60 ... -1) + Array(1...24)
        var profiles = ["case\t" + ds.map { "\($0)" }.joined(separator: "\t")]
        var text = ["case                  step  gstep(nat)  line(at)    strip(at)   photo  -12/-6/-2/+6          L step/line  R step/line  cut  cloud%  straight s/f(at)    grain  changed max"]
        for name in names {
            let img = FillPixels(try #require(Self.load(dir.appendingPathComponent(name))))
            let parts = name.dropLast(4).split(separator: "-").map(String.init)
            let a = Analysis(img, o: o, photoSkyRows: photoSky)
            let s = Straight(img, o: o)
            var changed = 0, maxChange = 0, maxAt = 0
            if parts[0] != "base", let bi = Self.load(dir.appendingPathComponent("base-\(parts[1])-\(parts[2]).png")) {
                let b = FillPixels(bi)
                // The change, as a picture: luma (after − base) × 8 about mid-grey, so a person can see where it went.
                var map = FillPixels(width: w, height: h)
                for i in 0..<(w * h) {
                    let dl = Analysis.luma(Analysis.px(img, i % w, i / w)) - Analysis.luma(Analysis.px(b, i % w, i / w))
                    let v = UInt8(min(255, max(0, 128 + 8 * dl)))
                    map.rgba[i * 4] = v; map.rgba[i * 4 + 1] = v; map.rgba[i * 4 + 2] = v; map.rgba[i * 4 + 3] = 255
                    guard mask.values[i] > 0 else { continue }
                    let dm = (0..<3).map { abs(Int(b.rgba[i * 4 + $0]) - Int(img.rgba[i * 4 + $0])) }.max()!
                    if dm > 0 { changed += 1; if dm > maxChange { maxChange = dm; maxAt = i } }
                }
                if parts[0] == "standard" { try Self.writePNG(map.cgImage, to: dir.appendingPathComponent("change-\(parts[1])-\(parts[2]).png")) }
                print("max change \(name): \(maxChange) at x \(maxAt % w) y \(maxAt / w)")
            }
            let seamsCut = [a.top.cutPx, a.left.cutPx, a.right.cutPx].filter { $0 > 0 }.count
            profiles.append(String(name.dropLast(4)) + "\t" + ds.map { String(format: "%.2f", a.top.prof($0)) }.joined(separator: "\t"))
            let f = { (v: Double) in String(format: "%.2f", v) }
            tsv.append([name, parts[0], parts[1], parts[2], f(a.top.step), f(a.top.gradientStep), f(a.top.natural), f(a.top.line), "\(a.top.lineAt)", f(a.top.stripLine),
                        "\(a.top.stripLineAt)", f(a.top.photoLine), f(a.top.prof(-12)), f(a.top.prof(-6)), f(a.top.prof(-2)),
                        f(a.top.prof(6)), f(a.left.step), f(a.left.line), f(a.right.step), f(a.right.line),
                        "\(a.top.cutPx)", "\(seamsCut)", f(a.cloudPct), f(a.extrap), String(format: "%.3f", s.seam),
                        String(format: "%.3f", s.far), "\(s.farAt)", f(a.grainNew), f(a.grainPhoto), "\(changed)", "\(maxChange)"]
                .joined(separator: "\t"))
            text.append(String(name.dropLast(4)).padding(toLength: 21, withPad: " ", startingAt: 0)
                + String(format: "%+5.1f  %+5.1f(%+4.1f)  %+5.1f(%3d)  %+5.1f(%3d)  %+4.1f  %5.1f/%5.1f/%5.1f/%5.1f  %+5.1f/%+4.1f  %+5.1f/%+4.1f  %3d  %5.1f   %.3f/%.3f(%3d)  %.1f/%.1f  %6d %2d",
                         a.top.step, a.top.gradientStep, a.top.natural, a.top.line, a.top.lineAt, a.top.stripLine, a.top.stripLineAt, a.top.photoLine,
                         a.top.prof(-12), a.top.prof(-6), a.top.prof(-2), a.top.prof(6),
                         a.left.step, a.left.line, a.right.step, a.right.line, a.top.cutPx, a.cloudPct,
                         s.seam, s.far, s.farAt, a.grainNew, a.grainPhoto, changed, maxChange))
        }
        try tsv.joined(separator: "\n").write(to: dir.appendingPathComponent("analysis.tsv"), atomically: true, encoding: .utf8)
        try profiles.joined(separator: "\n").write(to: dir.appendingPathComponent("profiles.tsv"), atomically: true, encoding: .utf8)
        try text.joined(separator: "\n").write(to: dir.appendingPathComponent("analysis.txt"), atomically: true, encoding: .utf8)
        print(text.joined(separator: "\n"))

        // Sheets: per prompt, rows = seeds, columns = variants. The top-left, top-centre and top-right seams at 1:1, and the
        // same with the luma contrast ×4 about the photo's top rows (a diagnostic: it shows a line a person might miss).
        let columns = (Self.env["LEVEL_SHEET"] ?? "base,band,sigma12,standard").split(separator: ",").map(String.init)
        let crops: [(String, CGRect)] = [("topleft", CGRect(x: 0, y: 40, width: 470, height: 260)),
                                         ("topcentre", CGRect(x: 215, y: 40, width: 470, height: 260)),
                                         ("topright", CGRect(x: 430, y: 40, width: 470, height: 260)),
                                         ("lefthorizon", CGRect(x: 0, y: 420, width: 300, height: 240)),
                                         ("righthorizon", CGRect(x: 600, y: 420, width: 300, height: 240))]
        for p in Self.prompts {
            let seeds = Array(Set(names.filter { $0.contains("-\(p.tag)-") }.map { $0.dropLast(4).split(separator: "-")[2].description })).sorted()
            guard !seeds.isEmpty else { continue }
            for (cropName, rect) in crops {
                for amplified in [false, true] {
                    var rows: [[(String, CGImage?)]] = []
                    for seed in seeds {
                        var row: [(String, CGImage?)] = []
                        for v in columns {
                            guard let img = Self.load(dir.appendingPathComponent("\(v)-\(p.tag)-\(seed).png")) else { continue }
                            let shown = amplified ? Self.amplified(FillPixels(img), about: o.y0).cgImage : img
                            row.append(("\(v) · seed \(seed)", shown.cropping(to: rect)))
                        }
                        rows.append(row)
                    }
                    let file = "sheet-\(p.tag)-\(cropName)-1to1\(amplified ? "-x4" : "").png"
                    try Self.writePNG(try Self.sheet(rows, scale: 1), to: dir.appendingPathComponent(file))
                }
            }
            var whole: [[(String, CGImage?)]] = []
            for seed in seeds {
                whole.append(columns.compactMap { v in
                    Self.load(dir.appendingPathComponent("\(v)-\(p.tag)-\(seed).png")).map { ("\(v) · seed \(seed)", $0) } })
            }
            try Self.writePNG(try Self.sheet(whole, scale: 0.5), to: dir.appendingPathComponent("sheet-\(p.tag).png"))
        }
    }

    /// Luma contrast ×4 about each column's photo level just below the top seam (rows y0 + 1…8), chroma kept.
    nonisolated static func amplified(_ p: FillPixels, about y0: Int) -> FillPixels {
        var out = p
        let w = p.width
        for x in 0..<w {
            var ref = 0.0
            for y in (y0 + 1)...(y0 + 8) { ref += Analysis.luma(Analysis.px(p, x, y)) }
            ref /= 8
            for y in 0..<p.height {
                let c = Analysis.px(p, x, y), l = Analysis.luma(c), shift = 3 * (l - ref)
                let i = (y * w + x) * 4
                for k in 0..<3 { out.rgba[i + k] = UInt8(min(255, max(0, (c[k] + shift).rounded()))) }
            }
        }
        return out
    }

    /// A straight edge parallel to the top seam (AB-T-0445's measure): per offset d (1…150 px into the strip), the share of
    /// the photo's columns whose 5 × 5-mean luma steps by more than 6 levels between d − 2 and d + 2. `seam` = the largest
    /// share at d ≤ 24, `far` the largest at 25…150 and where.
    struct Straight {
        var seam = 0.0, far = 0.0, farAt = 0
        init(_ img: FillPixels, o: (x0: Int, y0: Int, x1: Int, y1: Int)) {
            let (w, h) = (img.width, img.height)
            let lum = (0..<(w * h)).map { i in 0.299 * Float(img.rgba[i * 4]) + 0.587 * Float(img.rgba[i * 4 + 1]) + 0.114 * Float(img.rgba[i * 4 + 2]) }
            let lp = FillMatch.boxSum(lum, w, h, radius: 2).map { $0 / 25 }
            let cols = Array((o.x0 + 8)..<(o.x1 - 8))
            for d in 1...150 {
                let y = o.y0 - d
                guard y - 2 >= 0, y + 2 < h else { continue }
                let share = Double(cols.filter { abs(lp[(y - 2) * w + $0] - lp[(y + 2) * w + $0]) > 6 }.count) / Double(cols.count)
                if d <= 24 { seam = max(seam, share) } else if share > far { far = share; farAt = d }
            }
        }
    }

    // MARK: - The AB-R-0736 analysis, with the seam's line

    nonisolated static func photoSkyRows(_ c: FillPixels, _ o: (x0: Int, y0: Int, x1: Int, y1: Int)) -> (left: Set<Int>, right: Set<Int>) {
        func sky(_ x: Int, _ y: Int) -> Bool { let p = Analysis.px(c, x, y); return p[2] > p[1] + 10 && p[2] > p[0] + 10 }
        var l = Set<Int>(), r = Set<Int>()
        for y in o.y0..<o.y1 {
            if (0..<16).allSatisfy({ sky(o.x0 + 2 + $0, y) }) { l.insert(y) }
            if (0..<16).allSatisfy({ sky(o.x1 - 3 - $0, y) }) { r.insert(y) }
        }
        return (l, r)
    }

    struct Seam {
        var step = Double.nan, cutPx = 0, cutRuns = 0
        /// The mean luma profile across the seam over cloud-free positions, d = −100…+24 (d < 0 the new side, d > 0 the photo).
        var profile: [Int: Double] = [:]
        /// The deepest line: the largest |(prof(d − 4) + prof(d + 4)) / 2 − prof(d)| over d, signed (+ = darker than both
        /// sides). `line` in the band and its edge (d −24…−5), `stripLine` further out (−96…−25), `photoLine` the photo's own
        /// rows (+5…+20), the floor a seam is judged against.
        var line = Double.nan, lineAt = 0, stripLine = Double.nan, stripLineAt = 0, photoLine = Double.nan
        /// The step against the photo's own gradient: the new side's mean at −12…−4 minus the photo's rows +1…+16 fitted with
        /// a line and continued to −12…−4. `natural` is what `step` (AB-R-0736's −12…−4 against +4…+12) reads for a strip
        /// that continues that line exactly: the floor of `step` on a graded sky.
        var gradientStep = Double.nan, natural = Double.nan
        func prof(_ d: Int) -> Double { profile[d] ?? .nan }
    }

    struct Analysis {
        var top = Seam(), left = Seam(), right = Seam()
        var extrap = Double.nan, cloudPct = 0.0, grainNew = Double.nan, grainPhoto = Double.nan

        static func px(_ p: FillPixels, _ x: Int, _ y: Int) -> [Double] {
            let i = (y * p.width + x) * 4
            return [Double(p.rgba[i]), Double(p.rgba[i + 1]), Double(p.rgba[i + 2])]
        }
        static func luma(_ c: [Double]) -> Double { 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] }

        init(_ img: FillPixels, o: (x0: Int, y0: Int, x1: Int, y1: Int), photoSkyRows: (left: Set<Int>, right: Set<Int>)) {
            let (w, h) = (img.width, img.height)
            func ref(_ x: Int, _ y: Int) -> Double {
                if y < o.y0 {
                    let cx = min(max(x, o.x0), o.x1 - 1)
                    return (4..<12).map { Self.luma(Self.px(img, cx, o.y0 + $0)) }.reduce(0, +) / 8
                }
                let cx = x < o.x0 ? o.x0 + 4 : o.x1 - 12
                return (0..<8).map { Self.luma(Self.px(img, cx + $0, y)) }.reduce(0, +) / 8
            }
            var raw = [Bool](repeating: false, count: w * h)
            var newCount = 0
            for y in 0..<h {
                for x in 0..<w where !(x >= o.x0 && x < o.x1 && y >= o.y0) {
                    let c = Self.px(img, x, y)
                    newCount += 1
                    if c[2] >= c[1] - 10 && Self.luma(c) > ref(x, y) + 18 && c.max()! - c.min()! < 95 && c.min()! > 150 { raw[y * w + x] = true }
                }
            }
            var cloud = [Bool](repeating: false, count: w * h), cloudCount = 0
            for y in 0..<h {
                for x in 0..<w where raw[y * w + x] {
                    var n = 0
                    for dy in -2...2 { for dx in -2...2 { let (xx, yy) = (x + dx, y + dy); if xx >= 0, xx < w, yy >= 0, yy < h, raw[yy * w + xx] { n += 1 } } }
                    if n >= 18 { cloud[y * w + x] = true; cloudCount += 1 }
                }
            }
            cloudPct = 100 * Double(cloudCount) / Double(max(1, newCount))
            func isCloud(_ x: Int, _ y: Int) -> Bool { x >= 0 && x < w && y >= 0 && y < h && cloud[y * w + x] }
            func seam(_ positions: [Int], reach: Int, _ at: (Int, Int) -> (Int, Int)) -> Seam {
                var s = Seam(), steps: [Double] = [], prev = false
                var sums: [Int: Double] = [:], n = 0
                for p in positions {
                    let cut = (1...24).contains { let (x, y) = at(p, -$0); return isCloud(x, y) }
                    if cut { s.cutPx += 1; if !prev { s.cutRuns += 1 } }
                    prev = cut
                    guard !(1...40).contains(where: { let (x, y) = at(p, -$0); return isCloud(x, y) }) else { continue }
                    func lum(_ d: Int) -> Double { let (x, y) = at(p, d); return Self.luma(Self.px(img, x, y)) }
                    func mean(_ r: ClosedRange<Int>) -> Double { r.map(lum).reduce(0, +) / Double(r.count) }
                    steps.append(mean(-12 ... -4) - mean(4...12))
                    n += 1
                    for d in -reach...24 where d != 0 { sums[d, default: 0] += lum(d) }
                }
                if !steps.isEmpty { s.step = steps.reduce(0, +) / Double(steps.count) }
                guard n > 0 else { return s }
                s.profile = sums.mapValues { $0 / Double(n) }
                func dip(_ d: Int) -> Double { (s.prof(d - 4) + s.prof(d + 4)) / 2 - s.prof(d) }
                func deepest(_ r: ClosedRange<Int>) -> (Double, Int) {
                    r.map { (dip($0), $0) }.filter { !$0.0.isNaN }.max { abs($0.0) < abs($1.0) } ?? (.nan, 0)
                }
                (s.line, s.lineAt) = deepest(-24 ... -5)
                let fitD = Array(1...16).map(Double.init), fitL = (1...16).map { s.prof($0) }
                let md = fitD.reduce(0, +) / 16, ml = fitL.reduce(0, +) / 16
                var sxy = 0.0, sxx = 0.0
                for (dd, l) in zip(fitD, fitL) { sxy += (dd - md) * (l - ml); sxx += (dd - md) * (dd - md) }
                let slope = sxy / sxx
                // On the photo's scale the new side's d (−1 = the first new row) is d + 1 (photo row 1 is d = 1, so −1 → 0).
                var continued = 0.0
                for d in -12 ... -4 { continued += (ml + slope * (Double(d) + 1 - md)) / 9 }
                s.gradientStep = (-12 ... -4).map { s.prof($0) }.reduce(0, +) / 9 - continued
                s.natural = continued - (4...12).map { s.prof($0) }.reduce(0, +) / 9
                (s.stripLine, s.stripLineAt) = deepest(-(reach - 4) ... -25)
                s.photoLine = deepest(5...20).0
                return s
            }
            let topCols = Array((o.x0 + 8)..<(o.x1 - 8))
            top = seam(topCols, reach: 100) { p, d in (p, o.y0 + (d < 0 ? d : d - 1)) }
            left = seam((o.y0 + 8..<(o.y0 + 260)).filter { photoSkyRows.left.contains($0) }, reach: 44) { p, d in (o.x0 + (d < 0 ? d : d - 1), p) }
            right = seam((o.y0 + 8..<(o.y0 + 260)).filter { photoSkyRows.right.contains($0) }, reach: 44) { p, d in (o.x1 - 1 - (d < 0 ? d : d - 1), p) }
            let freeCols = topCols.filter { x in !(0..<o.y0).contains { isCloud(x, $0) } && !(560...760).contains(x) }
            if freeCols.count > 20 {
                func rowMean(_ y: Int) -> Double { freeCols.map { Self.luma(Self.px(img, $0, y)) }.reduce(0, +) / Double(freeCols.count) }
                let ys = Array((o.y0 + 4)...(o.y0 + 140)), ls = ys.map(rowMean)
                let my = Double(ys.reduce(0, +)) / Double(ys.count), ml = ls.reduce(0, +) / Double(ls.count)
                var sxy = 0.0, sxx = 0.0
                for (y, l) in zip(ys, ls) { let dy = Double(y) - my; sxy += dy * (l - ml); sxx += dy * dy }
                let slope = sxy / sxx
                let strip = Array(4..<(o.y0 - 2))
                extrap = strip.map { abs(rowMean($0) - (ml + slope * (Double($0) - my))) }.reduce(0, +) / Double(strip.count)
            }
            let lum = (0..<(w * h)).map { i in Float(Self.luma(Self.px(img, i % w, i / w))) }
            let lp = FillMatch.boxSum(lum, w, h, radius: 2).map { $0 / 25 }
            let fc = freeCols.isEmpty ? topCols : freeCols
            func sigma(_ ys: Range<Int>) -> Double {
                Double(FillMatch.robustSigma(ys.flatMap { y in fc.map { lum[y * w + $0] - lp[y * w + $0] } }))
            }
            grainNew = sigma(8..<(o.y0 - 8)); grainPhoto = sigma((o.y0 + 8)..<(o.y0 + 120))
        }
    }

    // MARK: - Helpers

    nonisolated static func phys() -> UInt64 {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return kr == KERN_SUCCESS ? info.phys_footprint : 0
    }

    final class Sampler: @unchecked Sendable {
        private let state = OSAllocatedUnfairLock(initialState: (peak: UInt64(0), running: true))
        init() {
            let state = self.state
            state.withLock { $0.peak = SeamLevelProbe.phys() }
            Thread {
                while state.withLock({ $0.running }) {
                    let v = SeamLevelProbe.phys()
                    state.withLock { $0.peak = max($0.peak, v) }
                    usleep(20_000)
                }
            }.start()
        }
        func stop() -> UInt64 { let v = SeamLevelProbe.phys(); return state.withLock { $0.running = false; $0.peak = max($0.peak, v); return $0.peak } }
    }

    nonisolated static func load(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    nonisolated static func writePNG(_ image: CGImage, to url: URL) throws {
        let dest = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, image, nil)
        try #require(CGImageDestinationFinalize(dest))
    }

    nonisolated static func sheet(_ rows: [[(String, CGImage?)]], scale: Double) throws -> CGImage {
        let tw = Int(Double(rows.flatMap { $0 }.compactMap { $0.1?.width }.max() ?? 100) * scale)
        let th = Int(Double(rows.flatMap { $0 }.compactMap { $0.1?.height }.max() ?? 100) * scale)
        let (gap, label) = (8, 22)
        let cols = rows.map(\.count).max() ?? 1
        let W = cols * (tw + gap) + gap, H = rows.count * (th + label + gap) + gap
        let ctx = try #require(CGContext(data: nil, width: W, height: H, bitsPerComponent: 8, bytesPerRow: 0,
                                         space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        ctx.setFillColor(CGColor(gray: 1, alpha: 1)); ctx.fill(CGRect(x: 0, y: 0, width: W, height: H))
        let font = CTFontCreateWithName("Helvetica" as CFString, 14, nil)
        for (r, row) in rows.enumerated() {
            for (c, tile) in row.enumerated() {
                let x = gap + c * (tw + gap), yTop = gap + r * (th + label + gap)
                let attr = NSAttributedString(string: tile.0, attributes: [NSAttributedString.Key(kCTFontAttributeName as String): font])
                ctx.textPosition = CGPoint(x: Double(x), y: Double(H - yTop - 16))
                CTLineDraw(CTLineCreateWithAttributedString(attr), ctx)
                if let img = tile.1 { ctx.interpolationQuality = .none; ctx.draw(img, in: CGRect(x: x, y: H - yTop - label - th, width: tw, height: th)) }
            }
        }
        return try #require(ctx.makeImage())
    }
}
