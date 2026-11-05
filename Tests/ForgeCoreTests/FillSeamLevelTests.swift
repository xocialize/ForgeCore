//
//  FillSeamLevelTests.swift
//  ForgeCoreTests
//
//  The expand's seam-band levelling (AB-T-0450) on a fixture shaped like the real model's fault: a photo of a sky, a strip
//  painted a few levels darker, and a darker line in the middle of the LaMa band. The line and the step come out, the photo
//  stays exact, the strip keeps its texture and its clouds, a large step is only limited, and the route without it is the
//  route before it.
//

import Testing
import Foundation
@testable import ForgeCore

@Suite struct FillSeamLevelTests {
    // A 600 × 260 canvas; the photo 568 × 160 at (16, 100): new strips 16 px left and right (too thin to level at a 12 px
    // band), 100 px on top.
    static let (width, height) = (600, 260)
    static let photo = FillBox(x: 16, y: 100, width: 568, height: 160)
    static let band = 12

    /// A sky: luma falls 0.05 a row going up, monochrome grain ±6. The strip (outside the photo) is `step` darker, its band's
    /// rows 4…8 out `line` darker again, and `cloud` (if any) a bright blob in the strip.
    static func canvas(step: Double = 6, line: Double = 5, cloud: (x: Int, y: Int, r: Int)? = nil) -> FillPixels {
        var p = FillPixels(width: width, height: height)
        var s: UInt64 = 11
        for y in 0..<height {
            for x in 0..<width {
                s = s &* 6364136223846793005 &+ 1442695040888963407
                let grain = Double(Int(s >> 60) - 8) * 0.75
                let inside = x >= photo.x && x < photo.x + photo.width && y >= photo.y
                var v = 120 + 0.05 * Double(y) + grain
                if !inside {
                    v -= step
                    let d = max(photo.x - x, x - (photo.x + photo.width - 1), photo.y - y)
                    if (4...8).contains(d) { v -= line }
                    if let c = cloud, (x - c.x) * (x - c.x) + (y - c.y) * (y - c.y) < c.r * c.r { v += 60 }
                }
                let i = (y * width + x) * 4
                p.rgba[i] = UInt8(max(0, min(255, v - 30))); p.rgba[i + 1] = UInt8(max(0, min(255, v)))
                p.rgba[i + 2] = UInt8(max(0, min(255, v + 50))); p.rgba[i + 3] = 255
            }
        }
        return p
    }

    static func luma(_ p: FillPixels, _ x: Int, _ y: Int) -> Double {
        let i = (y * p.width + x) * 4
        return 0.299 * Double(p.rgba[i]) + 0.587 * Double(p.rgba[i + 1]) + 0.114 * Double(p.rgba[i + 2])
    }

    /// The top seam's mean luma at `d` px from it over the photo's middle columns (d > 0 the strip, d ≤ 0 the photo's row 1 − d).
    static func row(_ p: FillPixels, _ d: Int) -> Double {
        let cols = (photo.x + 20)..<(photo.x + photo.width - 20)
        return cols.map { luma(p, $0, photo.y - d) }.reduce(0, +) / Double(cols.count)
    }

    @Test func theBandsLineAndTheStepComeOutAndThePhotoStaysExact() {
        let before = Self.canvas()
        let (after, report) = FillSeamLevel.levelled(before, photo: Self.photo, band: Self.band)
        // The fixture is what the real model leaves: a step of about 6 and a 5-deep line mid-band.
        #expect(Self.row(before, 6) < Self.row(before, 12) - 3)
        #expect(Self.row(before, 2) < Self.row(before, -2) - 4)
        // After: across the band no row is more than a level off a straight run, and the seam has no step.
        for d in 3...(Self.band - 2) {
            let dip = (Self.row(after, d - 2) + Self.row(after, d + 2)) / 2 - Self.row(after, d)
            #expect(abs(dip) < 1, "row \(d): dip \(dip)")
        }
        // Beyond the band the change is a smooth ramp: no row moves more than a little more than the next.
        for d in (Self.band + 1)...(3 * Self.band + 2) {
            let move = { (d: Int) in Self.row(after, d) - Self.row(before, d) }
            #expect(abs(move(d) - move(d + 1)) < 0.75, "row \(d)")
        }
        #expect(abs(Self.row(after, 2) - Self.row(after, -2)) < 1.5)
        #expect(abs((report.sides["top"]?[0] ?? 0) - 6) < 1.5)
        // The photo's pixels are untouched.
        for y in Self.photo.y..<Self.height {
            for x in Self.photo.x..<(Self.photo.x + Self.photo.width) {
                let i = (y * Self.width + x) * 4
                #expect(after.rgba[i..<i + 4] == before.rgba[i..<i + 4])
            }
        }
        // The thin side strips (16 px, under band + 6) are left alone.
        #expect(report.sides["left"] == nil && report.sides["right"] == nil)
        #expect(after.rgba[((Self.photo.y + 50) * Self.width + 5) * 4] == before.rgba[((Self.photo.y + 50) * Self.width + 5) * 4])
    }

    @Test func theStripKeepsItsGrainAndItsCloud() {
        let before = Self.canvas(cloud: (x: 150, y: 50, r: 18))
        let (after, _) = FillSeamLevel.levelled(before, photo: Self.photo, band: Self.band)
        // Pixel-to-pixel detail is unchanged: the correction is smooth along the seam.
        var worst = 0.0
        for y in (Self.photo.y - Self.band)..<(Self.photo.y - 1) {
            for x in 60..<240 {
                let a = (Self.luma(after, x + 1, y) - Self.luma(after, x, y)) - (Self.luma(before, x + 1, y) - Self.luma(before, x, y))
                worst = max(worst, abs(a))
            }
        }
        #expect(worst < 1.5)
        // The cloud, beyond the band, keeps its contrast against the sky beside it.
        let contrast = { (p: FillPixels) in Self.luma(p, 150, 50) - Self.luma(p, 100, 50) }
        #expect(abs(contrast(after) - contrast(before)) < 1.5)
        // And far out, past the ramp, nothing moves.
        #expect(after.rgba[(10 * Self.width + 150) * 4..<(10 * Self.width + 150) * 4 + 4]
                == before.rgba[(10 * Self.width + 150) * 4..<(10 * Self.width + 150) * 4 + 4])
    }

    @Test func aStepPastTwiceTheLimitIsContentAndLeftAlone() {
        let before = Self.canvas(step: 40, line: 0)
        let (after, report) = FillSeamLevel.levelled(before, photo: Self.photo, band: Self.band)
        #expect(after == before)
        #expect(report.sides["top"]?[1] == 0)
        // Half-way between the limit and twice it, half the correction.
        let (half, _) = FillSeamLevel.levelled(Self.canvas(step: 40, line: 0), photo: Self.photo, band: Self.band,
                                               settings: FillSeamLevel.Settings(limit: 40 / 1.5))
        let moved = Self.row(half, Self.band + 2) - Self.row(before, Self.band + 2)
        #expect(moved > 14 && moved < 26, "moved \(moved)")
    }

    @Test func aCloudCutAtTheSeamIsLeftAloneWhileTheSkyBesideItIsLevelled() {
        // A wide cloud sitting on the band's far side over the middle third of the seam.
        var before = Self.canvas()
        for y in 60..<(Self.photo.y - Self.band) {
            for x in 110..<190 {
                let i = (y * Self.width + x) * 4
                for c in 0..<3 { before.rgba[i + c] = UInt8(min(255, Int(before.rgba[i + c]) + 70)) }
            }
        }
        let (after, _) = FillSeamLevel.levelled(before, photo: Self.photo, band: Self.band)
        func change(_ x: Int, _ d: Int) -> Double { Self.luma(after, x, Self.photo.y - d) - Self.luma(before, x, Self.photo.y - d) }
        #expect(abs(change(150, 6)) < 1 && abs(change(150, Self.band + 4)) < 1)     // under the cloud: untouched
        #expect(change(450, 6) > 6)                                                 // the sky's band: its line and step out
    }

    @Test func anEdgeCrossingTheSeamIsLeftAlone() {
        // A dark hill whose top crosses the seam: the photo's columns 130…170 dark from the seam down, the strip's from 4 px
        // further out (the painting drew it a little higher). Levelling across it would smear the hill's edge.
        var before = Self.canvas()
        for y in (Self.photo.y - 4)..<Self.height {
            for x in 130..<170 {
                let i = (y * Self.width + x) * 4
                for c in 0..<3 { before.rgba[i + c] = UInt8(max(0, Int(before.rgba[i + c]) - 60)) }
            }
        }
        let (after, _) = FillSeamLevel.levelled(before, photo: Self.photo, band: Self.band)
        for y in (Self.photo.y - 3 * Self.band)..<Self.photo.y {
            for x in 125..<175 {
                #expect(abs(Self.luma(after, x, y) - Self.luma(before, x, y)) < 1.5, "x \(x) y \(y)")
            }
        }
    }

    @Test func theBandOnlySettingLeavesTheStripAsPainted() {
        let before = Self.canvas()
        let (after, _) = FillSeamLevel.levelled(before, photo: Self.photo, band: Self.band, settings: .band)
        #expect(abs(Self.row(after, Self.band + 3) - Self.row(before, Self.band + 3)) < 0.01)
        let dip = (Self.row(after, 2) + Self.row(after, 10)) / 2 - Self.row(after, 6)
        #expect(abs(dip) < 1)
    }

    @Test func theRouteLevelsOnlyWhenAskedAndNeverTouchesTheOriginal() async throws {
        let image = FillPixels(FillFixture.photo())
        // The model continues the photo's top row upward, 3 levels darker; LaMa's band comes back 5 darker again.
        let (padded, _) = FillRoute.pad(image, left: 0, top: 96, right: 0, bottom: 0)
        let lineLaMa: GenerativeFillRoute.Inpainter = { canvas, hole in
            var out = canvas
            for i in 0..<(canvas.width * canvas.height) where hole.values[i] > 0 {
                for c in 0..<3 { out.rgba[i * 4 + c] = UInt8(max(0, Int(canvas.rgba[i * 4 + c]) - 5)) }
            }
            return out
        }
        func run(_ settings: FillSeamLevel.Settings?) async throws -> (FillPixels, GenerativeFillTrace) {
            let calls = FillCalls()
            var route = GenerativeFillRoute(maxWindowPixels: 1024 * 1024, multipleOf: 32)
            route.seamLevel = settings
            let (canvas, patch, trace) = try await route.expand(image, left: 0, top: 96, right: 0, bottom: 0, prompt: "", seed: 5,
                                                                paint: FakeModel.painter(original: padded, content: nil, drift: -3, calls: calls),
                                                                inpaint: lineLaMa)
            return (patch.applied(to: canvas), trace)
        }
        let (plain, plainTrace) = try await run(nil)
        let (defaulted, _) = try await run(GenerativeFillRoute(maxWindowPixels: 1, multipleOf: 1).seamLevel)
        #expect(plain == defaulted && plainTrace.seamLevel == nil)     // off by default: the route before 0.45.0
        let (levelled, trace) = try await run(.standard)
        #expect((trace.seamLevel?.sides["top"]?[1] ?? 0) > 0.5)        // drift: levelled
        #expect(levelled != plain)
        for y in 96..<(96 + image.height) {
            let a = (y * image.width) * 4, b = ((y - 96) * image.width) * 4
            #expect(levelled.rgba[a..<a + image.width * 4] == image.rgba[b..<b + image.width * 4])
        }
    }
}
