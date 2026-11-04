//
//  FillMatchTests.swift
//  ForgeCoreTests
//
//  The level and grain match (AB-T-0425) on a fixture, with fake models: a painting that is smooth, or drifted in tone,
//  comes back with the photo's grain and level, every unselected pixel stays exact, and the bake-off's route is still
//  there with `matchesSurroundings` off.
//

import Testing
import CoreGraphics
import Foundation
@testable import ForgeCore

extension FillFixture {
    /// `image` with monochrome grain: the same uniform offset in −16…15 on every channel of a pixel, as the editor's
    /// hand-test photo has.
    static func grainy(_ image: FillPixels, seed: UInt64 = 7) -> FillPixels {
        var out = image, s = seed
        for i in 0..<(image.width * image.height) {
            s = s &* 6364136223846793005 &+ 1442695040888963407
            let n = Int(s >> 59) - 16
            for c in 0..<3 { out.rgba[i * 4 + c] = UInt8(max(0, min(255, Int(image.rgba[i * 4 + c]) + n))) }
        }
        return out
    }
}

@Suite struct FillMatchTests {
    static let (width, height) = (320, 240)
    static let box = FillBox(x: 120, y: 70, width: 90, height: 80)

    /// The grain (σ of luma's high-pass, trimmed) and mean luma of `p` within `r`.
    static func stats(_ p: FillPixels, _ r: FillBox) -> (grain: Double, level: Double) {
        let hp = FillMatch.HighPass(p, box: r)
        return (Double(FillMatch.robustSigma(hp.luma)), Double(hp.base.reduce(0, +)) / Double(hp.base.count))
    }

    /// The σ of the high-pass of R − G within `r`: about 0 for monochrome grain.
    static func chromaGrain(_ p: FillPixels, _ r: FillBox) -> Double {
        let hp = FillMatch.HighPass(p, box: r)
        return Double(FillMatch.robustSigma((0..<(r.width * r.height)).map { hp.rgb[$0 * 3] - hp.rgb[$0 * 3 + 1] }))
    }

    /// A painter that paints the marked area from `clean` (no grain) and re-renders everything `drift` levels brighter
    /// and smooth: SenseNova as measured (AB-T-0425), grain lost and tone off.
    static func smoothPainter(clean: FillPixels, drift: Int, calls: FillCalls) -> GenerativeFillRoute.Painter {
        { marked, instruction, seed, width, height in
            calls.paints.withLock { $0.append(.init(marked: marked, instruction: instruction, seed: seed, width: width, height: height)) }
            var out = clean.resized(width, height)
            for i in 0..<(width * height) { for c in 0..<3 { out.rgba[i * 4 + c] = UInt8(min(255, Int(out.rgba[i * 4 + c]) + drift)) } }
            return out
        }
    }

    @Test func aSmoothDriftedFillComesBackWithThePhotosGrainAndLevel() async throws {
        let clean = FillPixels(FillFixture.photo(width: Self.width, height: Self.height))
        let image = FillFixture.grainy(clean)
        let mask = FillFixture.mask(width: Self.width, height: Self.height, box: Self.box)
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        let (patch, trace) = try #require(try await route.fill(image, mask: mask, prompt: "a red bicycle", seed: 11,
                                                              paint: Self.smoothPainter(clean: clean, drift: 10, calls: calls),
                                                              inpaint: FakeModel.lama(calls: calls)))
        let out = patch.applied(to: image)
        // Every unselected pixel is the input's, exactly.
        for i in 0..<(Self.width * Self.height) where mask.values[i] == 0 { #expect(out.rgba[i * 4..<i * 4 + 4] == image.rgba[i * 4..<i * 4 + 4]) }
        // The grain inside is the photo's (it would be ~0 without the match), and stays monochrome.
        let inside = FillBox(x: Self.box.x + 4, y: Self.box.y + 4, width: Self.box.width - 8, height: Self.box.height - 8)
        let photoGrain = Self.stats(image, inside).grain
        let grain = Self.stats(out, inside).grain
        #expect(abs(grain - photoGrain) < 0.15 * photoGrain, "grain inside \(grain), the photo's \(photoGrain)")
        #expect(Self.chromaGrain(out, inside) < 0.25 * photoGrain, "the photo's grain is monochrome; so is the copy")
        // The level: the +10 drift is gone in the middle too, not just at the seam (the clean photo is the truth there).
        let middle = FillBox(x: Self.box.x + 30, y: Self.box.y + 25, width: 30, height: 30)
        let step = Self.stats(out, middle).level - Self.stats(clean, middle).level
        #expect(abs(step) < 2.5, "the middle's level should match the photo's; step \(step)")
        // The trace says what was done.
        let matched = try #require(trace.matched)
        #expect(matched.level.allSatisfy { abs($0 + 10) < 2 }, "level shift \(matched.level)")
        #expect(matched.paintingGrain < 1 && abs(matched.grain - photoGrain) < 0.15 * photoGrain && matched.resultGrain > 0.8 * matched.grain)
    }

    @Test func theBakeOffsRouteKeepsTheSmoothDriftedPainting() async throws {
        let clean = FillPixels(FillFixture.photo(width: Self.width, height: Self.height))
        let image = FillFixture.grainy(clean)
        let mask = FillFixture.mask(width: Self.width, height: Self.height, box: Self.box)
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32, matchesSurroundings: false)
        let (patch, trace) = try #require(try await route.fill(image, mask: mask, prompt: "a red bicycle", seed: 11,
                                                              paint: Self.smoothPainter(clean: clean, drift: 10, calls: calls),
                                                              inpaint: FakeModel.lama(calls: calls)))
        let out = patch.applied(to: image)
        let middle = FillBox(x: Self.box.x + 30, y: Self.box.y + 25, width: 30, height: 30)
        #expect(trace.matched == nil)
        #expect(Self.stats(out, middle).grain < 1, "the bake-off's route paints smooth")
        #expect(Self.stats(out, middle).level - Self.stats(clean, middle).level > 7, "and keeps the drift in the middle")
    }

    @Test func theGrainIsReproducibleFromTheSeed() async throws {
        let clean = FillPixels(FillFixture.photo(width: Self.width, height: Self.height))
        let image = FillFixture.grainy(clean)
        let mask = FillFixture.mask(width: Self.width, height: Self.height, box: Self.box)
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        func run(_ seed: UInt64) async throws -> FillPixels {
            let calls = FillCalls()
            return try #require(try await route.fill(image, mask: mask, prompt: "", seed: seed,
                                                     paint: Self.smoothPainter(clean: clean, drift: 0, calls: calls),
                                                     inpaint: FakeModel.lama(calls: calls))).patch.pixels
        }
        let (a, b, c) = (try await run(3), try await run(3), try await run(4))
        #expect(a == b)
        #expect(a != c)
    }

    @Test func aPhotoWithNoGrainGetsNone() async throws {
        let clean = FillPixels(FillFixture.photo(width: Self.width, height: Self.height))
        let mask = FillFixture.mask(width: Self.width, height: Self.height, box: Self.box)
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        let (patch, trace) = try #require(try await route.fill(clean, mask: mask, prompt: "a red bicycle", seed: 11,
                                                              paint: Self.smoothPainter(clean: clean, drift: 0, calls: calls),
                                                              inpaint: FakeModel.lama(calls: calls)))
        let inside = FillBox(x: Self.box.x + 4, y: Self.box.y + 4, width: Self.box.width - 8, height: Self.box.height - 8)
        #expect(Self.stats(patch.applied(to: clean), inside).grain < 1)
        let matched = try #require(trace.matched)
        #expect(matched.grain < 1 && matched.resultGrain == matched.paintingGrain)
    }

    @Test func aPaintingThatAlreadyHasTheGrainGetsNoMore() {
        let image = FillFixture.grainy(FillPixels(FillFixture.photo(width: Self.width, height: Self.height)))
        let mask = FillFixture.mask(width: Self.width, height: Self.height, box: Self.box)
        let painting = FillFixture.grainy(FillPixels(FillFixture.photo(width: Self.width, height: Self.height)), seed: 99)
        let (out, grain, before, after) = FillMatch.grained(painting, original: image, region: mask, seed: 1)
        #expect(out == painting)
        #expect(abs(before - grain) < 0.1 * grain && after == before)
    }

    @Test func whatTheModelChangedOutsideTheAreaDoesNotShiftIt() {
        // The model re-rendered everything 10 brighter, and also drew a white cloud beside the area: the drift is −10, the
        // cloud isn't drift. The cloud comes back as the original's, so the harmonize won't carry it in either.
        let original = FillPixels(FillFixture.photo(width: Self.width, height: Self.height))
        let mask = FillFixture.mask(width: Self.width, height: Self.height, box: Self.box)
        var painting = original
        for i in 0..<(Self.width * Self.height) { for c in 0..<3 { painting.rgba[i * 4 + c] = UInt8(min(255, Int(original.rgba[i * 4 + c]) + 10)) } }
        let cloud = FillBox(x: Self.box.x + Self.box.width + 2, y: Self.box.y, width: 40, height: 60)
        for y in cloud.y..<(cloud.y + cloud.height) { for x in cloud.x..<(cloud.x + cloud.width) { for c in 0..<3 { painting.rgba[(y * Self.width + x) * 4 + c] = 250 } } }
        let (levelled, shift) = FillMatch.levelled(painting, toward: original, region: mask)
        #expect(shift.allSatisfy { abs($0 + 10) < 1.5 }, "the shift is the drift alone: \(shift)")
        let i = ((cloud.y + 20) * Self.width + cloud.x + 20) * 4
        #expect(levelled.rgba[i..<i + 4] == original.rgba[i..<i + 4])
    }

    @Test func anExpandGetsThePhotosGrainOnItsNewSides() async throws {
        let image = FillFixture.grainy(FillPixels(FillFixture.photo(width: 256, height: 192)))
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 1024 * 1024, multipleOf: 32)
        // The fake paints the new side a flat colour, and LaMa's seam band a flat grey: both smooth.
        let (canvas, patch, trace) = try await route.expand(image, left: 0, top: 0, right: 64, bottom: 0, prompt: "", seed: 5,
                                                           paint: FakeModel.painter(original: nil, content: (120, 150, 200), drift: 0, calls: calls),
                                                           inpaint: FakeModel.lama(calls: calls))
        let out = patch.applied(to: canvas)
        let newSide = FillBox(x: 256 + 4, y: 8, width: 56, height: 176)
        let photoGrain = Self.stats(image, FillBox(x: 8, y: 8, width: 240, height: 176)).grain
        let grain = Self.stats(out, newSide).grain
        #expect(trace.seamBridged && trace.matched != nil)
        #expect(trace.matched?.level == [0, 0, 0], "an expand's new area gets no level shift, only the harmonize's")
        #expect(abs(grain - photoGrain) < 0.2 * photoGrain, "grain on the new side \(grain), the photo's \(photoGrain)")
        for y in 0..<192 { for x in stride(from: 0, to: 256, by: 7) {
            let (o, n) = ((y * 256 + x) * 4, (y * out.width + x) * 4)
            #expect(out.rgba[n..<n + 4] == image.rgba[o..<o + 4])
        } }
    }
}
