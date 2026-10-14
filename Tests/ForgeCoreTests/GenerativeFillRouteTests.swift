//
//  GenerativeFillRouteTests.swift
//  ForgeCoreTests
//
//  The Generative Fill / Expand route on a fixture image and mask, with the edit model and LaMa replaced by fakes —
//  no GPU, no weights, so it runs everywhere (AB-A-0166's fixture test; the real-model run is
//  `GenerativeFillLiveTests`). It pins the route's contract with the Image Engine's: the instruction text, the marker,
//  the guard's retry-then-LaMa, the harmonize, the composite that leaves every unselected pixel exact (16 bits
//  included), the window, and expand's padded canvas, per-side guard and seam bridge.
//

import Testing
import CoreGraphics
import Foundation
import Synchronization
@testable import ForgeCore

// MARK: - Fixtures

enum FillFixture {
    /// A deterministic outdoor-like picture: sky gradient, a sun, a ground band and a few blocks. Drawn, so it is
    /// licence-clean and needs no file.
    static func photo(width: Int = 640, height: Int = 480) -> CGImage {
        let ctx = CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                            space: FillPixels.sRGB, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let w = CGFloat(width), h = CGFloat(height)
        let sky = CGGradient(colorsSpace: FillPixels.sRGB, colors: [CGColor(srgbRed: 0.35, green: 0.55, blue: 0.85, alpha: 1),
                                                                    CGColor(srgbRed: 0.85, green: 0.9, blue: 0.95, alpha: 1)] as CFArray,
                             locations: [0, 1])!
        ctx.drawLinearGradient(sky, start: CGPoint(x: 0, y: h), end: CGPoint(x: 0, y: h * 0.35), options: [])
        ctx.setFillColor(CGColor(srgbRed: 0.35, green: 0.45, blue: 0.25, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: w, height: h * 0.35))
        ctx.setFillColor(CGColor(srgbRed: 1, green: 0.85, blue: 0.4, alpha: 1))
        ctx.fillEllipse(in: CGRect(x: w * 0.7, y: h * 0.7, width: w * 0.1, height: w * 0.1))
        for (i, shade) in [0.55, 0.42, 0.6, 0.5].enumerated() {
            ctx.setFillColor(CGColor(srgbRed: shade, green: shade * 0.9, blue: shade * 0.8, alpha: 1))
            ctx.fill(CGRect(x: w * (0.08 + 0.2 * CGFloat(i)), y: h * 0.3, width: w * 0.12, height: h * (0.15 + 0.05 * CGFloat(i))))
        }
        return ctx.makeImage()!
    }

    /// The same scene at 16 bits per channel, with a fine ramp so most samples sit between 8-bit levels.
    static func deepPhoto(width: Int = 320, height: Int = 256) -> CGImage {
        var data = [UInt16](repeating: 0, count: width * height * 4)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * 4
                data[i] = UInt16(9000 + x * 131 + y * 7)
                data[i + 1] = UInt16(20000 + y * 113)
                data[i + 2] = UInt16(40000 - x * 61)
                data[i + 3] = 65535
            }
        }
        return data.withUnsafeMutableBytes { buffer in
            CGContext(data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 16, bytesPerRow: width * 8,
                      space: FillPixels.sRGB, bitmapInfo: EngineImageCoding.rgba16BitmapInfo)!.makeImage()!
        }
    }

    /// A selection: white inside `box`, a `feather`-px linear ramp to black around it (a soft selection's edge).
    static func mask(width: Int, height: Int, box: FillBox, feather: Int = 0) -> FillMask {
        var m = FillMask(width: width, height: height)
        for y in 0..<height {
            for x in 0..<width {
                let dx = max(box.x - x, 0, x - (box.x + box.width - 1))
                let dy = max(box.y - y, 0, y - (box.y + box.height - 1))
                let d = max(dx, dy)
                if d == 0 { m.values[y * width + x] = 255 } else if d <= feather {
                    m.values[y * width + x] = UInt8(255 - 255 * d / (feather + 1))
                }
            }
        }
        return m
    }

    static func sixteenBit(_ image: CGImage) -> [UInt16] {
        var out = [UInt16](repeating: 0, count: image.width * image.height * 4)
        out.withUnsafeMutableBytes { buffer in
            let ctx = CGContext(data: buffer.baseAddress, width: image.width, height: image.height, bitsPerComponent: 16,
                                bytesPerRow: image.width * 8, space: FillPixels.sRGB, bitmapInfo: EngineImageCoding.rgba16BitmapInfo)!
            ctx.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return out
    }
}

/// Records every call a fake model or LaMa receives.
final class FillCalls: Sendable {
    struct Paint: Sendable { let marked: FillPixels; let instruction: String; let seed: UInt64; let width: Int; let height: Int }
    struct Inpaint: Sendable { let canvas: FillPixels; let hole: FillMask }
    let paints = Mutex<[Paint]>([])
    let inpaints = Mutex<[Inpaint]>([])
    var paintCount: Int { paints.withLock { $0.count } }
    var inpaintCount: Int { inpaints.withLock { $0.count } }
}

enum FakeModel {
    /// An edit model that does what SenseNova did in AB-R-0533: paints the marked area with `content`, and re-renders
    /// the rest of the picture `drift` levels brighter. `original` is the unmarked picture (window = whole image).
    static func painter(original: FillPixels?, content: (UInt8, UInt8, UInt8)?, drift: Int,
                        calls: FillCalls) -> GenerativeFillRoute.Painter {
        { marked, instruction, seed, width, height in
            calls.paints.withLock { $0.append(.init(marked: marked, instruction: instruction, seed: seed, width: width, height: height)) }
            var out = marked
            for i in 0..<(width * height) {
                let isMarker = marked.rgba[i * 4] == 255 && marked.rgba[i * 4 + 1] == 0 && marked.rgba[i * 4 + 2] == 255
                for c in 0..<3 {
                    var v: Int
                    if isMarker {
                        if let content { v = Int([content.0, content.1, content.2][c]) }
                        else if let original { v = Int(original.rgba[i * 4 + c]) } else { v = Int(marked.rgba[i * 4 + c]) }
                    } else {
                        v = Int(marked.rgba[i * 4 + c])
                    }
                    out.rgba[i * 4 + c] = UInt8(min(255, max(0, v + drift)))
                }
                out.rgba[i * 4 + 3] = 255
            }
            return out
        }
    }

    /// An edit model that hands the marker back untouched (Klein's failure mode in the bake-off's removes).
    static func keepsMarker(calls: FillCalls) -> GenerativeFillRoute.Painter {
        { marked, instruction, seed, width, height in
            calls.paints.withLock { $0.append(.init(marked: marked, instruction: instruction, seed: seed, width: width, height: height)) }
            return marked
        }
    }

    /// LaMa: fills the hole with a flat grey.
    static func lama(calls: FillCalls) -> GenerativeFillRoute.Inpainter {
        { canvas, hole in
            calls.inpaints.withLock { $0.append(.init(canvas: canvas, hole: hole)) }
            var out = canvas
            for i in 0..<(canvas.width * canvas.height) where hole.values[i] > 0 {
                out.rgba[i * 4] = 90; out.rgba[i * 4 + 1] = 90; out.rgba[i * 4 + 2] = 90; out.rgba[i * 4 + 3] = 255
            }
            return out
        }
    }
}

// MARK: - Tests

@Suite struct GenerativeFillRouteTests {

    // The Image Engine's instruction text, verbatim (RunPlan.swift:288–291, 376–380) — the bake-off's prompts.
    @Test func instructionsAreTheImageEnginesVerbatim() {
        #expect(FillRoute.fillInstruction(prompt: "") == "Fill the bright magenta area so it blends seamlessly into the rest of the picture, as if nothing had been there. Keep everything else unchanged.")
        #expect(FillRoute.fillInstruction(prompt: "a small wooden rowing boat") == "Replace the bright magenta area with a small wooden rowing boat. Match the picture's lighting, perspective and style, and keep everything else unchanged.")
        #expect(FillRoute.extendInstruction(left: 0, top: 0, right: 256, bottom: 0, prompt: "") == "Replace the bright magenta area at the right edge with a seamless continuation of the scene. Keep everything else unchanged.")
        #expect(FillRoute.extendInstruction(left: 256, top: 192, right: 0, bottom: 0, prompt: "a harbour") == "Replace the bright magenta areas at the left and top edges with a seamless continuation of the scene showing a harbour. Keep everything else unchanged.")
        #expect(FillRoute.extendInstruction(left: 8, top: 8, right: 8, bottom: 8, prompt: "") == "Replace the bright magenta areas at the left, right, top and bottom edges with a seamless continuation of the scene. Keep everything else unchanged.")
        #expect(FillRoute.checkable(prompt: "a PINK flamingo") == false)
        #expect(FillRoute.checkable(prompt: "a red boat") == true)
        #expect(FillRoute.marker == [255, 0, 255, 255])
    }

    @Test func fillPaintsTheSelectionHarmonizesAndLeavesTheRestExact() async throws {
        let image = FillPixels(FillFixture.photo())
        let box = FillBox(x: 240, y: 260, width: 160, height: 120)
        let mask = FillFixture.mask(width: image.width, height: image.height, box: box)
        let calls = FillCalls()
        // The fake re-renders the whole picture 10 levels brighter, the area included (the original under the marker):
        // the harmonize must take that step back out where the painting meets the picture.
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        let (patch, trace) = try #require(try await route.fill(image, mask: mask, prompt: "  a red bicycle \n", seed: 7,
                                                              paint: FakeModel.painter(original: image, content: nil, drift: 10, calls: calls),
                                                              inpaint: FakeModel.lama(calls: calls)))
        let out = patch.applied(to: image)

        // One model call, on the whole picture (it fits the cap), 1:1, with the route's instruction and the seed asked.
        #expect(calls.paintCount == 1 && calls.inpaintCount == 0)
        let call = try #require(calls.paints.withLock { $0.first })
        #expect(call.instruction == FillRoute.fillInstruction(prompt: "a red bicycle"))
        #expect(call.seed == 7 && call.width == 640 && call.height == 480)
        #expect(trace.window == FillWindow(box: FillBox(x: 0, y: 0, width: 640, height: 480), modelWidth: 640, modelHeight: 480))
        #expect(trace.seed == 7 && trace.retries == 0 && !trace.fellBackToInpaint)
        // The model saw magenta exactly where the selection is non-black, and the picture elsewhere.
        for i in 0..<(image.width * image.height) {
            let marked = Array(call.marked.rgba[i * 4..<i * 4 + 4])
            if mask.values[i] > 0 { #expect(marked == FillRoute.marker) } else if i % 97 == 0 {
                #expect(marked == Array(image.rgba[i * 4..<i * 4 + 4]))
            }
        }
        // Every unselected pixel is the input's, exactly.
        var outsideChanged = 0
        for i in 0..<(image.width * image.height) where mask.values[i] == 0 {
            if out.rgba[i * 4..<i * 4 + 4] != image.rgba[i * 4..<i * 4 + 4] { outsideChanged += 1 }
        }
        #expect(outsideChanged == 0)
        // Along the seam, the +10 drift is harmonized away (no tone step where the painting meets the picture) ...
        func meanStep(rows: Range<Int>, columns: Range<Int>) -> Double {
            var step = 0.0, n = 0
            for y in rows {
                for x in columns {
                    let i = (y * image.width + x) * 4
                    for c in 0..<3 { step += Double(Int(out.rgba[i + c]) - Int(image.rgba[i + c])) }
                    n += 3
                }
            }
            return step / Double(n)
        }
        let atSeam = meanStep(rows: box.y..<(box.y + 4), columns: (box.x + 10)..<(box.x + box.width - 10))
        #expect(abs(atSeam) < 3, "harmonize should cancel the 10-level drift at the seam; mean step \(atSeam)")
        // ... and fades with distance, as the Image Engine's does: the middle keeps the model's own tone.
        let middle = meanStep(rows: (box.y + box.height / 2 - 2)..<(box.y + box.height / 2 + 2),
                              columns: (box.x + box.width / 2 - 20)..<(box.x + box.width / 2 + 20))
        #expect(middle > 7, "the correction should fade away from the seam; mean step \(middle)")
    }

    @Test func aKeptMarkerIsRetriedWithTheNextSeedThenFilledByLaMa() async throws {
        let image = FillPixels(FillFixture.photo())
        let mask = FillFixture.mask(width: image.width, height: image.height, box: FillBox(x: 100, y: 100, width: 120, height: 90))
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        let (patch, trace) = try #require(try await route.fill(image, mask: mask, prompt: "", seed: 41,
                                                              paint: FakeModel.keepsMarker(calls: calls),
                                                              inpaint: FakeModel.lama(calls: calls)))
        #expect(calls.paints.withLock { $0.map(\.seed) } == [41, 42])
        #expect(trace.rejected == "marker" && trace.fellBackToInpaint && trace.seed == nil)
        let lama = try #require(calls.inpaints.withLock { $0.first })
        #expect(lama.hole == mask.hole)                                  // LaMa fills the binary hole of the window
        // No magenta reaches the output.
        let out = patch.applied(to: image)
        let magenta = stride(from: 0, to: out.rgba.count, by: 4).filter {
            out.rgba[$0] == 255 && out.rgba[$0 + 1] == 0 && out.rgba[$0 + 2] == 255
        }
        #expect(magenta.isEmpty)
    }

    @Test func aSecondPaintingThatWorksIsUsed() async throws {
        let image = FillPixels(FillFixture.photo())
        let mask = FillFixture.mask(width: image.width, height: image.height, box: FillBox(x: 300, y: 40, width: 100, height: 100))
        let calls = FillCalls()
        let good = FakeModel.painter(original: nil, content: (40, 120, 60), drift: 0, calls: calls)
        let bad = FakeModel.keepsMarker(calls: calls)
        let flaky: GenerativeFillRoute.Painter = { marked, instruction, seed, w, h in
            try await (seed == 3 ? bad : good)(marked, instruction, seed, w, h)
        }
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        let (_, trace) = try #require(try await route.fill(image, mask: mask, prompt: "a hedge", seed: 3, paint: flaky,
                                                           inpaint: FakeModel.lama(calls: calls)))
        #expect(trace.retries == 1 && trace.seed == 4 && trace.rejected == "marker" && !trace.fellBackToInpaint)
        #expect(calls.inpaintCount == 0)
    }

    @Test func aPromptAskingForTheMarkersColourIsNotGuarded() async throws {
        let image = FillPixels(FillFixture.photo())
        let mask = FillFixture.mask(width: image.width, height: image.height, box: FillBox(x: 300, y: 40, width: 100, height: 100))
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        let (_, trace) = try #require(try await route.fill(image, mask: mask, prompt: "a magenta umbrella", seed: 1,
                                                           paint: FakeModel.keepsMarker(calls: calls),
                                                           inpaint: FakeModel.lama(calls: calls)))
        #expect(calls.paintCount == 1 && !trace.fellBackToInpaint && trace.rejected == nil)
    }

    @Test func anEmptySelectionRunsNothing() async throws {
        let image = FillPixels(FillFixture.photo())
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        let result = try await route.fill(image, mask: FillMask(width: image.width, height: image.height), prompt: "x", seed: 0,
                                          paint: FakeModel.keepsMarker(calls: calls), inpaint: FakeModel.lama(calls: calls))
        #expect(result == nil && calls.paintCount == 0)
    }

    // MARK: Window

    @Test func theWindowIsTheWholePictureWhenItFitsTheCap() {
        let route = GenerativeFillRoute(maxWindowPixels: 1024 * 1024, multipleOf: 32)
        let w = route.window(for: FillBox(x: 10, y: 10, width: 50, height: 50), width: 1184, height: 880)
        #expect(w == FillWindow(box: FillBox(x: 0, y: 0, width: 1184, height: 880), modelWidth: 1184, modelHeight: 864))
    }

    @Test func aLargerPictureGetsASquareAroundTheSelectionAtTheCapsSide() {
        // The 16 GB tier on a 1024² picture: a 768² window, 1:1, containing the selection (the bake-off's fill-768² shape).
        let small = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        let region = FillBox(x: 820, y: 700, width: 180, height: 140)
        let w = small.window(for: region, width: 1024, height: 1024)
        #expect(w.box.width == 768 && w.box.height == 768 && w.isOneToOne)
        #expect(w.box.x == 1024 - 768 && w.box.y == 1024 - 768)          // pushed inside the picture
        // A 4000×3000 photo with a small selection: still a 1:1 768² window, not the whole photo squeezed into 768².
        let big = small.window(for: FillBox(x: 1900, y: 1400, width: 200, height: 160), width: 4000, height: 3000)
        #expect(big.box == FillBox(x: 1616, y: 1096, width: 768, height: 768) && big.isOneToOne)
        // A selection whose 2.5× context exceeds the cap: the square grows (on the 32 grid) and renders at the cap.
        let wide = small.window(for: FillBox(x: 1000, y: 1000, width: 500, height: 400), width: 4000, height: 3000)
        #expect(wide.box.width == 1248 && wide.box.height == 1248 && wide.modelWidth == 768 && wide.modelHeight == 768)
    }

    @Test func aSelectionLongerThanTheShortSideGetsTheWholePictureScaledIntoTheCap() {
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        let w = route.window(for: FillBox(x: 100, y: 500, width: 3500, height: 200), width: 4000, height: 3000)
        #expect(w.box == FillBox(x: 0, y: 0, width: 4000, height: 3000))
        #expect(w.modelWidth % 32 == 0 && w.modelHeight % 32 == 0 && w.modelWidth * w.modelHeight <= 768 * 768)
        #expect(w.modelWidth == 864 && w.modelHeight == 640)
    }

    @Test func everyWindowHoldsItsSelectionAndRespectsTheCap() {
        var rng = SystemRandomNumberGenerator()
        for cap in [768 * 768, 1024 * 1024] {
            for multiple in [16, 32] {
                let route = GenerativeFillRoute(maxWindowPixels: cap, multipleOf: multiple)
                for _ in 0..<400 {
                    let (w, h) = (Int.random(in: 64...6000, using: &rng), Int.random(in: 64...6000, using: &rng))
                    let rw = Int.random(in: 1...w, using: &rng), rh = Int.random(in: 1...h, using: &rng)
                    let region = FillBox(x: Int.random(in: 0...(w - rw), using: &rng), y: Int.random(in: 0...(h - rh), using: &rng),
                                         width: rw, height: rh)
                    let win = route.window(for: region, width: w, height: h)
                    #expect(win.box.x >= 0 && win.box.y >= 0 && win.box.x + win.box.width <= w && win.box.y + win.box.height <= h)
                    #expect(region.x >= win.box.x && region.y >= win.box.y
                            && region.x + region.width <= win.box.x + win.box.width
                            && region.y + region.height <= win.box.y + win.box.height, "\(region) outside \(win) on \(w)×\(h)")
                    #expect(win.modelWidth % multiple == 0 && win.modelHeight % multiple == 0)
                    #expect(win.modelWidth * win.modelHeight <= cap || win.modelWidth == multiple || win.modelHeight == multiple)
                }
            }
        }
    }

    @Test func aWindowedFillTouchesOnlyItsWindowAndPaintsAtTheModelsSize() async throws {
        // 1024² at the 16 GB cap: the model sees the 768² window, and the composite lands back in place.
        let image = FillPixels(FillFixture.photo(width: 1024, height: 1024))
        let box = FillBox(x: 600, y: 380, width: 150, height: 110)
        let mask = FillFixture.mask(width: 1024, height: 1024, box: box, feather: 4)
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32)
        let (patch, trace) = try #require(try await route.fill(image, mask: mask, prompt: "a fountain", seed: 9,
                                                              paint: FakeModel.painter(original: nil, content: (200, 40, 40), drift: 0, calls: calls),
                                                              inpaint: FakeModel.lama(calls: calls)))
        #expect(trace.window.box.width == 768 && trace.window.isOneToOne)
        let call = try #require(calls.paints.withLock { $0.first })
        #expect(call.width == 768 && call.height == 768 && call.marked.width == 768)
        let out = patch.applied(to: image)
        #expect(out.width == 1024 && out.height == 1024)
        var outsideChanged = 0, insidePainted = 0
        for i in 0..<(1024 * 1024) {
            if mask.values[i] == 0, out.rgba[i * 4..<i * 4 + 4] != image.rgba[i * 4..<i * 4 + 4] { outsideChanged += 1 }
            if mask.values[i] == 255, abs(Int(out.rgba[i * 4]) - 200) < 40 { insidePainted += 1 }
        }
        #expect(outsideChanged == 0)
        #expect(insidePainted > box.width * box.height * 9 / 10)
    }

    // MARK: Expand

    @Test func expandKeepsTheOriginalExactAndBridgesTheSeam() async throws {
        let image = FillPixels(FillFixture.photo())
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 1024 * 1024, multipleOf: 32)
        let (canvas, patch, trace) = try await route.expand(image, left: 0, top: 0, right: 192, bottom: 0, prompt: "", seed: 5,
                                                            paint: FakeModel.painter(original: nil, content: (120, 140, 90), drift: 3, calls: calls),
                                                            inpaint: FakeModel.lama(calls: calls))
        let out = patch.applied(to: canvas)
        #expect(out.width == 832 && out.height == 480)
        #expect(trace.instruction == FillRoute.extendInstruction(left: 0, top: 0, right: 192, bottom: 0, prompt: ""))
        #expect(trace.window.modelWidth == 832 && trace.window.modelHeight == 480)
        #expect(trace.seamBridged && !trace.fellBackToInpaint && trace.seed == 5)
        // The original is at (0, 0), unchanged.
        var changed = 0
        for y in 0..<480 {
            for x in 0..<640 where out.rgba[(y * 832 + x) * 4..<(y * 832 + x) * 4 + 4] != image.rgba[(y * 640 + x) * 4..<(y * 640 + x) * 4 + 4] {
                changed += 1
            }
        }
        #expect(changed == 0)
        // LaMa bridged a band on the new side of the seam, the route's width wide.
        let bridge = try #require(calls.inpaints.withLock { $0.first })
        let bandWidth = FillRoute.seamBandWidth(width: 832, height: 480)
        let banded = (0..<(832 * 480)).filter { bridge.hole.values[$0] > 0 }.map { $0 % 832 }
        #expect(!banded.isEmpty && banded.allSatisfy { $0 >= 640 && $0 <= 640 + bandWidth })
    }

    @Test func aDarkExtendIsRejectedAndFilledByLaMa() async throws {
        let image = FillPixels(FillFixture.photo())
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 1024 * 1024, multipleOf: 32)
        let (_, _, trace) = try await route.expand(image, left: 128, top: 0, right: 0, bottom: 0, prompt: "", seed: 0,
                                                   paint: FakeModel.painter(original: nil, content: (4, 4, 4), drift: 0, calls: calls),
                                                   inpaint: FakeModel.lama(calls: calls))
        #expect(trace.rejected == "dark" && trace.fellBackToInpaint && !trace.seamBridged && calls.paintCount == 2)
    }

    // MARK: Provider pixel path (the static half `EngineGenerativeFillProvider` runs off the main actor)

    @Test func aSixteenBitImageKeepsEveryUnselectedSampleExactly() async throws {
        let image = FillFixture.deepPhoto()
        let box = FillBox(x: 100, y: 90, width: 80, height: 60)
        let mask = FillFixture.mask(width: image.width, height: image.height, box: box, feather: 5)
        let calls = FillCalls()
        let (out, trace) = try await EngineGenerativeFillProvider.fill(
            image, mask: mask.cgImage, prompt: "a stone", seed: 2,
            route: GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32),
            paint: FakeModel.painter(original: nil, content: (100, 100, 100), drift: 0, calls: calls),
            inpaint: FakeModel.lama(calls: calls))
        #expect(trace != nil && out.bitsPerComponent == 16 && out.width == image.width && out.height == image.height)
        let before = FillFixture.sixteenBit(image), after = FillFixture.sixteenBit(out)
        var differ = 0, offGrid = 0
        for i in 0..<(image.width * image.height) where mask.values[i] == 0 {
            for c in 0..<4 where before[i * 4 + c] != after[i * 4 + c] { differ += 1 }
            if after[i * 4] % 257 != 0 { offGrid += 1 }
        }
        #expect(differ == 0)
        #expect(offGrid > image.width * image.height / 2)                // the depth survived, not just the levels
    }

    @Test func anEightBitExpandComesBackOnTheLargerCanvas() async throws {
        let calls = FillCalls()
        let (out, trace) = try await EngineGenerativeFillProvider.expand(
            FillFixture.photo(), left: 64, top: 32, right: 64, bottom: 0, prompt: "", seed: 1,
            route: GenerativeFillRoute(maxWindowPixels: 768 * 768, multipleOf: 32),
            paint: FakeModel.painter(original: nil, content: (110, 130, 100), drift: 0, calls: calls),
            inpaint: FakeModel.lama(calls: calls))
        #expect(out.width == 768 && out.height == 512 && out.bitsPerComponent == 8)
        #expect(trace.seamBridged)
    }

    // MARK: LaMa's bound

    @Test func laMaRunsAtTheCanvasSizeUpToItsBoundAndScaledAboveIt() async throws {
        let calls = FillCalls()
        let route = GenerativeFillRoute(maxWindowPixels: 256 * 256, multipleOf: 32, inpaintMaxPixels: 300 * 300)
        // 640×480 is above both caps: the selection is long, so the window is the whole picture and the fallback's LaMa
        // runs scaled into its 300² bound.
        let image = FillPixels(FillFixture.photo())
        let mask = FillFixture.mask(width: 640, height: 480, box: FillBox(x: 20, y: 200, width: 600, height: 40))
        _ = try await route.fill(image, mask: mask, prompt: "", seed: 0, paint: FakeModel.keepsMarker(calls: calls),
                                 inpaint: FakeModel.lama(calls: calls))
        let lama = try #require(calls.inpaints.withLock { $0.first })
        #expect(lama.canvas.width * lama.canvas.height <= 300 * 300 && lama.hole.width == lama.canvas.width)
    }
}
