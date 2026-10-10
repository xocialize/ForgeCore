//
//  ChainedUpscalePlanTests.swift
//  ForgeCoreTests
//
//  AB-D-0119 — ×6 / ×8 on tiers that stop at ×4. The plan is pure, so these need no engine: ≤ ×4 stays a single
//  pass; ×6 is ×4 → NERVE ×2 → Lanczos down to 6×; ×8 is ×4 → ×2 exactly; an intermediate over the cap falls back to
//  a named enlargement; the receipt name lists every model and the fit. Plus the resampler's exact-size contract.
//

import Testing
import CoreGraphics
import ForgeOptimizerKit
@testable import ForgeCore

@Suite struct ChainedUpscalePlanTests {
    typealias Plan = ChainedUpscalePlan
    let hd = Plan.Size(width: 1920, height: 1080)

    @Test func nativeFactorsAreASinglePass() {
        for requested in [1, 2, 3, 4] {
            let plan = Plan.make(requested: requested, source: hd)
            #expect(!plan.isChained)
            #expect(plan.primaryScale == requested)
            #expect(plan.secondaryScale == nil && plan.fit == nil && !plan.fitsUp)
            #expect(plan.outputSize(for: hd) == Plan.Size(width: 1920 * requested, height: 1080 * requested))
        }
    }

    @Test func sixIsFourThenTwoFittedDown() {
        let plan = Plan.make(requested: 6, source: hd)
        #expect(plan.isChained)
        #expect(plan.primaryScale == 4)
        #expect(plan.secondaryScale == 2)
        #expect(plan.fit == Plan.Size(width: 11520, height: 6480))
        #expect(!plan.fitsUp)
        #expect(plan.outputSize(for: hd) == Plan.Size(width: 11520, height: 6480))
    }

    @Test func eightIsFourThenTwoExactly() {
        let plan = Plan.make(requested: 8, source: hd)
        #expect(plan.primaryScale == 4 && plan.secondaryScale == 2)
        #expect(plan.fit == nil)
        #expect(plan.outputSize(for: hd) == Plan.Size(width: 15360, height: 8640))
    }

    @Test func pastEightIsServedAsEight() {
        let plan = Plan.make(requested: 12, source: hd)
        #expect(plan.requested == 8)
        #expect(plan.outputSize(for: hd) == Plan.Size(width: 15360, height: 8640))
    }

    /// A ×8 intermediate over the cap (a 4K source → 30720×17280 = 531 MP) is not built: one ×4 pass, then an
    /// enlargement to the target, and the plan says it enlarged.
    @Test func anIntermediateOverTheCapFallsBackToAnEnlargementAndSaysSo() {
        let uhd = Plan.Size(width: 3840, height: 2160)
        let plan = Plan.make(requested: 6, source: uhd)
        #expect(plan.primaryScale == 4)
        #expect(plan.secondaryScale == nil)
        #expect(plan.fit == Plan.Size(width: 23040, height: 12960))
        #expect(plan.fitsUp)
        // and the cap is the only reason: lift it and the chain comes back
        let lifted = Plan.make(requested: 6, source: uhd, intermediatePixelCap: .max)
        #expect(lifted.secondaryScale == 2 && !lifted.fitsUp)
    }

    @Test func theReceiptNamesEveryModelAndTheFit() {
        let six = Plan.make(requested: 6, source: hd)
        #expect(six.modelName(primaryName: "RealPLKSR", secondaryName: "NERVE") == "RealPLKSR ×4 + NERVE ×2 → ×6 (Lanczos)")
        let eight = Plan.make(requested: 8, source: hd)
        #expect(eight.modelName(primaryName: "NERVE", secondaryName: "NERVE") == "NERVE ×4 + NERVE ×2")
        // a guarded pass 1 keeps its composite name
        #expect(six.modelName(primaryName: "Nacre + NERVE (protected words)", secondaryName: "NERVE")
                == "Nacre + NERVE (protected words) ×4 + NERVE ×2 → ×6 (Lanczos)")
        let enlarged = Plan.make(requested: 6, source: Plan.Size(width: 3840, height: 2160))
        #expect(enlarged.modelName(primaryName: "RealPLKSR", secondaryName: "NERVE") == "RealPLKSR ×4 → ×6 (Lanczos, enlarged)")
    }

    /// The Kit's factors map to the integers the plan expects.
    @Test func kitFactorsMapToScales() {
        #expect(UpscaleFactor.none.requestedScale == nil)
        #expect(UpscaleFactor.x2.requestedScale == 2)
        #expect(UpscaleFactor.x4.requestedScale == 4)
        #expect(UpscaleFactor.x6.requestedScale == 6)
        #expect(UpscaleFactor.x8.requestedScale == 8)
    }

    /// The resampler lands on the exact size asked, down and up, and keeps a flat colour flat.
    @Test func lanczosLandsOnTheExactSize() throws {
        let src = try Self.solid(width: 64, height: 48, r: 200, g: 40, b: 90)
        let down = try ImageResampling.lanczos(src, toWidth: 48, height: 36)
        #expect(down.width == 48 && down.height == 36)
        let up = try ImageResampling.lanczos(src, toWidth: 96, height: 72)
        #expect(up.width == 96 && up.height == 72)
        let centre = try Self.pixel(down, x: 24, y: 18)
        #expect(abs(Int(centre.r) - 200) <= 2 && abs(Int(centre.g) - 40) <= 2 && abs(Int(centre.b) - 90) <= 2, "\(centre)")
        // same size is the identity
        #expect(try ImageResampling.lanczos(src, toWidth: 64, height: 48) === src)
    }

    static func solid(width: Int, height: Int, r: UInt8, g: UInt8, b: UInt8) throws -> CGImage {
        let ctx = try #require(CGContext(data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
                                         space: EngineImageCoding.bgraColorSpace, bitmapInfo: EngineImageCoding.bgraBitmapInfo))
        ctx.setFillColor(CGColor(srgbRed: CGFloat(r) / 255, green: CGFloat(g) / 255, blue: CGFloat(b) / 255, alpha: 1))
        ctx.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return try #require(ctx.makeImage())
    }

    static func pixel(_ image: CGImage, x: Int, y: Int) throws -> (r: UInt8, g: UInt8, b: UInt8) {
        let raw = try EngineImageCoding.rawImage(from: image)
        let row = raw.bytesPerRow ?? image.width * 4
        let i = y * row + x * 4   // BGRA
        return (raw.data[i + 2], raw.data[i + 1], raw.data[i])
    }
}
