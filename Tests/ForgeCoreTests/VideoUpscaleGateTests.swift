//
//  VideoUpscaleGateTests.swift — ForgeCore
//
//  The admission arithmetic a UI turns into "you can run this" or "you cannot". Pure arithmetic, no
//  engine, no weights — which is the point: the gate has to be answerable before anything is loaded.
//
//  🔑 These pin the model from `GAP-PROGRAM.md` V12-D/V12-B: base residency plus ~0.82 GiB per extra
//  spatial tile position, capped by the bank residency budget when eviction is enabled.
//

import Foundation
import Testing

@testable import ForgeCore

@Suite("VideoUpscaleGate")
struct VideoUpscaleGateTests {

    static let gib = 1024.0 * 1024.0 * 1024.0

    /// One tile position is the base cost — no bank term at all.
    @Test func singleTileIsBaseResidency() {
        let (fits, predicted) = ForgeEngineService.videoUpscaleFits(
            outputTilePositions: 1, budgetBytes: UInt64(16 * Self.gib))
        #expect(fits)
        #expect(abs(Double(predicted) / Self.gib - 11.93) < 0.01)
    }

    /// 🔑 The measured anchor: 9 tile positions (a 512² output) came in at 17.95–18.16 GiB across four
    /// runs, and the model predicts 18.49. If this drifts, the receipt and the gate have diverged.
    @Test func nineTilesMatchesTheMeasuredAnchor() {
        let (_, predicted) = ForgeEngineService.videoUpscaleFits(
            outputTilePositions: 9, budgetBytes: UInt64(64 * Self.gib))
        let gib = Double(predicted) / Self.gib
        #expect(gib > 17.9 && gib < 18.6, "predicted \(gib) GiB is outside the measured 17.95–18.16 band")
    }

    /// A 16 GB machine must be told "no" for a 512² output — the honest refusal that keeps the tier
    /// from being offered where it would OOM.
    @Test func nineTilesDoesNotFitSixteenGigabytes() {
        let (fits, _) = ForgeEngineService.videoUpscaleFits(
            outputTilePositions: 9, budgetBytes: UInt64(14 * Self.gib))
        #expect(!fits)
    }

    /// 🔑 **Eviction changes the answer** — the same geometry that does not fit un-evicted fits with a
    /// bank budget, because the excess spills instead of accumulating.
    @Test func aBankBudgetMakesLargeGeometriesFit() {
        let positions = 35     // SD ×2
        let budget = UInt64(14 * Self.gib)

        let (fitsUnevicted, unevicted) = ForgeEngineService.videoUpscaleFits(
            outputTilePositions: positions, budgetBytes: budget)
        #expect(!fitsUnevicted)
        #expect(Double(unevicted) / Self.gib > 26.0, "SD ×2 un-evicted should be ~39 GiB")

        let (fitsEvicted, evicted) = ForgeEngineService.videoUpscaleFits(
            outputTilePositions: positions, budgetBytes: budget,
            bankResidencyBudgetBytes: Int(1.0 * Self.gib))
        #expect(fitsEvicted, "a 1 GiB bank budget should bring SD ×2 inside 14 GiB")
        #expect(evicted < unevicted)
    }

    /// A budget of 0 is legal and distinct from `nil`: everything spills, so the prediction collapses
    /// to the base cost.
    @Test func zeroBankBudgetCollapsesToBaseCost() {
        let (_, predicted) = ForgeEngineService.videoUpscaleFits(
            outputTilePositions: 170, budgetBytes: UInt64(128 * Self.gib),
            bankResidencyBudgetBytes: 0)
        #expect(abs(Double(predicted) / Self.gib - 11.93) < 0.01)
    }

    /// `bankBudgetToFit` is the inverse: it answers "with what budget?" rather than "does it fit?", so a
    /// host can offer eviction instead of a refusal.
    @Test func bankBudgetToFitRoundTripsThroughTheGate() throws {
        let positions = 35
        let budget = UInt64(14 * Self.gib)
        let suggested = try #require(
            ForgeEngineService.bankBudgetToFit(outputTilePositions: positions, budgetBytes: budget))
        #expect(suggested > 0)
        let (fits, _) = ForgeEngineService.videoUpscaleFits(
            outputTilePositions: positions, budgetBytes: budget,
            bankResidencyBudgetBytes: suggested)
        #expect(fits, "the budget bankBudgetToFit returned must actually fit")
    }

    /// ⚠️ When the BASE residency alone exceeds the budget, no bank budget helps — and the honest
    /// answer is `nil`, not a number that would still OOM.
    @Test func bankBudgetToFitRefusesWhenBaseAloneExceedsBudget() {
        #expect(ForgeEngineService.bankBudgetToFit(
            outputTilePositions: 9, budgetBytes: UInt64(8 * Self.gib)) == nil)
    }

    /// The budget suggested for a geometry never exceeds what that geometry's banks would occupy —
    /// suggesting more would be an instruction to reserve memory nothing will use.
    @Test func suggestedBudgetIsCappedByActualBankNeed() throws {
        let suggested = try #require(
            ForgeEngineService.bankBudgetToFit(outputTilePositions: 3,
                                               budgetBytes: UInt64(128 * Self.gib)))
        #expect(Double(suggested) / Self.gib <= 2 * 0.82 + 0.01,
                "3 positions have 2 extra banks; suggesting more than that reserves unusable memory")
    }
}
