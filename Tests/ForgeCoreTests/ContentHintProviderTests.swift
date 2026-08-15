//
//  ContentHintProviderTests.swift — ForgeCore
//
//  The §6.3 planner-hint adapter (AB-T-0045). Live V-JEPA2 classification is not verifiable here —
//  it needs weights, a model store and the app's metallib boundary — so this suite pins the parts
//  that can be wrong *silently* and that no live run would catch:
//
//    1. **The inert clamp**, on every path including a head that scores 0.99. This is the whole
//       safety property of the chip: an uncalibrated head must not be able to steer the planner.
//    2. **The probe's arithmetic and sign convention.** A mirror-image probe scores plausibly and is
//       wrong on every item; a dimension mismatch silently truncates under `zip`. Both are the
//       "confident, plausible, wrong number" failure this codebase keeps paying for.
//    3. **The wiring shape** — `contentClassify` is in the registration pass, and ForgeCore's
//       provider satisfies the Kit's `ContentHintProvider` seam at `ForgeOptimizer`'s constructor.
//
//  Deliberately NOT asserted: that `contentClassify` registers on THIS machine. Registration runs
//  the license + device-eligibility gates, so its outcome is a fact about the host, not about the
//  code — the suite asserts the outcome is *recorded* and leaves the verdict to the machine.
//

import Foundation
import ForgeOptimizerKit
import MLXToolKit
import Testing

@testable import ForgeCore

@Suite("content hint adapter")
@MainActor
struct ContentHintProviderTests {

    // MARK: - The inert gate

    /// The clamp, stated as the number it must stay under.
    ///
    /// ⚠️ `HintedStartPolicy.minConfidence` is internal to `ForgeOptimizerKit`, so this cannot read
    /// the Kit's constant directly — it is mirrored here with the file that owns it named, and the
    /// Kit proves its own half (`ContentHintTests.testNonActionableHintsChangeNothing`).
    /// `ForgeOptimizerKit/Sources/ForgeOptimizerKit/ContentHint.swift`.
    static let kitMinConfidence = 0.9

    @Test("the inert ceiling sits below the Kit's action gate")
    func ceilingIsBelowTheKitGate() {
        #expect(EngineContentHintProvider.inertConfidenceCeiling < Self.kitMinConfidence)
    }

    @Test("with no head: general, at the ceiling, label carried verbatim")
    func noHeadObservesAndDoesNotSteer() {
        let hint = EngineContentHintProvider.hint(
            from: ContentClassifyResponse(labels: [ContentScore(label: "ssv2_113", score: 0.42)],
                                          embedding: [Float](repeating: 0.1, count: 1024)),
            verdict: nil)

        #expect(hint.contentClass == .general)
        #expect(hint.confidence == EngineContentHintProvider.inertConfidenceCeiling)
        // The tag travels into receipts untouched — it is provenance, never a routing input.
        #expect(hint.label == "ssv2_113")
    }

    /// The load-bearing one: a head that is *certain* still cannot raise a floor.
    @Test("a confident head is still clamped — certainty is not authority")
    func confidentHeadIsClamped() throws {
        // Weights and embedding both positive → a large positive logit → P(graphic) ≈ 1.
        let probe = GraphicProbe(weights: [Float](repeating: 0.5, count: 8), bias: 0,
                                 plattA: 1, plattB: 0, provenance: "AB-R-TEST")
        let verdict = try #require(probe.evaluate(embedding: [Float](repeating: 1.0, count: 8)))

        // Prove the clamp did the work rather than the probe being weak.
        #expect(verdict.confidence > Self.kitMinConfidence)
        #expect(verdict.contentClass == .graphic)

        let hint = EngineContentHintProvider.hint(
            from: ContentClassifyResponse(labels: [], embedding: []), verdict: verdict)
        #expect(hint.contentClass == .graphic)
        #expect(hint.confidence == EngineContentHintProvider.inertConfidenceCeiling)
        #expect(hint.confidence < Self.kitMinConfidence)
    }

    // MARK: - The probe

    /// Sign convention: a positive logit means `.graphic`. A fitter that hands over scikit-learn's
    /// `1/(1+exp(A·z+B))` coefficients without negating them produces the mirror image of this.
    @Test("positive logit means graphic, negative means general")
    func signConventionHoldsBothWays() throws {
        let probe = GraphicProbe(weights: [1, 1, 1, 1], bias: 0,
                                 plattA: 1, plattB: 0, provenance: "AB-R-TEST")

        let graphic = try #require(probe.evaluate(embedding: [1, 1, 1, 1]))
        #expect(graphic.margin == 4)
        #expect(graphic.contentClass == .graphic)
        #expect(graphic.probabilityGraphic > 0.5)

        let general = try #require(probe.evaluate(embedding: [-1, -1, -1, -1]))
        #expect(general.margin == -4)
        #expect(general.contentClass == .general)
        #expect(general.probabilityGraphic < 0.5)
        // Confidence is confidence in the REPORTED class, so both ends read the same distance out.
        #expect(abs(graphic.confidence - general.confidence) < 1e-9)
    }

    @Test("bias and Platt scale enter where they are documented to")
    func coefficientsEnterCorrectly() throws {
        // Zero embedding isolates the bias into the logit.
        let biased = GraphicProbe(weights: [0, 0], bias: 2, plattA: 1, plattB: 0,
                                  provenance: "AB-R-TEST")
        let v = try #require(biased.evaluate(embedding: [0, 0]))
        #expect(v.margin == 2)
        #expect(abs(v.probabilityGraphic - 1 / (1 + exp(-2.0))) < 1e-12)

        // Platt scaling moves the probability without moving the raw margin.
        let scaled = GraphicProbe(weights: [0, 0], bias: 2, plattA: 0.5, plattB: -1,
                                  provenance: "AB-R-TEST")
        let s = try #require(scaled.evaluate(embedding: [0, 0]))
        #expect(s.margin == 2)
        #expect(abs(s.probabilityGraphic - 0.5) < 1e-12)   // 0.5·2 − 1 = 0 → sigmoid(0)
    }

    /// `zip` truncates to the shorter sequence, so a probe fitted on a different checkpoint would
    /// score a *partial* dot product and return a confident wrong answer. A mismatch must be nil.
    @Test("a probe that does not match the embedding refuses rather than truncating")
    func dimensionMismatchRefuses() {
        let probe = GraphicProbe(weights: [Float](repeating: 1, count: 1024), bias: 0,
                                 plattA: 1, plattB: 0, provenance: "AB-R-TEST")
        #expect(probe.evaluate(embedding: [Float](repeating: 1, count: 768)) == nil)
        #expect(probe.evaluate(embedding: []) == nil)
        #expect(probe.evaluate(embedding: [Float](repeating: 1, count: 1024)) != nil)

        // An empty probe cannot score anything, including an empty embedding.
        let empty = GraphicProbe(weights: [], bias: 5, plattA: 1, plattB: 0, provenance: "x")
        #expect(empty.evaluate(embedding: []) == nil)
    }

    /// The loadable-parameters seam: fitted coefficients ship as data beside their receipt.
    @Test("probe round-trips through JSON with its provenance intact")
    func probeIsLoadableFromDisk() throws {
        let probe = GraphicProbe(weights: [0.25, -0.5, 0.125], bias: -0.75,
                                 plattA: 1.5, plattB: 0.25, provenance: "AB-R-0056")
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("probe-\(UUID().uuidString).json")
        try JSONEncoder().encode(probe).write(to: url)
        defer { try? FileManager.default.removeItem(at: url) }

        let loaded = try GraphicProbe(contentsOf: url)
        #expect(loaded == probe)
        // A head without provenance is a rumour — the receipt id must survive the trip.
        #expect(loaded.provenance == "AB-R-0056")
        #expect(loaded.evaluate(embedding: [1, 1, 1])?.margin == probe.evaluate(embedding: [1, 1, 1])?.margin)
    }

    // MARK: - Input inspection

    @Test("only containers the canonical Video artifact can express are accepted")
    func containerGuardMatchesTheArtifact() {
        #expect(EngineContentHintProvider.videoFormat(for: URL(fileURLWithPath: "/a/b.mp4")) == .mp4)
        #expect(EngineContentHintProvider.videoFormat(for: URL(fileURLWithPath: "/a/b.M4V")) == .mp4)
        #expect(EngineContentHintProvider.videoFormat(for: URL(fileURLWithPath: "/a/b.mov")) == .mov)
        // The Kit normalizes MKV/WebM to mp4 before the hint call, so these are a guard rather than
        // a limitation — but handing a still or a non-native container to a video contract is not.
        #expect(EngineContentHintProvider.videoFormat(for: URL(fileURLWithPath: "/a/b.mkv")) == nil)
        #expect(EngineContentHintProvider.videoFormat(for: URL(fileURLWithPath: "/a/b.png")) == nil)
        #expect(EngineContentHintProvider.videoFormat(for: URL(fileURLWithPath: "/a/b")) == nil)
    }

    /// An unavailable capability is silent, not fatal — the Kit's contract is that missing advice is
    /// normal. Constructed without `registerAll()`, so this pins the *unregistered* path specifically:
    /// no `PackageID`, therefore no hint, therefore the planner runs its behavioral ratchet.
    @Test("an unavailable capability yields no hint, not an error")
    func unavailableCapabilityIsSilent() async {
        let provider = EngineContentHintProvider(service: ForgeEngineService())
        #expect(!provider.isAvailable)
        let onVideo = await provider.classify(URL(fileURLWithPath: "/nonexistent/clip.mp4"))
        let onStill = await provider.classify(URL(fileURLWithPath: "/nonexistent/still.png"))
        #expect(onVideo == nil)
        #expect(onStill == nil)
    }

    // MARK: - Wiring shape

    /// `contentClassify` joins the eager registration pass, and its outcome is recorded whichever
    /// way the machine's gates fall.
    @Test("contentClassify is in the model table and the registration pass")
    func contentClassifyIsRegistered() async throws {
        #expect(ForgeModel.contentClassify.capability == .contentClassify)
        #expect(ForgeModel.contentClassify.modelName == "V-JEPA2")
        #expect(ForgeModel.allCases.contains(.contentClassify))

        let service = ForgeEngineService()
        await service.registerAll()
        // The outcome exists either way — `isRegistered` is a fact about this host (license +
        // device eligibility), so asserting it would make the suite a hardware test.
        let outcome = try #require(service.outcome(for: .contentClassify))
        #expect(outcome.model == .contentClassify)
        #expect(outcome.isRegistered == (outcome.failure == nil))
    }

    /// The cross-package seam: ForgeCore's provider is what `ForgeOptimizer` accepts for
    /// `hintProvider:`. A Kit floor regression below 0.11.0 un-compiles this line, which is the
    /// point — same guarantee `WebOptimizeWiringTests` gives `webOptimize`.
    @Test("the adapter satisfies the Kit's hintProvider seam")
    func adapterFitsTheKitSeam() {
        let caps = ForgeCapabilities(service: ForgeEngineService())
        _ = ForgeOptimizer(enhancer: caps.enhancer, hintProvider: caps.contentHint)
        _ = ForgeOptimizerService(enhancer: caps.enhancer, hintProvider: caps.contentHint)
    }
}
