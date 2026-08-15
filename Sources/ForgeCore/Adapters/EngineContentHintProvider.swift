//
//  EngineContentHintProvider.swift — ForgeCore
//
//  ForgeOptimizerKit's §6.3 `ContentHintProvider` seam, engine-backed: V-JEPA2 `contentClassify`
//  → the [1024] router embedding → a calibrated graphic-vs-general head → `ContentHint`.
//
//  🚨 **THIS ADAPTER SHIPS INERT AND THAT IS THE POINT.** Every hint it returns has its confidence
//  clamped to `inertConfidenceCeiling` (0.5), which sits below the Kit's `HintedStartPolicy`
//  .minConfidence (0.9) — so the Kit observes each classification and acts on none of it. The clamp
//  is unconditional in this version: no initializer parameter, no environment variable, no flag an
//  app can flip. Unlocking is a visible, reviewable edit to this file, gated on the calibration
//  receipt existing (AB-T-0045 step 4).
//
//  Why so blunt: LESSONS' calibration rule is *"the search's landing IS a content classifier —
//  behavior outranks the axis label"*. The Kit already enforces that structurally (confidence gate,
//  post-hoc behavioral audit, disagreement receipts) — but an adapter reporting `confidence: 0.95`
//  from an unfitted head would be laundering a guess through a safety valve. A number the 0.9 gate
//  can be honest about has to be earned against encoder behavior first.
//
//  Full design + the calibration protocol: `Docs/VJEPA2-HINT-ADAPTER.md`.
//

import Foundation
import ForgeOptimizerKit
import MLXServeCore
import MLXToolKit

/// One classification's raw output, for whatever is collecting calibration rows.
///
/// The `ContentHint` the planner receives is deliberately narrow (class + confidence + label), so
/// the **feature vector never reaches the caller through it**. This is the tap that makes AB-T-0045
/// step 4 reachable without re-running the model: a harness injects a sink, optimizes the corpus,
/// and pairs each embedding with that item's behavioral receipt (`quality_floor` /
/// `floor_raised_from` / `content_class`) to build the training rows.
public struct ContentHintObservation: Sendable {
    /// The clip classified — the join key against the optimize receipt for the same item.
    public let url: URL
    /// V-JEPA2's pooled clip embedding, [1024] for the ViT-L checkpoint. **The signal.**
    public let embedding: [Float]
    /// The package's top-k tags, carried verbatim as provenance. ⚠️ `ssv2_<index>` ids from a
    /// hand-object action vocabulary — never interpret them, never map them to a content class.
    public let labels: [ContentScore]
    /// The head's raw logit, when a head was supplied. `nil` while no probe is loaded — which is
    /// the shipping state, and is why the embedding above is the row that matters.
    public let probeMargin: Double?
}

/// When the adapter is willing to pay for a classification.
public enum ContentHintLoadPolicy: Sendable {
    /// Classify every time, loading V-JEPA2 if it is not resident. The mode a calibration harness
    /// wants — it is collecting embeddings, and the load is the price of the row. Across a bulk
    /// `AsyncStream` run the load amortizes over the batch.
    case always
    /// Classify only when V-JEPA2 is **already resident**. `VJEPA2-HINT-ADAPTER.md`'s footprint
    /// rule: *"Don't load a ViT to save one search on a cold engine"* — ~1.2 GB fp32 resident to
    /// save at most one search is a bad trade for a one-off, and the behavioral ratchet covers that
    /// case for free. ⚠️ Nothing else in Forge uses `contentClassify`, so under this policy a hint
    /// fires only after something has warmed the model.
    case whenResident
}

/// V-JEPA2-backed pre-classification for the Kit's planner-hint seam.
///
/// Shape follows the house adapter pattern (`EngineFrameRestoreProvider`): the `@MainActor` init
/// resolves the engine + `PackageID` once, and `classify` is a plain `nonisolated` async call on a
/// `Sendable` value — the Kit calls it from its own executor, per item.
public struct EngineContentHintProvider: ContentHintProvider {

    /// The ceiling every reported confidence passes through while the head is uncalibrated.
    ///
    /// 🔑 **0.5 < 0.9 = `HintedStartPolicy.minConfidence`** (`ForgeOptimizerKit/ContentHint.swift`),
    /// so `HintedStartPolicy.startingFloor` returns nil for every hint this adapter can produce and
    /// the planner runs exactly as if no provider were injected. The Kit proves that end of the
    /// contract itself (`ContentHintTests.testNonActionableHintsChangeNothing`); this end is pinned
    /// by `ContentHintProviderTests`.
    ///
    /// ⚠️ Raising or removing this is the LAST step of AB-T-0045, not a tuning knob: it requires a
    /// fitted probe **and** a filed receipt carrying the held-out disagreement rate.
    public static let inertConfidenceCeiling: Double = 0.5

    /// Refuse inputs larger than this rather than pay a full copy for advice.
    ///
    /// ⚠️ **The `contentClassify` contract takes bytes, not a URL** — `ContentClassifyRequest`
    /// carries a canonical `Video` artifact, and the package writes those bytes back out to a temp
    /// file to run AVFoundation's frame sampler over them. So a classification costs one read plus
    /// one full disk copy of the source, while the sampling it enables is ~10² ms. This adapter maps
    /// the file rather than reading it resident, which keeps the *memory* honest; the temp copy is
    /// the package's contract and is filed as a follow-up ask against the engine/video area. Until
    /// that contract learns a URL form, a ceiling is the honest mitigation: above it there is no
    /// hint, which the Kit already treats as a normal outcome. 2 GiB clears every corpus master
    /// (largest: 429 MB) by a wide margin.
    public static let defaultMaxInputBytes = 2 << 30

    private let engine: MLXServeEngine
    private let packageID: PackageID?
    private let head: GraphicProbe?
    private let loadPolicy: ContentHintLoadPolicy
    private let maxInputBytes: Int
    private let observe: (@Sendable (ContentHintObservation) -> Void)?

    /// - Parameters:
    ///   - service: the registered engine service. A `contentClassify` registration that failed
    ///     (license, eligibility) leaves `packageID` nil and every `classify` returns nil — the
    ///     capability is simply absent, which is a fact to show, not an error to retry.
    ///   - head: the fitted graphic-vs-general probe over the router embedding. **`nil` is the
    ///     shipping state.** Supplying one today changes the reported *class*, never the reported
    ///     confidence — see `inertConfidenceCeiling`.
    ///   - loadPolicy: whether a cold V-JEPA2 is worth loading. See `ContentHintLoadPolicy`.
    ///   - maxInputBytes: see `defaultMaxInputBytes`.
    ///   - observe: calibration tap. Called once per successful classification, off the Kit's
    ///     critical path in the sense that it may not throw and must not block — a harness should
    ///     append a row and return.
    @MainActor
    public init(service: ForgeEngineService,
                head: GraphicProbe? = nil,
                loadPolicy: ContentHintLoadPolicy = .always,
                maxInputBytes: Int = EngineContentHintProvider.defaultMaxInputBytes,
                observe: (@Sendable (ContentHintObservation) -> Void)? = nil) {
        self.engine = service.engine
        self.packageID = service.packageID(for: .contentClassify)
        self.head = head
        self.loadPolicy = loadPolicy
        self.maxInputBytes = maxInputBytes
        self.observe = observe
    }

    /// Whether this provider can produce a hint at all — i.e. whether `contentClassify` registered.
    /// A consumer showing capability state should read this rather than inferring it from silence.
    public var isAvailable: Bool { packageID != nil }

    // MARK: - ContentHintProvider

    /// Non-throwing by contract: a hint is advice, and missing advice is not an error the optimize
    /// should surface. Every failure path — unregistered, unsupported container, oversized input,
    /// decode failure, engine error — returns nil, and the Kit behaves exactly as if no provider
    /// were injected (its behavioral ratchet is the fallback, and it stays armed regardless).
    public func classify(_ url: URL) async -> ContentHint? {
        guard let packageID else { return nil }
        guard let format = Self.videoFormat(for: url) else {
            // Non-native containers reach the Kit's hint call already normalized to mp4, so this is
            // a guard rather than a limitation — but a still, an audio file or an MKV that skipped
            // normalization must not be handed to a video contract.
            log.debug("content hint skipped — unsupported container for \(url.lastPathComponent, privacy: .public)")
            return nil
        }
        guard let size = Self.fileSize(url), size <= maxInputBytes else {
            log.notice("content hint skipped — \(url.lastPathComponent, privacy: .public) exceeds the \(maxInputBytes, privacy: .public)-byte ceiling for a bytes-carrying classify")
            return nil
        }
        if case .whenResident = loadPolicy {
            guard await engine.residentPackages[packageID] != nil else { return nil }
        }

        do {
            let data = try Data(contentsOf: url, options: .mappedIfSafe)
            _ = try await engine.prepare(.contentClassify, package: packageID)
            let response = try await engine.run(ContentClassifyRequest(video: Video(format: format, data: data)),
                                                package: packageID)
            guard let classified = response as? ContentClassifyResponse else {
                log.error("content hint: unexpected response type for \(url.lastPathComponent, privacy: .public)")
                return nil
            }
            let verdict = head?.evaluate(embedding: classified.embedding)
            observe?(ContentHintObservation(url: url,
                                            embedding: classified.embedding,
                                            labels: classified.labels,
                                            probeMargin: verdict?.margin))
            return Self.hint(from: classified, verdict: verdict)
        } catch is CancellationError {
            return nil
        } catch {
            // Absent advice is not an error — the behavioral ratchet covers it.
            log.notice("content hint unavailable for \(url.lastPathComponent, privacy: .public): \(String(describing: error), privacy: .public)")
            return nil
        }
    }

    // MARK: - Response → hint (pure; the part the tests pin)

    /// Map one classification to the planner's hint, applying the inert clamp.
    ///
    /// Two things are load-bearing here and both are asserted by the tests:
    /// 1. **The clamp is applied last and applies to every path**, probe or no probe. A head that
    ///    evaluates to 0.99 still reports `inertConfidenceCeiling`.
    /// 2. **With no head the class is `.general`.** Not "unknown", not a guess from the tags — the
    ///    Kit documents `.general` as the class that never changes planner behavior, so it is the
    ///    correct way to say "observed, no opinion".
    static func hint(from response: ContentClassifyResponse,
                     verdict: GraphicProbe.Verdict?) -> ContentHint {
        let label = response.labels.first?.label
        guard let verdict else {
            return ContentHint(contentClass: .general,
                               confidence: inertConfidenceCeiling,
                               label: label)
        }
        return ContentHint(contentClass: verdict.contentClass,
                           confidence: min(verdict.confidence, inertConfidenceCeiling),
                           label: label)
    }

    // MARK: - Input inspection

    /// The canonical `Video` format for a URL, or nil when the artifact cannot express it.
    /// `Video.Format` is mp4/mov only; `.m4v` is an MP4-family container and rides the mp4 case.
    static func videoFormat(for url: URL) -> Video.Format? {
        switch url.pathExtension.lowercased() {
        case "mp4", "m4v": return .mp4
        case "mov", "qt":  return .mov
        default:           return nil
        }
    }

    static func fileSize(_ url: URL) -> Int? {
        try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize
    }
}

/// The calibrated graphic-vs-general head over V-JEPA2's router embedding — a logistic probe,
/// Platt-scaled so the confidence it reports is a probability the Kit's 0.9 gate could be honest
/// about.
///
/// 🚨 **No fitted probe exists yet, and one existing is not sufficient to unlock anything** — see
/// `EngineContentHintProvider.inertConfidenceCeiling`. This type is the loadable-parameters seam so
/// that fitting (AB-T-0045 step 4) has somewhere to land: fit offline against the behavioral receipt
/// rows, ship the coefficients as data beside the calibration receipt, load with `init(contentsOf:)`.
///
/// ⚠️ **Sign convention, stated because getting it wrong silently inverts the classifier** — a
/// mirror-image probe scores plausibly and is wrong on every item:
/// - `z = bias + Σ wᵢ·eᵢ` is the raw logit, **positive toward `.graphic`**;
/// - `P(graphic) = sigmoid(plattA·z + plattB)`, so a fitter using scikit-learn's
///   `1/(1+exp(A·z+B))` convention must negate both `A` and `B` on the way in.
public struct GraphicProbe: Sendable, Codable, Equatable {
    /// One evaluation, in the terms the caller needs: a class for the planner, a confidence for the
    /// gate, and the raw logit for the calibration fitter's x-axis.
    public struct Verdict: Sendable, Equatable {
        public let contentClass: ContentHint.ContentClass
        /// P(graphic) after Platt scaling, in [0, 1].
        public let probabilityGraphic: Double
        /// Confidence **in the reported class** — `max(p, 1−p)`, which is what the Kit's gate means
        /// by confidence.
        public var confidence: Double { max(probabilityGraphic, 1 - probabilityGraphic) }
        /// The pre-Platt logit. Not a probability; do not gate on it.
        public let margin: Double
    }

    /// [1024] for the ViT-L checkpoint. Its count is the dimension check.
    public let weights: [Float]
    public let bias: Float
    public let plattA: Float
    public let plattB: Float
    /// The calibration receipt id this head came from. **A head without provenance is a rumour** —
    /// the whole reason the confidence gate exists is that an unattributed number cannot be audited.
    public let provenance: String

    public init(weights: [Float], bias: Float, plattA: Float, plattB: Float, provenance: String) {
        self.weights = weights
        self.bias = bias
        self.plattA = plattA
        self.plattB = plattB
        self.provenance = provenance
    }

    /// Load fitted coefficients from JSON.
    public init(contentsOf url: URL) throws {
        self = try JSONDecoder().decode(GraphicProbe.self, from: Data(contentsOf: url))
    }

    /// Score one router embedding, or nil when the embedding does not match this probe.
    ///
    /// 🔑 **The dimension check is not defensive boilerplate.** `zip` truncates to the shorter
    /// sequence, so a probe fitted on a different checkpoint's embedding would silently score a
    /// partial dot product and return a confident, plausible, wrong answer — the exact failure mode
    /// this codebase keeps paying for. A mismatch is nil: no hint, ratchet as usual.
    public func evaluate(embedding: [Float]) -> Verdict? {
        guard embedding.count == weights.count, !weights.isEmpty else { return nil }
        // Accumulate in Double: 1024 Float products summed in Float loses low-order bits near the
        // decision boundary, which is precisely where the class flips.
        var z = Double(bias)
        for i in 0..<weights.count {
            z += Double(weights[i]) * Double(embedding[i])
        }
        let p = 1 / (1 + exp(-(Double(plattA) * z + Double(plattB))))
        return Verdict(contentClass: p >= 0.5 ? .graphic : .general,
                       probabilityGraphic: p,
                       margin: z)
    }
}
