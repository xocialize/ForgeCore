//
//  SenseNovaTextReader.swift
//  ForgeCore
//
//  MaskKit's `TextSecondReader` backed by the resident SenseNova-U1.5 package's analysis surface (VQA on the
//  understanding stream — bit-identical across the base and distilled tiers, AB-L-0236). It exists for the text
//  guard's WORD CONFIRMATION (`TextLegibilityGuard.confirm`): the certain Vision words U1 also read.
//
//  Measured 2026-10-15 (AB-R-0556; SA-Text 200-subsets at lv3 / lv1 / hq): U1 alone reads more of the truth than
//  Vision (recall 0.18 / 0.31 / 0.78 vs 0.09 / 0.21 / 0.59) and invents far more (precision 0.23 / 0.59 / 0.33 raw —
//  greedy repetition loops, describing instead of transcribing, plausible wrong words on illegible crops; 800
//  confabulated words on one low-contrast page). So it is NEVER the truth on its own and never reads document
//  pages for the guard. The words BOTH readers read are right 97–98 % at 85–90 % of Vision's recall, which is what
//  this reader is for.
//
//  Residency: U1 is a 15–22 GB package. This reader is only free where the package is already loaded (Generative
//  Fill); `resident(in:packageID:)` returns nil when it is not registered, and the guard then falls back to Vision's
//  verdict. Cost measured: 0.17–0.27 s per 128² crop, 4.4 s per 512² page (greedy, ≤ 256 tokens for crops).
//

import CoreGraphics
import Foundation
import MaskKit
import MLXServeCore
import MLXToolKit

public struct SenseNovaTextReader: TextSecondReader {

    /// The transcription prompt the measurement used. Line-per-line output; `NONE` is an empty read.
    public static let prompt =
        "Transcribe all the text in this image exactly as written, one line of the image per line of output. "
        + "Do not describe the image. If there is no text, write NONE."

    public enum ReaderError: Error { case unexpectedResponse }

    private let service: ForgeEngineService
    private let packageID: PackageID

    /// `packageID`: the U1 package the app registered (a Generative Fill tier's). Use `resident(in:packageID:)` to get
    /// nil instead when it is absent.
    public init(service: ForgeEngineService, packageID: PackageID) {
        self.service = service
        self.packageID = packageID
    }

    /// The reader when `packageID` is a registered package of `service`, else nil — the guard then runs on Vision alone.
    @MainActor
    public static func resident(in service: ForgeEngineService, packageID: PackageID?) -> SenseNovaTextReader? {
        guard let packageID else { return nil }
        return SenseNovaTextReader(service: service, packageID: packageID)
    }

    public func transcribe(_ image: CGImage) async throws -> [String] {
        let raw = try EngineImageCoding.rawImage(from: image)
        _ = try await service.engine.prepare(.imageAnalysis, package: packageID)
        let response = try await service.engine.run(
            ImageAnalysisRequest(image: raw, prompt: Self.prompt), package: packageID)
        guard let analysis = response as? ImageAnalysisResponse else { throw ReaderError.unexpectedResponse }
        return Self.lines(from: analysis.text)
    }

    /// The answer as lines: trimmed, empties and a bare `NONE` dropped. Degenerate repeats are left in — confirmation
    /// only removes words, so they cannot add a protected word.
    static func lines(from text: String) -> [String] {
        text.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty && $0.uppercased() != "NONE" }
    }
}
