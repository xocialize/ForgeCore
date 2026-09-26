//
//  GuardedUpscaleLiveTests.swift
//  ForgeCoreTests
//
//  [VAL] — the guarded generative stills path END TO END through the real engine: register → guard assesses
//  the input → prepare + run the tier(s) the verdict calls for → composite. Env-gated (needs the GPU, the
//  SeedVR2 snapshot and real pages), so it never runs in CI:
//
//      FORGE_GUARD_E2E_DIR=<dir of PNGs> FORGE_GUARD_E2E_OUT=<dir> [FORGE_GUARD_E2E_SCALE=2] \
//        swift test --build-system swiftbuild --filter GuardedUpscaleLiveTests
//
//  Pass 1 runs the default policy (every route opt-in → generative) and asserts only the generative tier ran.
//  Pass 2 opts into the document gate and asserts keepBase ⇔ the verdict's document page, base tier only.
//  Pass 3 opts into word protection on any page with certain words and asserts the composite route ran both
//  tiers and produced a mask. Every output and a JSON receipt land in FORGE_GUARD_E2E_OUT for the OCR check
//  outside (`Tools/vosr-gates/scripts/`), which is what decides whether a route earns its opt-in for a tier pair:
//  2026-09-26, SeedVR2-3B ×2 + Real-ESRGAN ×2, three ScreenSR pages — generative read best on all three.
//

import Testing
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers
import MaskKit
@testable import ForgeCore

@MainActor
@Suite struct GuardedUpscaleLiveTests {
    nonisolated static let env = ProcessInfo.processInfo.environment

    @Test(.enabled(if: ProcessInfo.processInfo.environment["FORGE_GUARD_E2E_DIR"] != nil))
    func guardedPathRunsEndToEnd() async throws {
        let inDir = URL(fileURLWithPath: Self.env["FORGE_GUARD_E2E_DIR"]!)
        let outDir = URL(fileURLWithPath: Self.env["FORGE_GUARD_E2E_OUT"] ?? inDir.appendingPathComponent("out").path)
        let scale = Self.env["FORGE_GUARD_E2E_SCALE"].flatMap(Int.init) ?? 2
        try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

        let service = ForgeEngineService()
        await service.registerAll()
        try #require(service.packageID(for: .upscale) != nil,
                     "Real-ESRGAN must register: \(service.outcome(for: .upscale)?.failure ?? "no failure recorded")")
        try #require(service.packageID(for: .upscaleQuality) != nil,
                     "SeedVR2 must register: \(service.outcome(for: .upscaleQuality)?.failure ?? "no failure recorded")")

        let files = try FileManager.default.contentsOfDirectory(at: inDir, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension.lowercased() == "png" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
        try #require(!files.isEmpty, "no PNGs in \(inDir.path)")

        var receipts: [[String: Any]] = []
        for file in files {
            let name = file.deletingPathExtension().lastPathComponent
            let image = try #require(Self.loadPNG(file), "could not decode \(file.lastPathComponent)")

            // Pass 1 — the defaults: every route is opt-in, so the generative tier alone.
            let guarded = EngineTextGuardedUpscaler(service: service)
            let out = try await guarded.upscale(image, scale: scale)
            #expect(out.image.width == image.width * scale && out.image.height == image.height * scale,
                    "\(name): output \(out.image.width)×\(out.image.height) for ×\(scale) of \(image.width)×\(image.height)")
            #expect(out.route == .generative, "\(name): the default policy routes generative, got \(out.route)")
            #expect(out.baseSeconds == 0 && out.generativeSeconds > 0, "\(name): generative must run the generative tier only")
            try Self.writePNG(out.image, to: outDir.appendingPathComponent("\(name)__generative.png"))
            receipts.append(Self.receipt(name: name, pass: "default", scale: scale, out: out))
            print("e2e: \(name) default → \(out.route.rawValue)  lines \(out.verdict.regions.count) certain-words \(out.verdict.protected.count) share \(String(format: "%.2f", out.verdict.certainLineShare)) docPage \(out.verdict.isDocumentPage)  assess \(String(format: "%.2f", out.assessSeconds))s gen \(String(format: "%.2f", out.generativeSeconds))s  out \(out.image.width)×\(out.image.height)")

            // Pass 2 — document gate opted in: a document page goes to the base tier alone.
            var gatePolicy = TextLegibilityGuard.Policy.default
            gatePolicy.documentGate = true
            let gated = EngineTextGuardedUpscaler(service: service, policy: gatePolicy)
            let out2 = try await gated.upscale(image, scale: scale)
            #expect((out2.route == .keepBase) == out2.verdict.isDocumentPage, "\(name): gate on → keepBase iff document page")
            if out2.route == .keepBase {
                #expect(out2.generativeSeconds == 0 && out2.baseSeconds > 0, "\(name): keepBase must run the base tier only")
                try Self.writePNG(out2.image, to: outDir.appendingPathComponent("\(name)__keepBase.png"))
            }
            receipts.append(Self.receipt(name: name, pass: "documentGate", scale: scale, out: out2))
            print("e2e: \(name) gate-on → \(out2.route.rawValue)  gen \(String(format: "%.2f", out2.generativeSeconds))s base \(String(format: "%.2f", out2.baseSeconds))s")

            // Pass 3 — word protection opted in (gate off so the composite route is exercised on dense pages too).
            if out.verdict.firesOnText {
                var policy = TextLegibilityGuard.Policy.default
                policy.wordProtection = true
                let optIn = EngineTextGuardedUpscaler(service: service, policy: policy)
                let out3 = try await optIn.upscale(image, scale: scale)
                #expect(out3.route == .protectWords, "\(name): opt-in with certain words must take the composite route, got \(out3.route)")
                #expect(out3.mask != nil && out3.generativeSeconds > 0 && out3.baseSeconds > 0, "\(name): composite route runs both tiers and yields a mask")
                #expect(out3.image.width == image.width * scale)
                try Self.writePNG(out3.image, to: outDir.appendingPathComponent("\(name)__protectWords.png"))
                if let mask = out3.mask { try Self.writePNG(mask, to: outDir.appendingPathComponent("\(name)__mask.png")) }
                receipts.append(Self.receipt(name: name, pass: "wordProtection", scale: scale, out: out3))
                print("e2e: \(name) opt-in → \(out3.route.rawValue)  protected \(out3.verdict.protected.count) words, coverage \(String(format: "%.3f", out3.verdict.protectedCoverage))  gen \(String(format: "%.2f", out3.generativeSeconds))s base \(String(format: "%.2f", out3.baseSeconds))s")
            }
        }
        let json = try JSONSerialization.data(withJSONObject: receipts, options: [.prettyPrinted, .sortedKeys])
        try json.write(to: outDir.appendingPathComponent("receipt.json"))
        print("e2e: \(receipts.count) runs → \(outDir.path)")
    }

    // MARK: - helpers

    static func receipt(name: String, pass: String, scale: Int, out: EngineTextGuardedUpscaler.Output) -> [String: Any] {
        ["image": name, "pass": pass, "scale": scale, "route": out.route.rawValue,
         "lines": out.verdict.regions.count, "certainWords": out.verdict.protected.count,
         "certainLineShare": out.verdict.certainLineShare, "isDocumentPage": out.verdict.isDocumentPage,
         "protectedCoverage": out.verdict.protectedCoverage,
         "assessSeconds": out.assessSeconds, "generativeSeconds": out.generativeSeconds, "baseSeconds": out.baseSeconds,
         "outputWidth": out.image.width, "outputHeight": out.image.height]
    }

    static func loadPNG(_ url: URL) -> CGImage? {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { return nil }
        return CGImageSourceCreateImageAtIndex(source, 0, nil)
    }

    static func writePNG(_ image: CGImage, to url: URL) throws {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
            throw CocoaError(.fileWriteUnknown)
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw CocoaError(.fileWriteUnknown) }
    }
}
