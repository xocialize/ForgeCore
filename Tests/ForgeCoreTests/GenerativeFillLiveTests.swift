//
//  GenerativeFillLiveTests.swift
//  ForgeCoreTests
//
//  [VAL] — Generative Fill / Expand through the REAL engine (AB-A-0166). Env-gated (GPU + weights), never in CI.
//
//  1. Tier run — one tier on one fixture, timed cold then warm, `phys_footprint` sampled after every run. Its peak is
//     the PROCESS's lifetime peak, so run ONE configuration per process:
//
//      FORGE_FILL_LIVE=1 FORGE_FILL_STORE=<models dir> [FORGE_FILL_TIER=senseNova16GB|senseNova24GB|senseNova32GB|klein16GB]
//        [FORGE_FILL_IMAGE=<png> FORGE_FILL_MASK=<png> FORGE_FILL_PROMPT="…"] [FORGE_FILL_RUNS=3] [FORGE_FILL_SEED=7]
//        [FORGE_FILL_EXPAND=right:256] [FORGE_FILL_OUT=<dir>] \
//        swift test --build-system swiftbuild --filter GenerativeFillLiveTests/tierRun
//
//     Without an image it fills the drawn fixture. Outputs and `receipt-<tier>-<image>.json` land in FORGE_FILL_OUT.
//
//  2. Bake-off reproduction — the provider on the bake-off's own cases (pixel-agency-image-engine bench/fillbake), with
//     the bake-off's SenseNova arm (8step-v2-8bit) at the Image Engine's whole-canvas envelope (2048², so the window is
//     the whole picture as it was there) and seed 7, compared pixel for pixel with the Image Engine's output:
//
//      FORGE_FILL_PARITY=<…/bench/fillbake> FORGE_FILL_STORE=<models dir> [FORGE_FILL_PARITY_ITEMS=fill:stilts,extend:car]
//        swift test --build-system swiftbuild --filter GenerativeFillLiveTests/bakeOffReproduces
//

import Testing
import CoreGraphics
import Darwin
import Foundation
import ImageIO
import UniformTypeIdentifiers
@testable import ForgeCore

@MainActor
@Suite struct GenerativeFillLiveTests {
    nonisolated static let env = ProcessInfo.processInfo.environment

    static let tiers: [String: GenerativeFillTier] = [
        "senseNova16GB": .senseNova16GB, "senseNova24GB": .senseNova24GB, "senseNova32GB": .senseNova32GB, "klein16GB": .klein16GB,
    ]

    static func service(tier: GenerativeFillTier) async throws -> ForgeEngineService {
        let service = ForgeEngineService(generativeFillTier: tier)
        if let store = env["FORGE_FILL_STORE"] { await service.attachModelStore(root: URL(fileURLWithPath: store)) }
        await service.registerAll()
        try #require(service.packageID(for: .generativeFill) != nil,
                     "generativeFill must register: \(service.outcome(for: .generativeFill)?.failure ?? "?")")
        try #require(service.packageID(for: .inpaint) != nil, "LaMa must register")
        print("fill-live: MLX pool cap applied by the engine: \(service.engine.appliedGPUCacheLimitBytes.map { String(format: "%.2f GB", Double($0) / 1e9) } ?? "none")")
        return service
    }

    // MARK: - 1. Tier run

    @Test(.enabled(if: ProcessInfo.processInfo.environment["FORGE_FILL_LIVE"] != nil))
    func tierRun() async throws {
        let tierName = Self.env["FORGE_FILL_TIER"] ?? "senseNova24GB"
        let tier = try #require(Self.tiers[tierName], "unknown FORGE_FILL_TIER \(tierName)")
        let image: CGImage, mask: CGImage, prompt: String, name: String
        if let path = Self.env["FORGE_FILL_IMAGE"] {
            image = try #require(Self.load(URL(fileURLWithPath: path)))
            let maskPath = try #require(Self.env["FORGE_FILL_MASK"], "FORGE_FILL_IMAGE needs FORGE_FILL_MASK")
            mask = try #require(Self.load(URL(fileURLWithPath: maskPath)))
            prompt = Self.env["FORGE_FILL_PROMPT"] ?? ""
            name = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        } else {
            image = FillFixture.photo(width: 1024, height: 1024)
            mask = FillFixture.mask(width: 1024, height: 1024, box: FillBox(x: 560, y: 600, width: 220, height: 160), feather: 6).cgImage
            prompt = Self.env["FORGE_FILL_PROMPT"] ?? "a red wooden bench"
            name = "fixture-1024"
        }
        let runs = Int(Self.env["FORGE_FILL_RUNS"] ?? "3") ?? 3
        let seed = UInt64(Self.env["FORGE_FILL_SEED"] ?? "7") ?? 7
        let out = URL(fileURLWithPath: Self.env["FORGE_FILL_OUT"] ?? FileManager.default.temporaryDirectory.appendingPathComponent("fill-live").path)
        try FileManager.default.createDirectory(at: out, withIntermediateDirectories: true)

        let baseline = Self.physFootprint()
        let service = try await Self.service(tier: tier)
        let provider = EngineGenerativeFillProvider(service: service)
        print("fill-live: \(provider.name) on \(name) \(image.width)×\(image.height), prompt \"\(prompt)\"")

        var rows: [[String: Any]] = []
        for run in 0..<runs {
            let result = try await provider.fillReporting(image, mask: mask, prompt: prompt, seed: seed)
            let phys = Self.physFootprint()
            #expect(result.image.width == image.width && result.image.height == image.height)
            #expect(result.report.weightLicense == "Apache-2.0" && result.report.portCodeLicense == "MIT")
            let trace = try #require(result.report.trace)
            try Self.writePNG(result.image, to: out.appendingPathComponent("\(tierName)-\(name)-fill-\(run).png"))
            print(String(format: "fill-live: run %d %@ %.2f s · window %dx%d@%d,%d → model %dx%d · seed %@ retries %d fallback %@ · phys %.2f GB, peak %.2f GB",
                         run, run == 0 ? "cold" : "warm", result.report.seconds, trace.window.box.width, trace.window.box.height,
                         trace.window.box.x, trace.window.box.y, trace.window.modelWidth, trace.window.modelHeight,
                         trace.seed.map(String.init) ?? "-", trace.retries, trace.fellBackToInpaint ? "LaMa" : "no",
                         Double(phys?.current ?? 0) / 1e9, Double(phys?.peak ?? 0) / 1e9))
            rows.append(["operation": "fill", "run": run, "cold": run == 0, "seconds": result.report.seconds,
                         "physGB": Double(phys?.current ?? 0) / 1e9, "physPeakGB": Double(phys?.peak ?? 0) / 1e9,
                         "report": try JSONSerialization.jsonObject(with: JSONEncoder().encode(result.report))])
        }
        if let spec = Self.env["FORGE_FILL_EXPAND"] {
            let parts = spec.split(separator: ":"), px = Int(parts.last ?? "") ?? 256
            let side = parts.first.map(String.init) ?? "right"
            let result = try await provider.expandReporting(image, left: side == "left" ? px : 0, top: side == "top" ? px : 0,
                                                            right: side == "right" ? px : 0, bottom: side == "bottom" ? px : 0,
                                                            prompt: "", seed: seed)
            let phys = Self.physFootprint()
            try Self.writePNG(result.image, to: out.appendingPathComponent("\(tierName)-\(name)-expand-\(side)\(px).png"))
            let trace = try #require(result.report.trace)
            print(String(format: "fill-live: expand %@ %d %.2f s · canvas %dx%d → model %dx%d · seam %@ fallback %@ · phys peak %.2f GB",
                         side, px, result.report.seconds, trace.window.box.width, trace.window.box.height, trace.window.modelWidth,
                         trace.window.modelHeight, trace.seamBridged ? "LaMa" : "no", trace.fellBackToInpaint ? "LaMa" : "no",
                         Double(phys?.peak ?? 0) / 1e9))
            rows.append(["operation": "expand", "seconds": result.report.seconds, "physPeakGB": Double(phys?.peak ?? 0) / 1e9,
                         "report": try JSONSerialization.jsonObject(with: JSONEncoder().encode(result.report))])
        }
        let receipt: [String: Any] = [
            "tier": tierName, "provider": provider.name, "image": name, "width": image.width, "height": image.height,
            "prompt": prompt, "baselinePhysGB": Double(baseline?.current ?? 0) / 1e9,
            "poolCapGB": service.engine.appliedGPUCacheLimitBytes.map { Double($0) / 1e9 } ?? NSNull(), "runs": rows,
        ]
        try JSONSerialization.data(withJSONObject: receipt, options: [.prettyPrinted, .sortedKeys])
            .write(to: out.appendingPathComponent("receipt-\(tierName)-\(name).json"))
    }

    // MARK: - 2. Bake-off reproduction

    @Test(.enabled(if: ProcessInfo.processInfo.environment["FORGE_FILL_PARITY"] != nil))
    func bakeOffReproduces() async throws {
        let bench = URL(fileURLWithPath: Self.env["FORGE_FILL_PARITY"]!)
        // The bake-off's arm, at the Image Engine's SenseNova envelope (mlx-image-runtime: maxPixels 2048², multipleOf 32).
        let arm = GenerativeFillTier(label: "bake-off arm", model: .senseNova(variant: .fast8V2), maxWindowPixels: 2048 * 2048)
        let service = try await Self.service(tier: arm)
        let provider = EngineGenerativeFillProvider(service: service)
        let cases = try JSONSerialization.jsonObject(with: Data(contentsOf: bench.appendingPathComponent("cases.json"))) as! [String: Any]
        var outputs: [String: String] = [:]                              // "suite:item" → the Image Engine's output
        for file in ["run-fill", "run-rest"] {
            let lines = try String(contentsOf: bench.appendingPathComponent("\(file)/results.jsonl"), encoding: .utf8).split(separator: "\n")
            for line in lines {
                guard let row = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                      row["model"] as? String == "sensenova-u1-fast", let suite = row["suite"] as? String,
                      let item = row["item"] as? String, let out = (row["outputs"] as? [String])?.first else { continue }
                outputs["\(suite):\(item)"] = bench.appendingPathComponent("\(file)/\(out)").path
            }
        }
        let items = (Self.env["FORGE_FILL_PARITY_ITEMS"] ?? "fill:stilts,extend:car").split(separator: ",").map(String.init)
        for key in items {
            let (suite, item) = (String(key.split(separator: ":")[0]), String(key.split(separator: ":")[1]))
            let entry = try #require((cases[suite] as? [[String: Any]])?.first { $0["id"] as? String == item }, "no case \(key)")
            let reference = try #require(outputs[key].flatMap { Self.load(URL(fileURLWithPath: $0)) }, "no Image Engine output for \(key)")
            let image = try #require(Self.load(URL(fileURLWithPath: entry["image"] as! String)))
            let result: GenerativeFillResult
            if suite == "extend" {
                let c = entry["canvas"] as! [String: Int]
                result = try await provider.expandReporting(image, left: c["left"]!, top: c["top"]!, right: c["right"]!,
                                                            bottom: c["bottom"]!, prompt: "", seed: 7)
            } else {
                let mask = try #require(Self.load(URL(fileURLWithPath: entry["mask"] as! String)))
                result = try await provider.fillReporting(image, mask: mask, prompt: entry["prompt"] as? String ?? "", seed: 7)
            }
            let (a, b) = (FillPixels(result.image), FillPixels(reference))
            try #require(a.width == b.width && a.height == b.height)
            var differing = 0, maxDiff = 0, total = 0
            for i in 0..<(a.width * a.height) {
                var d = 0
                for c in 0..<3 { d = max(d, abs(Int(a.rgba[i * 4 + c]) - Int(b.rgba[i * 4 + c]))) }
                if d > 0 { differing += 1 }
                maxDiff = max(maxDiff, d); total += d
            }
            let trace = result.report.trace
            print(String(format: "fill-parity: %@ %dx%d %.2f s · window %@ model %dx%d · differing pixels %d of %d (%.4f%%), max |Δ| %d, mean |Δ| %.5f",
                         key, a.width, a.height, result.report.seconds, trace.map { "\($0.window.box.width)x\($0.window.box.height)" } ?? "-",
                         trace?.window.modelWidth ?? 0, trace?.window.modelHeight ?? 0, differing, a.width * a.height,
                         100 * Double(differing) / Double(a.width * a.height), maxDiff, Double(total) / Double(a.width * a.height)))
            if let out = Self.env["FORGE_FILL_OUT"] {
                try FileManager.default.createDirectory(atPath: out, withIntermediateDirectories: true)
                try Self.writePNG(result.image, to: URL(fileURLWithPath: out).appendingPathComponent("parity-\(suite)-\(item).png"))
            }
            #expect(differing == 0, "\(key): the provider should reproduce the Image Engine's bake-off output exactly")
        }
    }

    // MARK: - Helpers

    /// The process's `phys_footprint` now, and its lifetime peak.
    static func physFootprint() -> (current: UInt64, peak: UInt64)? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<natural_t>.size)
        let kr = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        guard kr == KERN_SUCCESS else { return nil }
        return (info.phys_footprint, UInt64(info.ledger_phys_footprint_peak))
    }

    static func load(_ url: URL) -> CGImage? {
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
