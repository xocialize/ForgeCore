//
//  WeightMaterializationLiveTests.swift
//  ForgeCoreTests
//
//  [VAL] — engine-owned weight delivery through ForgeCore's own registration path (AB-T-0196 / AB-T-0197): attach a
//  store, `registerAll()`, prepare each named model, and record the `PreparePhase` sequence the engine reports while it
//  runs. A package lane can prove `.downloading` on its own; this proves it for the configuration ForgeCore actually
//  registers, into the store ForgeCore actually attaches. Env-gated (network + GPU), so it never runs in CI:
//
//      FORGE_MAT_E2E_STORE=<dir on a volume with room> FORGE_MAT_E2E_MODELS=qualityScore,upscaleQuality \
//        [FORGE_MAT_E2E_EXPECT_DOWNLOAD=1] swift test --build-system swiftbuild --filter WeightMaterializationLiveTests
//
//  Per model it asserts the prepare reached `.ready`. With FORGE_MAT_E2E_EXPECT_DOWNLOAD=1 (an EMPTY store) it also asserts
//  that `.downloading` was observed and that a new `models--<org>--<name>` directory carrying the engine's
//  `mlx-package.json` marker landed in the store. With FORGE_MAT_E2E_EXPECT_DOWNLOAD=0 (a populated or legacy-layout
//  store) it asserts that NO download happened — the adopt-don't-re-download contract. Unset, it only reports. Each
//  prepared model then runs once on a small gradient, so "ready" means "runs": a no-reference score in 0…1 for
//  `.qualityScore`, a ×2 image of the right size for an `imageUpscale` tier.
//

import Testing
import CoreGraphics
import Foundation
import MLXServeCore
import MLXToolKit
@testable import ForgeCore

@MainActor
@Suite struct WeightMaterializationLiveTests {
    nonisolated static let env = ProcessInfo.processInfo.environment

    @MainActor final class Flag { var done = false }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["FORGE_MAT_E2E_STORE"] != nil))
    func registeredModelsMaterializeIntoTheAttachedStore() async throws {
        let store = URL(fileURLWithPath: Self.env["FORGE_MAT_E2E_STORE"]!)
        try FileManager.default.createDirectory(at: store, withIntermediateDirectories: true)
        let names = (Self.env["FORGE_MAT_E2E_MODELS"] ?? "qualityScore").split(separator: ",").map(String.init)
        let models = try names.map { try #require(ForgeModel(rawValue: $0), "unknown ForgeModel '\($0)'") }
        let expectDownload: Bool? = Self.env["FORGE_MAT_E2E_EXPECT_DOWNLOAD"].map { $0 == "1" }

        let service = ForgeEngineService()
        #expect(await service.attachModelStore(root: store))
        await service.registerAll()

        for model in models {
            let id = try #require(service.packageID(for: model),
                                  "\(model.modelName) did not register: \(service.outcome(for: model)?.failure ?? "no failure recorded")")
            let before = Self.markedRepoDirs(in: store)

            // Prepare on a child task and sample the engine's monitor until it returns.
            let flag = Flag()
            let prep = Task { @MainActor in
                defer { flag.done = true }
                _ = try await service.engine.prepare(model.capability, package: id)
            }
            var kinds: [String] = []
            var downloadPeak = 0.0
            let start = Date()
            while !flag.done {
                let phase = service.engine.preparation.phase(for: model.capability, package: id.rawValue)
                let kind = Self.kind(phase)
                if kinds.last != kind { kinds.append(kind) }
                if case .downloading(let fraction, _) = phase { downloadPeak = max(downloadPeak, fraction) }
                try await Task.sleep(for: .milliseconds(20))
            }
            try await prep.value
            let final = service.engine.preparation.phase(for: model.capability, package: id.rawValue)
            if kinds.last != Self.kind(final) { kinds.append(Self.kind(final)) }
            let after = Self.markedRepoDirs(in: store)
            let landed = after.subtracting(before).sorted()
            let sawDownload = kinds.contains("downloading")
            print("mat-e2e: \(model.modelName) [\(id.rawValue)] phases \(kinds.joined(separator: " → "))  "
                  + "download peak \(String(format: "%.2f", downloadPeak))  new store dirs \(landed)  "
                  + String(format: "prepare %.1f s", Date().timeIntervalSince(start)))

            #expect(Self.kind(final) == "ready", "\(model.modelName) ended \(final)")
            if expectDownload == true {
                #expect(sawDownload, "\(model.modelName): no .downloading phase into an empty store")
                #expect(!landed.isEmpty, "\(model.modelName): nothing new with a marker landed in the store")
            } else if expectDownload == false {
                #expect(!sawDownload, "\(model.modelName): downloaded although the store already held its weights")
            }

            // Run it once, so "ready" means "runs".
            let probe = try Self.gradient(64, 64)
            switch model.capability {
            case .imageQualityScore:
                let score = try await service.scoreNoReferenceQuality(probe)
                print("mat-e2e: \(model.modelName) score \(score)")
                #expect((0.0...1.0).contains(score), "\(model.modelName) score \(score)")
            case .imageUpscale:
                let m = try await service.measureUpscale(probe, model: model, scale: 2)
                print("mat-e2e: \(model.modelName) ×2 → \(m.image.width)×\(m.image.height)")
                #expect(m.image.width == 128 && m.image.height == 128, "\(model.modelName): \(m.image.width)×\(m.image.height)")
            default:
                break
            }
        }
    }

    static func kind(_ phase: PreparePhase) -> String {
        switch phase {
        case .idle: return "idle"
        case .registering: return "registering"
        case .prewarming: return "prewarming"
        case .downloading: return "downloading"
        case .loading: return "loading"
        case .ready: return "ready"
        case .failed(let why): return "failed(\(why))"
        }
    }

    /// `models--<org>--<name>` directories under `store` that carry the engine's materialization marker.
    static func markedRepoDirs(in store: URL) -> Set<String> {
        let fm = FileManager.default
        let names = (try? fm.contentsOfDirectory(atPath: store.path)) ?? []
        return Set(names.filter {
            $0.hasPrefix("models--")
                && fm.fileExists(atPath: store.appendingPathComponent($0).appendingPathComponent("mlx-package.json").path)
        })
    }

    static func gradient(_ w: Int, _ h: Int) throws -> CGImage {
        let cs = CGColorSpaceCreateDeviceRGB()
        let ctx = try #require(CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: 0, space: cs,
                                         bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue))
        let grad = try #require(CGGradient(colorsSpace: cs,
                                           colors: [CGColor(red: 0.1, green: 0.2, blue: 0.85, alpha: 1),
                                                    CGColor(red: 0.95, green: 0.7, blue: 0.1, alpha: 1)] as CFArray,
                                           locations: [0, 1]))
        ctx.drawLinearGradient(grad, start: .zero, end: CGPoint(x: w, y: h), options: [])
        return try #require(ctx.makeImage())
    }
}
