//
//  GenerativeFillTierTests.swift
//  ForgeCoreTests
//
//  The tier → model/quant/window mapping is configuration (AB-A-0166): it says what AB-D-0141 decided, it round-trips
//  as JSON so a host can store it, and `.generativeFill` registers whichever model the tier names — under one id, so a
//  tier switch replaces the registration rather than adding one. No weights load here.
//

import Testing
import Foundation
import MLXServeCore
import MLXToolKit
import MLXSenseNovaU1
import MLXKlein
@testable import ForgeCore

@MainActor
@Suite struct GenerativeFillTierTests {

    private static let gib: UInt64 = 1 << 30

    @Test func theDefaultIsAB_D_0141AsAmended() {
        let tiers = GenerativeFillTiers.forgeDefault
        #expect(tiers.tier(forPhysicalMemory: 8 * Self.gib) == nil)
        for gb: UInt64 in [16, 18] {
            #expect(tiers.tier(forPhysicalMemory: gb * Self.gib) == .senseNova16GB)
        }
        #expect(tiers.tier(forPhysicalMemory: 24 * Self.gib) == .senseNova24GB)
        for gb: UInt64 in [32, 36, 48, 64, 128] {
            #expect(tiers.tier(forPhysicalMemory: gb * Self.gib) == .senseNova32GB)
        }
        #expect(GenerativeFillTier.senseNova16GB.model == .senseNova(variant: .fast4V2, window: .fill768))
        #expect(GenerativeFillTier.senseNova16GB.maxWindowPixels == 768 * 768)
        #expect(GenerativeFillTier.senseNova24GB.model == .senseNova(variant: .fast4V2))
        #expect(GenerativeFillTier.senseNova24GB.maxWindowPixels == 1024 * 1024)
        #expect(GenerativeFillTier.senseNova32GB.model == .senseNova(variant: .fast8V2))
        #expect(GenerativeFillTier.senseNova32GB.maxWindowPixels == 1024 * 1024)
        // Klein is not the 16 GB fallback (AB-D-0141, amended 2026-10-22): no Klein tier in the default table.
        #expect(!GenerativeFillTiers.forgeDefault.entries.contains { if case .klein = $0.tier.model { true } else { false } })
        // `kleinAt16GB` stays a host-chosen table, not a shipping one: Klein-4B int8, encoder evicted, ≤ 768², at 16 GB only.
        #expect(GenerativeFillTiers.kleinAt16GB.tier(forPhysicalMemory: 16 * Self.gib)?.model == .klein(quant: .int8, evictEncoder: true))
        #expect(GenerativeFillTiers.kleinAt16GB.tier(forPhysicalMemory: 24 * Self.gib) == .senseNova24GB)
    }

    @Test func theMappingRoundTripsAsJSON() throws {
        for tiers in [GenerativeFillTiers.forgeDefault, .kleinAt16GB] {
            let data = try JSONEncoder().encode(tiers)
            #expect(try JSONDecoder().decode(GenerativeFillTiers.self, from: data) == tiers)
        }
        // A host can write a tier by hand, e.g. Klein int8 evicted at 768² (a host's choice; not a 16 GB fallback).
        let json = #"{"label":"16 GB","model":{"klein":{"quant":"int8","evictEncoder":true}},"maxWindowPixels":589824}"#
        let tier = try JSONDecoder().decode(GenerativeFillTier.self, from: Data(json.utf8))
        #expect(tier.model == .klein(quant: .int8, evictEncoder: true) && tier.model.displayName == "FLUX.2-klein-4B int8 (encoder evicted)")
        #expect(tier == .klein16GB)
        // The 16 GB tier stores its window; a SenseNova tier stored before the window existed decodes as native (nil).
        let stored = String(decoding: try JSONEncoder().encode(GenerativeFillTier.senseNova16GB), as: UTF8.self)
        #expect(stored.contains(#""window":"fill-768""#))
        let legacy = #"{"label":"24 GB","model":{"senseNova":{"variant":"8step-v2-4bit"}},"maxWindowPixels":1048576}"#
        #expect(try JSONDecoder().decode(GenerativeFillTier.self, from: Data(legacy.utf8)) == .senseNova24GB)
    }

    @Test func theSenseNovaFillTiersServeImageEdit() {
        // The registry row's "8step distill tiers are T2I-only" was superseded by sensenova-u1-swift 0.5.1 (ec6525a,
        // AB-R-0490): every tier serves imageEdit, and run() checks exactly this set.
        for variant in [SenseNovaVariant.fast4V2, .fast8V2] {
            #expect(variant.capabilities.contains(.imageEdit))
            #expect(variant.isDistilled && variant.defaultSteps == 8 && variant.defaultGuidance == 1)
        }
        #expect(SenseNovaVariant.fast4V2.quant == .int4 && SenseNovaVariant.fast8V2.quant == .int8)
        let manifest = SenseNovaU1Package.manifest
        #expect(manifest.surfaces.contains { $0.capability == .imageEdit })
        #expect(manifest.license.weightLicense == .apache2 && manifest.license.portCodeLicense == .mit)
        let klein = Klein4BT2IPackage.manifest
        #expect(klein.surfaces.contains { $0.capability == .imageEdit })
        #expect(klein.license.weightLicense == .apache2 && klein.license.portCodeLicense == .mit)
    }

    @Test func generativeFillRegistersTheTiersModelUnderOneID() async {
        // Klein first: it declares no chip floor, so it registers on any Mac this suite runs on.
        let service = ForgeEngineService(generativeFillTier: .klein16GB)
        await service.registerAll()
        #expect(service.packageID(for: .generativeFill)?.rawValue == "generative-fill",
                "\(service.outcome(for: .generativeFill)?.failure ?? "")")
        #expect(service.registeredGenerativeFillTier == .klein16GB)
        #expect(ForgeModel.generativeFill.capability == .imageEdit)

        // Switch the tier: the same id, now SenseNova — or, on a chip below SenseNova's declared floor (.max), an honest
        // refusal, never a silent fallback to the previous model.
        service.generativeFillTier = .senseNova24GB
        await service.registerAll()
        if let id = service.packageID(for: .generativeFill) {
            #expect(id.rawValue == "generative-fill" && service.registeredGenerativeFillTier == .senseNova24GB)
            #expect(await service.engine.manifest(for: id)?.license.weightLicense == .apache2)
        } else {
            #expect(service.registeredGenerativeFillTier == nil)
            #expect(service.outcome(for: .generativeFill)?.failure?.contains("cannot run") == true)
        }

        let provider = EngineGenerativeFillProvider(service: service)
        #expect(provider.requiredModels == [.generativeFill, .inpaint])
    }

    @Test func noTierIsAnHonestOutcome() async throws {
        let service = ForgeEngineService(generativeFillTier: nil)
        await service.registerAll()
        #expect(service.packageID(for: .generativeFill) == nil)
        #expect(service.outcome(for: .generativeFill)?.failure?.contains("no Generative Fill tier") == true)
        let provider = EngineGenerativeFillProvider(service: service)
        #expect(provider.name == "Generative Fill (no tier on this Mac)")
        await #expect(throws: EngineGenerativeFillProvider.GenerativeFillError.self) {
            _ = try await provider.fill(FillFixture.photo(), mask: FillMask(width: 640, height: 480).cgImage, prompt: "x")
        }
    }

    /// The tier registers the declaration the engine charges: the 16 GB tier its `.fill768` window, the 24 GB and 32 GB+
    /// tiers the native envelope (sensenova-u1-swift 0.6.0, AB-T-0278).
    @Test func theSixteenGigabyteTierRegistersTheFill768Window() throws {
        let config16 = try #require(GenerativeFillTier.senseNova16GB.model.registration.1 as? SenseNovaU1Configuration)
        #expect(config16.variant == .fast4V2 && config16.window == .fill768)
        #expect(config16.residentBytesHint == 11_800_000_000 && config16.peakActivationBytesHint == 750_000_000)
        for tier in [GenerativeFillTier.senseNova24GB, .senseNova32GB] {
            let config = try #require(tier.model.registration.1 as? SenseNovaU1Configuration)
            #expect(config.window == .native && config.residentBytesHint == nil, "\(tier.label)")
        }
        #expect(GenerativeFillTier.senseNova16GB.model.declaredMaxPixels == 768 * 768)
        #expect(GenerativeFillTier.senseNova24GB.model.declaredMaxPixels == nil)
    }

    /// AB-T-0278 landed (sensenova-u1-swift 0.6.0): on a 16 GB budget (16 GiB × 0.74 ≈ 12.7 GB) the engine admits the
    /// 16 GB tier's `.fill768` window (11.8 + 0.75 GB), and still refuses SenseNova int4's native declaration (12.2 GB
    /// resident + 4.5 GB 2048²-envelope activation, AB-R-0537). Confirmed on a real 16 GB Mac, AB-R-0602.
    @Test func atASixteenGigabyteBudgetSenseNovaFill768IsAdmitted() async {
        let budget = MemoryGovernor(budgetBytes: UInt64(Double(16 * Self.gib) * 0.74))
        let window = ForgeEngineService(governor: budget, generativeFillTier: .senseNova16GB)
        await window.registerAll()
        guard let verdict = await window.admissibility(for: .generativeFill) else { return }   // below the chip floor
        #expect(verdict.fitsBudget, "SenseNova .fill768 should fit a 16 GB budget (footprint \(verdict.footprint))")
        let native = GenerativeFillTier(label: "16 GB native", model: .senseNova(variant: .fast4V2), maxWindowPixels: 768 * 768)
        let refused = ForgeEngineService(governor: budget, generativeFillTier: native)
        await refused.registerAll()
        guard let nativeVerdict = await refused.admissibility(for: .generativeFill) else { return }
        #expect(!nativeVerdict.fitsBudget, "SenseNova int4 native now fits a 16 GB budget? footprint \(nativeVerdict.footprint)")
    }

    /// The window refuses an output over 768² by area, a second reference, or a guidance branch (`outsideWindow`), so the
    /// 16 GB tier's route must never ask for one: a 1024² fill renders a 768² window with ONE reference at the render size,
    /// a 1024² expand scales its canvas into 768², and a hand-written tier whose cap exceeds its declaration is held to it.
    @Test func aTenTwentyFourRequestOnTheSixteenGigabyteTierIsScaledIntoTheWindow() async throws {
        let maxPixels = try #require(SenseNovaWindow.fill768.maxPixels)
        let route = GenerativeFillTier.senseNova16GB.route
        #expect(route.maxWindowPixels == maxPixels && route.multipleOf == 32)

        for box in [FillBox(x: 600, y: 380, width: 150, height: 110),        // small selection: a 1:1 768² square
                    FillBox(x: 50, y: 300, width: 940, height: 200)] {       // context over the cap: 1024² square at 768²
            let image = FillPixels(FillFixture.photo(width: 1024, height: 1024))
            let mask = FillFixture.mask(width: 1024, height: 1024, box: box, feather: 4)
            let calls = FillCalls()
            _ = try await route.fill(image, mask: mask, prompt: "a fountain", seed: 9,
                                     paint: FakeModel.painter(original: nil, content: (200, 40, 40), drift: 0, calls: calls),
                                     inpaint: FakeModel.lama(calls: calls))
            let paints = calls.paints.withLock { $0 }
            #expect(!paints.isEmpty)
            for call in paints {
                #expect(call.width * call.height <= maxPixels, "\(call.width)×\(call.height) is over the window")
                #expect(call.marked.width == call.width && call.marked.height == call.height)   // one reference, render size
            }
        }

        let expand = route.modelSize(width: 1024 + 256, height: 1024)
        #expect(expand.width * expand.height <= maxPixels)

        let handWritten = GenerativeFillTier(label: "16 GB, too wide",
                                             model: .senseNova(variant: .fast4V2, window: .fill768),
                                             maxWindowPixels: 1024 * 1024)
        #expect(handWritten.route.maxWindowPixels == maxPixels)
        #expect(GenerativeFillTier.senseNova24GB.route.maxWindowPixels == 1024 * 1024)
    }

    /// SenseNova never renders under its 512² reference floor on any tier (AB-A-0199); Klein has no floor.
    @Test func everySenseNovaTierCarriesTheReferenceFloor() {
        for tier in GenerativeFillTiers.forgeDefault.entries.map(\.tier) {
            #expect(tier.model.minimumRenderPixels == 512 * 512)
            #expect(tier.route.minimumRenderPixels == 512 * 512, "\(tier.label)")
            // PCIntelligence's window for a 75 × 75 selection (300²) renders at 512², inside the tier's cap.
            let w = tier.route.window(for: FillBox(x: 112, y: 112, width: 75, height: 75), width: 300, height: 300)
            #expect(w.modelWidth == 512 && w.modelHeight == 512, "\(tier.label): \(w)")
            #expect(w.modelWidth * w.modelHeight <= tier.route.maxWindowPixels)
        }
        #expect(GenerativeFillTier.klein16GB.model.minimumRenderPixels == nil)
        #expect(GenerativeFillTier.klein16GB.route.minimumRenderPixels == 0)
        let klein = GenerativeFillTier.klein16GB.route.window(for: FillBox(x: 112, y: 112, width: 75, height: 75), width: 300, height: 300)
        #expect(klein.modelWidth == 288 && klein.modelHeight == 288)
    }
}
