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
        #expect(GenerativeFillTier.senseNova16GB.model == .senseNova(variant: .fast4V2))
        #expect(GenerativeFillTier.senseNova16GB.maxWindowPixels == 768 * 768)
        #expect(GenerativeFillTier.senseNova24GB.model == .senseNova(variant: .fast4V2))
        #expect(GenerativeFillTier.senseNova24GB.maxWindowPixels == 1024 * 1024)
        #expect(GenerativeFillTier.senseNova32GB.model == .senseNova(variant: .fast8V2))
        #expect(GenerativeFillTier.senseNova32GB.maxWindowPixels == 1024 * 1024)
        // The documented fallback: Klein-4B int8, encoder evicted, ≤ 768², at 16 GB only (int4 failed AB-R-0544's gate).
        #expect(GenerativeFillTiers.kleinAt16GB.tier(forPhysicalMemory: 16 * Self.gib)?.model == .klein(quant: .int8, evictEncoder: true))
        #expect(GenerativeFillTiers.kleinAt16GB.tier(forPhysicalMemory: 24 * Self.gib) == .senseNova24GB)
    }

    @Test func theMappingRoundTripsAsJSON() throws {
        for tiers in [GenerativeFillTiers.forgeDefault, .kleinAt16GB] {
            let data = try JSONEncoder().encode(tiers)
            #expect(try JSONDecoder().decode(GenerativeFillTiers.self, from: data) == tiers)
        }
        // A host can write a tier by hand. This one is the shipped 16 GB fallback since AB-A-0182: Klein int8 evicted at 768².
        let json = #"{"label":"16 GB","model":{"klein":{"quant":"int8","evictEncoder":true}},"maxWindowPixels":589824}"#
        let tier = try JSONDecoder().decode(GenerativeFillTier.self, from: Data(json.utf8))
        #expect(tier.model == .klein(quant: .int8, evictEncoder: true) && tier.model.displayName == "FLUX.2-klein-4B int8 (encoder evicted)")
        #expect(tier == .klein16GB)
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

    /// 🪤 Tripwire for AB-T-0278. On a 16 GB budget (12.7 GB) the engine's admission refuses SenseNova int4 as declared
    /// (12.2 GB resident + 4.5 GB 2048²-envelope activation — AB-R-0537), so the 16 GB tier cannot run through the engine
    /// until sensenova-u1-swift ships its 16 GB footprint declaration; Klein-4B evicted (declared 5.2 + 5.0 GB at either quant until AB-T-0290) is admitted.
    /// When this starts failing on SenseNova, AB-T-0278 has landed: update the tier notes and REQUIREMENTS.md.
    @Test func atASixteenGigabyteBudgetOnlyKleinIsAdmittedToday() async {
        let budget = MemoryGovernor(budgetBytes: UInt64(Double(16 * Self.gib) * 0.74))
        let klein = ForgeEngineService(governor: budget, generativeFillTier: .klein16GB)
        await klein.registerAll()
        if let verdict = await klein.admissibility(for: .generativeFill) {
            #expect(verdict.fitsBudget, "Klein int8 evicted should fit a 16 GB budget (footprint \(verdict.footprint))")
        }
        let senseNova = ForgeEngineService(governor: budget, generativeFillTier: .senseNova16GB)
        await senseNova.registerAll()
        guard let verdict = await senseNova.admissibility(for: .generativeFill) else { return }   // below the chip floor
        #expect(!verdict.fitsBudget, "SenseNova int4 now fits a 16 GB budget — AB-T-0278 landed? footprint \(verdict.footprint)")
    }
}
