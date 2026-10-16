import Testing
import CoreGraphics
import Foundation
import Anime4KMetal
import MLXServeCore
import MLXToolKit
@testable import ForgeCore

/// AB-A-0183 — the Instant tier as a weightless engine registration (`.upscaleInstant`, `Anime4KUpscalePackage`), so an
/// engine client reaches it by name. These drive the real engine: register, prepare, run.
@MainActor
@Suite struct InstantPackageTests {

    @Test func isASeventhImageUpscaleBackerUnderItsOwnName() {
        #expect(ForgeModel.upscaleInstant.capability == .imageUpscale)
        #expect(ForgeModel.upscaleInstant.modelName == "Anime4K VL")
        #expect(ForgeModel.upscaleInstant.modelName == EngineImageEnhancer.instantModelName)
        // The Optimizer keeps running Instant directly: the enhancer's map leaves the registration out.
        #expect(EngineImageEnhancer.model(for: .instant) == nil)
        #expect(EngineImageEnhancer.tier(for: .upscaleInstant) == nil)
    }

    /// Weightless and permissive: it clears `.permissiveOnly` + `.blocking` on MIT/MIT, needs nothing downloaded, keeps 16
    /// bits, and lands on its explicit id.
    @Test func registersWeightlessUnderItsExplicitID() async throws {
        let service = ForgeEngineService()
        await service.registerAll()
        #expect(service.outcome(for: .upscaleInstant)?.failure == nil,
                "Anime4K did not register: \(service.outcome(for: .upscaleInstant)?.failure ?? "")")
        let id = try #require(service.packageID(for: .upscaleInstant))
        #expect(id.rawValue == "anime4k-upscale")
        let manifest = try #require(await service.engine.manifest(for: id))
        #expect(manifest.license.weightLicenses.allSatisfy(\.isPermissive) && manifest.license.portCodeLicense.isPermissive)
        #expect(manifest.requirements.requiredBackends == [.metalGPU])
        #expect(await service.keepsSixteenBits(.upscaleInstant))
        // Nothing to download, as the engine reads it (BundledWeightSourcing: the shaders live in the binary).
        #expect(await service.engine.needsDownload(.imageUpscale, package: id) == false)
        #expect(await service.engine.residentPackages.isEmpty)
    }

    /// The declared working set covers what one call allocates at the default tile, on both lanes, with alpha.
    @Test func footprintCoversTheUpscalersOwnEstimate() throws {
        try #require(Anime4KUpscaler.isAvailable, "needs a Metal device")
        let upscaler = try Anime4KUpscaler()
        let byQuant = Dictionary(uniqueKeysWithValues: Anime4KUpscalePackage.footprints.map { ($0.quant, $0) })
        let eight = UInt64(upscaler.memoryEstimate(width: 1024, height: 1024, deep: false, alpha: true))
        let deep = UInt64(upscaler.memoryEstimate(width: 1024, height: 1024, deep: true, alpha: true))
        print("Anime4K VL working set at 1024²: 8-bit \(eight) B, 16-bit \(deep) B")
        #expect(try #require(byQuant[.fp16]).peakActivationBytes >= eight)
        #expect(try #require(byQuant[.fp32]).peakActivationBytes >= deep)
    }

    /// Through `engine.run`: ×2, ×4 and ×6 land on the exact size with `appliedScale` saying so, an 8-bit request comes back
    /// `.rawBGRA8`, and a 16-bit one `.rawRGBA16`.
    @Test func runsThroughTheEngineAtExactSizes() async throws {
        try #require(Anime4KUpscaler.isAvailable, "needs a Metal device")
        let service = ForgeEngineService()
        await service.registerAll()
        let id = try #require(service.packageID(for: .upscaleInstant))
        _ = try await service.engine.prepare(.imageUpscale, package: id)

        let eight = try EngineImageCoding.rawImage(from: try InstantTierTests.drawing(width: 48, height: 36))
        for scale in [2, 4, 6] {
            let response = try await service.engine.run(ImageUpscaleRequest(image: eight, scale: scale), package: id)
            let out = try #require(response as? ImageUpscaleResponse)
            #expect(out.appliedScale == scale)
            #expect(out.image.format == .rawBGRA8)
            #expect(out.image.width == 48 * scale && out.image.height == 36 * scale, "×\(scale)")
        }
        let deep = try EngineImageCoding.rawImage(from: try InstantTierTests.drawing(width: 40, height: 30, bits: 16),
                                                  keepingDepth: true)
        let response = try await service.engine.run(ImageUpscaleRequest(image: deep, scale: 2), package: id)
        let out = try #require(response as? ImageUpscaleResponse)
        #expect(out.image.format == .rawRGBA16)
        #expect(out.image.width == 80 && out.image.height == 60)
        await service.engine.evict(package: id)
    }
}
