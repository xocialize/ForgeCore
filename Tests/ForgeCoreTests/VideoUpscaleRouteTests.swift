//
//  VideoUpscaleRouteTests.swift — ForgeCore
//
//  The live-action video tier (FlashVSR, `.upscaleVideoLiveAction`, registered 2026-10-04) and the DECLARED route to
//  it. Offline: no weights, no GPU. What these pin:
//
//  • Only content the caller declares `.liveAction` reaches FlashVSR. Drawn and undeclared content keep the SeedVR2
//    route `upscaleVideo(_:scale:)` has always taken — FlashVSR turns anime and graphics into photographic texture, and
//    Forge has no validated detector to stand in for the declaration.
//  • A routed tier that is not available fails with ITS reason — never a silent swap to the other tier.
//  • The fit gate reads the PACKAGE'S declaration (resident + max(scalar, ActivationScaling)), so it cannot drift from
//    what the engine admits against; the measured ceiling is 1920×1152 output for the bf16 lane.
//

import Foundation
import MLXFlashVSR
import MLXToolKit
import Testing

@testable import ForgeCore

@Suite("VideoUpscaleRoute")
struct VideoUpscaleRouteTests {

    @Test func flashVSRIsAVideoUpscaleBackerNotAStillsOne() {
        #expect(ForgeModel.upscaleVideoLiveAction.capability == .videoUpscale)
        #expect(ForgeModel.upscaleVideoLiveAction.modelName == "FlashVSR")
        #expect(ForgeModel.allCases.contains(.upscaleVideoLiveAction))
        let stills = ForgeModel.allCases.filter { $0.capability == .imageUpscale }
        #expect(!stills.contains(.upscaleVideoLiveAction))
    }

    /// The whole policy in one table: exactly one declaration reaches the generative live-action tier.
    @Test func onlyDeclaredLiveActionReachesFlashVSR() {
        #expect(ForgeEngineService.videoUpscaleRoute(for: .liveAction) == .upscaleVideoLiveAction)
        #expect(ForgeEngineService.videoUpscaleRoute(for: .drawn) == .upscaleQuality)
        #expect(ForgeEngineService.videoUpscaleRoute(for: .unspecified) == .upscaleQuality)
        let toFlashVSR = VideoContent.allCases.filter {
            ForgeEngineService.videoUpscaleRoute(for: $0) == .upscaleVideoLiveAction
        }
        #expect(toFlashVSR == [.liveAction])
    }

    /// Registration is offline (the engine materializes mlx-community/FlashVSR-v1.1-bf16 on first prepare, not here):
    /// FlashVSR must clear `.permissiveOnly` + `.blocking` (Apache-2.0 weights and port) and land on its explicit id.
    @Test @MainActor func flashVSRRegistersUnderItsExplicitID() async {
        let service = ForgeEngineService()
        await service.registerAll()
        #expect(service.outcome(for: .upscaleVideoLiveAction)?.failure == nil,
                "FlashVSR did not register: \(service.outcome(for: .upscaleVideoLiveAction)?.failure ?? "")")
        #expect(service.packageID(for: .upscaleVideoLiveAction)?.rawValue == "flashvsr-upscale")
        #expect(service.packageID(for: .upscaleVideoLiveAction) != service.packageID(for: .upscaleQuality))
    }

    /// ForgeCore registers FlashVSR's package defaults; these are the ones it relies on: the bf16 lane from the
    /// mlx-community repo, upstream's sparsity defaults, the colour fix on, and the live-action warning in the manifest.
    @Test func flashVSRDefaultsAreThePackages() {
        let c = FlashVSRConfiguration()
        #expect(c.precision == .bf16)
        #expect(c.repo == FlashVSRConfiguration.bf16Repo)
        #expect(c.topkRatio == nil && c.localRange == 11 && c.colorFix)
        let summary = FlashVSRUpscalePackage.manifest.surfaces.map(\.summary).joined()
        #expect(summary.contains("live action"))
        #expect(FlashVSRUpscalePackage.manifest.license.weightLicense == .apache2)
    }

    /// A tier that cannot run fails with its own reason — the drawn route never borrows FlashVSR and the live-action
    /// route never borrows SeedVR2. (Nothing is registered here, so each route reports ITS model.)
    @Test @MainActor func aMissingTierFailsWithItsOwnReason() async {
        let service = ForgeEngineService()                      // no registerAll
        let clip = Video(format: .mp4, data: Data([0]))
        for (content, expected) in [(VideoContent.liveAction, ForgeModel.upscaleVideoLiveAction),
                                    (.drawn, .upscaleQuality), (.unspecified, .upscaleQuality)] {
            do {
                _ = try await service.upscaleVideo(clip, scale: 4, content: content)
                Issue.record("\(content) ran with nothing registered")
            } catch let ForgeEngineService.DiagnosticsError.notRegistered(model, _) {
                #expect(model == expected, "\(content) reported \(model)")
            } catch {
                Issue.record("\(content): unexpected \(error)")
            }
        }
    }

    // MARK: - fit gate

    static let gb = 1_000_000_000.0

    /// The padded pixel count is the package's own workload unit (its WorkloadDeclaring test: 200×120 ×4 → 896×512).
    @Test func outputPixelsMatchThePackagesWorkloadUnit() {
        #expect(ForgeEngineService.liveActionOutputPixels(sourceWidth: 200, sourceHeight: 120, scale: 4) == 896 * 512)
        #expect(ForgeEngineService.liveActionOutputPixels(sourceWidth: 200, sourceHeight: 120, scale: 2) == 512 * 256)
        #expect(ForgeEngineService.liveActionOutputPixels(sourceWidth: 320, sourceHeight: 192) == 1280 * 768)
        #expect(ForgeEngineService.liveActionOutputPixels(sourceWidth: 960, sourceHeight: 540, scale: 2) == 1920 * 1152)
    }

    /// 1280×768 out (the representative case): the scalar governs — 3.6 GB resident + 19.0 GB activation.
    @Test func representativeCaseIsTheScalar() {
        let r = ForgeEngineService.liveActionVideoUpscaleFits(sourceWidth: 320, sourceHeight: 192,
                                                              budgetBytes: UInt64(32 * Self.gb))
        #expect(r.withinCeiling && r.fits)
        #expect(abs(Double(r.predictedBytes) / Self.gb - 22.6) < 0.05)
        // A 16 GB Mac's budget (~12.7 GB) is told no, with the number to say why.
        let small = ForgeEngineService.liveActionVideoUpscaleFits(sourceWidth: 320, sourceHeight: 192,
                                                                  budgetBytes: UInt64(12.7 * Self.gb))
        #expect(small.withinCeiling && !small.fits)
    }

    /// 1920×1152 out — AT the measured ceiling: the line governs (4.73 GB + 14.23 KB/px), above the scalar.
    @Test func ceilingCaseFollowsTheLine() {
        let r = ForgeEngineService.liveActionVideoUpscaleFits(sourceWidth: 480, sourceHeight: 288,
                                                              budgetBytes: UInt64(64 * Self.gb))
        #expect(r.withinCeiling && r.fits)
        let expected = 3.6 + 4.73 + 14_230 * 1920 * 1152 / Self.gb
        #expect(abs(Double(r.predictedBytes) / Self.gb - expected) < 0.05)
    }

    /// Beyond the ceiling the engine refuses before loading anything; the gate says so instead of offering the control.
    @Test func beyondTheCeilingIsRefused() {
        let hd4x = ForgeEngineService.liveActionVideoUpscaleFits(sourceWidth: 1920, sourceHeight: 1080,
                                                                 budgetBytes: UInt64(512 * Self.gb))
        #expect(!hd4x.withinCeiling && !hd4x.fits)
        // …while the same HD source at ×2 (1920×1080 → padded 1920×1152) is inside it.
        let hd2x = ForgeEngineService.liveActionVideoUpscaleFits(sourceWidth: 960, sourceHeight: 540, scale: 2,
                                                                 budgetBytes: UInt64(64 * Self.gb))
        #expect(hd2x.withinCeiling)
    }
}
