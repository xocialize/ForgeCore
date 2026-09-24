//
//  RestoreRoutingTests.swift — ForgeCore
//
//  Pins `EngineFrameRestoreProvider.routing`, the measured job → backer table. One edit to it re-routes
//  every app that restores through ForgeCore, and nothing else would notice: the receipt just names a
//  different engine. So a re-route has to change this file too, in the same change as its evidence.
//

import RestoreTier2
import Testing

@testable import ForgeCore

@Suite("Restore routing table")
struct RestoreRoutingTests {

    /// AB-T-0180: on real blur (C2, RealBlur-J) Restormer `.motionDeblur` measured +1.29 dB over the
    /// untouched input; FFTformer measured −0.02 dB and hurt 41% of images.
    @Test func motionDeblurRoutesToRestormer() {
        #expect(EngineFrameRestoreProvider.routing[.motionDeblur] == .restormerMotion)
        #expect(EngineFrameRestoreProvider.routing[.motionDeblur]?.modelName == "Restormer (motion)")
    }

    /// FFTformer stays registered as an explicit choice, but no job reaches it by default.
    @Test func fftformerIsNotADefaultRoute() {
        #expect(!EngineFrameRestoreProvider.routing.values.contains(.deblurMotion))
    }

    @Test func tableIsPinned() {
        #expect(EngineFrameRestoreProvider.routing == [
            .denoise: .restore,
            .motionDeblur: .restormerMotion,
            .defocusDeblur: .restormerDefocus,
        ])
    }

    /// A job without a backer only surfaces as `.unavailable` at run time, in front of a user.
    @Test func everyJobIsRouted() {
        for job in RestoreJob.allCases {
            #expect(EngineFrameRestoreProvider.routing[job] != nil, "\(job) has no backer")
        }
    }
}

/// The table above maps jobs to `ForgeModel`s; this pins that each model reaches its OWN registration. The engine keys
/// a registration by its manifest's first surface name unless `register` is given an explicit `id:`, and a second
/// registration under the same id REPLACES the first. Restormer's three variants share one manifest, so without
/// explicit ids all three collapsed onto "restormer-restore" and the last one registered (`.realDenoise`) answered for
/// motion and defocus too — under their names. Found by AB-T-0180's live run, not by a static test.
@Suite("Registration identity")
@MainActor
struct RegistrationIdentityTests {

    @Test func everyModelHasItsOwnPackageID() async {
        let service = ForgeEngineService()
        await service.registerAll()
        let byID = Dictionary(grouping: ForgeModel.allCases.filter { service.packageID(for: $0) != nil },
                              by: { service.packageID(for: $0)!.rawValue })
        let shared = byID.filter { $0.value.count > 1 }.map { "\($0.key) ← \($0.value.map(\.rawValue))" }
        #expect(shared.isEmpty, "models sharing a PackageID (the last registration wins): \(shared)")
    }

    @Test func restoreRoutesRegister() async {
        let service = ForgeEngineService()
        await service.registerAll()
        for model in Set(EngineFrameRestoreProvider.routing.values) {
            #expect(service.packageID(for: model) != nil, "\(model) is routed but did not register")
        }
    }
}
