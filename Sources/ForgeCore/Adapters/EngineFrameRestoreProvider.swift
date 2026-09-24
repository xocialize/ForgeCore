//
//  EngineFrameRestoreProvider.swift — ForgeCore
//
//  RestoreTier2's `FrameRestoreProvider`, implemented over the engine (BRIDGE-070, adapter half).
//
//  🔑 **The routing table lives HERE, in one place, and the receipt names what it chose.** The Kit
//  names jobs; this adapter maps each job to a registered backer. When BRIDGE-069's head-to-head
//  (SCUNet vs the shipping NAFNet `.signage`) decides the denoise default, THIS table is the single
//  line that changes — no Kit edit, no app edit, and every receipt after it names the new backer.
//  That line lives in `routing` below, and `RestoreRoutingTests` pins it, so re-routing a job means
//  changing a test in the same change as the evidence.
//
//  ✅ **Every route is a measured default:**
//  - `denoise` → NAFNet `.signage` (BRIDGE-069, 2026-07-31 —
//    `probes/r2_bridge069_restore_default.out`) — RE-CONFIRMED, not merely incumbent. The engine-path
//    head-to-head: `.signage` TIES SCUNet at ISO1600 (+1.38 vs +1.40) and is **6–9 dB closer to a
//    no-op on clean content**; SCUNet dominates only at heavy noise (+4.91 vs +2.25 at ISO6400,
//    +8.25 vs +2.66 at ISO25600). The default must win where being wrong is costly, and that is
//    clean content. SCUNet stays registered (`.denoiseBlind`) as the heavy-noise backer; **the
//    per-source router that picks between them is T4's decision layer, not this table** — when T4
//    lands, `denoise` routing becomes measurement-driven and this comment moves there.
//  - `motionDeblur` → Restormer `.motionDeblur` (C2, 2026-07-27; re-routed by AB-T-0180, 2026-09-24).
//    RealBlur-J, 300 seeded pairs, the dataset authors' ECC-aligned protocol with the untouched input
//    scored through the same alignment: Restormer **+1.29 ± 0.13 dB**, 95% of images improved, worst
//    case −0.69 dB. FFTformer GoPro, this job's backer until AB-T-0180: **−0.02 ± 0.22 dB**, 41% of
//    images hurt (18% by more than 1 dB), worst case −18.70 dB. A default has to be safe on the tail.
//    FFTformer stays registered (`.deblurMotion`) as an explicit, non-default choice.
//    ⚠️ Both are GoPro-trained, and FFTformer WINS GoPro (34.21 vs 32.92): the ranking inverts on real
//    blur, so a GoPro score is not evidence for this table. The comment this replaces said C2 found
//    every GoPro-trained method below the blurry input — that was the *Deblurring in the Wild* premise
//    C2 was run to test, not what it measured. Still unmeasured: SMARTPHONE blur (RealBlur is a DSLR
//    rig; C3 / AB-T-0032). Record: `corpus/realblur/RESULTS.md` (mlxengine-image), AB-R-0331.
//  - `defocusDeblur` → Restormer `.defocusDeblur` — the only shippable defocus deblur (C4/DPDD: +2.09 dB,
//    the published 25.98 reproduced exactly).
//  🔴 Until AB-T-0180 both Restormer routes silently ran `.realDenoise`: the three Restormer registrations
//  shared one PackageID and the last replaced the others (see `ForgeEngineService`, Restormer cases). The
//  receipt still named the variant the table chose, so nothing downstream could tell.
//

import Foundation
import MLXServeCore
import MLXToolKit
import RestoreTier2

public struct EngineFrameRestoreProvider: FrameRestoreProvider {
    /// Which registered model backs each job — the measured table the header documents. Static so a
    /// test can pin it without an engine.
    public static let routing: [RestoreJob: ForgeModel] = [
        .denoise: .restore,                  // ✅ BRIDGE-069 re-confirmed .signage
        .motionDeblur: .restormerMotion,     // ✅ C2: +1.29 dB on real blur; FFTformer −0.02 (AB-T-0180)
        .defocusDeblur: .restormerDefocus,   // ✅ C4: +2.09 dB on DPDD
    ]

    private let engine: MLXServeEngine
    private let table: [RestoreJob: (id: PackageID, name: String)]

    /// Build from a registered service. Jobs whose backer failed registration are absent from the
    /// table and throw `.unavailable` at call time — a fact the caller can show, not a crash.
    @MainActor
    public init(service: ForgeEngineService) {
        self.engine = service.engine
        var t: [RestoreJob: (PackageID, String)] = [:]
        for (job, model) in Self.routing {
            if let id = service.packageID(for: model) {
                t[job] = (id, model.modelName)
            }
        }
        self.table = t
    }

    public func engineName(for job: RestoreJob) -> String {
        table[job]?.name ?? "unavailable"
    }

    public func restore(imageData: Data, job: RestoreJob,
                        strength: Float?) async throws -> FrameRestoreResult {
        guard let backer = table[job] else {
            throw FrameRestoreProviderError.unavailable(job)
        }
        _ = try await engine.prepare(.imageRestore, package: backer.id)
        let response = try await engine.run(
            ImageRestoreRequest(image: Image(format: .png, data: imageData), strength: strength),
            package: backer.id)
        guard let restore = response as? ImageRestoreResponse else {
            throw FrameRestoreProviderError.unexpectedResponse(job)
        }
        return FrameRestoreResult(imageData: restore.image.data,
                                  appliedStrength: restore.appliedStrength,
                                  engineName: backer.name)
    }
}

public enum FrameRestoreProviderError: Error, CustomStringConvertible {
    /// The job's backer did not register on this machine (license or eligibility) — show the
    /// registration outcome, don't retry.
    case unavailable(RestoreJob)
    case unexpectedResponse(RestoreJob)

    public var description: String {
        switch self {
        case .unavailable(let job): return "no registered backer for \(job.rawValue)"
        case .unexpectedResponse(let job): return "unexpected response type for \(job.rawValue)"
        }
    }
}
