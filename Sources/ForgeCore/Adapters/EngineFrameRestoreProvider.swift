//
//  EngineFrameRestoreProvider.swift — ForgeCore
//
//  RestoreTier2's `FrameRestoreProvider`, implemented over the engine (BRIDGE-070, adapter half).
//
//  🔑 **The routing table lives HERE, in one place, and the receipt names what it chose.** The Kit
//  names jobs; this adapter maps each job to a registered backer. When BRIDGE-069's head-to-head
//  (SCUNet vs the shipping NAFNet `.signage`) decides the denoise default, THIS table is the single
//  line that changes — no Kit edit, no app edit, and every receipt after it names the new backer.
//
//  ⚠️ **Interim table, recorded honestly (2026-07-31):**
//  - `denoise` → NAFNet `.signage` — the INCUMBENT, kept per BRIDGE-066's "no default change".
//    V4 measured SCUNet as the strongest blind denoiser on NIND, but against a *sibling* NAFNet
//    checkpoint, not `.signage` — the head-to-head is exactly BRIDGE-069's job.
//  - `motionDeblur` → FFTformer — the dedicated motion deblurrer (fp32). ⚠️ C2's RealBlur result is
//    weak for the whole class (every GoPro-trained method scored below the blurry input on real
//    smartphone blur), so this routing is "best of the backers we hold", not "measured good".
//  - `defocusDeblur` → Restormer `.defocusDeblur` — the only shippable defocus deblur (C4/DPDD).
//

import Foundation
import MLXServeCore
import MLXToolKit
import RestoreTier2

public struct EngineFrameRestoreProvider: FrameRestoreProvider {
    private let engine: MLXServeEngine
    private let table: [RestoreJob: (id: PackageID, name: String)]

    /// Build from a registered service. Jobs whose backer failed registration are absent from the
    /// table and throw `.unavailable` at call time — a fact the caller can show, not a crash.
    @MainActor
    public init(service: ForgeEngineService) {
        self.engine = service.engine
        var t: [RestoreJob: (PackageID, String)] = [:]
        let routing: [(RestoreJob, ForgeModel)] = [
            (.denoise, .restore),               // ⚠️ BRIDGE-069 decides this row
            (.motionDeblur, .deblurMotion),
            (.defocusDeblur, .restormerDefocus),
        ]
        for (job, model) in routing {
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
