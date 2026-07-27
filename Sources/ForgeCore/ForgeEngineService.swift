//
//  ForgeEngineService.swift
//  ForgeCore
//
//  The single owner of the MLX engine, the model store, and every package registration.
//
//  The app this was extracted from spread this across six adapters: each lazily called `useModelStore`
//  and then registered its own package on first use. That made the engine-wide store a race between
//  adapters (whoever ran first won), hid registration failures inside the first inference, and meant a
//  license refusal surfaced as "the Colorize button doesn't work" rather than as a fact about the model.
//
//  Here registration is one eager, ordered pass: every package's outcome is recorded, and the adapters
//  are handed a `PackageID` that is already known-good.
//

import Foundation
import Observation
import OSLog
import MLXServeCore
import MLXToolKit
import MLXNAFNet
import MLXRealESRGAN
import MLXBiRefNet
import MLXDDColor
import MLXInpaint
import MLXSEARAFT
import MLXEdgeTAM

let log = Logger(subsystem: "com.xocialize.ForgeCore", category: "ForgeCore")

/// The seven model packages Forge registers. One case per registration, *not* per capability —
/// Optimize alone uses two (restore + upscale), and one flow registration feeds four consumers.
public enum ForgeModel: String, CaseIterable, Sendable {
    case restore        // NAFNet
    case upscale        // Real-ESRGAN
    case matte          // BiRefNet
    case colorize       // DDColor
    case inpaint        // LaMa / MI-GAN
    case flow           // SEA-RAFT
    case promptMask     // EdgeTAM

    public var modelName: String {
        switch self {
        case .restore:    return "NAFNet"
        case .upscale:    return "Real-ESRGAN"
        case .matte:      return "BiRefNet"
        case .colorize:   return "DDColor"
        case .inpaint:    return "LaMa / MI-GAN"
        case .flow:       return "SEA-RAFT"
        case .promptMask: return "EdgeTAM"
        }
    }

    /// The engine capability this registration backs — use it to bind a progress view
    /// (`MLXEngineUI.ModelStateView(monitor:capability:)`) in a consuming app.
    public var capability: Capability {
        switch self {
        case .restore:    return .imageRestore
        case .upscale:    return .imageUpscale
        case .matte:      return .matting
        case .colorize:   return .imageColorize
        case .inpaint:    return .imageInpaint
        case .flow:       return .opticalFlow
        case .promptMask: return .promptSegment
        }
    }
}

/// What happened when a package was registered. A failure here is a fact worth showing, not an error to
/// retry: under `.permissiveOnly` + `.blocking` a `licenseRejected` means the model's weights are not
/// shippable, and `ineligible` means this Mac cannot run it. Neither improves by trying again.
public struct RegistrationOutcome: Sendable {
    public let model: ForgeModel
    public let packageID: PackageID?
    public let failure: String?

    public var isRegistered: Bool { packageID != nil }
}

@MainActor
@Observable
public final class ForgeEngineService {

    /// The engine.
    ///
    /// ⚠️ **`licenseEnforcement: .blocking` is load-bearing and must not be dropped.** As of contract
    /// 1.28.0 (engine v0.37.0) C7/C8 are *declaration* requirements, not refusals: the default
    /// `.advisory` **registers** a package with non-commercial weights and merely records a
    /// `LicenseAdvisory`. Forge is a commercial product, so it refuses instead.
    ///
    /// Nothing catches a regression here at compile time — the parameter is defaulted, so deleting it
    /// still builds and silently loosens every consuming app. Constructing the engine in exactly one
    /// place, in the shared package, is the mitigation. Runtime tell that enforcement has slipped: a
    /// `[License]` log line per advisory.
    public let engine: MLXServeEngine

    /// Registration result per model, in `ForgeModel.allCases` order once `registerAll()` has run.
    public private(set) var outcomes: [RegistrationOutcome] = []

    /// The store root actually handed to the engine — for display, and for diagnosing "it downloaded
    /// into the wrong place".
    public private(set) var attachedStoreRoot: URL?

    private var packageIDs: [ForgeModel: PackageID] = [:]

    /// - Parameters:
    ///   - policy: license gate. Leave at `.permissiveOnly` for a shipping product; a research build
    ///     may widen it.
    ///   - enforcement: leave `.blocking`. See the note on `engine`.
    public init(policy: LicensePolicy = .permissiveOnly,
                enforcement: LicenseEnforcement = .blocking) {
        self.engine = MLXServeEngine(policy: policy, licenseEnforcement: enforcement)
    }

    // MARK: - Model store

    /// Point the engine at the models folder.
    ///
    /// **Must precede every `register()`** — the engine stamps the store root onto a `ModelStorable`
    /// configuration at registration time, so a package registered before this call downloads into the
    /// default cache instead of the chosen folder.
    ///
    /// `root` is a plain URL by design: owning the folder pick and its security-scoped bookmark is the
    /// app's job (`MLXEngineUI.ModelStorageModel` does it), which keeps this package headless.
    ///
    /// - Returns: false when no folder is available. Recoverable, not fatal — a consumer can run, let
    ///   the user choose a folder, then call this and `registerAll()` again.
    @discardableResult
    public func attachModelStore(root: URL?) async -> Bool {
        await engine.useModelStore(ModelStore(root: root))
        attachedStoreRoot = root
        if let root {
            log.notice("Model store attached: \(root.path, privacy: .public)")
            return true
        }
        log.error("No models folder — weights would land in the default cache")
        return false
    }

    // MARK: - Registration

    /// Register all seven packages, recording each outcome. Registration is cheap: it runs the license
    /// and device-eligibility gates only; weights load lazily on the first `run`.
    public func registerAll() async {
        var results: [RegistrationOutcome] = []
        for model in ForgeModel.allCases {
            results.append(await register(model))
        }
        outcomes = results

        let failed = results.filter { !$0.isRegistered }
        if failed.isEmpty {
            log.notice("All \(results.count, privacy: .public) model packages registered")
        } else {
            for outcome in failed {
                log.error("\(outcome.model.modelName, privacy: .public) did not register: \(outcome.failure ?? "unknown", privacy: .public)")
            }
        }

        if let applied = await engine.appliedGPUCacheLimitBytes {
            log.notice("GPU buffer-pool cap applied by the engine: \(applied, privacy: .public) bytes")
        } else {
            log.notice("GPU buffer-pool cap not applied (unmanaged policy, or MLX's Metal device is unavailable here)")
        }
    }

    private func register(_ model: ForgeModel) async -> RegistrationOutcome {
        do {
            let id: PackageID
            switch model {
            case .restore:
                id = try await engine.register(NAFNetRestorePackage.registration,
                                               configuration: NAFNetConfiguration(variant: .siddWidth64))
            case .upscale:
                id = try await engine.register(RealESRGANUpscalePackage.registration,
                                               configuration: RealESRGANConfiguration(variant: .general))
            case .matte:
                id = try await engine.register(BiRefNetPackage.registration,
                                               configuration: BiRefNetConfiguration())
            case .colorize:
                id = try await engine.register(DDColorPackage.registration,
                                               configuration: DDColorConfiguration())
            case .inpaint:
                id = try await engine.register(InpaintPackage.registration,
                                               configuration: InpaintConfiguration())
            case .flow:
                id = try await engine.register(SEARAFTOpticalFlowPackage.registration,
                                               configuration: SEARAFTConfiguration(variant: .spring, iters: 4))
            case .promptMask:
                id = try await engine.register(EdgeTAMPackage.registration,
                                               configuration: EdgeTAMConfiguration())
            }
            packageIDs[model] = id
            return RegistrationOutcome(model: model, packageID: id, failure: nil)
        } catch {
            return RegistrationOutcome(model: model, packageID: nil, failure: Self.describe(error))
        }
    }

    /// Name *why* a registration failed, in the user's terms. `EngineError`'s payloads carry the detail
    /// (which license layer, which device dimension); dropping it turns an actionable fact into
    /// "something went wrong".
    private static func describe(_ error: Error) -> String {
        guard let engineError = error as? EngineError else { return String(describing: error) }
        switch engineError {
        case .licenseRejected(let gate):
            return "license not admitted under the active policy — \(gate)"
        case .ineligible(let eligibility):
            return "this Mac cannot run it — \(eligibility)"
        default:
            return String(describing: engineError)
        }
    }

    // MARK: - Lookup

    /// The registered package backing `model`, or nil if it failed to register. Adapters read this per
    /// call; nil means the capability is unavailable and should be presented as such rather than failing
    /// at the moment the user clicks.
    public func packageID(for model: ForgeModel) -> PackageID? { packageIDs[model] }

    public func outcome(for model: ForgeModel) -> RegistrationOutcome? {
        outcomes.first { $0.model == model }
    }
}
