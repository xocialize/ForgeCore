//
//  Anime4KUpscalePackage.swift
//  ForgeCore
//
//  The Instant tier (AB-D-0146) as an engine registration: Anime4K's upscale-only VL network as Metal compute shaders
//  (`anime4k-metal`), wrapped as a weightless `imageUpscale` ModelPackage so an engine client — mlx-image-runtime and the
//  Pixel Agency Image Engine's MES `image.upscale` with `model: "anime4k-upscale"` (AB-A-0183) — reaches it through the
//  same routing, admission, licence stamp and descriptor every other upscale tier gets.
//
//  It is the `mlx-audio-polish-swift` precedent for a visual capability: no weights, nothing to download, no MLX.
//  Unlike AudioPolish it runs on the GPU, so it requires `.metalGPU` and declares the GPU working set of one tile.
//
//  ForgeOptimizer's own `EngineImageEnhancer` keeps running Instant directly (no engine hop, works with nothing
//  registered); this package shares its upscaler and its factor arithmetic (`InstantUpscalePlan`), so the two paths
//  produce the same pixels for the same request.
//

import Foundation
import CoreGraphics
import Anime4KMetal
import MLXToolKit

/// Init-time configuration for `Anime4KUpscalePackage` (C9). Nothing to choose: the mode is the upscaler's default
/// (upscale-only VL, AB-D-0146), and its 1024-px tile is what `footprints` was measured for.
public struct Anime4KUpscaleConfiguration: PackageConfiguration {
    public init() {}
}

/// An MLXEngine `imageUpscale` package over **Anime4K VL** (bloc97/Anime4K, MIT; the Metal port is xocialize's, MIT): the
/// **efficiency** tier — instant and model-free, never a quality rival to NERVE or HEART (AB-R-0559).
///
/// Scale routing (`appliedScale` reports what ran): Anime4K doubles per pass, so ×2, ×4 and ×8 are one, two and three
/// doublings, and ×6 is ×8 fitted down with Lanczos (`InstantUpscalePlan`, AB-D-0119's rule). `nil` is the native ×2.
/// ×1 returns the image as it came; ×3 is ×4 fitted down.
///
/// Output mirrors the input: `.rawBGRA8` in ⇒ `.rawBGRA8` out, `.rawRGBA16` in ⇒ `.rawRGBA16` out (the chain runs at fp32
/// for a deep image), anything else ⇒ PNG (16-bit when the source was deep). Alpha is kept.
@InferenceActor
public final class Anime4KUpscalePackage: ModelPackage {
    public typealias Configuration = Anime4KUpscaleConfiguration

    /// What receipts and offers call it (`EngineImageEnhancer.instantModelName` names the same pixels).
    public nonisolated static let modelName = "Anime4K VL"

    public nonisolated static var manifest: PackageManifest {
        PackageManifest(
            // C7/C8: no weights — the network's constants are bloc97/Anime4K's shaders (MIT), and the port is MIT.
            license: LicenseDeclaration(weightLicense: .mit, portCodeLicense: .mit),
            provenance: Provenance(sourceRepo: "xocialize/anime4k-metal", revision: "v0.1.0", tier: 1),
            requirements: RequirementsManifest(
                footprints: Anime4KUpscalePackage.footprints,
                // Metal compute shaders, no CPU fallback (`Anime4KUpscaler.isAvailable`).
                requiredBackends: [.metalGPU],
                os: OSRequirement(minMacOS: SemanticVersion(major: 26, minor: 0, patch: 0)),
                chipFloor: nil
            ),
            specialties: [],
            surfaces: [
                ImageUpscaleContract.descriptor(
                    name: "anime4k-upscale",
                    summary: "Anime4K VL: instant, model-free ×2 super-resolution on Metal shaders (×4/×8 by doubling, "
                        + "×6 fitted from ×8). The efficient choice when speed matters more than the learned tiers' quality; "
                        + "nothing to download."
                )
                .declaring(rawImageFormats: [.rawBGRA8, .rawRGBA16])
            ]
        )
    }

    /// What one call holds at most: one tile's GPU textures (`Anime4KUpscaler.memoryEstimate` at the default 1024-px tile,
    /// transparent image — the largest case: 231 MB 8-bit, 445 MB 16-bit, measured by `InstantPackageTests`) plus the
    /// whole-image buffers around it (source, doubled output and the raw copy), sized here for a 4096² output (~0.2 GB at
    /// 8 bits, ~0.4 GB at 16). fp16 is the 8-bit lane, fp32 the 16-bit one; with no `QuantConfigured` the governor
    /// charges the largest that fits, i.e. the deep case. Resident is the compiled pipelines only, an upper bound: the
    /// textures are released after every call.
    nonisolated static var footprints: [QuantFootprint] {
        [
            QuantFootprint(quant: .fp16, residentBytes: residentBytes, peakActivationBytes: 500_000_000),
            QuantFootprint(quant: .fp32, residentBytes: residentBytes, peakActivationBytes: 1_000_000_000),
        ]
    }

    nonisolated static let residentBytes: UInt64 = 16_000_000

    private var upscaler: Anime4KUpscaler?

    public nonisolated init(configuration: Configuration) {}

    /// Compiles the Metal pipelines. Nothing to download or page in. Without a Metal device the engine refuses the
    /// registration first (`.metalGPU`); should one still get here, `Anime4KError.noDevice` says why.
    public func load() async throws {
        guard upscaler == nil else { return }
        upscaler = try Anime4KUpscaler()
    }

    public func unload() async {
        upscaler?.purge()
        upscaler = nil
    }

    public func run(_ request: any CapabilityRequest) async throws -> any CapabilityResponse {
        // CAN-1: the entry checkpoint is the FIRST act of run(). Mid-run: one checkpoint per doubling (each is a
        // millisecond-scale GPU pass, the CAN-3 sub-second exemption within a pass).
        try Task.checkCancellation()
        guard let upscaler else { throw PackageError.notLoaded }
        guard request.capability == .imageUpscale, let req = request as? ImageUpscaleRequest else {
            throw PackageError.unsupportedCapability(request.capability)
        }

        let source = try EngineImageCoding.cgImage(from: req.image)
        let asked = req.scale ?? 2
        let plan = InstantUpscalePlan.make(requested: asked, source: .init(width: source.width, height: source.height))

        var out = source
        RunProgress.report(.upsample, step: 0, totalSteps: plan.doublings)
        for pass in 0 ..< plan.doublings {
            try Task.checkCancellation()
            out = try upscaler.upscale(out)
            RunProgress.report(.upsample, step: pass + 1, totalSteps: plan.doublings)
        }
        if let fit = plan.fit {
            try Task.checkCancellation()
            RunProgress.report(.postprocess)
            out = try ImageResampling.lanczos(out, toWidth: fit.width, height: fit.height)
        }

        let image: Image
        switch req.image.format {
        case .rawBGRA8: image = try EngineImageCoding.rawImage(from: out, keepingDepth: false)
        case .rawRGBA16: image = try EngineImageCoding.rawImage(from: out, keepingDepth: true)
        default: image = try EngineImageCoding.pngImage(from: out)
        }
        return ImageUpscaleResponse(image: image, appliedScale: plan.requested)
    }
}

extension Anime4KUpscalePackage {
    /// The author one-liner the engine registers.
    public nonisolated static var registration: PackageRegistration {
        .of(Anime4KUpscalePackage.self)
    }
}
