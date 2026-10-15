//
//  GenerativeFillTier.swift
//  ForgeCore
//
//  Which model fills on which Mac, as CONFIGURATION (AB-A-0166). The provider never names a model in code: a host
//  hands `ForgeEngineService` a tier, and `.generativeFill` registers whatever that tier says. mlx-image's quality gate
//  (AB-A-0178, AB-R-0544) passed both SenseNova int4 tiers and failed Klein int4, so the 16 GB fallback is Klein int8. A
//  host still swaps a tier (e.g. to `klein16GB`) without a new provider or a new ForgeCore.
//

import Foundation
import MLXToolKit
import MLXSenseNovaU1
import MLXKlein

/// The model behind a Generative Fill tier.
public enum GenerativeFillModel: Sendable, Hashable, Codable {
    /// SenseNova-U1.5-8B-MoT at an artifact tier. Fill wants a distill tier — `.fast4V2` / `.fast8V2` (the 8-step V2
    /// adapter): every tier serves `imageEdit` since sensenova-u1-swift 0.5.1 (AB-R-0490), and the V2 8-step cfg-free edit
    /// is what the bake-off measured (AB-R-0534, at `.fast8V2`).
    case senseNova(variant: SenseNovaVariant)
    /// FLUX.2-klein-4B (the distilled 4-step turbo) at `quant`; `evictEncoder` loads the Qwen3-4B conditioner per call and
    /// evicts it before the denoise — the 16 GB light tier (MLX peak 8.9 GB at 768², AB-R-0534).
    case klein(quant: Quant, evictEncoder: Bool)

    /// What a receipt names: the model and the exact tier.
    public var displayName: String {
        switch self {
        case .senseNova(let variant): "SenseNova-U1.5-8B-MoT \(variant.rawValue)"
        case .klein(let quant, let evict): "FLUX.2-klein-4B \(quant.rawValue)" + (evict ? " (encoder evicted)" : "")
        }
    }

    /// The weight quantization the tier runs.
    public var quant: Quant {
        switch self {
        case .senseNova(let variant): variant.quant
        case .klein(let quant, _): quant
        }
    }

    /// The model's size grid: SenseNova's pixel token is 32 px (`sizeNotTokenAligned` otherwise), Klein's latent patch 16.
    public var multipleOf: Int {
        switch self {
        case .senseNova: 32
        case .klein: 16
        }
    }

    /// The registration and configuration `.generativeFill` hands the engine. Package defaults otherwise (the BRIDGE-067
    /// discipline): SenseNova's distill tiers run 8 steps at cfg 1, Klein's turbo 4 steps at guidance 1 — the bake-off's
    /// settings, which the route does not override.
    var registration: (PackageRegistration, any PackageConfiguration) {
        switch self {
        case .senseNova(let variant):
            (SenseNovaU1Package.registration, SenseNovaU1Configuration(variant: variant))
        case .klein(let quant, let evict):
            (Klein4BT2IPackage.registration, KleinConfiguration.turbo(quant: quant, evictEncoder: evict))
        }
    }
}

/// One memory tier's Generative Fill setting.
public struct GenerativeFillTier: Sendable, Hashable, Codable {
    /// The tier's name in receipts and UI ("16 GB", "24 GB", "32 GB+").
    public var label: String
    public var model: GenerativeFillModel
    /// The largest window the model is asked to render, in pixels. A larger picture is filled through a window around
    /// the selection (`GenerativeFillRoute.window`); an expand's whole canvas is scaled into it.
    public var maxWindowPixels: Int

    public init(label: String, model: GenerativeFillModel, maxWindowPixels: Int) {
        self.label = label; self.model = model; self.maxWindowPixels = maxWindowPixels
    }

    // AB-D-0141 as amended 2026-10-14: SenseNova on every tier.

    /// 16 GB: SenseNova 8step-v2-4bit at a ≤ 768² window. 12.51 GB working set under the 12.7 GB budget (AB-R-0537), with
    /// three conditions the provider cannot meet for the host: every other MLX resident evicted first, a 16 GB footprint
    /// declaration in sensenova-u1-swift (AB-T-0278 — today's 12.2 + 4.5 GB declaration is refused on a 16 GB budget), and
    /// a real-16-GB-Mac confirmation before it ships.
    public static let senseNova16GB = GenerativeFillTier(label: "16 GB", model: .senseNova(variant: .fast4V2),
                                                         maxWindowPixels: 768 * 768)
    /// 24 GB: SenseNova 8step-v2-4bit at a ≤ 1024² window (12.2 GB MLX peak at 1024², AB-R-0536).
    public static let senseNova24GB = GenerativeFillTier(label: "24 GB", model: .senseNova(variant: .fast4V2),
                                                         maxWindowPixels: 1024 * 1024)
    /// 32 GB+: SenseNova 8step-v2-8bit at a ≤ 1024² window (19.9 GB MLX peak at 1024², AB-R-0536) — the bake-off's arm.
    public static let senseNova32GB = GenerativeFillTier(label: "32 GB+", model: .senseNova(variant: .fast8V2),
                                                         maxWindowPixels: 1024 * 1024)
    /// AB-D-0141's documented 16 GB fallback: Klein-4B int8, encoder evicted, ≤ 768² (10.7 GB MLX peak, AB-R-0534). Int4
    /// failed mlx-image's quality gate against bf16 (8/15 prompt failures vs 6/15, blind 1:4); int8 reproduces bf16
    /// (AB-R-0544, AB-A-0182). Its fill quality is still below SenseNova's (AB-R-0534). KleinConfiguration's evict hints
    /// (5.2 + 5.0 GB) don't vary with the quant, so int8 is under-declared until flux2-klein-swift re-declares (AB-T-0290).
    public static let klein16GB = GenerativeFillTier(label: "16 GB", model: .klein(quant: .int8, evictEncoder: true),
                                                     maxWindowPixels: 768 * 768)
}

/// The tier → model/quant/window mapping, keyed by the Mac's unified memory.
public struct GenerativeFillTiers: Sendable, Hashable, Codable {
    public struct Entry: Sendable, Hashable, Codable {
        /// The smallest Mac (in GiB of unified memory) this entry serves.
        public var minimumMemoryGiB: Int
        public var tier: GenerativeFillTier

        public init(minimumMemoryGiB: Int, tier: GenerativeFillTier) {
            self.minimumMemoryGiB = minimumMemoryGiB; self.tier = tier
        }
    }

    public var entries: [Entry]

    public init(entries: [Entry]) { self.entries = entries }

    /// AB-D-0141 (amended): SenseNova on every tier — 16 GB 8step-v2-4bit ≤ 768², 24 GB 8step-v2-4bit ≤ 1024², 32 GB+
    /// 8step-v2-8bit ≤ 1024². Nothing below 16 GB.
    public static let forgeDefault = GenerativeFillTiers(entries: [
        Entry(minimumMemoryGiB: 16, tier: .senseNova16GB),
        Entry(minimumMemoryGiB: 24, tier: .senseNova24GB),
        Entry(minimumMemoryGiB: 32, tier: .senseNova32GB),
    ])

    /// `forgeDefault` with AB-D-0141's documented 16 GB fallback (Klein-4B int8, encoder evicted, ≤ 768²) — for when the
    /// real-16-GB-Mac run or AB-T-0278's 16 GB declaration refuses SenseNova there (its quality gate passed, AB-R-0544).
    public static let kleinAt16GB = GenerativeFillTiers(entries: [
        Entry(minimumMemoryGiB: 16, tier: .klein16GB),
        Entry(minimumMemoryGiB: 24, tier: .senseNova24GB),
        Entry(minimumMemoryGiB: 32, tier: .senseNova32GB),
    ])

    /// The tier for a Mac with `bytes` of unified memory: the entry with the largest minimum at or below it, or nil
    /// below every entry. Counted in whole GiB, so 16 and 18 GB Macs both land on the 16 GB entry.
    public func tier(forPhysicalMemory bytes: UInt64) -> GenerativeFillTier? {
        let gib = Int(bytes / (1 << 30))
        return entries.filter { $0.minimumMemoryGiB <= gib }.max { $0.minimumMemoryGiB < $1.minimumMemoryGiB }?.tier
    }
}
