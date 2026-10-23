//
//  GenerativeFillTier.swift
//  ForgeCore
//
//  Which model fills on which Mac, as CONFIGURATION (AB-A-0166). The provider never names a model in code: a host
//  hands `ForgeEngineService` a tier, and `.generativeFill` registers whatever that tier says. mlx-image's quality gate
//  (AB-A-0178, AB-R-0544) passed both SenseNova int4 tiers, and the 16 GB tier runs SenseNova inside its own declared
//  window (`SenseNovaWindow.fill768`, AB-T-0278, AB-R-0602). Klein is not a 16 GB fallback (AB-D-0141, amended
//  2026-10-22). A host still swaps a tier without a new provider or a new ForgeCore.
//

import Foundation
import MLXToolKit
import MLXSenseNovaU1
import MLXKlein

/// The model behind a Generative Fill tier.
public enum GenerativeFillModel: Sendable, Hashable, Codable {
    /// SenseNova-U1.5-8B-MoT at an artifact tier. Fill wants a distill tier — `.fast4V2` / `.fast8V2` (the 8-step V2
    /// adapter): every tier serves `imageEdit` since sensenova-u1-swift 0.5.1 (AB-R-0490), and the V2 8-step cfg-free edit
    /// is what the bake-off measured (AB-R-0534, at `.fast8V2`). `window` is the declaration the engine charges: nil is
    /// the package's native envelope; `.fill768` is the 16 GB tier's (11.8 + 0.75 GB over the same artifact, ≤ 768² by
    /// area, one reference, cfg-free — sensenova-u1-swift 0.6.0, AB-T-0278). A tier stored without a window decodes as nil.
    case senseNova(variant: SenseNovaVariant, window: SenseNovaWindow? = nil)
    /// FLUX.2-klein-4B (the distilled 4-step turbo) at `quant`; `evictEncoder` loads the Qwen3-4B conditioner per call and
    /// evicts it before the denoise — a light tier (MLX peak 8.9 GB at 768², AB-R-0534), not a 16 GB fallback (AB-D-0141).
    case klein(quant: Quant, evictEncoder: Bool)

    /// What a receipt names: the model and the exact tier.
    public var displayName: String {
        switch self {
        case .senseNova(let variant, _): "SenseNova-U1.5-8B-MoT \(variant.rawValue)"
        case .klein(let quant, let evict): "FLUX.2-klein-4B \(quant.rawValue)" + (evict ? " (encoder evicted)" : "")
        }
    }

    /// The weight quantization the tier runs.
    public var quant: Quant {
        switch self {
        case .senseNova(let variant, _): variant.quant
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

    /// The largest render, in pixels, the model's declaration covers; nil = no cap beyond the tier's own.
    public var declaredMaxPixels: Int? {
        switch self {
        case .senseNova(_, let window): window?.maxPixels
        case .klein: nil
        }
    }

    /// The registration and configuration `.generativeFill` hands the engine. Package defaults otherwise (the BRIDGE-067
    /// discipline): SenseNova's distill tiers run 8 steps at cfg 1, Klein's turbo 4 steps at guidance 1 — the bake-off's
    /// settings, which the route does not override.
    var registration: (PackageRegistration, any PackageConfiguration) {
        switch self {
        case .senseNova(let variant, let window):
            (SenseNovaU1Package.registration, SenseNovaU1Configuration(variant: variant, window: window ?? .native))
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

    /// The route this tier fills through: its window cap, held inside the model's declared window, so a hand-written tier
    /// whose cap is larger than its declaration renders at the declaration's size instead of being refused by it.
    var route: GenerativeFillRoute {
        GenerativeFillRoute(maxWindowPixels: min(maxWindowPixels, model.declaredMaxPixels ?? .max),
                            multipleOf: model.multipleOf)
    }

    // AB-D-0141 as amended 2026-10-14: SenseNova on every tier.

    /// 16 GB: SenseNova 8step-v2-4bit in its `.fill768` window (≤ 768²), declared 11.8 GB resident + 0.75 GB activation
    /// = 12.55 GB, which the engine admits on a 16 GB Mac's 12.7 GB budget (sensenova-u1-swift 0.6.0, AB-T-0278). The
    /// native int4 declaration (12.2 + 4.5 GB) is still refused there. Confirmed on a real 16 GB Mac (M1 Pro, AB-R-0602):
    /// 768² MLX peak 11,930 MiB, process footprint 12.60 GB, about 41–43 s per fill warm. One condition the provider
    /// cannot meet for the host: every other MLX resident evicted first (AB-D-0141).
    public static let senseNova16GB = GenerativeFillTier(label: "16 GB", model: .senseNova(variant: .fast4V2, window: .fill768),
                                                         maxWindowPixels: 768 * 768)
    /// 24 GB: SenseNova 8step-v2-4bit at a ≤ 1024² window (12.2 GB MLX peak at 1024², AB-R-0536).
    public static let senseNova24GB = GenerativeFillTier(label: "24 GB", model: .senseNova(variant: .fast4V2),
                                                         maxWindowPixels: 1024 * 1024)
    /// 32 GB+: SenseNova 8step-v2-8bit at a ≤ 1024² window (19.9 GB MLX peak at 1024², AB-R-0536) — the bake-off's arm.
    public static let senseNova32GB = GenerativeFillTier(label: "32 GB+", model: .senseNova(variant: .fast8V2),
                                                         maxWindowPixels: 1024 * 1024)
    /// Klein-4B int8, encoder evicted, ≤ 768² (10.7 GB MLX peak, AB-R-0534). **Not a shipping configuration and not the
    /// 16 GB fallback**: Klein stays off 16 GB (AB-D-0141, amended 2026-10-22) — its light tiers were never confirmed on
    /// a 16 GB Mac, and KleinConfiguration's evict hints (5.2 + 5.0 GB) don't vary with the quant, so int8 is
    /// under-declared (AB-R-0541). Int4 failed mlx-image's quality gate (AB-R-0544). Kept for a host that chooses it.
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

    /// `forgeDefault` with Klein-4B int8 (encoder evicted, ≤ 768²) at 16 GB. **Not a shipping configuration and not the
    /// 16 GB fallback** (AB-D-0141, amended 2026-10-22): SenseNova's `.fill768` window is the 16 GB tier.
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
