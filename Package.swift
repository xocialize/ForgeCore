// swift-tools-version: 6.2
import PackageDescription

// ForgeCore — category B (engine adapters), per CLAUDE.md "Package categories".
//
// B owns the heavy pins: mlx-engine-swift and every mlx-*-swift wrapper resolve HERE and nowhere else,
// so two consuming projects cannot diverge on Package.resolved.
//
// What this package deliberately does NOT depend on:
//
//   • vpx-swift — it vendors vpx.xcframework, and B's net-consumable bar forbids vendored binaries.
//     Linking it is also what accepts the libvpx PATENTS grant, which is a consumer's decision to make,
//     not one to inherit silently. The app registers `VpxVideoDecoder` into media-bridge itself.
//   • MLXEngineUI — ForgeCore stays headless so its tests run without SwiftUI and a CLI can consume it.
//     The models-folder root arrives as a URL; owning `ModelStorageModel` (folder pick, security-scoped
//     bookmark) is the app's job, and the reference app shows how.
//   • Any *UI package — the SwiftUI capability classes are category C/A-UI, above this seam.

let package = Package(
    name: "ForgeCore",
    // Floats to what MLX needs. This is precisely why B cannot fold into A: the net-clean Kits keep
    // their macOS-14 floor because this package carries the 26.x/Metal requirement instead.
    platforms: [.macOS(.v26)],
    products: [
        .library(name: "ForgeCore", targets: ["ForgeCore"]),
    ],
    dependencies: [
        // A — net-clean Kits owning the provider protocols these adapters implement.
        // 0.11.0 = the first tag carrying the §6.3 planner-hint seam (`ContentHintProvider` /
        // `ContentHint` / `HintedStartPolicy`), which `EngineContentHintProvider` implements. It
        // supersedes the old 0.5.0 floor (`webOptimize`); that verb is still in the surface, a
        // higher floor being the stronger guarantee. ⚠️ URL-at-a-tag, never a path — mixing the two
        // for one package identity jams the whole graph at the lowest pin any consumer holds
        // (`Docs/LESSONS.md` "SPM gotcha", 2026-06-24).
        .package(url: "https://github.com/xocialize/ForgeOptimizerKit.git", from: "0.11.0"),
        .package(url: "https://github.com/xocialize/ExtractKit.git", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/ColorizeKit.git", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/EraseKit.git", from: "0.2.0"),
        .package(url: "https://github.com/xocialize/MaskKit.git", from: "0.1.0"),
        // 0.13.0 = ImageQualityTarget.encodePNG + VideoQualityTarget.EncodeProfile (the web tier).
        .package(url: "https://github.com/xocialize/media-bridge.git", from: "0.13.0"),

        // The engine. v0.37.0 is the FIRST tag carrying `licenseEnforcement` (contract 1.28.0) —
        // do not lower this floor, or `.blocking` silently stops being expressible.
        .package(url: "https://github.com/xocialize/mlx-engine-swift.git", from: "0.43.0"),

        // Model wrappers — the heavy pins.
        .package(url: "https://github.com/xocialize/mlx-nafnet-swift.git", from: "0.4.0"),
        // The frame-restore wave (BRIDGE-066) + face pair (064-face) + relight (BRIDGE-067) —
        // registered 2026-07-31, ending the state where shipped packages were invisible to apps.
        .package(url: "https://github.com/xocialize/mlx-fftformer-swift.git", from: "0.1.1"),
        .package(url: "https://github.com/xocialize/mlx-restormer-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/mlx-drunet-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/mlx-scunet-swift.git", from: "0.1.1"),
        .package(url: "https://github.com/xocialize/mlx-cidnet-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/mlx-gfpgan-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/mlx-restoreformer-swift.git", from: "0.1.0"),
        .package(url: "https://github.com/xocialize/mlx-bopbtl-swift.git", from: "0.2.0"),
        // Category A Kit — RestoreTier2 is the provider-protocol seam this package implements
        // (BRIDGE-070). One-way: the Kit never depends back on ForgeCore.
        .package(url: "https://github.com/xocialize/ForgeRestoreKit.git", from: "0.10.0"),
        .package(url: "https://github.com/xocialize/mlx-realesrgan-swift.git", from: "0.7.0"),
    // The quality tier behind `imageUpscale` — a second backer, not a replacement.
    .package(url: "https://github.com/xocialize/mlx-seedvr2-swift.git", from: "0.9.1"),
        .package(url: "https://github.com/xocialize/mlx-birefnet-swift.git", from: "0.4.0"),
        // 🔑 **The no-reference half of the two-oracle rule** (`CLAUDE.md`: *"SigLIP2 NR-IQA gates
        // whether to restore … SSIMULACRA2 gates how hard to compress"*). ForgeCore shipped the
        // full-reference oracle only (MediaMeasure, via media-bridge), which made every quality
        // comparison a fidelity comparison by default — and V10 measured the cost of that: a
        // full-reference metric structurally penalises a generative upscaler for inventing
        // plausible-but-different detail more than it penalises an interpolator for inventing
        // nothing. Both oracles must be linkable here or the rule cannot be honoured by a consumer.
        .package(url: "https://github.com/xocialize/mlx-siglip2-iqa-swift.git", from: "0.3.0"),
        .package(url: "https://github.com/xocialize/mlx-ddcolor-swift.git", from: "0.2.0"),
        .package(url: "https://github.com/xocialize/mlx-lama-swift.git", from: "0.2.0"),
        .package(url: "https://github.com/xocialize/mlx-sea-raft-swift.git", from: "0.5.0"),
        .package(url: "https://github.com/xocialize/mlx-edgetam-swift.git", from: "0.4.1"),
        // frameInterpolate — RIFE 4.25 (AB-T-0038). Ported and parity-locked 2026-06-18, but no
        // registration existed anywhere, so the capability was unreachable by every app until
        // 2026-08-14. 0.8.0 = the CAN-gate retrofit tag.
        .package(url: "https://github.com/xocialize/mlx-rife-swift.git", from: "0.8.0"),
        // contentClassify — V-JEPA2 ViT-L, the §6.3 planner hint's feature extractor (AB-T-0045).
        // 0.5.0 = the CAN-gate retrofit tag, the same floor discipline RIFE gets; the adapter sketch
        // said ≥ 0.2.0 before the gate landed. 🔑 It is here for the **[1024] router embedding**, not
        // the tags: the SSv2 attentive probe emits bare `ssv2_<index>` ids (the checkpoint ships no
        // label names) and that vocabulary is hand-object actions, semantically wrong for signage.
        // Never build a label→class mapping from it.
        .package(url: "https://github.com/xocialize/mlx-vjepa2-swift.git", from: "0.5.0"),
    ],
    targets: [
        .target(
            name: "ForgeCore",
            dependencies: [
                .product(name: "ForgeOptimizerKit", package: "ForgeOptimizerKit"),
                .product(name: "ExtractKit", package: "ExtractKit"),
                .product(name: "ColorizeKit", package: "ColorizeKit"),
                .product(name: "EraseKit", package: "EraseKit"),
                .product(name: "MaskKit", package: "MaskKit"),
                .product(name: "MediaMeasure", package: "media-bridge"),
                .product(name: "MLXServeCore", package: "mlx-engine-swift"),
                .product(name: "MLXToolKit", package: "mlx-engine-swift"),
                .product(name: "MLXNAFNet", package: "mlx-nafnet-swift"),
                .product(name: "MLXFFTformer", package: "mlx-fftformer-swift"),
                .product(name: "MLXRestormer", package: "mlx-restormer-swift"),
                .product(name: "MLXDRUNet", package: "mlx-drunet-swift"),
                .product(name: "MLXSCUNet", package: "mlx-scunet-swift"),
                .product(name: "MLXCIDNet", package: "mlx-cidnet-swift"),
                .product(name: "MLXGFPGAN", package: "mlx-gfpgan-swift"),
                .product(name: "MLXRestoreFormer", package: "mlx-restoreformer-swift"),
                .product(name: "MLXBOPBTL", package: "mlx-bopbtl-swift"),
                .product(name: "RestoreTier2", package: "ForgeRestoreKit"),
                .product(name: "MLXRealESRGAN", package: "mlx-realesrgan-swift"),
                .product(name: "MLXSeedVR2", package: "mlx-seedvr2-swift"),
                .product(name: "MLXBiRefNet", package: "mlx-birefnet-swift"),
                .product(name: "MLXSigLIP2IQA", package: "mlx-siglip2-iqa-swift"),
                .product(name: "MLXDDColor", package: "mlx-ddcolor-swift"),
                .product(name: "MLXInpaint", package: "mlx-lama-swift"),
                .product(name: "MLXSEARAFT", package: "mlx-sea-raft-swift"),
                .product(name: "MLXEdgeTAM", package: "mlx-edgetam-swift"),
                .product(name: "MLXRIFE", package: "mlx-rife-swift"),
                .product(name: "MLXVJEPA2", package: "mlx-vjepa2-swift"),
            ]
        ),
        // ⚠️ Run with `swift test --build-system swiftbuild`. Importing MLX inherits the metallib
        // requirement, and the deprecated `native` build system ships no metallib — it aborts the whole
        // xctest process while the log still reads "all tests passed".
        .testTarget(name: "ForgeCoreTests", dependencies: ["ForgeCore"]),
    ]
)
