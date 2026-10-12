# ForgeCore

The MLX seam for Forge: the engine service, the image-coding helper, and the engine-backed provider
adapters — extracted so **more than one app can consume one capability surface**.

Category **B** (engine adapters) under `XocializeForge/CLAUDE.md` § *Package categories*. It may import
MLX; it owns the heavy pins; nothing in category A may depend on it.

## Why this package exists

The net-clean Kits (`ForgeOptimizerKit`, `ExtractKit`, `ColorizeKit`, `EraseKit`, `MaskKit`) each declare
a narrow provider protocol for anything they need a model for, and stay free of MLX so a host that wants
the planner isn't forced into a GPU toolchain. Something still has to implement those protocols against a
real engine. That implementation used to live in an app target — which meant a second app could only get
it by copying the files, and copies drift.

So: the adapters live here, and both the reference app and the production Metal-UI project consume this
package at a tag. One `Package.resolved` decides which engine and which model wrappers everyone runs.

## Using it

```swift
let forge = ForgeEngineService()                  // .permissiveOnly + .blocking
await forge.attachModelStore(root: modelsFolder)  // MUST precede registration
await forge.registerAll()
let capabilities = ForgeCapabilities(service: forge)

let restored = try await capabilities.enhancer.enhance(image, options: options)
let matte    = try await capabilities.matte.matte(image, quality: .best)
```

Three ordering facts that are easy to get wrong, and each cost a bug once:

1. **`attachModelStore` before `registerAll`.** The engine stamps the store root onto a `ModelStorable`
   configuration *at registration*, so a package registered first downloads into the default cache
   instead of the chosen folder. If the user changes folders, call **both** again.
2. **Never pass `modelsRootDirectory:` to a package configuration.** The engine auto-stamps it (contract
   1.24). Passing it by hand is redundant at best and contradicts the store at worst.
3. **Never write `GPU.set(cacheLimit:)` or `MLX.Memory.cacheLimit`.** Since engine 0.21.0 the buffer pool
   is bounded at `MLXServeEngine` construction; an app-side write fights it. Read
   `engine.appliedGPUCacheLimitBytes` to see what was applied.

## Video upscale: declare what the clip is

Two `videoUpscale` tiers, and the route between them is the **caller's declaration**, not a guess:

```swift
let route = ForgeEngineService.videoUpscaleRoute(for: .liveAction)   // .upscaleVideoLiveAction (FlashVSR)
let gate  = ForgeEngineService.liveActionVideoUpscaleFits(sourceWidth: 640, sourceHeight: 360, budgetBytes: budget)
let out   = try await forge.upscaleVideo(video, scale: 4, content: .liveAction)   // out.model names the tier that ran
```

- **`.liveAction` → FlashVSR** (`.upscaleVideoLiveAction`, id `flashvsr-upscale`): one-step streaming diffusion VSR,
  much faster than SeedVR2 and sharper on resolution-limited faces — 0.72 s per output frame end to end at 1280×768,
  ~2.1 at 1920×1152 (M5 Max, idle GPU; REQUIREMENTS.md). It is **generative** and renders anime, cartoons
  and graphics as photographic texture — so it is reached only by declaration.
- **`.drawn` / `.unspecified` → SeedVR2** (`.upscaleQuality`): exactly what `upscaleVideo(_:scale:)` always did.

Forge has no validated detector for live action vs drawn content (the mechanical classifier is off by measurement and
the V-JEPA2 hint ships inert), so **ask the user** — a "Live action / Animation & graphics" choice — rather than
defaulting to `.liveAction`. A routed tier that cannot run throws its own reason (`DiagnosticsError.notRegistered`, or
the engine's refusal above FlashVSR's measured 1920×1152 output ceiling); it is never swapped for the other tier.
`liveActionVideoUpscaleFits` answers "can this machine run it" from the package's own declared footprint before
anything loads.

**In the Optimizer** that choice is the upscale control's third segment, *Live action*, offered for a video source
only (ForgeOptimizerKit 0.20.0, `UpscaleTier.liveAction`). `ForgeCapabilities.videoUpscaler` (`EngineVideoUpscaler`)
implements the Kit's `VideoUpscaler` seam over the `.liveAction` route: `availability(of:width:height:factor:)` answers
per clip — not registered, not eligible, past the measured output ceiling (and whether ×2 is inside it), or over this
Mac's budget, each with its numbers — and `upscale` reports the package's own frame count as progress (the engine's
`RunProgress` `.upsample` reports). The per-frame `EngineImageEnhancer` refuses the tier on a still.

## The planner hint (§6.3) ships inert — deliberately

`EngineContentHintProvider` implements `ForgeOptimizerKit`'s `ContentHintProvider` over V-JEPA2
`contentClassify`, so the optimizer can classify a clip *before* its first quality search instead of
discovering the content class from that search's landing and paying a second one.

**It cannot currently change anything, and that is the shipped behavior.** Every hint's confidence is
clamped to `inertConfidenceCeiling` (0.5), below the Kit's `HintedStartPolicy.minConfidence` (0.9), so
the planner observes each classification and acts on none of it. The clamp is unconditional — no
parameter, no environment variable — because the alternative is worse than useless: an adapter
reporting high confidence from an unfitted head would be laundering a guess through the very gate that
exists to catch guesses. LESSONS' rule is that *the search's landing IS a content classifier; behavior
outranks the axis label*, so a number the 0.9 gate can be honest about has to be earned against
encoder behavior first.

Two consequences worth stating plainly:

- **Injecting it today costs and buys nothing yet.** Each classification pays a ~1.2 GB fp32 V-JEPA2
  load plus a full copy of the input, and saves no search. The one reason to inject it now is to
  collect calibration rows — pass `observe:` and pair each `[1024]` embedding with that item's
  optimize receipt. An app that just wants faster optimize leaves `hintProvider` nil and gets the
  Kit's behavioral ratchet for free.
- **The tags are not the signal and never will be.** V-JEPA2's checkpoint ships no label names, so
  its top-k comes back as bare `ssv2_<index>` ids from an SSv2 hand-object action vocabulary that is
  semantically wrong for signage. `GraphicProbe` reads the pooled router embedding. Do not build a
  label→class mapping; `MODEL-CATALOG.md` has said so since the port.

Unlocking is the last step of the work order, in this order: fit the logistic probe on
behaviorally-labeled rows → receipt the held-out disagreement rate → raise the ceiling in
`EngineContentHintProvider.swift`. The Kit-side post-hoc audit (`hint_outcome`) stays on forever
afterward; it is the drift alarm. Full protocol: `Docs/VJEPA2-HINT-ADAPTER.md`, AB-T-0045.

`registerAll()` is eager and records a `RegistrationOutcome` per model rather than failing lazily inside
the first inference. A failure is a fact to display — under `.permissiveOnly` + `.blocking`,
`licenseRejected` means the weights aren't shippable and `ineligible` means this Mac can't run it. Neither
improves by retrying.

## License enforcement

The engine is constructed `licenseEnforcement: .blocking`, and that is load-bearing. Contract 1.28.0
(engine v0.37.0) made C7/C8 *declaration* requirements: the default `.advisory` **registers** a package
with non-commercial weights and merely records a `LicenseAdvisory`. Forge is commercial, so it refuses.

The parameter is defaulted, so **nothing catches a regression at compile time** — deleting it still
builds and silently loosens every consumer. Constructing the engine in exactly one shared place is the
mitigation. Runtime tell that enforcement has slipped: a `[License]` log line per advisory.

This is also why the `mlx-engine-swift` floor is `0.37.0` and must not be lowered — v0.37.0 is the first
tag on which `.blocking` is expressible at all.

## What is deliberately absent

- **`vpx-swift`** — it vendors `vpx.xcframework`, and category B forbids vendored binaries. Linking it is
  also what accepts the libvpx PATENTS grant, which is a consumer's decision, not one to inherit
  silently. A consumer that wants VP9/WebM ingest links it itself and calls
  `MediaBridge.register(externalDecoder: VpxVideoDecoder())` once at startup.
- **`MLXEngineUI`** — this package stays headless so its tests run without SwiftUI and a CLI can consume
  it. The models root arrives as a `URL`; owning the folder pick and its security-scoped bookmark
  (`ModelStorageModel`) is the app's job. The reference app shows the wiring.
- **Product logic** — adapters and composition only. The planner/executor stays in the Kit so it remains
  testable with no GPU, no weights, and no metallib.

## Image coding

`EngineImageCoding` is the single `CGImage ⇄ MLXToolKit.Image` conversion. Callers state which encoding
they need rather than letting it guess, because handing raw bytes to a PNG-only wrapper fails as a decode
error deep inside that package instead of at the boundary.

| Wrapper | Mirrors `.rawBGRA8` | Keeps 16 bits (`.rawRGBA16`) |
|---|---|---|
| `mlx-realplksr-swift` ≥ 0.2.0 | ✅ | ✅ — Best keeps a 16-bit still's depth through the model |
| `mlx-nerve-swift` ≥ 0.2.0, `mlx-heart-swift` ≥ 0.2.0 | ✅ | ✅ — Fast, the ×6/×8 chain's pass 2, and the opt-in HEART variant (AB-T-0229) |
| `mlx-nafnet-swift` ≥ 0.6.0 | ✅ — Optimize runs raw end to end | ✅ — Optimize's restore hop keeps a 16-bit still's depth (AB-T-0230) |
| `mlx-gfpgan-swift`, `mlx-restoreformer-swift` ≥ 0.3.0 | ✅ | ✅ — the face pass warps, pastes and masks at 16 bits (AB-T-0230) |
| `mlx-vosr-swift`, `mlx-nacre-swift` | ✅ | ❌ — the generative tiers stay 8-bit (AB-T-0230's decisions) |
| BiRefNet, DDColor, LaMa, SEA-RAFT, EdgeTAM | ❌ — PNG leg (tracked as BRIDGE-041) | ❌ |

`Matte` is PNG-only by contract regardless.

**16 bits** (MLXToolKit contract 1.52.0, AB-A-0158): `rawImage(from:keepingDepth:)` sends a deep (more than 8 bits
per component) image as `.rawRGBA16` only when asked to. `EngineImageEnhancer` asks per hop, with
`ForgeEngineService.keepsSixteenBits(_:)`, which reads the registered surface's `rawImageFormats` declaration. So a
16-bit source stays 16-bit through every model that declares it, and the engine refuses `.rawRGBA16` anywhere else.
`cgImage(from:)` decodes by format and returns a 16-bit image whenever the package answered in 16 bits. The ×6/×8
chain's Lanczos fit keeps 16 bits for a deep image. The restore hop's 16-bit answer goes on to the upscale hop as it is
only when that tier keeps 16 bits too (`EngineImageEnhancer.image(_:forHopKeepingDepth:)`). The generative tiers' text
guard sends both tiers 8 bits and composites at 8 bits, and those tiers stay 8-bit.

## Dependency visibility

Every dependency in this package's closure is **public** — the five Forge Kits, `media-bridge`, the
`mlx-*-swift` wrappers, and `mlx-engine-swift` — so CI needs no credential. ForgeCore itself is private,
which its own workflow token can clone.

Keep it that way when adding a dependency. A private one makes SwiftPM fail with *"could not read
Username for `https://github.com`"*, which reads like a network fault and isn't: a workflow's default
`GITHUB_TOKEN` cannot clone sibling private repos. The fix is a PAT in a repo secret plus a
`git config url.<token>.insteadOf` rewrite — not a retry.

## Testing

```bash
swift test --build-system swiftbuild
```

**The flag is required, not stylistic.** Importing MLX inherits the metallib requirement, and the
deprecated `native` build system ships no metallib — it aborts the whole xctest process while the log
still reads "all tests passed". The same trap once had `mlx-engine-swift`'s CI green while broken.

The tests here are CoreGraphics-only on purpose: image-coding boundary arithmetic and the model table,
which are the parts that can be wrong *silently*. A BGRA/RGBA channel swap yields images that look
plausible with red and blue exchanged, so only a byte-exact assertion catches it. Live capability
verification belongs in a consumer that has a model store — the reference app's smoke runner does it.
