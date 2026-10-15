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

## The Instant upscale tier runs outside the engine

`UpscaleTier.instant` (ForgeOptimizerKit ≥ 0.25.0, AB-D-0146) is the efficiency tier: instant and model-free, for when
speed is all a job needs. `EngineImageEnhancer` runs it as Anime4K's upscale-only VL network on Metal compute shaders
(`anime4k-metal`, MIT), directly on the GPU. It needs no registration, no admission, no download and no memory tier,
so it is available on any Mac with a Metal device, before `registerAll()` has run. ×2, ×4 and ×8 are one, two and three
doublings; ×6 is ×8 fitted down with Lanczos (`InstantUpscalePlan`, AB-D-0119's rule). Depth and alpha are kept. The
receipt names the model `Anime4K VL`. It is not a quality rival to Fast or Best, which score higher on every content
class measured (AB-R-0559); `EngineImageEnhancer.model(for: .instant)` is nil because there is no registration.

## Generative Fill and Expand

`ForgeCapabilities.generativeFill` (`EngineGenerativeFillProvider`, AB-A-0166) fills a selection from a prompt and
extends a canvas, through the **same route the Generative Fill bake-off measured** (AB-R-0534): the Image Engine's
magenta marker, its paint guard (marker or dark → retry with the next seed → LaMa), its harmonize and its composite
through the mask. Expand is a sibling method on the same provider, not a second seam: the Image Engine's `image.extend`
route, including its LaMa seam bridge.

```swift
let fill = capabilities.generativeFill
let filled = try await fill.fill(image, mask: selection, prompt: "a small wooden rowing boat")   // the editor's seam
let result = try await fill.fillReporting(image, mask: selection, prompt: "", seed: 7)           // + model, quant, licence
let wider  = try await fill.expand(image, left: 0, top: 0, right: 256, bottom: 0)
```

- **The seam.** `fill(_:mask:prompt:)` matches the editor's `GenerativeFillProvider` exactly: a grey mask (white =
  generate) at the image's size in, the whole image at the same size out. A soft selection blends; every unselected
  pixel is the input's exactly, at 16 bits for a 16-bit image. A host conforms it in one line,
  `extension EngineGenerativeFillProvider: @retroactive GenerativeFillProvider {}`. An empty prompt blends the area
  away, which is also how this route removes an object (AB-R-0534: SenseNova beat LaMa that way, 17.5 vs 13.8 dB).
- **Which model is configuration.** `ForgeEngineService(generativeFillTier:)` takes a `GenerativeFillTier`. The default
  comes from `GenerativeFillTiers.forgeDefault` for this Mac's memory, which is AB-D-0141 as amended:

  | Mac | model | window cap |
  |---|---|---|
  | 16 GB | SenseNova-U1.5-8B-MoT 8step-v2-4bit | 768² |
  | 24 GB | SenseNova-U1.5-8B-MoT 8step-v2-4bit | 1024² |
  | 32 GB+ | SenseNova-U1.5-8B-MoT 8step-v2-8bit | 1024² |

  `GenerativeFillTiers.kleinAt16GB` swaps 16 GB to the documented fallback, FLUX.2-klein-4B int8 with the encoder
  evicted at 768² (int4 failed mlx-image's quality gate, AB-R-0544). The tables are `Codable`, so a host can store its own. `.generativeFill` registers the tier's model
  under one id, `generative-fill`; set `generativeFillTier` and call `registerAll()` again to switch.
- **The window.** When the whole picture fits the tier's cap, the model sees the whole picture, exactly as in the
  bake-off. When it doesn't, the model sees a square around the selection. The square is 2.5× the selection's long side,
  at least the cap's side and at most the picture's short side, which is the Image Engine's own rule for a square
  around a region. A small selection on a big photo is therefore painted 1:1, not squeezed. A selection longer than
  the short side gets the whole picture, scaled into the cap. Expand always sends the whole new canvas, scaled into the
  cap, as the route does.
- **Report.** Every call reports the tier, model, quant, the weight and port licences (read from the registered
  manifest), the window, the seed that was used, retries, and whether LaMa ran. `fillReporting` returns the report, and
  `onReport:` delivers it to hosts that use the plain seam.
- **Needs two models.** `requiredModels` is `[.generativeFill, .inpaint]`: LaMa is the guard's fallback and expand's
  seam bridge. Offer both downloads before the first fill.
- ⚠️ **Not every Mac can run it yet.** SenseNova declares `chipFloor: .max` (measured on M5 Max only), so the engine
  refuses it on base and Pro chips. Its int4 declaration (12.2 + 4.5 GB) is refused on a 16 GB budget until
  sensenova-u1-swift ships a 16 GB footprint declaration (AB-T-0278). Both refusals show up as `.generativeFill`'s
  outcome or admissibility, never as a silent swap. The 16 GB tier also needs every other MLX resident evicted first,
  and a real-16-GB-Mac confirmation before it ships (AB-D-0141).

The route lives in `Sources/ForgeCore/GenerativeFill/` and is engine-free: CoreGraphics over 8-bit premultiplied RGBA,
with the model and LaMa injected as closures. `FillRoute`'s functions are the Image Engine's, line for line after a
type rename. Through this provider, with the bake-off's arm (8step-v2-8bit, seed 7) at the Image Engine's envelope,
four bake-off cases (two prompted fills, a seamless fill, an extend) come back **bit-identical** to the Image Engine's
own outputs (`GenerativeFillLiveTests/bakeOffReproduces`).

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
| `mlx-realplksr-swift` ≥ 0.2.1 | ✅ | ✅ — Best keeps a 16-bit still's depth through the model; ×2 is a registered Lanczos reduction of the ×4 (AB-T-0225) |
| `mlx-nerve-swift` ≥ 0.2.1, `mlx-heart-swift` ≥ 0.2.2 | ✅ | ✅ — Fast, the ×6/×8 chain's pass 2, and the opt-in HEART variant (AB-T-0229) |
| `mlx-nafnet-swift` ≥ 0.6.1 | ✅ — Optimize runs raw end to end | ✅ — Optimize's restore hop keeps a 16-bit still's depth (AB-T-0230) |
| `mlx-gfpgan-swift`, `mlx-restoreformer-swift` ≥ 0.3.1 | ✅ | ✅ — the face pass warps, pastes and masks at 16 bits (AB-T-0230); no dark line at the image edge under a close-up's crop (AB-T-0227) |
| `mlx-vosr-swift` ≥ 0.3.1 | ✅ | ✅ — VOSR2 writes 16 bits (8-bit reference prep, float32 VAE decode; AB-D-0127, AB-T-0233) |
| `mlx-nacre-swift` | ✅ | ❌ — Nacre stays 8-bit (AB-D-0128) |
| BiRefNet, DDColor, LaMa, SEA-RAFT, EdgeTAM | ❌ — PNG leg (tracked as BRIDGE-041) | ❌ |

`Matte` is PNG-only by contract regardless.

**16 bits** (MLXToolKit contract 1.52.0, AB-A-0158): `rawImage(from:keepingDepth:)` sends a deep (more than 8 bits
per component) image as `.rawRGBA16` only when asked to. `EngineImageEnhancer` asks per hop, with
`ForgeEngineService.keepsSixteenBits(_:)`, which reads the registered surface's `rawImageFormats` declaration. So a
16-bit source stays 16-bit through every model that declares it, and the engine refuses `.rawRGBA16` anywhere else.
`cgImage(from:)` decodes by format and returns a 16-bit image whenever the package answered in 16 bits. The ×6/×8
chain's Lanczos fit keeps 16 bits for a deep image. The restore hop's 16-bit answer goes on to the upscale hop as it is
only when that tier keeps 16 bits too (`EngineImageEnhancer.image(_:forHopKeepingDepth:)`). The generative tiers' text
guard sends each tier its own depth (VOSR2 and NERVE 16 bits, Nacre 8), and MaskKit ≥ 0.4.1 composites at 16 bits when
either tier answered at 16.

## Dependency visibility

Every dependency in this package's closure is **public** — the five Forge Kits, `media-bridge`, the
`mlx-*-swift` wrappers, and `mlx-engine-swift` — so CI needs no credential. ForgeCore itself has been public since
2026-10-10 (AB-A-0148), so nothing here may rely on it being private: no local paths, credentials or research-only
corpus files in commits or tags.

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
