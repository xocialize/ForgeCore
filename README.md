# ForgeCore

The MLX seam for Forge: the engine service, the image-coding helper, and the six engine-backed provider
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

| Wrapper | Mirrors `.rawBGRA8` |
|---|---|
| `mlx-nafnet-swift`, `mlx-realesrgan-swift` | ✅ — Optimize runs raw end to end |
| BiRefNet, DDColor, LaMa, SEA-RAFT, EdgeTAM | ❌ — PNG leg (tracked as BRIDGE-041) |

`Matte` is PNG-only by contract regardless.

## CI setup (one manual step)

ForgeCore resolves five **private** xocialize Kits, and a workflow's default `GITHUB_TOKEN` is scoped to
its own repository — so CI cannot clone them. It fails with *"could not read Username for
`https://github.com`"*, which reads like a network fault and isn't one.

Add a repository secret **`FORGE_CI_TOKEN`**: a fine-grained PAT with read access to `ForgeOptimizerKit`,
`ExtractKit`, `ColorizeKit`, `EraseKit` and `MaskKit`. The workflow rewrites the URL scheme with it. (The
alternative is making those Kits public.) Until it exists, CI fails fast with that instruction rather
than a wall of clone errors.

Local builds are unaffected — your own git credentials already reach the private repos.

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
