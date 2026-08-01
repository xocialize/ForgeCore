# Capability requirements — what each registered package needs to run

**For integrators.** ForgeCore is what an app links, so this is where "can this machine do X?" gets
answered. Opened 2026-07-30.

⚠️ **This file is the human-readable view. It is NOT the authority.** The authority is each package's
`RequirementsManifest` — `footprints: [QuantFootprint(quant:residentBytes:peakActivationBytes:)]`,
`requiredBackends`, `os`, `chipFloor` — because that is what the engine's memory governor actually
admits against. If the two disagree, the manifest wins at runtime and **this file's job is to say so
loudly**, which is exactly what the first entry below has to do.

---

## 🚨 Read this before trusting any single number

**A footprint is not a constant. For some packages it is a function of input size and which code path
the input takes — and `QuantFootprint` is a scalar per quant, so it cannot express that.**

Concretely, Real-ESRGAN declares **6.26 GB** (resident 1.02 + activation 5.24), measured on the
**tiled** path where cost is bounded by the tile and does not scale with input area. That declaration
is accurate for tiling and **wrong for the whole-frame path**, which the package takes by default below
`wholeFrameMaxPixels` and which *does* scale with input area:

| input | path | measured `phys_footprint` | vs declared 6.26 GB |
|---|---|---|---|
| SD (624×352) | whole-frame | **3.22 GB** | 0.5× |
| **1920×1080** | whole-frame | **21.24 GB** | **3.4×** |
| 1080×1920 | whole-frame | **24.64 GB** | **3.9×** |
| any, tiled | tiled 128/8 | ~6 GB, flat | 1.0× ✅ |

✅ **Tile geometry changed 64/8 → 128/8 in `mlx-realesrgan-swift` 0.7.0** (ForgeCore ≥ 0.8.3). The old
64 was both the slowest of {64,128,256} on *both* backends and the only one with a detectable seam:
2048²→8192² ×4 went **7.68 s → 1.65 s (4.65×)**, and the flat-block seam z-test reads **+1.50 at t64
(a seam) vs +0.46 at t128 (none above a ~166-stride null)**. Output is otherwise near-identical
(p99 |Δ| = 2/255). The tiled footprint row above is unchanged — it is bounded by the tile either way.
⚠️ **`tileOverlap` is absolute, not a ratio** — it covers the receptive field, which does not grow with
the tile, so 8 px at 128 is exactly the margin 8 px at 64 was, across 4.2× fewer seams. Do not "fix"
the ratio.

*(`GAP-PROGRAM.md` V12, in-app `phys_footprint`, the admission basis — not MLX peak, which under-reads
by ~2.5–2.9×.)*

🔑 **Why it is this shape:** SRVGGNetCompact runs at its **native ×4 internally**, so a 1080p input
builds a 7680×4320 intermediate regardless of the requested scale. And `processAdaptive` tests `<=`,
so **1920×1080 sits exactly on the default ceiling and takes the whole-frame path** — the most common
video resolution, on the expensive side by one comparison.

⚠️ **Consequence for admission: the governor can accept a 1080p job on its 6.26 GB declaration and then
be asked for 21 GB.** This is an open defect, recorded rather than silently patched — raising the
declaration to the whole-frame worst case would make the governor refuse SD jobs that currently succeed
comfortably, so the honest fix is conditional footprints, not a bigger scalar. **Until then, a host that
cares about the memory ceiling must set `wholeFrameMaxPixels` itself** (`RealESRGANConfiguration`,
`mlx-realesrgan-swift` ≥ 0.6.3) rather than rely on admission.

---

## Registered capabilities

| model | capability | backend | min macOS | footprint basis |
|---|---|---|---|---|
| NAFNet | `imageRestore` | Metal GPU | 26.0 | declared per variant |
| Real-ESRGAN | `imageUpscale` (fast) | Metal GPU | 26.0 | ⚠️ **input-dependent — see above** |
| SeedVR2-3B | `imageUpscale` (quality) | Metal GPU | 26.0 | int8 resident **4.8 GB** + activation |
| BiRefNet | `matting` | Metal GPU | 26.0 | declared |
| DDColor | `imageColorize` | Metal GPU | 26.0 | declared |
| LaMa / MI-GAN | `imageInpaint` | Metal GPU | 26.0 | declared |
| SEA-RAFT | `opticalFlow` | Metal GPU | 26.0 | declared |
| EdgeTAM | `promptSegment` | Metal GPU | 26.0 | declared |
| SigLIP2-IQA | `imageQualityScore` | Metal GPU | 26.0 | declared |

✅ **`videoUpscale` IS BOUND as of ForgeCore 0.8.x** — `ForgeEngineService.upscaleVideo(_:scale:)`
(`Adapters/EngineVideoUpscaler.swift`). ⚠️ This line previously read *"not registered, and therefore not
available to any app today"*, which was true when written and stale afterwards. There is **no separate
registration**: `SeedVR2UpscalePackage` declares both `ImageUpscaleContract` and `VideoUpscaleContract`,
so the adapter resolves the same `.upscaleQuality` PackageID the still path uses — one model load, and a
prepared image tier leaves the video tier warm. UI is one dial {off/conservative/default/max} →
`temporalWindow` {1,5,9,13}. 🚨 **Read the geometry ceiling below before defaulting it on**, and treat it
as a long job: ~3.3 s per output frame at `.default`, so a 10 s 24 fps clip is roughly 13 minutes.

**Still not registered, and therefore unavailable to any app:** `frameInterpolate` (RIFE is ported but
never registered).

✅ **A second `imageUpscale` backend exists for macOS 27+ hosts: CoreAI/ANE Real-ESRGAN**
(`coreai-realesrgan-swift` ≥ 0.2.0, product `CoreAIRealESRGAN` — MLXToolKit-only, zero MLX). ForgeCore
floors at 26 and cannot link it; a 27-floored app injects it via `ExternalRegistration` (ForgeCore
≥ 0.12.0) and it registers beside the MLX backer under its own PackageID. Measured through the package
surface at 1080p→×4: **resident 19 MB, peak 0.86 GB, 1.60 s** — ~25× lighter than the MLX whole-frame
path, ~7× lighter than tiled, at **≈4.5–4.9× less energy** (V13). fp16 parity per variant: min 58.15 dB
(wdn) to 68.56 dB (general). macOS 27 releases publicly ~October; below 27 the MLX backer is the path,
by design.

---

## Measured operating points — the numbers an integrator actually needs

These come from `GAP-PROGRAM.md` V10 / V12 / V12-F receipts. All `phys_footprint`, all Release.

**SeedVR2-3B int8 — stills, 512→2048 ×4**
- **8.40 GB**, ~30 s/frame. Tiled at 256/32 above 1 MP output; **bounded by the tile**, so it does not
  grow with source size. Below ~1 MP it single-passes and is cheaper.
- ⚠️ Before `mlx-seedvr2-swift` v0.7.2 this was **43.71 GB** and the output above 1 MP was broken.
  Anything pinned older than that is unusable for stills.

**SeedVR2-3B int8 — video, temporal (T = frames per window)**

🚨 **CORRECTED 2026-07-31. The first ladder published here was measured with an UNBOUNDED MLX buffer
cache and overstated every rung by 40–70%.** At T=5 the 20.13 GiB figure decomposes as **5.21 GiB MLX
active + 14.69 GiB MLX cache**; capping the cache at 2 GiB gives 9.85 GiB for identical output. The
shipping engine *does* cap it — `PORT-QUEUE.md` cross-cutting note 3 has said *"set
`GPU.set(cacheLimit:)`, never `memoryLimit`"* all along — so the uncapped numbers described a
configuration the product never runs.

Ladder as the product actually runs it (`GAP-PROGRAM.md` V12-S, streaming on, three measurements per
rung):

| T | s/frame | peak (capped) | superseded (uncapped) | 16 GB machine |
|---|---|---|---|---|
| 1 | — | **7.82 / 7.90 / 7.80 GiB** | 8.21 | ✅ |
| 5 | — | **9.83 / 9.86 / 9.90 GiB** | 12.18 | ✅ |
| **9** | **0.306** | **11.76 / 11.78 / 11.54 GiB** | 15.73 | ✅ **74% — the recommended tier** |
| 13 | — | **13.47 / 13.57 / 13.65 GiB** | 18.91 ❌ | ✅ **84% — fits, was reported as not fitting** |

🔑 **Every rung fits 16 GB, including T=13**, which the earlier row called impossible. Streaming's own
cost is ~1.0–1.5 GiB over independent chunks; the retained tails are **784 MiB and constant in T**.

**Recommended for a 16 GB machine: T=9 at 256² tiles** — 11.78 GiB (74%), boundary max |ΔL\*| 0.3071,
0.306 s/frame. T=13 is better on quality and still fits at 84%, but that is a headroom call rather than
a default. ⚠️ T=5 is now the *weakest* of the three: its steady-state chunks carry latT = 1.

✅ **The chunk-boundary defect that blocked shipping is resolved, not merely reduced.** Boundary
max |ΔL\*| 1.4089 → **0.3426** at T=5, and at every T the boundary max is now *smaller* than the
interior max — the seam is no longer where the clip's worst frame pair lives. Chunked T>1 now beats
T=1 on max|ΔL\*| (0.700/0.586/0.494 vs 0.851).

✅ **Scene cuts: SOLVED end-to-end (2026-07-31).** `SceneCutDetector` (media-bridge **v0.11.0**) runs
inline in the temporal driver; a detected cut ENDS the window early and flushes state — proven through
the shipping surface: post-cut mean |ΔL\*| **0.34** (driver) vs **2.97** (no reset), identical in the
boundary-aligned and mid-window cases (`probes/v12d_temporal_driver.out`).

✅ **The temporal chunking driver EXISTS and is the shipping surface** (`mlx-seedvr2-swift` **v0.8.0**:
`TemporalWindowPlanner` + `SeedVR2TemporalRefiner`; `temporalWindow` on the config, default 9;
`availableBudgetBytes` clamps T per clip geometry; requires `frame-stream-native` ≥ 0.4.0 for
per-output PTS). T=1 through the new driver is bit-identical to v0.7.7. 🔑 **T=9 is 20–30% CHEAPER per
frame than T=1** (3.27 vs 4.10–4.68 s bracketed) — coherence is now cheaper, not a trade.

🚨 **NEW MEASURED FACT THAT CHANGES SIZING — the streaming banks are PER SPATIAL TILE POSITION
(~0.82 GiB each, ~constant in T), so OUTPUT GEOMETRY is a memory multiplier the T-ladder above does not
capture.** Single 256² output tile: the ladder holds (7.75/10.10/11.93 GiB at T=1/5/9). **Nine tile
positions (512² output): 17.95–18.16 GiB** — the per-tile model predicts it within 0.3 GiB. This is the
third instance of "a footprint is a function, not a constant": here it is a function of *output* area.
- **≤256² output on 16 GB: T=9 quality tier works today.**
- **Real content on 16 GB** (e.g. SD ×2 ≈ 15 tile positions ≈ ~12 GiB of banks alone): does NOT fit at
  any T — clamping T does not shrink banks. Needs **bank offload/eviction** before the quality tier
  defaults on above tiny outputs; larger-memory machines run it today, which is exactly the app-side
  capability gating the product intends.

🚨 **"Per tile position" is a unit of convenience — the real unit is PER OUTPUT PIXEL, and that rules
out the two obvious workarounds.** The bank is `CausalConv3d`'s retained tail, which keeps the tile's
full spatial extent, so bank bytes scale with tile *area* and `positions × area` is simply the output
area: **≈ 13.4 KB of retained tail per output pixel**, invariant to tiling.
- ❌ **Lowering T does not help** — tails are ~constant in T (784 MiB across the ladder, measured).
- ❌ **Enlarging the tile does not help** — fewer tiles, proportionally larger tails. ⚠️ Note this is
  the opposite of the Real-ESRGAN result above, where enlarging the tile was a pure win. Different
  mechanism, do not carry the conclusion across.

✅ **Both of the above are now CI-enforced**, not remembered: `VAEStreamingBankSizeTests` in
`mlx-seedvr2-swift` pins the law at shipping precision — 57 tails, all bfloat16,
`12525·s² + 1024·s + 1536` bytes, **0.765 GiB per 256² tile position**, within 7% of the V12-D
process-footprint number by a completely different method.

🚨 **Sized at the product's real targets, this is a hard ceiling, and `bank offload/eviction` — carried
as the planned fix since V12-D — does NOT clear it:**

| output | tile positions | banks alone |
|---|---|---|
| 256² | 1 | 0.8 GiB |
| 512² | 9 | 6.9 GiB |
| SD ×2 (1440×960) | 35 | 26.8 GiB |
| 720p ×2 | 84 | 64.2 GiB |
| **HD→4K (3840×2160)** | **170** | **130.0 GiB** |

✅ **SHIPPED 2026-07-31 — bank eviction** (`mlx-seedvr2-swift` v0.9.0, ForgeCore ≥ 0.9.0). Set
`ForgeEngineService.bankResidencyBudgetBytes` (or `FORGE_SEEDVR2_BANK_BUDGET`) and banks over the budget
round-trip through a scratch directory **bit-identically**. That turns the table above from a hard
ceiling into a cost: predicted peak becomes `base + min(extra banks, budget)`, which
`videoUpscaleFits(outputTilePositions:budgetBytes:bankResidencyBudgetBytes:)` models and
`bankBudgetToFit` inverts — so a host can offer "run it with eviction" instead of a refusal.

✅ **MEASURED end to end 2026-07-31** (`GAP-PROGRAM.md` V12-B2, `probes/v12b2_bank_eviction.out`):
17 frames at **40 tile positions** through the shipping surface, arm A bracketed before and after B.

| | peak `phys_footprint` | wall (17 frames) |
|---|---|---|
| no budget (v0.8.0) | **42.02 / 41.92 GiB** | 212.4 / 241.5 s |
| **1 GiB bank budget** | **12.69 GiB** ✅ 3.31× | 335.6 s (**+48%**) |

✅ **Output is BIT-IDENTICAL** — max \|Δ\| = 0 across all 18 emitted frames, over two real bank
round-trip boundaries. ✅ **And the sizing model extrapolates**: fitted at 1 and 9 tile positions, it
predicts 43.9 GiB un-evicted (measured 42.02, within 4.5%) and 12.93 GiB evicted (measured 12.69,
within 2%) at 40 positions. 🔑 **42 GiB does not fit a 16 GB machine and 12.7 GiB does** — that is the
claim, now measured rather than argued.

🚨 **The cost is ~48%, not the ~6–13% previously published here — and that was the third wrong
estimate for this work.** V12-B said "2.7×, offload cannot work" (I/O at 170 tile positions against
compute measured at 9); the replacement said "~6–13%" (bank bytes ÷ raw disk bandwidth). **A ratio of
two quantities measured separately, under different conditions, is not a measurement.** The round trip
runs ~7× below raw bandwidth because it is *serialization*-bound — 57 tensors per bank through
safetensors framing — which is headroom (a contiguous bank format), not a floor.

⚠️ **1 GiB is the EXTREME budget** (39 of 40 banks spill). Set the largest budget that fits, not the
smallest that works — `bankBudgetToFit(40, 14 GiB)` returns ~2.07 GiB on a 16 GB machine, and overhead
falls as fewer banks spill.

⚠️ **The real price is write volume** — 59.7 GiB measured for 17 frames at 40 tile positions, i.e.
~840 GiB of scratch for a 10 s 24 fps clip.
Eviction is **off by default** and is how a machine runs a job it otherwise could not; a machine with
headroom should leave it unset and write nothing. ⚠️ It also only moves `phys_footprint` because the
engine caps the MLX buffer cache — freed buffers beyond the cap go back to the OS rather than parking
in the pool.

**Un-evicted, the ceiling for the quality tier is ~9 tile positions (512² output) on 16 GB.** With a
bank budget set, SD ×2 (35 positions) fits inside 14 GiB — which is the case the two named customer
segments actually need.
⚠️ Still owed: the ByteDance multi-frame golden (no reference parity oracle for T>1 output values).

**Real-ESRGAN — see the warning above.** Sub-second per frame at SD. Tiled path is 128/8 since 0.7.0 — 4.65× faster than the old 64/8 at 8192² output, and the seam the old geometry had is gone (see the tiling note above).

---

## Competitive floor, for context

Topaz Starlight Mini documents a **36 GB minimum on macOS** and states Preview *"may also not work"*
for Mac users (`mlxengine-todo/TOPAZ-MATCHUP.md` §3b). Every number in this file is below that floor.
Aiarty publishes **no memory requirement at all** (§3c).

---

## How to keep this file honest

- **Numbers here must come from a receipt in `mlxengine-todo/probes/`, with the row named.** No
  estimates, no extrapolation. If it was not measured, say "unmeasured".
- 🚨 **State the measurement CONFIGURATION, not just the units.** Two separate basis errors have now
  inflated numbers in this row: MLX peak read as the admission basis (under-reads ~2.5–2.9×), and an
  **unbounded buffer cache** read as working set (over-reads 40–70%). Both looked like clean
  measurements. A footprint is only meaningful with its cache policy stated — cap it the way the engine
  does, and say that you did.
- **Use `phys_footprint`, never MLX peak** — MLX peak under-reads the admission basis by ~2.5–2.9×,
  and that confusion has already produced one wrong "fits in 16 GB" claim (V12 §6).
- **When a footprint is input-dependent, say so and give the range.** A single number for a function is
  the defect this file opens with.
- When a package's manifest is corrected, update the row *and* note what changed, so an integrator
  pinned to an older version can tell whether it applies to them.
