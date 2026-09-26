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
| SAM 2.1-S | `promptSegment` (quality, image only) | Metal GPU | 26.0 | declared split **0.25 GB** resident + **2.6 GB** activation — measured phys 2.35 GB at 1024², 2.79 GB at 1800×1200 (AB-T-0173); registered as `.promptMaskQuality`, id `sam21-prompt-segment` |
| SigLIP2-IQA | `imageQualityScore` | Metal GPU | 26.0 | declared |
| BOPBTL | `defectDetect` | CPU (pinned — see note) | 26.0 | **measured**: 1.63 GB active / 3.9 GB at the 2 GiB cache, ~1 s/image |
| RIFE 4.25 | `frameInterpolate` | Metal GPU | 26.0 | declared: fp32 resident 150 MB + peak 4.0 GB — **measured 3.58 GB** phys at 1080p whole-frame |
| V-JEPA2 ViT-L | `contentClassify` | Metal GPU | 26.0 | declared split: fp32 resident **1.3 GB** + peak activation **1.7 GB** (activation is an ESTIMATE — in-app `phys_footprint` re-baseline pending) |

⚠️ **BOPBTL runs on the CPU stream by its core's own pin**: the Metal fp32 conv2d kernel is wrong in
its geometry window (inC ≥ 128 ∧ outC ≥ 128 ∧ spatial ≥ ~64 — receipted in `mlx-bopbtl-swift`, with
a `--gpu-probe` to re-test on every mlx-swift bump). Detection runs at short-side 256, so CPU costs
~1 s/image. Its mask composes with the registered `imageInpaint` (detect → mask → fill). Corpus
validation on real scratched photographs is deferred to the restore-app scaffolding stage (P7).

✅ **`videoUpscale` IS BOUND as of ForgeCore 0.8.x** — `ForgeEngineService.upscaleVideo(_:scale:)`
(`Adapters/EngineVideoUpscaler.swift`). ⚠️ This line previously read *"not registered, and therefore not
available to any app today"*, which was true when written and stale afterwards. There is **no separate
registration**: `SeedVR2UpscalePackage` declares both `ImageUpscaleContract` and `VideoUpscaleContract`,
so the adapter resolves the same `.upscaleQuality` PackageID the still path uses — one model load, and a
prepared image tier leaves the video tier warm. UI is one dial {off/conservative/default/max} →
`temporalWindow` {1,5,9,13}. 🚨 **Read the geometry ceiling below before defaulting it on**, and treat it
as a long job: ~3.3 s per output frame at `.default`, so a 10 s 24 fps clip is roughly 13 minutes.

✅ **`frameInterpolate` IS REGISTERED AND BOUND as of ForgeCore 0.15.0 (2026-08-14, AB-T-0038)** —
`ForgeEngineService.interpolateVideo(_:factor:)` (`Adapters/EngineFrameInterpolator.swift`), the 20th
registration. ⚠️ This line previously read *"Still not registered, and therefore unavailable to any
app"* — true from the port's landing (2026-06-18) until today: RIFE was parity-locked and shipping at
v0.8.0 the whole time; the gap was registration, not the model. Factor 2 default (one synthesized
midpoint per adjacent pair at t=0.5); ≤ 1080p runs whole-frame, above it the package's tiler bounds
memory (tile 1024 / overlap 128). Output is HEVC via frame-stream-native's **software**-encoder
default — load-bearing under AB-B-0002 (hardware HEVC right after Metal work stalls, and panics
quarantined builds on teardown); no Forge surface may opt back into hardware. Smoke (M5 Max 128 GB,
Release, software encode): 1080p H.264 14.06 s fixture → ×2 = 447 frames @ 31.58 fps (exactly 2× the
source 15.79), duration preserved, **143.4 s wall = 0.32 s/output frame**, peak `phys_footprint`
**3.58 GB**; first-run weight materialization (~22 MB) ≈ 2.3 s. Treat it as a long job: ~0.3 s per
output frame at 1080p, and the software encode is part of why — that price buys not panicking the
machine.

🚨 **`contentClassify` IS REGISTERED on `main` as of 2026-08-15 (AB-T-0045; the 21st registration —
tag pending) — and its adapter is deliberately INERT, so an integrator must not size a machine for it
by default.** V-JEPA2 backs the
optimizer's §6.3 planner hint (`EngineContentHintProvider` → `ForgeOptimizerKit.ContentHintProvider`),
which classifies a clip before its first quality search so the search can start at the class floor
instead of discovering the class from its own landing and paying a second search. The Kit half is
measured: **−50% wall on graphic masters, byte-identical deliverable** (AB-R-0056).

⚠️ **None of that saving is available yet.** The adapter clamps every hint's confidence to 0.5, under
the Kit's 0.9 action gate, until the embedding probe is fitted against encoder behavior and its
held-out disagreement rate is receipted. So today the capability is **pure cost** wherever it is
injected: ~1.3 GB resident for the ViT plus a full copy of the input per classification. Inject it only
to collect calibration rows; leave `hintProvider` nil otherwise and the Kit's behavioral ratchet covers
the case for free.

⚠️ **Cost note the footprint table cannot express — the classify contract takes BYTES, not a URL.**
`ContentClassifyRequest` carries a canonical `Video` artifact, and the package writes those bytes back
out to a temp file to run AVFoundation's frame sampler. So one classification costs a read plus a full
disk copy of the source, while the sampling it enables is ~10² ms — on a 429 MB master the transport
dominates the work by orders of magnitude. The adapter maps rather than reads the file (memory stays
honest) and refuses inputs over 2 GiB rather than pay silently. The real fix is a URL form of the
contract, filed as an ask against the engine/video area; until it lands, this is a documented tax, not
a solved problem.

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

✅ **MEASURED end to end 2026-07-31** (`GAP-PROGRAM.md` V12-B2, `probes/v12b2_bank_eviction.out`),
**cost REMEASURED 2026-08-03 with repeated arms** (`probes/v12b3g_bank_granule_io.out` — arms in both
orders, 150 s cooldowns, baselines woven through): 17 frames at **40 tile positions** through the
shipping surface.

| | peak `phys_footprint` | wall (17 frames, 2026-08-03 arms) |
|---|---|---|
| no budget (v0.8.0) | **42.02 / 41.83–42.04 GiB** | 225.0 / 244.0 s |
| **1 GiB bank budget** | **12.69 / 12.68–12.74 GiB** ✅ 3.31× | 241.2–266.8 s (**≈+7%**, spread 0–19%) |
| `bankBudgetToFit` 2.07 GiB | **13.50 GiB** (13.57 predicted) | 258.4 s |

✅ **Output is BIT-IDENTICAL** — max \|Δ\| = 0 across all 18 emitted frames, over two real bank
round-trip boundaries (re-confirmed across 7 arms in the remeasure). ✅ **And the sizing model
extrapolates**: fitted at 1 and 9 tile positions, it predicts 43.9 GiB un-evicted (measured 42.02,
within 4.5%) and 12.93 GiB evicted (measured 12.69, within 2%) at 40 positions; the remeasure landed
within 0.6% at BOTH budgets. 🔑 **42 GiB does not fit a 16 GB machine and 12.7 GiB does** — that is the
claim, now measured rather than argued.

🚨 **Cost-figure history, kept honest: 2.7× → ~6–13% → +48% → ≈+7% (current).** The first two divided
quantities measured under different conditions. The +48% (2026-07-31) measured the composite but ran
the treatment arm ONCE, and it absorbed session drift — the remeasure's four evicted arms all run
20–28% below it with matching baselines. Its "serialization-bound, ~7× below raw" decomposition was
then falsified directly: a contiguous single-buffer bank format (BlockStreamKit `GranuleSpillFile`)
A/B'd through the same path is composite-invisible — the round trip runs at disk rate on both
transports. **Measure the composite, AND repeat the arms.** ⚠️ Cost band measured on a 128 GB host;
the 16 GB-arm composite (where buffered spill cannot ride the page cache) is the open receipt.

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
