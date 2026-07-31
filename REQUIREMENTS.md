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
| any, tiled | tiled 64/8 | ~6 GB, flat | 1.0× ✅ |

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

**Not registered, and therefore not available to any app today:** `videoUpscale` (no binding at all),
`frameInterpolate` (RIFE is ported but never registered).

---

## Measured operating points — the numbers an integrator actually needs

These come from `GAP-PROGRAM.md` V10 / V12 / V12-F receipts. All `phys_footprint`, all Release.

**SeedVR2-3B int8 — stills, 512→2048 ×4**
- **8.40 GB**, ~30 s/frame. Tiled at 256/32 above 1 MP output; **bounded by the tile**, so it does not
  grow with source size. Below ~1 MP it single-passes and is cheaper.
- ⚠️ Before `mlx-seedvr2-swift` v0.7.2 this was **43.71 GB** and the output above 1 MP was broken.
  Anything pinned older than that is unusable for stills.

**SeedVR2-3B int8 — video, temporal (T = frames per window)**

| T | s/frame | peak | 16 GB machine |
|---|---|---|---|
| 1 | 0.329 | 8.21 GiB | ✅ comfortable |
| **5** | **0.222** | **12.18 GiB** | ✅ **the 16 GB tier** |
| 9 | 0.223 | 15.73 GiB | ⚠️ marginal |
| 13 | 0.242 | 18.91 GiB | ❌ |

🔑 **T=5 is both the cheapest per frame and the best on flicker** — an unusually clean choice.
⚠️ **Not shippable yet:** independent windows produce boundary pops of 4–12× the content floor
(`GAP-PROGRAM.md` V12-F). Streaming memory is the outstanding fix.

**Real-ESRGAN — see the warning above.** Sub-second per frame at SD.

---

## Competitive floor, for context

Topaz Starlight Mini documents a **36 GB minimum on macOS** and states Preview *"may also not work"*
for Mac users (`mlxengine-todo/TOPAZ-MATCHUP.md` §3b). Every number in this file is below that floor.
Aiarty publishes **no memory requirement at all** (§3c).

---

## How to keep this file honest

- **Numbers here must come from a receipt in `mlxengine-todo/probes/`, with the row named.** No
  estimates, no extrapolation. If it was not measured, say "unmeasured".
- **Use `phys_footprint`, never MLX peak** — MLX peak under-reads the admission basis by ~2.5–2.9×,
  and that confusion has already produced one wrong "fits in 16 GB" claim (V12 §6).
- **When a footprint is input-dependent, say so and give the range.** A single number for a function is
  the defect this file opens with.
- When a package's manifest is corrected, update the row *and* note what changed, so an integrator
  pinned to an older version can tell whether it applies to them.
