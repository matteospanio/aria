# aria vs stable-audio-tools — benchmark

## 2026-10 optimization round (byte-identical output)

Same machine as below (i9-10900KF 20T + RTX 3070), `--uncond`, 8 steps; warm-min inside each
process, median of 3 interleaved base/new rounds (the i9 drifts about ±8 % run to run). Every row's
output is `cmp`-identical to the previous build (CPU and GPU, both models, 10 s and 60 s, with and
without a prompt; also fp16 storage, W8A8, `--continue`, `--stream`, steering, `ARIA_NO_GRAPH`, and
1 thread vs default). The decoder mapping conv-as-GEMM is the one fp32 reordering (rel_rmse ≤ 2.1e-7).

| config | before | after | total (DiT / decode) after |
|---|---|---|---|
| CPU small 10 s | 2.61 s | **2.08 s** (−20 %) | 1.69 / 0.37 |
| CPU small 60 s | 12.29 s | **9.59 s** (−22 %) | 7.46 / 2.10 |
| CPU medium 10 s | 10.03 s | **7.84 s** (−22 %) | 5.36 / 2.37 |
| GPU small 10 s | 0.13 s | **0.10 s** (−23 %) | 0.09 / 0.01 |
| GPU small 60 s | 0.49 s | **0.37 s** (−24 %) | 0.30 / 0.06 |
| GPU medium 10 s | 0.34 s | **0.29 s** (−15 %) | 0.22 / 0.07 |
| GPU medium 60 s | 1.26 s | **1.14 s** (−10 %) | 0.78 / 0.36 |

One-shot GPU CLI with a text prompt (process start to WAV, warm page cache): small 10 s
1.36 → **0.72 s**, medium 10 s 2.77 → **1.78 s** — T5Gemma now runs only the prompt's real tokens
(0.2–0.35 s → 0.05 s) and weights upload as host-converted fp16. What remains is mostly the upload
(`weights=` 0.30 s small / 0.84 s medium).

Where it came from:
- **Threads:** one OpenMP thread per physical core. HT siblings only contend for the FMA ports
  the GEMMs saturate: the DiT is ~5–10 % faster at 10 threads than 20 on the i9, and decode too.
- **Small decoder:** the per-token GEMMs (qkv/out/FF) run over 32-chunk groups (M = 1088) instead of
  one 34-token chunk each, so every block's weights stream once per group, not once per chunk:
  CPU decode −40 %.
- **Medium decoder:** the serial glue (GLU `sinf`/`expf`, RoPE, residuals) is parallel; banded attention
  splits over (head × 64-query block) and vectorizes the q·k dot 8 keys at a time with the same
  mul-then-add rounding; decode windows grow to what 1/32 of RAM holds (a 10 s medium clip no longer
  decodes 132 frames for 108).
- **GEMM:** A packs in parallel; packed-A ≥ 8 MB takes a cache-blocked driver (KC/MC from `sysconf`).
  Measured on the real model, the default dispatch beats both "always panels" and "always blocked"
  (small 60 s DiT 7.48 vs 7.6 vs 8.2 s); the blocked driver only wins on large-K, large-M GEMMs
  (medium decoder FF-out 410 → 589 GFLOP/s). The CLI sets `OMP_WAIT_POLICY=active`; without it the
  panels driver was erratic in a standalone microbenchmark. `aria-server` and libaria programs don't
  set it (active waiting also keeps idle cores spinning) — export it for CPU-heavy resident use.
- **GPU:** the pingpong loop runs on the device (gcond precomputed, noise drawn on the host into
  pinned buffers while the DiT runs, `__fmul_rn` pingpong kernel); warp-per-row softmax that
  reproduces the 256-thread tree's pairing; differential subtract fused into the head merge; the small
  decoder's 34×34 chunk attention is one fused kernel (cuBLAS ran it as 128×128-tile SGEMMs) — and
  it comes out bit-identical, since both are sequential-k FMA chains.
- Tried and dropped: padding the DiT score rows to 8 for cuBLAS `align8` kernels — cuBLAS keeps the
  `align1/2` kernels unless the key count itself is a multiple of 8.

**Sampling schedule.** aria samples with LogSNR `rate=1` (length-dependent); `stable_audio_3`
`generate()` uses `rate=0` (its wrapper default when the config has no sampling shift), and the
stable-audio-3 TensorRT runtime uses the training `DistributionShift`. All three are now available
via `--schedule ref|len|full` and parity-checked against the reference `build_schedule` (≤ 6e-8).
A/B on the 3070 (12 prompts × 4 seeds × {10, 30, 60 s} × both models, CLAP prompt similarity +
wav2taste): `ref` vs `len` ΔCLAP is within ±0.008 in every cell with every 95 % CI spanning 0, and
wav2taste drift stays under the seed-noise floor — no measurable quality difference, while `ref`
ends clips more naturally (last second 7–10 dB quieter at 10 s). The pre-registered non-inferiority
margin (δ = ¼ seed-sd) was not met in 5 of 6 cells for lack of power, so the default stays `len`
for now. `full` is significantly worse on medium 10 s (ΔCLAP −0.015); `--headroom 6` shows no gain.

### New defaults (output-changing, A/B-gated)

A speed setting that changes the output becomes a default only on hardware where it is faster *and*
passes the same A/B against the exact default (12 prompts × 4 paired seeds per cell; pass = lower
95 % bootstrap CI of ΔCLAP ≥ −¼ seed-sd and median wav2taste drift below the seed-noise floor):

| knob | cells | ΔCLAP [95 % CI] range | wav2taste drift / seed noise | verdict |
|---|---|---|---|---|
| 6 steps (vs 8), GPU | small, medium × 10, 60 s | +0.002 … +0.008 (small 10 s CI > 0) | 0.053–0.068 / 0.112–0.143 | pass → default everywhere |
| q8 + W8A8 (vs fp16), GPU | small, medium × 10, 60 s | −0.0014 … +0.0025 | 0.029–0.041 / 0.112–0.143 | pass → default on sm_75+ |
| q8 + W8A8 (vs fp32), CPU | small, medium × 10 s | −0.002 … +0.003 | 0.033–0.040 / 0.136–0.143 | pass, but 1.3–1.4× **slower** on the i9 → not a default |
| decoder q8 (vs fp32), CPU | small × 10 s | +0.0013 [−0.0018, +0.0044] | 0.004 / 0.136 | pass, but 1.2× **slower** on the i9 → not a default |

The CPU int8 paths are slower on AVX2 without VNNI. On ARM with dotprod they were faster in
earlier Pi sessions (see the Raspberry Pi table), but the fp32 path has moved since; they become
the ARM default once a same-session Pi timing on the current build confirms it.

Plain defaults, before (previous release: 8 steps, fp16 GPU / fp32 CPU) → after (6 steps, q8a8 on
the GPU, fp32 on the CPU); same protocol as above, and peak VRAM sampled with `nvidia-smi` (whole
GPU, ~330 MB idle included):

| config | before | after | exact 6-step (`-s 6 --precision fp32`) | peak VRAM before → after |
|---|---|---|---|---|
| GPU small 10 s | 0.13 s | **0.06 s** | 0.08 s | 1679 → 1357 MB |
| GPU small 60 s | 0.50 s | **0.25 s** | 0.30 s | 2575 → 2259 MB |
| GPU medium 10 s | 0.34 s | **0.18 s** | 0.24 s | 4479 → 3417 MB |
| GPU medium 60 s | 1.32 s | **0.85 s** | 0.96 s | 5735 → 4675 MB |
| CPU small 10 s | 2.66 s | **1.65 s** | — | |
| CPU small 60 s | 12.59 s | **7.74 s** | — | |
| CPU medium 10 s | 10.21 s | **6.47 s** | — | |

`-s 8 --precision fp32` reproduces the previous default (byte-identical on the GPU; on the CPU within
the conv-as-GEMM rounding, rel_rmse ≤ 2.1e-7).

aria (this runtime) vs **stable-audio-tools** (the PyTorch reference), same machine,
same model weights, same workload: **uncond, 8-step pingpong, DiT + autoencoder
decode** (no text encoder — isolates the core diffusion runtime). Both measure
**warm** generation with the model resident (aria = min of 3, SAT = mean of 3;
medium CPU = single run). Hardware: **Alienware — RTX 3070 (8 GB, sm_86) + i9-10900KF (20 threads).**

GPU: aria runs fp16 weights / fp32 compute (cuBLAS tensor cores); SAT runs fp16
(`.half()`). VRAM is the **actual process footprint** (`cudaMemGetInfo` for aria,
`mem_get_info` total−free for SAT) so the CUDA context + allocator pool are counted
for both; SAT's `max_memory_allocated` (tensors only) is shown in parentheses.

## small-music (10 s clip, 8 steps)

| metric | aria | stable-audio-tools | winner |
|---|---|---|---|
| **GPU time** | **0.19 s** | 0.34 s | aria (1.8×) |
| **GPU VRAM** (process) | **1548 MB** | 2770 MB (alloc 1161) | aria (−44 %) |
| **CPU time** (20 threads) | 3.5 s | **2.8 s** | SAT (1.25×) |
| **CPU peak RAM** | **2089 MB** | 5421 MB | aria (−61 %) |

## medium (GPU 10 s, CPU 5 s, 8 steps)

| metric | aria | stable-audio-tools | winner |
|---|---|---|---|
| **GPU time** | 0.54 s | **0.50 s** | ≈ parity (1.08×) |
| **GPU VRAM** (process) | **4314 MB** | 5452 MB (alloc 4561) | aria (−21 %) |
| **CPU time** (5 s, 20 thr) | 8.1 s | **7.0 s** | SAT (1.15×) |
| **CPU peak RAM** (5 s) | **7204 MB** | 18866 MB | aria (−62 %) |

aria GPU breakdown — small: setup 0.02 + dit ~0.14 + decode 0.02; medium: setup
**0.02** + dit **0.38** + decode **0.13**. Three profile-guided GPU optimizations took
medium **0.98 → 0.54 s** (≈ SAT parity) and small-music **0.29 → 0.19 s** (1.8× SAT):
1. **on-device cross-K/V projection** (was a per-request host GEMM): medium setup
   0.33 → **0.02 s** (`ca_to_kv` device-resident, +340 MB).
2. **fp16 tensor-core attention** for the DiT (the QK/AV ran as fp32 `sgemm` = 21 % of
   GPU time, no tensor cores). Small decoder stays fp32 (fp16 there cost accuracy for
   ~0 gain).
3. **fused sliding-window decoder attention** — the medium decoder ran full N² (N=1836)
   then masked to ±17; a warp-per-query kernel computes only the 35-key window
   (O(N·35), exact), dropping decode **0.23 → 0.13 s** and a 324 MB N² score buffer.
What's left (medium 0.54 vs 0.50) is fp32 elementwise + launch overhead — diminishing
returns. aria's DiT/decoder GEMMs are already fp16 tensor cores.

## Long audio — 60 s reference (8 steps)

The motivating workload is a **60 s** clip. Below is aria-vs-SAT at 10 s and 60 s so
the *scaling* is visible (uncond, warm; Alienware RTX 3070 + i9-10900KF).

**medium:**

| backend | aria 10 s | aria 60 s | aria 10→60 | SAT 10 s | SAT 60 s | SAT 10→60 | @ 60 s |
|---|---|---|---|---|---|---|---|
| **CPU** (20 thr) | 11.2 s | **55.9 s** | 5.0× | 11.3 s | **108.4 s** | 9.6× | **aria 1.94× faster** |
| **GPU** (fp16) | 0.46 s | **1.82 s** | 4.0× | 0.49 s | **1.32 s** | 2.7× | SAT 1.38× faster |

**small-music:**

| backend | aria 10 s | aria 60 s | aria 10→60 | SAT 10 s | SAT 60 s | SAT 10→60 | @ 60 s |
|---|---|---|---|---|---|---|---|
| **CPU** (20 thr) | 3.58 s | **16.3 s** | 4.6× | 2.79 s | **11.1 s** | 4.0× | SAT 1.47× faster |
| **GPU** (fp16) | 0.16 s | **0.60 s** | 3.8× | 0.34 s | **0.41 s** | 1.2× | SAT 1.46× faster |

aria 60 s stage split — medium GPU dit **1.28** + decode 0.78 s, CPU dit 35.1 + decode 20.6 s
(CPU dit after query-blocked attention, −4% on the 20-core); small-music GPU dit 0.50 +
decode 0.16 s, CPU dit 12.5 + decode 4.3 s. GPU VRAM @60 s: medium aria 4514 vs SAT 6720 MB.
The medium GPU dit dropped **1.45 → 1.28 s (−12 %)** by fusing the elementwise long tail
(see below); the gap to SAT closed 1.68× → 1.57×. Folding 1/√D into the QK^T alpha and the
fp16 cast into softmax (commit `42962e0`, byte-identical output) removed two more S²·H passes
per attention: 60 s GPU **medium 2.07 → 1.82 s (−12 %), small 0.67 → 0.60 s (−10 %)**;
the SAT gap closed further, 1.57× → **1.38×** (medium) and 1.63× → **1.46×** (small).
Fusing the medium-decoder head glue (E16.2, `cab0808`, byte-identical, 156 → 60 launches per
decode) took 60 s medium to **1.73 s** — the gap is now **1.31×**.

**small-music has no banded-decoder win** (its decoder already chunks → O(N·34) linear),
so on CPU it's pure GEMM and MKL edges aria (1.3–1.5×); the medium CPU win is specifically
the banded decoder fix. On GPU both models show the same long-audio gap (parity/ahead at
10 s → ~1.7–1.8× behind at 60 s).

**The decoder fix (CPU + GPU).** The medium decoder's ±17 sliding-window attention
ran as a full N×N score matrix + mask, where the transformer sequence `N = T×17`
(10,982 tokens at 60 s) — pure **O(N²)**: CPU decode measured 18.6 → 78.0 s when the
latent length doubled (15→30 s). A banded kernel computes only the 35-key window
(`aria_attention_band` on CPU; `k_attn_band` on GPU; both numerically exact, A/B
audio rel 1e-5). CPU decode at 60 s ~300 → 21–32 s; the win grows with duration. This
is why **aria CPU now scales *better* than PyTorch** for long audio (SAT CPU 9.6× vs
aria 5.0× from 10→60 s) — PyTorch's medium decoder still pays the growth aria shed.

**The remaining gap — GPU.** aria GPU is parity/ahead at 10 s but **~1.7–1.8× behind
at 60 s** (both models). The DiT self-attention is full O(S²) (S = 710 at 60 s) and
aria materializes the fp16 S×S scores + a softmax pass, vs PyTorch SDPA's flash
attention. **Two flash kernels were tried and reverted**, both losing to cuBLAS:
(1) an **fp32** flash made the DiT 4× slower (no tensor cores; latency-bound per-thread
online softmax); (2) a **WMMA tensor-core** flash was numerically correct (small-music
GPU-vs-CPU 1.18 %) but still slower (medium DiT 1.45→1.9 s, small 0.52→0.75 s) and
regressed medium fidelity to 5.8 %. At S ≤ 710 (S² = 0.5 M) cuBLAS GemmEx + a parallel
softmax reduction beats a hand-rolled single-warp-per-tile flash; matching PyTorch
needs a cuDNN-class **multi-warp/pipelined FlashAttention-2** kernel — large effort
for an already-fast (≤ 2.2 s) path, so deferred. The cuBLAS attention stands.

**What did help — fusing the long tail.** nsys (medium, GPU) shows the GEMMs are only
~41 % of GPU time, the attention softmax ~9 %, and a **~32 % long tail** of small
memory-bound elementwise/reshape kernels (fp16-convert, rmsnorm, RoPE, extract/merge-
heads, adaLN/gate). Fusing the biggest of these — **extract-head + per-head qk-rmsnorm +
RoPE** into one kernel (`k_extract_normrope`), plus rmsnorm+adaLN and gate+residual-add —
cut the **medium GPU DiT 1.45 → 1.28 s (−12 %)**, whole gen 2.22 → 2.07 s, parity
unchanged. It's a bigger, lower-risk lever than flash (the softmax FA-2 would touch is
only ~9 %). Remaining long-tail headroom: the per-attention fp16 conversions (have
`k_extract_normrope` emit fp16 so `attn_dev` skips them).

**CUDA Graphs (landed).** The per-step DiT is now captured once and replayed (`--default-
stream per-thread` + cuBLAS pinned to it + a fixed workspace; robust inline fallback;
`ARIA_NO_GRAPH=1` to A/B). Output is byte-identical to the inline path (rel_rmse 0). The
launch-overhead win is **modest at these sizes** — ~4 % warm-min on the medium DiT (s=24,
0.96→0.92 s), within noise on small-music — because the small audio latent keeps the DiT
compute-bound so the ~14 k launches mostly hide behind compute. The graph's clearer benefit
is **steadier per-step latency** (less launch jitter); the win grows with steps and the
medium differential path's kernel count. Also drops the redundant per-block residual copies
(3 `cudaMemcpyD2D`/block) on the same path.

**True Megakernel feasibility (measured, not adopted).** `tests/bench_megakernel.cu` benches a
from-scratch WMMA tensor-core GEMM vs cuBLAS `GemmEx` at the DiT FFN/qkv shapes (3070, sm_86).
The WMMA kernel is **bit-identical** to cuBLAS but **0.12–0.56×** its speed, and cuBLAS runs
these GEMMs at **18–31 TFLOP/s** (≈50–78 % of the ~40 TFLOP/s fp16-tensor peak) at only
**49–138 GB/s** (≪ 448 GB/s HBM) — i.e. **compute-bound, not memory-bound**, so a megakernel's
HBM-traffic savings buy nothing on the GEMMs. Crossover model: per-block weights 33.6/75.5 MB
(small/medium) stream regardless; activation share 35 %→69 % of HBM traffic over S=172→710;
ideal free-GEMM fusion ceiling 1.18×→1.53× (crossover S≈315/473), much of it already L2-resident.
Verdict: a 3–8× GEMM penalty vs a ≤1.53× ceiling on compute-bound kernels → **not worth it**;
cuBLAS+graph+glue-fusion is at/near the roofline. The crossover analysis is the paper result.

**fp16-emit attention epilogue (exact, landed).** The extract-head kernels
(`k_extract_normrope`, `k_extract_heads_h`) now emit fp16 q/k/v directly and the cross-K/V
request cache is stored fp16 (converted once per request, not per step), so `attn_dev_h`
consumes fp16 with zero per-call conversion passes. `__float2half` at the producer is the
same RTNE value the old fp32-store + convert pass produced → **output byte-identical**
(small + medium 60 s, graph and inline). 60 s warm: small DiT 0.41 → **0.40 s**, medium DiT
1.04 → **0.97 s** (−7 %); VRAM −50/−64 MB (small/medium).

**W8A8 int8 tensor-core GEMMs (`ARIA_W8A8=1` + `--precision q8`).** The q8 block weights are
already packed int8+scale in VRAM; instead of dequantizing to fp16 per GEMM, the activations
are quantized per-token (row absmax, symmetric, mirrors `aria_q8_quant`) and the 6 per-step
block GEMMs run **int8×int8 → int32 IMMA** (2× fp16 tensor throughput on sm_86), dequantized
in place in the fp32 output (`y = acc·sx[m]·sw[n] + b`). q8 flips from the *slowest* GPU mode
(dequant overhead) to the **fastest**; cross-K/V stays fp16 (fidelity-sensitive), attention
QK/AV stays fp16. Graph-capturable (graph-vs-inline byte-identical). 60 s warm, RTX 3070:

| mode | small 60 s | medium 60 s | GPU VRAM (med) |
|---|---|---|---|
| fp16 (default, exact) | 0.55 s (DiT 0.40 + dec 0.13) | 1.43 s (DiT 0.96 + dec 0.43) | 4.46 GB |
| q8, fp16-dequant (old) | 0.62 s (DiT 0.47) | 1.62 s (DiT 1.11) | 3.40 GB |
| **q8 W8A8** | **0.46 s (DiT 0.30)** | **1.24 s (DiT 0.75)** | 3.40 GB |

(with the decoder mapping-conv-as-GEMM: decode small 0.16 → **0.13 s**, medium 0.48 →
**0.44 s**, audio rel 7e-7 — the fp32 conv ran as a naive kernel, 29/42 ms at 60 s.)
vs stable-audio-tools at 60 s: small 0.46 vs 0.41 (gap 1.46× → **1.12×**), medium 1.24 vs
1.32 — **aria now beats SAT on long-audio medium** (was 1.31× behind at the last audit);
fp16-exact is at ≈ parity (1.43 vs 1.32, 1.08×). 10 s small DiT 0.12 → 0.08 s.
Fidelity (single-step `-s 1` probe, isolates one DiT call): medium q8-dequant 3.2 % →
W8A8 7.3 % velocity-class error — between q8 and the supported q4's 9.3 %, i.e. a
legitimate point on the existing precision dial (opt-in, off by default).

**W8A8 on the CPU (NEON sdot, E15.2c).** The same `ARIA_W8A8` knob routes the CPU q8
block GEMMs through per-token int8 activations + exact-int32 sdot accumulation
(`aria_linear_q8a8`; sdot / vmull / scalar paths are bit-identical). **Pi 5, 10 s
small-music: q8 DiT 57.3 → 22.8 s (2.5×), total 67.5 → 32.6 s** — q8 flips from the
slowest CPU mode to the fastest (fp32 measured 44.3 s in the same session; the box ran
soft-thermal-limited, so absolute numbers are conservative).

**Banded DiT self-attention: measured and REJECTED.** Per-layer softmax mass at 60 s
(S=710, small-music, CPU instrumentation): the 64 memory tokens absorb 17–53 % of every
audio query's attention, and even memory-prefix + a ±256 band (79 % of all keys) covers only
**86–96 %** of the mass depending on layer (±128: 70–92 %). The DiT's self-attention is
genuinely global — unlike the medium *decoder*'s architectural ±17 window — so a windowed
approximation would discard 4–30 % of attention mass per layer and was not wired in. This is
the flash-attention counterpoint: the only remaining long-audio attention lever is a real
fused-softmax kernel (cuDNN-class), not sparsity.

## Streaming / continuation on GPU (small-music, 8 steps, RTX 3070)

Continue/inpaint used to be gated to the CPU DiT (the local-additive inpaint cond was
CPU-only). Porting it to the device DiT (reuse the CPU-projected per-block `local_emb`,
upload it, add after cross-attn via the existing `k_add`) makes `--continue`/`--inpaint`/
`--stream` run on the GPU. The taae context encoder stays on the CPU.

- **Parity:** GPU vs CPU continuation, same seed → regen region **corr = 1.0000, rel_rmse
  0.45 %** (pure fp16 DiT noise; the local-cond is uploaded + added in f32, so bit-identical).
  CPU and GPU `--stream` outputs are identical to 2 decimals per second.
- **Speed:** an 18 s continuation **6.92 → 2.54 s (2.7×)**. Streaming per-chunk (4 s emit,
  14.5 s window): CPU **RTF ~0.6×** → GPU **RTF ~4×** (chunk ~1.0 s). Crosses real-time with
  ~3× headroom → smooth live playback (`--stream -o - | play …`, writer thread overlaps gen).
- **Decay ceiling (device-independent):** for prompts whose chunk-0 emit reaches SA3's
  natural outro fade (e.g. "ambient pads", "funk groove"), the chained continuations inherit
  the fade and drift to silence past ~9 s. Steady-energy prompts ("melodic techno") hold.
  This is the inpaint-envelope-drift limit of the post-hoc method, not the GPU — CPU and GPU
  decay identically (beat-sync loop / streaming-trained model).
- **Re-anchoring (`--anchor <beta>`, E14b) bounds the decay.** Each continuation context is
  blended toward the chunk-0 reference tail (convex, conditioning-only) and its RMS pulled
  back to the reference level (boost-only, ≤4×, then a ≤0.97 peak guard for the encoder).
  Measured on the worst-case "ambient pads" (8 chunks, seed 0, RTX 3070): baseline sinks
  −15.5 dB tail-vs-head and keeps falling (−34 dB); `--anchor 0.35` settles at ~−26 dB and
  *recovers* (−8.8 dB drop; β 0.5/0.7 similar) — the stream no longer dies. The energy
  correction is what matters: a plain amplitude blend only softened the drop (−13.1 dB).
  `--evolve <n>` (E14c) adds the SA3-Realtime-style seed policy: default = fixed seed every
  chunk (the voice carries forward); `--evolve N` re-seeds every N chunks for variation.
  Defaults byte-identical (anchor/evolve off: GPU + CPU streams cmp-equal to pre-change).

## Steering efficiency — aria vs sf-api (E12)

Same residual steer (`delta = α·norm·unit`) through the aria C runtime and the PyTorch
reference. 10 s / 8 steps, steered, RTX 3070.
**Warm = in-process resident (`--bench` warm-min) on both sides**; a one-shot aria CLI
invocation additionally re-pays per-process setup (T5 encode on CPU + weight upload —
"invocation" row):

| metric | aria small | sf-api small | aria medium | sf-api medium |
|---|---|---|---|---|
| GPU warm steered gen | **0.16 s** (2.2×) | 0.358 s | **0.46 s** (1.2×) | 0.547 s |
| GPU per-invocation (one-shot CLI) | 1.23 s | — | 2.55 s | — |
| cold start (process+load+gen) | **1.6 s** (5×) | 7.9 s | **2.9 s** (6×) | 17.5 s |
| GPU peak VRAM | **1395 MB** (2.2×) | 2999 MB | **4215 MB** (1.3×) | 5375 MB |
| CPU warm steered gen | **3.5 s** | — | **11.2 s** | — |

aria wins every axis measured apples-to-apples. Two earlier claims were measurement artifacts,
now retired: "aria 0.10 s / 3.4× faster" (aria *unsteered* graph path vs sf-api steered) and
"steering disables the graph → aria 3.4–4.6× slower warm" (aria *fresh-process* vs sf-api
in-process). In-process at 10 s scale, graph/no-graph/steered are all ≈0.16 s — per-step launch
overhead is not the bottleneck there; the per-invocation gap is process setup (→ `--batch`).

Steering is bit-exact parity-verified (`tests/steer_verify.sh`: scale-0 byte-identical,
`steer(2α,d)≡steer(α,2d)`), CPU + GPU. Since commit `bf00c9e` the CUDA graph **stays captured
while steering**: the steer kernels are recorded into the graph and read a device-resident
effective scale (0 outside the step window → bit-exact no-op) refreshed per step like
`dx`/`dgcond`; steered graph-vs-inline output is byte-identical (full + windowed).

## Raspberry Pi 5 (edge target, E15)

First measured deployment on the paper's edge device — **Raspberry Pi 5, 4× Cortex-A76,
8 GB**, small-music fp32, 8-step, CPU only (`make ARCHFLAGS="-mcpu=native"`), windowed
decode (E16.1) active:

| model / build | 10 s clip | peak RSS |
|---|---|---|
| small fp32, scalar / gcc autovec | 50.5 s | ~1.9 GB |
| **small fp32, NEON (E15.2)** | **28.4 s** | ~1.9 GB |
| small q8, scalar dequant | 276 s | 2.9 GB |
| small q8, **NEON dequant (E15.2b)** | **67.1 s** (4.1×) | 2.9 GB |
| small q8, **W8A8 NEON sdot (E15.2c)** | **32.6 s** (2.1× over dequant; DiT 57.3→22.8 s) | 2.9 GB |
| **medium q4, NEON** | **242 s** | **6.79 GB — fits the 8 GB board** |

(small fp32 NEON: 4 s clip 16.6 s; GEMM 34 → 58 GFLOP/s, K-blocked 12.9 → 52.6.)

≈ **0.35× realtime** for small on a $80 board, and — the paper's edge claim, measured —
**the 1.2 B-parameter medium variant generates on the Pi at 4-bit** (windowed decode +
q4 + NEON dequant are jointly the enabler: fp32 medium would not fit, and the pre-E16.1
monolithic decode would have blown the 8 GB budget). NEON-vs-scalar parity 3.8e-5; full
test suite green on ARM. Remaining levers: E15.3 params-on-disk (headroom below 6.8 GB),
sdot int8 path, threads-vs-thermals tuning.

## Generation-speed optimization pass (2026-07, audit-driven)

Eight measured levers from a multi-lens code audit. Each is byte-/parity-gated; the
default (fp16 GPU / fp32 CPU, 8 steps) is untouched unless noted.

| # | lever | where | measured | gate |
|---|---|---|---|---|
| A1 | `--fast` = 6 steps | sampler | ~25 % off every DiT-bound path (med GPU 1.43→1.19, CPU 60 s 55.9→44, Pi q8 32.6→27) | wav2taste: 8→6 drift < seed noise on both tiers |
| A2 | GPU micro-pack (softmax writeback drop, FF-bias fold, fp16-emit producers) | `aria_cuda.cu` | 60 s exact: small 0.55→0.49, **medium 1.43→1.28 (beats SAT 1.38)**; W8A8 med 1.24→1.17 | fp16 byte-identical (graph+inline) |
| A3 | warm-request reuse + 8-way T5 LRU | `aria_cuda.cu`, model | per-request setup 0.03→0.01 s; batch/sweeps skip graph rebuild + re-encode | bench/batch byte-identical |
| A4 | CPU glue pack (parallelize+vectorize glue, collapse(head,qblock), conv-interchange, `OMP_WAIT_POLICY=active`) | `aria_cpu.c`, `aria_sa3_dit.c` | i9 small 60 s CPU 16.3→11.3 s; SAT gap 1.47×→1.09×; Pi small 10 s 39.2→28.4 s | CPU + GPU byte-identical |
| B2 | fp16/bf16 CPU storage **+ overlay source-release** | `aria_quant.c`, `aria_sa3_dit.c` | Pi peak RSS fp32 1913 → fp16 1199 (−37 %) → **q8 835 MB (−56 %)**; i9 W8A8 2426→982 MB | fp16 rel 5.5e-3; W8A8/GPU byte-identical; unit-tested |
| B3 | opt-in decoder q8 (`ARIA_DEC_Q8`, ARM sdot) | `aria_taae.c`, decoders | **ARM-only**: Pi small decode 5.03→3.54 s (1.4×); **x86 is a 2× loss** (packed-fp32 wins) → do not enable on x86 | rel 9.0 % small / 2.7 % med (q4-class); off by default |
| B4 | x86 AVX2 int8 GEMM (maddubs W8A8 + widening q8) | `aria_quant.c` | i9 W8A8 DiT 5.11→1.99 s (2.6×), q8-dequant 27→3.6 s (7.5×) vs scalar; fp32 still fastest x86 mode | W8A8 byte-identical (exact int32); unit-tested |
| B5 | streaming partial-emit range decode (E14.1) | decoders, model, `--stream` | Pi small-music continuation decode **5.22→1.67 s (3.1×)**; i9 medium 3.5→2.3 s (wider halo) | emitted audio byte-identical x86 **and** ARM; `ARIA_NO_RANGE` A/B |

The **overlay source-release** (B2) is the sleeper win: every CPU overlay (q8/q4/fp16)
was previously a footprint *add* (packed copy + still-resident fp32 mmap); `MADV_DONTNEED`
of the six big DiT block matrices after the overlay copies them makes it a *replace*,
so q8 on the Pi drops from RAM-negative to the smallest footprint. Two levers stay
opt-in because they lose on the wrong hardware: **W8A8** (`ARIA_W8A8`) is a CPU win only on
ARM sdot / a GPU win everywhere; **decoder q8** (`ARIA_DEC_Q8`) is ARM-only and carries a
q4-class fidelity cost on the final audio, so it is never on by default. Deferred with a
reason: a **cuDNN FlashAttention-2 build flag** (`FLASH=cudnn`, long-audio GPU only —
libcudnn dependency, must be graph-captured) and the **GPU-IMMA decoder** (device decode
already 0.13–0.44 s; 2 structs × 9 sites for a marginal, fidelity-risky gain).

## Takeaways

- **GPU memory:** aria's footprint is consistently smaller (−27 to −47 %) — a bump
  allocator with no caching-allocator slack vs PyTorch's reserved pool + CUDA context.
  Notably, **medium fp16 OOMs stable-audio-tools' naïve `.to(cuda).half()` load** on
  the 8 GB card (transient fp32 copy); aria runs medium GPU in 3.97 GB with headroom.
- **GPU speed:** aria **wins small-music (1.8×)** and reaches **≈ parity on medium**
  (0.54 vs 0.50 s) after the three GPU optimizations above.
- **CPU speed:** within **1.15–1.5× of PyTorch/MKL** after the **packed outer-product
  GEMM + K-blocking** (was 2.4× behind); aria's AVX2 GEMM runs **~460–622 GFLOP/s** on
  the big DiT shapes (MKL is ~600–700). For **long-audio medium aria now *beats*
  PyTorch** — the banded decoder keeps aria near-linear while SAT's medium decode grows
  superlinearly: at **60 s, aria CPU is 1.94× faster** (55.9 vs 108.4 s). small-music
  (decoder already chunked, no banded win) stays pure-GEMM where MKL leads ~1.3–1.5×.
- **CPU/host memory:** aria uses **~60 % less RAM** everywhere (mmap'd weights, no
  framework). Medium on CPU: aria 7.2 GB vs PyTorch 18.9 GB.

**Where aria wins:** dependency-free single binary, the smallest memory footprint
(GPU and CPU, −44 to −62 %), faster small-music GPU, ≈-parity medium GPU, and
**faster long-audio CPU** (1.94× at 60 s). **Where the PyTorch stack still edges
ahead:** short-clip CPU throughput (MKL, ~1.3×) and **long-audio GPU** (1.68× at 60 s
— PyTorch's cuDNN tensor-core flash attention; a matching aria kernel is deferred).
aria's niche is exactly the plan's:
run these models on budget/low-VRAM hardware without the framework — now also at
competitive (often better) speed.

## Memory notes (no-speed-loss optimizations)

The dominant consumer is the **weights** — fp16 on the GPU, F32 (mmap, zero-copy) on
the host — which is the speed-optimal storage, so it's off-limits without a precision
(speed) cost. The clean wins are in waste:

- **Host RAM on the GPU path: 7.26 GB → 0.44 GB steady-state** (medium). The host F32
  weights stay mmap-resident even after the fp16 copies are on the GPU; once uploaded
  they aren't read on the hot path, so `madvise(MADV_DONTNEED)` reclaims them (file-
  backed → re-faults cheaply if a small per-request MLP is touched). Biggest win.
- **Decoder arena −~90–100 MB** (GPU peak + CPU): the attention scratch and the FFN
  scratch were held in one arena scope though never live together — freeing the former
  before the latter drops the per-block peak from `16·N·D + 3·N·INNER` to `~17·N·D`.
- **Tried and reverted:** Q8 on `ca_to_kv` (cross-K/V) saved 170 MB VRAM but the cross
  projection is fidelity-sensitive (GPU-vs-CPU 0.7 % → 2.6 %) — not worth it.
- **Hard floor:** pure-CPU RAM is weights-bound (medium ~5.4 GB F32, needed for the
  fp32 compute that gives exact parity); reducing it means fp16 compute = slower.

_Reproduce: aria `./aria -m <model> --uncond -d 10 -s 8 --device cuda|cpu --bench 3`
(`ARIA_PROFILE=1` for the breakdown); SAT via `scripts/bench_sat.py` (uncond DiT+decode)._
