# Changelog

All notable changes to *aria* are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Faster by default, with byte-identical output (checked with `cmp` against the previous build on CPU
  and GPU, both models, 10 s and 60 s). Measured on an i9-10900KF + RTX 3070: CPU 18–22 % faster
  (small 60 s 12.3 → 9.6 s, medium 10 s 10.0 → 7.8 s), GPU 10–24 % faster (small 60 s 0.49 → 0.37 s,
  medium 60 s 1.26 → 1.14 s), and a one-shot GPU run with a text prompt 36–47 % faster.
  - CPU: one OpenMP thread per physical core by default (`OMP_NUM_THREADS` still wins); the small
    decoder/encoder run their per-token GEMMs over groups of chunks instead of one 34-token chunk at
    a time; the medium decoder's glue loops, RoPE, and banded attention are parallel and
    vectorized; activations pack in parallel; large-A GEMMs (decoders, long clips) use a cache-blocked
    driver; T5Gemma runs only the prompt's real tokens and reads its BF16 weights from the mmap
    (−340 MB RAM); the decode window is sized from physical RAM instead of a fixed 8 s.
  - GPU: the denoise loop runs on the device with no per-step host round trip; warp-per-row softmax;
    the differential-attention subtract is fused into the head merge; the small decoder's chunk
    attention is one fused kernel; weights upload as host-converted fp16 (half the PCIe traffic);
    decoder arenas are kept across requests.
- The decoders' mapping convolution runs as a GEMM (output differs from before by ~2e-7 relative,
  fp32 rounding).
- New defaults, each one faster and A/B-checked for quality against the old default (CLAP prompt
  similarity non-inferior, wav2taste drift below seed noise; see BENCHMARKS.md): **6 denoising steps**
  instead of 8, and on GPUs with int8 tensor cores (sm_75+) a **q8 DiT with int8 activations (W8A8)**
  instead of fp16. These change the output; `-s 8 --precision fp32` reproduces the previous default.
  The CPU default stays fp32: on x86 the int8 paths are slower.

### Added

- `--schedule ref|len|full`: the sampling schedule, now read from the model config. `ref` is what
  `stable_audio_3` `generate()` samples with (LogSNR, rate 0), `len` is aria's previous length-dependent
  schedule (still the default, see BENCHMARKS.md), `full` is the training shift the stable-audio-3
  TensorRT runtime uses.
- `--headroom <sec>`: generate past the requested duration and crop, as the reference does (6 s).
- `--precision auto` (the default) and `--precision q8a8` (q8 weights + W8A8 GEMMs, same as
  `ARIA_W8A8=1` with `q8`); `ARIA_DTYPE_AUTO` / `ARIA_Q8A8` in the C API.
- `prompt=` and `weights=` timers on the `ARIA_PROFILE` line.

### Fixed

- `--continue` / `--inpaint` / `--stream` on the medium model now fail with a clear error instead of
  encoding garbage: only the small models' 768-d SAME-S encoder is implemented.

## [0.1.0] - 2026-07-08

First public release: a dependency-free C/CUDA runtime for Stable Audio 3.

### Added

- Native runtime for the full Stable Audio 3 text-to-music pipeline — tokenizer, T5Gemma text
  encoder, diffusion-transformer denoiser, and audio autoencoder — with no BLAS, deep-learning
  framework, or third-party dependency beyond a C math library and OpenMP.
- Support for both released model variants: `small-music` and the 1.2B-parameter `medium`.
- Two backends behind one op surface: a vectorized, multi-threaded CPU path (AVX2/FMA, ARM/NEON)
  and a CUDA path with half-precision tensor-core kernels and a per-step denoise loop captured and
  replayed as a CUDA graph. Byte-identical output where a path is shared.
- Precision options: fp16 / bf16 / 8-bit (q8) / 4-bit (q4) weight storage, plus an int8-activation
  mode (W8A8) on GPU tensor cores and the equivalent ARM instruction. A source-releasing overlay
  frees the full-precision weights once packed, so a lower precision reduces resident memory rather
  than adding to it.
- Offline quantizer (`aria-quantize`) and `--load-quant` for pre-packed overlays.
- Built-in activation steering: three injection sites (residual stream, latent, text conditioning),
  additive and projection operators, optional step windows, and `.atns` direction files. Live
  steering ramps during streaming; output is bit-identical to the base model at zero strength.
- Per-axis LoRA adapters via `--lora`.
- Long-form and streaming generation: bounded-memory windowed decoding, `--stream` sliding-window
  generation, `--continue` / `--inpaint`, `--anchor` re-anchoring, and `--evolve` seed control.
- Serving: a resident batch mode (`--batch`) and an HTTP server (`aria-server`) that keep the model
  loaded across requests.
- CLI utilities: `--fast` preset, `--bench`, `--info`, `--list-tensors`, and WAV round-trip tools.
- Runs on NVIDIA GPUs, x86 CPUs, and the Raspberry Pi 5, including the medium model at 4-bit within
  an 8 GB memory budget.
- Weight downloader (`download_model.sh`), runnable examples, and benchmark/design notes
  (`README.md`, `BENCHMARKS.md`, `PROFILING.md`).

[Unreleased]: https://github.com/matteospanio/aria/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/matteospanio/aria/releases/tag/v0.1.0
