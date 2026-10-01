# aria

[![arXiv](https://img.shields.io/badge/arXiv-2607.08526-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.08526)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

aria is a native inference engine for audio diffusion models, written from scratch in C with no third-party dependencies. It runs Stable Audio 3 (small-music and medium) end to end, from a text prompt to a stereo WAV, with no Python or deep-learning framework underneath.

It is CPU-first by design and runs on ordinary laptops, on old and cheap NVIDIA cards through an optional CUDA backend, and on an 8 GB Raspberry Pi 5. The point is to make capable open-weight audio models usable on hardware people already own, where the usual PyTorch stack is impractical.

## What it does

- **text → audio** — GemmaTokenizer BPE → T5Gemma text encoder → DiT denoiser over a LogSNR pingpong sampler → taae_v2 decoder → stereo WAV.
- **continue / inpaint** — extend a clip (`--continue`) or regenerate a region (`--inpaint --from --to`); kept regions are preserved bit-for-bit.
- **quantization** — `--precision fp32|q8|q4`. On the GPU, weights stay packed in VRAM and dequantize on use. q4 (asymmetric int4 + q8 attention) shrinks the small-music DiT from 1.6 GB to about 0.3 GB, at roughly 9% velocity error.
- **steering** — because the runtime owns every intermediate tensor, training-free activation steering is a single tensor add at a known boundary (the diffusion latent, a DiT residual, the conditioning vectors, or the pre-decode latent). See [Steering](#steering).
- **medium model** — differential DiT attention plus a sliding-window decoder, both device-resident on CUDA.

Every stage is parity-checked against the `stable-audio-tools` PyTorch reference; correctness is measured, not assumed. On an RTX 3070, a 10 s / 8-step small-music clip generates in about 0.29 s warm; the medium model runs fully on the GPU in about 2.2 s.

The CPU path (AVX2/FMA + OpenMP, zero-copy mmap weights) is the reference and must always build with `make`. CUDA is an optional accelerator.

This software was built with heavy assistance from large language models, with a human leading the ideas, testing, and debugging. We say so because it shaped how the project was built.

## Model weights

Stable Audio 3 weights live on the Hugging Face Hub and are gated: accept the license on the model page and log in first. The download script pulls exactly what aria needs — `model_config.json`, `model.safetensors`, and the `t5gemma-b-b-ul2/` text-encoder subfolder — into `./models/<name>/`.

```sh
./download_model.sh small-music   # ~2.3 GB DiT+autoencoder + ~1.2 GB T5Gemma encoder
./download_model.sh small-sfx     # sound-effects variant
./download_model.sh medium        # larger; needs quantization to fit a 2 GB GPU
```

The script authenticates with `--token TOKEN`, the `HF_TOKEN` environment variable, or your local token cache (`~/.cache/huggingface/token`), in that order. Get a token at <https://huggingface.co/settings/tokens> after accepting the license. Downloads resume if interrupted; run the same command again.

It requires the Hugging Face CLI (`hf` or `huggingface-cli`):

```sh
python3 -m pip install -U huggingface_hub
```

## Build

### Linux / POSIX

```sh
make                        # CPU build -> ./aria + build/libaria.a (default)
make test                   # hermetic unit tests (no model, no GPU)
make cuda CUDA_ARCH=sm_86   # CUDA build (sm_61 for a GT 1030)
make clean
```

Requirements: a C11 compiler (gcc/clang), `make`, libm, and OpenMP. No BLAS, no libsndfile, no JSON library. The CPU build stays warning-clean under `-Wall -Wextra`.

### Windows (MSVC, optional CUDA)

`build.bat` is the Makefile's Windows counterpart. Run it from an *x64 Native Tools Command Prompt for VS 2019/2022*:

```bat
rem CPU build: build\aria.exe + build\libaria.lib
build.bat
rem CPU build + hermetic unit tests (no model, no GPU)
build.bat test
rem CUDA build: needs the CUDA Toolkit; set ARIA_CUDA_ARCH first (default sm_86)
build.bat cuda
```

Usage is the same as on Linux, with `build\aria.exe` in place of `./aria`. The CUDA build needs the NVIDIA driver and the CUDA Toolkit's cuBLAS DLLs on `PATH` at runtime. Not ported yet: live prompt re-steering during `--stream`, and `aria-server` (both POSIX-only). The MSVC build compiles the scalar, single-threaded CPU kernels (no OpenMP or AVX2 yet), so use `--device cuda` on Windows.

## Usage

```sh
# one-time: export the tokenizer to a compact binary aria loads at runtime
python scripts/export_tokenizer.py models/small-music

# text -> audio  (--device auto picks the GPU when one fits, else CPU)
./aria -m models/small-music -p "warm romantic piano, slow, tender" -d 15 -s 8 --seed 0 -o out.wav

# unconditional, or from a precomputed [256,768] prompt embedding
./aria -m models/small-music --uncond -d 10 -o out.wav
./aria -m models/small-music --prompt-embed prompt.atns -d 10 -o out.wav

# pick backend / precision; --rng torch matches PyTorch's randn for reproduction
./aria -m models/small-music -p "..." --device cuda --precision q4 -o out.wav
./aria -m models/small-music --uncond -d 5 --seed 0 --rng torch -o out.wav

# continue / inpaint (CPU, init WAV at 44.1 kHz)
./aria -m models/small-music -p "..." -d 30 --continue in.wav -o out.wav
./aria -m models/small-music -p "..." --inpaint in.wav --from 5 --to 10 -o out.wav

# medium model (differential DiT + sliding-window decoder; full GPU on CUDA)
./aria -m models/medium -p "..." -d 10 -s 8 -o out.wav

# offline-quantize the DiT to a packed .aria, then load it
make quantize && ./aria-quantize models/small-music dit.q4.aria q4
./aria -m models/small-music --uncond -d 10 --load-quant dit.q4.aria -o out.wav

# inspect / all flags
./aria -m models/small-music --info
./aria -h          # full flag list;  make help  for build targets
```

`-m <dir>` is any directory with `model_config.json` and `model.safetensors` (what `download_model.sh` produces, or a raw Hugging Face snapshot). Text prompts need the T5Gemma weights plus the exported tokenizer; `--uncond` and `--prompt-embed` work without them.

## Steering

Steering directions are plain tensor files, applied as `x += scale * direction` at a chosen boundary — the 256-D diffusion latent, a per-layer DiT residual (1024-D), the conditioning vectors (768-D), or the pre-decode latent. The `--steer site:layer:dir.atns:scale:lo-hi` flag drives it, and the parity harness doubles as the path for extracting the directions. The paper's "sonic seasoning" case study uses this interface to bias generations toward taste associations.

## Parity and testing

Correctness is checked against the real `stable-audio-tools` PyTorch reference. The `make parity` harness dumps reference tensors from PyTorch and compares them in C with magnitude-aware tolerances. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to run and extend it.

## Documentation

- [AGENTS.md](AGENTS.md) — conventions and ground rules for anyone (human or AI) working in this repo. Read this before changing code.
- [CONTRIBUTING.md](CONTRIBUTING.md) — the parity and build regression tracks. Read this before sending a pull request.
- [BENCHMARKS.md](BENCHMARKS.md) — measured performance and memory across CPU, CUDA, and the Raspberry Pi 5.

## Acknowledgements

aria draws on two reference runtimes: [iris.c](https://github.com/antirez/iris.c) (its diffusion architecture and modularity) and [ds4](https://github.com/antirez/ds4) (its CUDA, quantization, and streaming ideas). It builds on Stable Audio 3 and [stable-audio-tools](https://github.com/Stability-AI/stable-audio-tools) by Stability AI, which provide the model and the parity ground truth. Thank you.

## License

The code is distributed under the MIT license; see [LICENSE](LICENSE). The model
weights are distributed under their own licenses — see the [Stability AI Community
License](https://stability.ai/license).

If you use this work, please cite this paper:

```bibtex
@misc{spanio2026quantizednativeruntimeondevice,
      title={A Quantized Native Runtime for On-Device Semantic Audio Generation}, 
      author={Matteo Spanio and Antonio Rodà},
      year={2026},
      eprint={2607.08526},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2607.08526}, 
}
```
