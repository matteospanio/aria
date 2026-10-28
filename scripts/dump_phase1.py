"""dump_phase1.py - Dump PyTorch reference tensors for aria's Phase 1 parity tests.

Run with a venv that has stable_audio_tools installed, e.g.:

  SA3_MODEL=~/.cache/huggingface/hub/models--stabilityai--stable-audio-3-small-music/snapshots/<snap>
  python dump_phase1.py "$SA3_MODEL" out_dir

Currently dumps:
  number_cond/seconds_<i>.atns   reference seconds_total embeddings [768]
"""
import os
import sys
import json
import math
import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(__file__))
from parity_io import save_atns

# Shared with tests/test_number_cond.c -- keep in sync.
SECONDS_VALUES = [0.0, 5.0, 15.0, 30.0, 47.3, 95.0, 200.0, 384.0]


def dump_number_cond(model_dir, out_dir):
    from stable_audio_tools.models.conditioners import NumberConditioner

    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    # locate the seconds_total conditioner config
    cond_cfgs = cfg["model"]["conditioning"]["configs"]
    sec = next(c for c in cond_cfgs if c["id"] == "seconds_total")
    cond_dim = cfg["model"]["conditioning"].get("cond_dim", 768)
    min_val = sec["config"].get("min_val", 0)
    max_val = sec["config"].get("max_val", 384)
    fft = sec["config"].get("fourier_features_type", "expo")

    nc = NumberConditioner(output_dim=cond_dim, min_val=min_val, max_val=max_val,
                           fourier_features_type=fft).eval()

    # load the Linear weights (embedder.embedding.1) from the model safetensors
    st_path = os.path.join(model_dir, "model.safetensors")
    with safe_open(st_path, framework="pt") as f:
        w = f.get_tensor("conditioner.conditioners.seconds_total.embedder.embedding.1.weight")
        b = f.get_tensor("conditioner.conditioners.seconds_total.embedder.embedding.1.bias")
    nc.embedder.embedding[1].weight.data.copy_(w)
    nc.embedder.embedding[1].bias.data.copy_(b)
    W = w.float().cpu().numpy()  # [768, 256]
    B = b.float().cpu().numpy()  # [768]

    def canonical(s):
        # ExpoFourierFeatures computed in float64 (stable at large args), cast to
        # f32, then the float32 Linear -- matches aria_number_embed in C. torch's
        # all-float32 path is rounding noise at high freq*duration; see dump log.
        t = min((max(s, min_val) - min_val) / (max_val - min_val), 1.0)
        half = 128
        ramp = np.linspace(0.0, 1.0, half)
        freqs = np.exp(ramp * (math.log(10000.0) - math.log(0.5)) + math.log(0.5))
        args = t * freqs * 2 * math.pi
        fourier = np.concatenate([np.cos(args), np.sin(args)]).astype(np.float32)  # [256]
        return (fourier @ W.T + B).astype(np.float32)  # [768]

    d = os.path.join(out_dir, "number_cond")
    os.makedirs(d, exist_ok=True)
    worst_torch_dev = 0.0
    with torch.no_grad():
        for i, s in enumerate(SECONDS_VALUES):
            ref = canonical(s)
            save_atns(os.path.join(d, f"seconds_{i}.atns"), ref)
            t_emb, _mask = nc([s], device="cpu")  # torch float32 path, [1,1,768]
            dev = float(np.max(np.abs(t_emb.squeeze().float().cpu().numpy() - ref)))
            worst_torch_dev = max(worst_torch_dev, dev)
    print(f"dumped {len(SECONDS_VALUES)} seconds_total embeddings to {d} "
          f"(min={min_val} max={max_val} type={fft} dim={cond_dim}); "
          f"canonical(f64)-vs-torch(f32) worst dev = {worst_torch_dev:.3e}")


def dump_ops(out_dir):
    """Op-level references with seeded synthetic inputs, driving the real
    stable_audio_tools ops. Inputs are dumped too, so C consumes identical data."""
    import torch.nn.functional as F
    from stable_audio_tools.models.transformer import (
        RotaryEmbedding, apply_rotary_pos_emb, FeedForward,
    )

    d = os.path.join(out_dir, "ops")
    os.makedirs(d, exist_ok=True)

    def save(name, t):
        save_atns(os.path.join(d, name), t.detach().float().cpu().numpy())

    # --- RoPE (rotate-half) : H=2, N=6, D=64, rot_dim=64 ---
    torch.manual_seed(0)
    H, N, D, rot = 2, 6, 64, 64
    q = torch.randn(H, N, D)
    rotary = RotaryEmbedding(rot)
    freqs, _ = rotary.forward_from_seq_len(N)
    q_rot = apply_rotary_pos_emb(q, freqs)
    save("rope_q_in.atns", q)
    save("rope_q_out.atns", q_rot)

    # --- attention (SDPA) : H=2, Nq=4, Nk=6, D=8 ---
    torch.manual_seed(1)
    H, Nq, Nk, Dh = 2, 4, 6, 8
    qa = torch.randn(H, Nq, Dh)
    ka = torch.randn(H, Nk, Dh)
    va = torch.randn(H, Nk, Dh)
    out = F.scaled_dot_product_attention(qa, ka, va, is_causal=False)
    save("attn_q.atns", qa)
    save("attn_k.atns", ka)
    save("attn_v.atns", va)
    save("attn_out.atns", out)

    # --- GLU/SwiGLU FeedForward : dim=64, mult=4 -> inner=256 ---
    torch.manual_seed(2)
    dim, mult, Nf = 64, 4, 5
    ff = FeedForward(dim, mult=mult, glu=True, zero_init_output=False, no_bias=False).eval()
    x = torch.randn(1, Nf, dim)
    with torch.no_grad():
        y = ff(x)
    save("ff_x.atns", x.squeeze(0))
    save("ff_y.atns", y.squeeze(0))
    save("ff_Win.atns", ff.ff[0].proj.weight)   # [2*inner, dim]
    save("ff_bin.atns", ff.ff[0].proj.bias)     # [2*inner]
    save("ff_Wout.atns", ff.ff[2].weight)       # [dim, inner]
    save("ff_bout.atns", ff.ff[2].bias)         # [dim]
    print(f"dumped op references (rope/attn/ff) to {d}")


def dump_dit_block(model_dir, out_dir):
    """Load DiT block 0's real weights into a TransformerBlock and dump a
    seeded forward (inputs + output). C loads the same weights from the model
    and must reproduce the output -- validates both name mapping and the block."""
    from stable_audio_tools.models.transformer import TransformerBlock, RotaryEmbedding

    dim, num_heads, head_dim = 1024, 16, 64
    block = TransformerBlock(
        dim, dim_heads=head_dim, cross_attend=True, dim_context=dim,
        global_cond_dim=dim, local_add_cond_dim=257,
        norm_type="rms_norm", norm_kwargs={"force_fp32": True},
        attn_kwargs={"qk_norm": "rms", "differential": False},
        ff_kwargs={"mult": 4.0},
    ).eval()

    prefix = "model.model.transformer.layers.0."
    sd = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = f.get_tensor(k)
    missing, unexpected = block.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "rope" not in m]  # rope buffer is recomputed
    if missing:
        print(f"  WARNING block0 missing keys: {missing}")

    torch.manual_seed(3)
    S, Sc = 40, 257
    x = torch.randn(1, S, dim)
    context = torch.randn(1, Sc, dim)
    global_cond = torch.randn(1, 6 * dim)  # already passed through global_cond_embedder
    rope = RotaryEmbedding(max(head_dim // 2, 32)).forward_from_seq_len(S)
    with torch.no_grad():
        out = block(x, context=context, global_cond=global_cond,
                    rotary_pos_emb=rope, local_add_cond=None)

    d = os.path.join(out_dir, "dit")
    os.makedirs(d, exist_ok=True)
    save_atns(os.path.join(d, "block0_x.atns"), x.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "block0_context.atns"), context.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "block0_global.atns"), global_cond.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "block0_out.atns"), out.squeeze(0).float().cpu().numpy())
    print(f"dumped DiT block-0 reference to {d} (S={S}, Sc={Sc})")


def dump_inpaint(model_dir, out_dir):
    """Inpaint conditioning + DiT velocity reference. Builds the local-additive
    conditioning exactly as inference/generation.py does (audio-space mask ->
    nearest-interp to latent -> masked_input = latent*mask -> concat
    [inpaint_mask, inpaint_masked_input] = [257,T]) and runs the real DiT
    _forward with it. C must reproduce the mask interp, the local cond, and the
    velocity. Validates E7.2 (mask build) + E7.3 (live local_add_cond)."""
    from stable_audio_tools.models.dit import DiffusionTransformer
    from torch.nn.functional import interpolate

    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    dcfg = cfg["model"]["diffusion"]["config"]
    objective = cfg["model"]["diffusion"].get("diffusion_objective", "rf_denoiser")
    dit = DiffusionTransformer(diffusion_objective=objective, **dcfg).eval()
    prefix = "model.model."
    sd = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = f.get_tensor(k)
    missing, _ = dit.load_state_dict(sd, strict=False)
    missing = [x for x in missing if "rope" not in x and "inv_freq" not in x]
    if missing:
        print(f"  WARNING inpaint dit missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")

    torch.manual_seed(9)
    T, n_cond = 16, 257
    audio_len = T * 4096
    x = torch.randn(1, dcfg["io_channels"], T)
    t = torch.tensor([0.3])
    cross = torch.randn(1, n_cond, dcfg["cond_token_dim"])
    glob = torch.randn(1, dcfg["global_cond_dim"])
    init_latent = torch.randn(1, dcfg["io_channels"], T)  # stand-in for encode(init_audio)

    # audio-space inpaint mask: keep everything except a [start,end) region (=0)
    start_s, end_s = 1.2, 2.7
    sr = cfg["sample_rate"]
    mask_audio = torch.ones(1, audio_len)
    mask_audio[:, int(start_s * sr):int(end_s * sr)] = 0.0
    mask_lat = interpolate(mask_audio.unsqueeze(1), size=T, mode='nearest').squeeze(1)  # [1,T]
    masked_input = init_latent * mask_lat.unsqueeze(1)                                  # [1,256,T]
    local = torch.cat([mask_lat.unsqueeze(1), masked_input], dim=1)                     # [1,257,T]

    with torch.no_grad():
        out = dit._forward(x, t, cross_attn_cond=cross, global_embed=glob, local_add_cond=local)

    d = os.path.join(out_dir, "inpaint")
    os.makedirs(d, exist_ok=True)
    save_atns(os.path.join(d, "x.atns"), x.squeeze(0).float().cpu().numpy())          # [256,T]
    save_atns(os.path.join(d, "cross.atns"), cross.squeeze(0).float().cpu().numpy())  # [257,768]
    save_atns(os.path.join(d, "global.atns"), glob.squeeze(0).float().cpu().numpy())  # [768]
    save_atns(os.path.join(d, "t.atns"), np.array([0.3], dtype=np.float32))
    save_atns(os.path.join(d, "init_latent.atns"), init_latent.squeeze(0).float().cpu().numpy())  # [256,T]
    save_atns(os.path.join(d, "mask_audio.atns"), mask_audio.squeeze(0).float().cpu().numpy())     # [audio_len]
    save_atns(os.path.join(d, "mask_lat.atns"), mask_lat.squeeze(0).float().cpu().numpy())          # [T]
    save_atns(os.path.join(d, "local.atns"), local.squeeze(0).float().cpu().numpy())                # [257,T]
    save_atns(os.path.join(d, "out.atns"), out.squeeze(0).float().cpu().numpy())                    # [256,T]
    print(f"dumped inpaint cond + DiT velocity to {d} (T={T}, mask {start_s}-{end_s}s)")


def _ref_shifts(model_dir):
    """The schedule shifts as the reference builds them from model_config.json: the
    wrapper's sampling shift (what generate() samples with -- aria `--schedule ref`) and
    its training shift (`--schedule full`, the stable-audio-3 TensorRT runtime's), plus
    aria's legacy LogSNRShift(rate=1) (`--schedule len`). Built by the real
    ConditionedDiffusionModelWrapper so the default is never re-derived by hand."""
    try:
        from stable_audio_3.models.diffusion import ConditionedDiffusionModelWrapper as W
        from stable_audio_3.inference.sampling import build_schedule
        from stable_audio_3.inference.distribution_shift import LogSNRShift
    except ImportError:
        from stable_audio_tools.models.diffusion import ConditionedDiffusionModelWrapper as W
        from stable_audio_tools.inference.sampling import build_schedule, LogSNRShift
    dif = json.load(open(os.path.join(model_dir, "model_config.json")))["model"]["diffusion"]
    w = W(torch.nn.Identity(), torch.nn.Identity(), io_channels=256, sample_rate=44100,
          min_input_length=1, distribution_shift_options=dif.get("distribution_shift_options"),
          sampling_distribution_shift_options=dif.get("sampling_distribution_shift_options"))
    shifts = {"ref": w.sampling_dist_shift, "full": w.dist_shift,
              "len": LogSNRShift(anchor_length=2000, anchor_logsnr=-6.2, rate=1.0, logsnr_end=2.0)}
    return shifts, build_schedule


def dump_schedule(model_dir, out_dir):
    """Schedule references (build_schedule + each shift) at reference effective lengths."""
    shifts, build_schedule = _ref_shifts(model_dir)
    d = os.path.join(out_dir, "sampler")
    os.makedirs(d, exist_ok=True)
    for name, shift in shifts.items():
        if shift is None:
            continue
        for steps, seq_len in [(8, 108), (6, 108), (8, 323), (8, 1293), (16, 512)]:
            sched = build_schedule(steps=steps, sigma_max=1.0, dist_shift=shift,
                                   effective_seq_len=seq_len, include_endpoint=True)
            save_atns(os.path.join(d, f"sched_{name}_s{steps}_L{seq_len}.atns"),
                      sched.float().cpu().numpy())
    print(f"dumped ref/full/len schedules to {d}")


def dump_dit_block_diff(out_dir):
    """Differential DiT block (medium's attention variant) with random weights, so
    the C differential path can be parity-checked without the medium model. Small
    dims (head_dim 64 like medium); both self- and cross-attention are differential:
    out = attn(q,k,v) - attn(q_diff,k_diff,v)."""
    from stable_audio_tools.models.transformer import TransformerBlock, RotaryEmbedding

    dim, num_heads, head_dim = 256, 4, 64
    block = TransformerBlock(
        dim, dim_heads=head_dim, cross_attend=True, dim_context=dim,
        global_cond_dim=dim, norm_type="rms_norm", norm_kwargs={"force_fp32": True},
        attn_kwargs={"qk_norm": "rms", "differential": True}, ff_kwargs={"mult": 4.0},
    ).eval()
    # the block zero-inits branch outputs (to_out, ff.2) -> randomize so the test is non-degenerate
    torch.manual_seed(31)
    with torch.no_grad():
        for name, p in block.named_parameters():
            if name.endswith("gamma"):
                p.copy_(1.0 + 0.02 * torch.randn_like(p))      # RMSNorm gains ~ 1
            elif "to_scale_shift_gate" in name:
                p.copy_(0.02 * torch.randn_like(p))
            elif name.endswith("bias"):
                p.copy_(0.01 * torch.randn_like(p))
            else:
                p.copy_(0.04 * torch.randn_like(p))

    torch.manual_seed(32)
    S, Sc = 12, 7
    x = torch.randn(1, S, dim) * 0.5
    context = torch.randn(1, Sc, dim) * 0.5
    global_cond = torch.randn(1, 6 * dim) * 0.1
    rope = RotaryEmbedding(max(head_dim // 2, 32)).forward_from_seq_len(S)
    with torch.no_grad():
        out = block(x, context=context, global_cond=global_cond, rotary_pos_emb=rope, local_add_cond=None)

    d = os.path.join(out_dir, "dit_diff")
    os.makedirs(d, exist_ok=True)
    save_atns(os.path.join(d, "x.atns"), x.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "context.atns"), context.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "global.atns"), global_cond.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "out.atns"), out.squeeze(0).float().cpu().numpy())
    sd = block.state_dict()
    wmap = {
        "pre_norm": "pre_norm.gamma", "cross_norm": "cross_attend_norm.gamma", "ff_norm": "ff_norm.gamma",
        "sa_to_qkv": "self_attn.to_qkv.weight", "sa_q_norm": "self_attn.q_norm.gamma",
        "sa_k_norm": "self_attn.k_norm.gamma", "sa_to_out": "self_attn.to_out.weight",
        "ca_to_q": "cross_attn.to_q.weight", "ca_to_kv": "cross_attn.to_kv.weight",
        "ca_q_norm": "cross_attn.q_norm.gamma", "ca_k_norm": "cross_attn.k_norm.gamma",
        "ca_to_out": "cross_attn.to_out.weight", "ssg": "to_scale_shift_gate",
        "ff_in_w": "ff.ff.0.proj.weight", "ff_in_b": "ff.ff.0.proj.bias",
        "ff_out_w": "ff.ff.2.weight", "ff_out_b": "ff.ff.2.bias",
    }
    for fn, key in wmap.items():
        save_atns(os.path.join(d, fn + ".atns"), sd[key].float().cpu().numpy())
    print(f"dumped differential DiT block reference to {d} (dim={dim}, S={S}, Sc={Sc})")


def dump_dit_full(model_dir, out_dir):
    """Full DiT denoiser_forward: instantiate the real DiffusionTransformer from
    the config, load model.model.* weights, run _forward on seeded synthetic
    conditioning (bypasses T5Gemma). C must reproduce the velocity output."""
    from stable_audio_tools.models.dit import DiffusionTransformer

    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    dcfg = cfg["model"]["diffusion"]["config"]
    objective = cfg["model"]["diffusion"].get("diffusion_objective", "rf_denoiser")
    dit = DiffusionTransformer(diffusion_objective=objective, **dcfg).eval()

    prefix = "model.model."
    sd = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for k in f.keys():
            if k.startswith(prefix):
                sd[k[len(prefix):]] = f.get_tensor(k)
    missing, unexpected = dit.load_state_dict(sd, strict=False)
    missing = [x for x in missing if "rope" not in x and "inv_freq" not in x]
    if missing:
        print(f"  WARNING dit missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")

    torch.manual_seed(5)
    T, n_cond = 16, 257
    x = torch.randn(1, dcfg["io_channels"], T)
    t = torch.tensor([0.3])
    cross = torch.randn(1, n_cond, dcfg["cond_token_dim"])
    glob = torch.randn(1, dcfg["global_cond_dim"])
    with torch.no_grad():
        out = dit._forward(x, t, cross_attn_cond=cross, global_embed=glob)

    d = os.path.join(out_dir, "dit")
    os.makedirs(d, exist_ok=True)
    save_atns(os.path.join(d, "full_x.atns"), x.squeeze(0).float().cpu().numpy())       # [C,T]
    save_atns(os.path.join(d, "full_cross.atns"), cross.squeeze(0).float().cpu().numpy())  # [257,768]
    save_atns(os.path.join(d, "full_global.atns"), glob.squeeze(0).float().cpu().numpy())  # [768]
    save_atns(os.path.join(d, "full_out.atns"), out.squeeze(0).float().cpu().numpy())     # [C,T]
    save_atns(os.path.join(d, "full_t.atns"), np.array([0.3], dtype=np.float32))
    print(f"dumped full DiT denoiser reference to {d} (T={T})")


def _load_decoder_ae(model_dir):
    from safetensors.torch import load_file
    from stable_audio_tools.models.autoencoders import create_autoencoder_from_config
    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    ae_cfg = cfg["model"]["pretransform"]["config"]
    ae = create_autoencoder_from_config(
        {"model": ae_cfg, "sample_rate": cfg["sample_rate"]}).eval()
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    prefix = "pretransform.model."
    ae_sd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
    missing, unexpected = ae.load_state_dict(ae_sd, strict=False)
    missing = [x for x in missing if "encoder" not in x and "inv_freq" not in x]
    if missing:
        print(f"  WARNING decoder missing keys: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    # neutralize noise so the reference is deterministic (matches the C path)
    for mod in ae.modules():
        if hasattr(mod, "noise_regularize"):
            mod.noise_regularize = False
        if hasattr(mod, "mask_noise"):
            mod.mask_noise = 0.0
    return ae


def dump_taae_medium_block(out_dir):
    """Synthetic medium-decoder resampling block (random weights): differential
    self-attention + DyT norms + SLIDING-WINDOW (banded) attention + a SiLU and a
    Sin (sinusoidal FF) variant. Validates the C medium decoder block kernel on the
    dev box (no medium model needed). Window [17,17]."""
    from stable_audio_tools.models.transformer import TransformerBlock

    dim, head_dim = 512, 64          # 8 heads
    W = [17, 17]                     # sliding window (medium: stride16 -> [17,17])
    d = os.path.join(out_dir, "taae_med")
    os.makedirs(d, exist_ok=True)
    for tag, sinus in [("silu", False), ("sin", True)]:
        blk = TransformerBlock(dim, dim_heads=head_dim, cross_attend=False, norm_type="dyt",
                               attn_kwargs={"qk_norm": "dyt", "qk_norm_eps": 1e-3, "differential": True},
                               ff_kwargs={"mult": 3, "no_bias": False, "sinusoidal": sinus},
                               norm_kwargs={"eps": 1e-3}, add_rope=True).eval()
        torch.manual_seed(41)
        with torch.no_grad():
            for name, p in blk.named_parameters():
                if name.endswith("alpha"):   p.copy_(0.5 + 0.1 * torch.randn_like(p))
                elif name.endswith("gamma"): p.copy_(1.0 + 0.05 * torch.randn_like(p))
                elif name.endswith("beta"):  p.copy_(0.02 * torch.randn_like(p))
                elif name.endswith("bias"):  p.copy_(0.01 * torch.randn_like(p))
                else:                         p.copy_(0.04 * torch.randn_like(p))
        torch.manual_seed(42)
        N = 51                       # 3 * 17
        x = torch.randn(1, N, dim) * 0.5
        with torch.no_grad():
            out = blk(x, self_attention_flash_sliding_window=W)
        save_atns(os.path.join(d, f"{tag}_x.atns"), x.squeeze(0).float().cpu().numpy())
        save_atns(os.path.join(d, f"{tag}_out.atns"), out.squeeze(0).float().cpu().numpy())
        sd = blk.state_dict()
        wmap = {
            "pre_alpha": "pre_norm.alpha", "pre_gamma": "pre_norm.gamma", "pre_beta": "pre_norm.beta",
            "to_qkv": "self_attn.to_qkv.weight",
            "qn_alpha": "self_attn.q_norm.alpha", "qn_gamma": "self_attn.q_norm.gamma", "qn_beta": "self_attn.q_norm.beta",
            "kn_alpha": "self_attn.k_norm.alpha", "kn_gamma": "self_attn.k_norm.gamma", "kn_beta": "self_attn.k_norm.beta",
            "to_out": "self_attn.to_out.weight",
            "ff_alpha": "ff_norm.alpha", "ff_gamma": "ff_norm.gamma", "ff_beta": "ff_norm.beta",
            "ff_in_w": "ff.ff.0.proj.weight", "ff_in_b": "ff.ff.0.proj.bias",
            "ff_out_w": "ff.ff.2.weight", "ff_out_b": "ff.ff.2.bias",
        }
        for fn, key in wmap.items():
            v = sd[key]
            save_atns(os.path.join(d, f"{tag}_{fn}.atns"),
                      (v.reshape(()) if v.numel() == 1 else v).float().cpu().numpy())
    print(f"dumped synthetic medium taae blocks (silu+sin, differential, sliding-window {W}) to {d}")


def dump_decoder(model_dir, out_dir):
    """Staged taae_v2 decode reference (softnorm -> SAMEDecoder -> unpatch)."""
    ae = _load_decoder_ae(model_dir)
    d = os.path.join(out_dir, "dec")
    os.makedirs(d, exist_ok=True)

    orig_randn_like = torch.randn_like
    torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)  # belt-and-suspenders determinism
    try:
        torch.manual_seed(0)
        T = 16
        latent = torch.randn(1, 256, T)
        with torch.no_grad():
            z = ae.bottleneck.decode(latent)        # [1,256,T]
            dec = ae.decoder(z)                     # [1,512,16T]
            audio = ae.pretransform.decode(dec)     # [1,2,4096T]
    finally:
        torch.randn_like = orig_randn_like

    save_atns(os.path.join(d, "latent.atns"), latent.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "after_softnorm.atns"), z.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "after_decoder.atns"), dec.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "audio.atns"), audio.squeeze(0).float().cpu().numpy())
    print(f"dumped taae decoder stages to {d} (T={T})")


def dump_encoder(model_dir, out_dir):
    """Staged taae_v2 encode reference (patchify -> SAMEEncoder -> softnorm).
    Mirror of dump_decoder. Two cases: a clean length (T_patch a multiple of 32)
    with staged intermediates, and a short length that exercises the encoder's
    zero-pad-to-multiple-of-32 path (full encode only)."""
    ae = _load_decoder_ae(model_dir)   # loads the full AE, incl. the encoder
    d = os.path.join(out_dir, "enc")
    os.makedirs(d, exist_ok=True)

    orig_randn_like = torch.randn_like
    torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)
    try:
        torch.manual_seed(0)
        audio = torch.randn(1, 2, 16 * 4096)          # T_patch=256 (multiple of 32)
        audio_pad = torch.randn(1, 2, 250 * 256)      # T_patch=250 -> pad to 256
        with torch.no_grad():
            patches = ae.pretransform.encode(audio)
            enc = ae.encoder(patches)
            latent = ae.bottleneck.encode(enc)
            latent_pad = ae.bottleneck.encode(ae.encoder(ae.pretransform.encode(audio_pad)))
    finally:
        torch.randn_like = orig_randn_like

    save_atns(os.path.join(d, "audio.atns"), audio.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "patches.atns"), patches.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "enc.atns"), enc.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "latent.atns"), latent.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "audio_pad.atns"), audio_pad.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "latent_pad.atns"), latent_pad.squeeze(0).float().cpu().numpy())
    print(f"dumped taae encoder stages to {d} "
          f"(patches {tuple(patches.shape)}, latent {tuple(latent.shape)}, "
          f"pad latent {tuple(latent_pad.shape)})")


def dump_taae_block(model_dir, out_dir):
    """Standalone transformers[0] block of the decoder resampling stack (best-effort,
    for localization). Skipped if the block can't be constructed standalone."""
    try:
        from stable_audio_tools.models.transformer import TransformerBlock
        from safetensors.torch import load_file
        blk = TransformerBlock(768, dim_heads=64, cross_attend=False, norm_type="dyt",
                               attn_kwargs={"qk_norm": "dyt", "differential": True},
                               add_rope=True).eval()
        sd = load_file(os.path.join(model_dir, "model.safetensors"))
        prefix = "pretransform.model.decoder.layers.3.transformers.0."
        bsd = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)
               and "inv_freq" not in k}
        blk.load_state_dict(bsd, strict=False)
        torch.manual_seed(7)
        N = 34
        x = torch.randn(1, N, 768)
        with torch.no_grad():
            out = blk(x)
        d = os.path.join(out_dir, "dec")
        os.makedirs(d, exist_ok=True)
        save_atns(os.path.join(d, "block_x.atns"), x.squeeze(0).float().cpu().numpy())
        save_atns(os.path.join(d, "block_out.atns"), out.squeeze(0).float().cpu().numpy())
        print(f"dumped taae block-0 reference to {d} (N={N})")
    except Exception as e:
        print(f"  (skipping standalone taae block dump: {type(e).__name__}: {e})")


def dump_e2e(model_dir, out_dir):
    """End-to-end: init noise -> pingpong(DiT denoiser) -> latent -> decode -> audio,
    with explicit injected per-step noise (matches the C aria_pingpong path)."""
    from stable_audio_tools.models.dit import DiffusionTransformer
    from safetensors.torch import load_file

    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    dcfg = cfg["model"]["diffusion"]["config"]
    objective = cfg["model"]["diffusion"].get("diffusion_objective", "rf_denoiser")
    dit = DiffusionTransformer(diffusion_objective=objective, **dcfg).eval()
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    dit_sd = {k[len("model.model."):]: v for k, v in sd.items() if k.startswith("model.model.")}
    dit.load_state_dict(dit_sd, strict=False)
    ae = _load_decoder_ae(model_dir)

    torch.manual_seed(11)
    T, steps = 16, 8
    # unconditional (zero) conditioning so the rf_denoiser converges toward the
    # data manifold -> an in-distribution latent the decoder handles stably.
    cross = torch.zeros(1, 257, 768)
    glob = torch.zeros(1, 768)
    init_noise = torch.randn(1, 256, T)
    step_noise = torch.randn(steps, 256, T)
    shifts, build_schedule = _ref_shifts(model_dir)   # the shift generate() samples with
    sched = build_schedule(steps=steps, sigma_max=1.0, dist_shift=shifts["ref"], effective_seq_len=T)

    orig = torch.randn_like
    torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)
    try:
        with torch.no_grad():
            x = init_noise.clone()
            for i in range(steps):
                tc = sched[i].item(); tn = sched[i + 1].item()
                v = dit._forward(x, torch.tensor([tc]), cross_attn_cond=cross, global_embed=glob)
                denoised = x - tc * v
                x = (1 - tn) * denoised + tn * step_noise[i:i + 1]
            latent = x
            z = ae.bottleneck.decode(latent)
            dec = ae.decoder(z)
            audio = ae.pretransform.decode(dec)
    finally:
        torch.randn_like = orig

    d = os.path.join(out_dir, "e2e")
    os.makedirs(d, exist_ok=True)
    save_atns(os.path.join(d, "cross.atns"), cross.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "global.atns"), glob.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "init_noise.atns"), init_noise.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "step_noise.atns"), step_noise.float().cpu().numpy())
    save_atns(os.path.join(d, "sched.atns"), sched.float().cpu().numpy())
    save_atns(os.path.join(d, "latent.atns"), latent.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "audio.atns"), audio.squeeze(0).float().cpu().numpy())
    print(f"dumped end-to-end reference to {d} (T={T}, steps={steps})")


def dump_inpaint_e2e(model_dir, out_dir):
    """End-to-end inpaint (continue-style): encode a prepared init clip -> latent,
    build the local-additive cond (keep first half, regenerate second half), then
    pingpong(DiT denoiser with local_add_cond) from pure noise with injected
    per-step noise -> latent -> decode -> audio. Unconditional (zero cross/global)
    like dump_e2e. Validates the full E7.1-E7.4 inpaint chain in C."""
    from stable_audio_tools.models.dit import DiffusionTransformer
    from safetensors.torch import load_file
    from torch.nn.functional import interpolate

    cfg = json.load(open(os.path.join(model_dir, "model_config.json")))
    dcfg = cfg["model"]["diffusion"]["config"]
    objective = cfg["model"]["diffusion"].get("diffusion_objective", "rf_denoiser")
    dit = DiffusionTransformer(diffusion_objective=objective, **dcfg).eval()
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    dit.load_state_dict({k[len("model.model."):]: v for k, v in sd.items()
                         if k.startswith("model.model.")}, strict=False)
    ae = _load_decoder_ae(model_dir)

    torch.manual_seed(13)
    T, steps = 16, 8
    audio_len = T * 4096
    init_audio = torch.randn(1, 2, audio_len) * 0.3        # prepared init clip stand-in
    cross = torch.zeros(1, 257, 768)
    glob = torch.zeros(1, 768)
    init_noise = torch.randn(1, 256, T)
    step_noise = torch.randn(steps, 256, T)
    shifts, build_schedule = _ref_shifts(model_dir)   # the shift generate() samples with
    sched = build_schedule(steps=steps, sigma_max=1.0, dist_shift=shifts["ref"], effective_seq_len=T)

    # continue-style mask: keep [0, audio_len/2) (=1), regenerate the rest (=0)
    mask_audio = torch.ones(1, audio_len)
    mask_audio[:, audio_len // 2:] = 0.0

    orig = torch.randn_like
    torch.randn_like = lambda x, *a, **k: torch.zeros_like(x)
    try:
        with torch.no_grad():
            init_latent = ae.bottleneck.encode(ae.encoder(ae.pretransform.encode(init_audio)))
            mask_lat = interpolate(mask_audio.unsqueeze(1), size=T, mode='nearest').squeeze(1)
            masked_input = init_latent * mask_lat.unsqueeze(1)
            local = torch.cat([mask_lat.unsqueeze(1), masked_input], dim=1)   # [1,257,T]

            x = init_noise.clone()
            for i in range(steps):
                tc = sched[i].item(); tn = sched[i + 1].item()
                v = dit._forward(x, torch.tensor([tc]), cross_attn_cond=cross,
                                 global_embed=glob, local_add_cond=local)
                denoised = x - tc * v
                x = (1 - tn) * denoised + tn * step_noise[i:i + 1]
            latent = x
            audio = ae.pretransform.decode(ae.decoder(ae.bottleneck.decode(latent)))
    finally:
        torch.randn_like = orig

    d = os.path.join(out_dir, "inpaint_e2e")
    os.makedirs(d, exist_ok=True)
    save_atns(os.path.join(d, "init_audio.atns"), init_audio.squeeze(0).float().cpu().numpy())   # [2,audio_len]
    save_atns(os.path.join(d, "init_latent.atns"), init_latent.squeeze(0).float().cpu().numpy())  # [256,T]
    save_atns(os.path.join(d, "mask_audio.atns"), mask_audio.squeeze(0).float().cpu().numpy())    # [audio_len]
    save_atns(os.path.join(d, "cross.atns"), cross.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "global.atns"), glob.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "init_noise.atns"), init_noise.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "step_noise.atns"), step_noise.float().cpu().numpy())
    save_atns(os.path.join(d, "sched.atns"), sched.float().cpu().numpy())
    save_atns(os.path.join(d, "latent.atns"), latent.squeeze(0).float().cpu().numpy())
    save_atns(os.path.join(d, "audio.atns"), audio.squeeze(0).float().cpu().numpy())
    print(f"dumped end-to-end inpaint reference to {d} (T={T}, steps={steps}, keep first half)")


def dump_t5enc(model_dir, out_dir):
    """T5Gemma encoder reference via the real T5GemmaConditioner: dump token ids,
    pre-padding last_hidden, and final (post learned-padding) [256,768]."""
    from stable_audio_tools.models.conditioners import T5GemmaConditioner
    from safetensors.torch import load_file
    nc = T5GemmaConditioner(output_dim=768, max_length=256, padding_mode="learned",
                            repo_id="stabilityai/stable-audio-3-small-music",
                            subfolder="t5gemma-b-b-ul2").eval()
    nc.model = nc.model.float()   # fp32 reference to match the fp32 C runtime (weights are bf16 on disk)
    sd = load_file(os.path.join(model_dir, "model.safetensors"))
    with torch.no_grad():
        nc.padding_embedding.copy_(sd["conditioner.conditioners.prompt.padding_embedding"])

    prompts = ["warm romantic piano", "Amen break 174 BPM", "lofi house loop"]
    d = os.path.join(out_dir, "t5enc")
    os.makedirs(d, exist_ok=True)
    for idx, prompt in enumerate(prompts):
        enc = nc.tokenizer([prompt], truncation=True, max_length=256,
                           padding="max_length", return_tensors="pt")
        ids = enc["input_ids"]; am = enc["attention_mask"]
        with torch.no_grad():
            last_hidden = nc.model(input_ids=ids, attention_mask=am.bool()).last_hidden_state
            cond_final = nc([prompt], device="cpu")[0]   # [1,256,768] post learned-padding
        n_real = int(am.sum().item())
        save_atns(os.path.join(d, f"ids_{idx}.atns"), ids[0].float().cpu().numpy())
        save_atns(os.path.join(d, f"nreal_{idx}.atns"), np.array([n_real], dtype=np.float32))
        save_atns(os.path.join(d, f"last_hidden_{idx}.atns"), last_hidden[0].float().cpu().numpy())
        save_atns(os.path.join(d, f"cond_{idx}.atns"), cond_final[0].float().cpu().numpy())
    print(f"dumped t5gemma encoder reference to {d} ({len(prompts)} prompts)")


if __name__ == "__main__":
    model_dir = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else "parity_dumps"
    os.makedirs(out_dir, exist_ok=True)
    dump_number_cond(model_dir, out_dir)
    dump_ops(out_dir)
    dump_dit_block(model_dir, out_dir)
    dump_dit_block_diff(out_dir)
    dump_dit_full(model_dir, out_dir)
    dump_schedule(model_dir, out_dir)
    dump_taae_block(model_dir, out_dir)
    dump_taae_medium_block(out_dir)
    dump_decoder(model_dir, out_dir)
    dump_encoder(model_dir, out_dir)
    dump_inpaint(model_dir, out_dir)
    dump_inpaint_e2e(model_dir, out_dir)
    dump_e2e(model_dir, out_dir)
    dump_t5enc(model_dir, out_dir)
