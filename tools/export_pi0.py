#!/usr/bin/env python3
"""Export REAL pi0.5 (openpi-comet pi05-b1kpt12-cs32) weights + activations for /pi0.

One BEHAVIOR demo frame (the same episode/frame as the k005b export) is pushed
through the local openpi-comet checkpoint (pi0.5 fine-tuned on BEHAVIOR-1K):

  3 cameras (HEVC, 2026-challenge-demos) -> resize_with_pad 224 -> SigLIP So400m/14
      (27 blocks) -> head 1152->2048 -> 3 x 256 image tokens
  "Task: ..., State: <23 discretised R1 Pro joints>;\nAction: " -> Gemma embed (x sqrt 2048)
  prefix [image | text] -> Gemma-2B stream (18 layers, bidirectional) -> per-layer KV cache
  noise 32x32 -> 10 Euler steps (t = 1.0 -> 0.1, dt = -0.1) of
      {action_in_proj -> 18-layer action-expert stream (width 1024, adaRMSNorm on
       time_mlp(t), gated residuals) attending to [prefix KV | own KV] -> action_out_proj}

The forward is re-implemented in plain torch (float32) so every intermediate is
visible; it follows openpi/models/{pi0,gemma,siglip}.py line by line. Checks:
cached-prefix sampling == one joint (non-cached) forward of both streams, Euler
identity, and (--jax-ref) the official openpi Pi0.sample_actions in JAX on CPU.

Outputs (public/), in the same schema as tools/export_k005b.py (the /pi0 page runs on
the shared VLA engine; config.variant carries pi0.5's structural differences):
pi0-weights.json, pi0-acts.json (every layer, step 0 = t 1), pi0-fm.json (every
Euler step, float16), pi0-check.json (spot values from the live tensors).
"""
import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

CKPT = "/home/frank/workspace/model/openpi_comet/pi05-b1kpt12-cs32"
OPENPI_SRC = "/home/frank/behavior_challenge_ws/behavior_challenge/third_party/openpi-comet/src"
TOKENIZER = os.path.expanduser("~/.cache/openpi/big_vision/paligemma_tokenizer.model")
PYDEPS = "/home/frank/vla-src/pydeps"
JAXENV = "/tmp/jaxenv"
HERE = os.path.dirname(os.path.abspath(__file__))

CAMS = [("base_0_rgb", "head", "observation.rgb.zed_link_camera_0"),
        ("left_wrist_0_rgb", "left_wrist", "observation.rgb.left_realsense_link_camera_0"),
        ("right_wrist_0_rgb", "right_wrist", "observation.rgb.right_realsense_link_camera_0")]
WT = 32       # token window: the last WT prefix tokens
WC = 64       # hidden-channel window [0, WC)
WD = 32       # head-dim window (head_dim 256)
WF = 64       # MLP hidden window
NPART = 8     # one-step particles (seeds) at t = 1
SEED = 0
NSTEP = 10
HOR, ADIM = 32, 32

# openpi-comet b1k_policy.extract_state_from_proprio (R1Pro PROPRIOCEPTION_INDICES)
STATE23 = [("base_qvel", 0, 3), ("trunk_qpos", 53, 57), ("arm_left_qpos", 3, 10),
           ("arm_right_qpos", 28, 35), ("gripper_left_width", 24, 26), ("gripper_right_width", 49, 51)]
# BEHAVIOR R1 Pro native action order (openpi configs/robots/b1k.py: grippers at 14 and 22)
ACTION_DIM_NAMES = (["base_vx", "base_vy", "base_yaw"] + [f"trunk_{i}" for i in range(1, 5)]
                    + [f"L_arm_j{i}" for i in range(1, 8)] + ["L_gripper"]
                    + [f"R_arm_j{i}" for i in range(1, 8)] + ["R_gripper"])


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


G = load_module("export_groot", os.path.join(HERE, "export_groot.py"))
stats, ent, ent16 = G.stats, G.ent, G.ent16


# ------------------------------------------------------------- weights --- #

class Params:
    """Minimal zarr-v2 reader for the orbax/openpi checkpoint (zstd chunks, '.' separator)."""

    def __init__(self, ckpt, device):
        self.root = os.path.join(ckpt, "params")
        self.names = {d[len("params."):-len(".value")]: d for d in os.listdir(self.root) if d.startswith("params.")}
        self.device = device
        self.cache = {}

    def numpy(self, name):
        import itertools
        import numpy as np
        from compression import zstd
        d = os.path.join(self.root, self.names[name])
        za = json.load(open(os.path.join(d, ".zarray")))
        assert za["dtype"] == "<f4" and za["compressor"]["id"] == "zstd" and za["order"] == "C"
        shape, ch = za["shape"], za["chunks"]
        sep = za.get("dimension_separator", ".")
        arr = np.zeros(shape, np.float32)
        grid = [range(math.ceil(s / c)) for s, c in zip(shape, ch)]
        for idx in itertools.product(*grid):
            fn = os.path.join(d, sep.join(map(str, idx)) if idx else "0")
            if not os.path.exists(fn):
                continue
            blk = np.frombuffer(zstd.decompress(open(fn, "rb").read()), "<f4").reshape(ch)
            sl = tuple(slice(i * c, min((i + 1) * c, s)) for i, c, s in zip(idx, ch, shape))
            arr[sl] = blk[tuple(slice(0, x.stop - x.start) for x in sl)]
        return arr

    def __call__(self, name, device=None):
        import torch
        if name not in self.cache:
            self.cache[name] = torch.from_numpy(self.numpy(name)).to(device or self.device)
        return self.cache[name]

    def size(self, prefix=""):
        tot = 0
        for n, d in self.names.items():
            if n.startswith(prefix):
                tot += math.prod(json.load(open(os.path.join(self.root, d, ".zarray")))["shape"])
        return tot


# --------------------------------------------------------------- input --- #

def resize_with_pad(img, size=224):
    """openpi_client.image_tools.resize_with_pad (PIL bilinear, centred black pad)."""
    import numpy as np
    from PIL import Image
    im = Image.fromarray(img)
    w, h = im.size
    ratio = max(w / size, h / size)
    rh, rw = int(h / ratio), int(w / ratio)
    r = im.resize((rw, rh), resample=Image.BILINEAR)
    z = Image.new(r.mode, (size, size), 0)
    z.paste(r, (max(0, int((size - rw) / 2)), max(0, int((size - rh) / 2))))
    return np.asarray(z)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=None, help="default: the k005b export's episode")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "public"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--jax-ref", action="store_true", help="also run the official openpi Pi0 (JAX, CPU)")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    sys.path.insert(0, PYDEPS)

    import numpy as np
    import sentencepiece
    import torch
    import torch.nn.functional as F
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    t0 = time.time()

    if args.episode is None or args.frame is None:
        s = json.load(open(os.path.join(out_dir, "k005b-acts.json")))["config"]["sample"]
        args.episode = s["episode"] if args.episode is None else args.episode
        args.frame = s["frame"] if args.frame is None else args.frame

    # ---------------- input (b1k policy + pi05 model transforms) ---------------- #
    G.CAMS = [(v, k) for _, v, k in CAMS]
    fr = G.load_frame(args.episode, args.frame)
    prompt = fr["task"]["task"]
    proprio = fr["state"]
    state23 = np.concatenate([proprio[a:b] if not n.startswith("gripper") else proprio[a:b].sum(keepdims=True)
                              for n, a, b in STATE23]).astype(np.float32)
    ns = json.load(open(os.path.join(CKPT, "assets/behavior-1k/2025-challenge-demos/norm_stats.json")))["norm_stats"]
    sq01, sq99 = np.asarray(ns["state"]["q01"], np.float32), np.asarray(ns["state"]["q99"], np.float32)
    aq01, aq99 = np.asarray(ns["actions"]["q01"], np.float32), np.asarray(ns["actions"]["q99"], np.float32)
    state_n = (state23 - sq01[:23]) / (sq99[:23] - sq01[:23] + 1e-6) * 2.0 - 1.0
    disc = np.digitize(state_n, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    cleaned = prompt.strip().replace("_", " ").replace("\n", " ")
    full_prompt = f"Task: {cleaned}, State: {' '.join(map(str, disc))};\nAction: "
    sp = sentencepiece.SentencePieceProcessor(model_file=TOKENIZER)
    ids = sp.encode(full_prompt, add_bos=True)
    T = len(ids)
    assert T <= 200, "prompt longer than max_token_len"
    imgs224 = {name: resize_with_pad(fr["raw"][view]) for name, view, _ in CAMS}
    print(f"[sample] episode={args.episode} frame={args.frame} prompt={full_prompt!r} T={T}", flush=True)

    Pm = Params(CKPT, dev)
    P = "PaliGemma."

    # ---------------- SigLIP ---------------- #
    def ln(x, pre, l=None):
        s, b = Pm(pre + ".scale"), Pm(pre + ".bias")
        if l is not None:
            s, b = s[l], b[l]
        return F.layer_norm(x, x.shape[-1:], s, b, 1e-6)

    VB = P + "img.Transformer.encoderblock."
    NV, VW, VH = 27, 1152, 16
    VD = VW // VH

    def siglip(img_u8):
        x = torch.from_numpy(img_u8).to(dev).float() / 255.0 * 2.0 - 1.0          # [224,224,3]
        k = Pm(P + "img.embedding.kernel").permute(3, 2, 0, 1)                    # HWIO -> OIHW
        x = F.conv2d(x.permute(2, 0, 1)[None], k, Pm(P + "img.embedding.bias"), stride=14)
        x = x[0].flatten(1).T                                                      # [256, 1152] row-major
        x = x + Pm(P + "img.pos_embedding")[0]
        for l in range(NV):
            y = ln(x, VB + "LayerNorm_0", l)
            A = VB + "MultiHeadDotProductAttention_0."
            q = torch.einsum("td,dnh->tnh", y, Pm(A + "query.kernel")[l]) + Pm(A + "query.bias")[l]
            kk = torch.einsum("td,dnh->tnh", y, Pm(A + "key.kernel")[l]) + Pm(A + "key.bias")[l]
            v = torch.einsum("td,dnh->tnh", y, Pm(A + "value.kernel")[l]) + Pm(A + "value.bias")[l]
            pr = (torch.einsum("tnh,snh->nts", q / math.sqrt(VD), kk)).softmax(-1)
            o = torch.einsum("nts,snh->tnh", pr, v)
            x = x + torch.einsum("tnh,nhd->td", o, Pm(A + "out.kernel")[l]) + Pm(A + "out.bias")[l]
            y = ln(x, VB + "LayerNorm_1", l)
            M = VB + "MlpBlock_0."
            y = F.gelu(y @ Pm(M + "Dense_0.kernel")[l] + Pm(M + "Dense_0.bias")[l], approximate="tanh")
            x = x + y @ Pm(M + "Dense_1.kernel")[l] + Pm(M + "Dense_1.bias")[l]
        enc = ln(x, P + "img.Transformer.encoder_norm")
        return enc @ Pm(P + "img.head.kernel") + Pm(P + "img.head.bias")          # [256, 2048]

    with torch.no_grad():
        img_tok = torch.cat([siglip(imgs224[name]) for name, _, _ in CAMS], 0)                                            # [768, 2048]
    print(f"[siglip] {tuple(img_tok.shape)} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- Gemma prefix + expert ---------------- #
    NL, NH, HD = 18, 8, 256
    DP, DE = 2048, 1024
    LY = P + "llm.layers."

    emb_table = Params(CKPT, "cpu")(P + "llm.embedder.input_embedding")
    txt_emb = (emb_table[torch.tensor(ids)] * math.sqrt(DP)).to(dev)              # [T, 2048]
    prefix0 = torch.cat([img_tok, txt_emb], 0)
    Np = prefix0.shape[0]
    NIMG = img_tok.shape[0]

    def rms(x, scale=None, dense=None, cond=None):
        n = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        if dense is None:
            return n * (1 + scale), None, None
        k, b = dense
        sc, sh, g = (cond @ k + b).chunk(3)
        return n * (1 + sc) + sh, g, (sc, sh, g)

    def rope(x, pos):                                                              # x [T, N, H]
        half = x.shape[-1] // 2
        ts = 10000.0 ** ((2.0 / x.shape[-1]) * torch.arange(half, dtype=torch.float32, device=dev))
        rad = (pos.float()[:, None] / ts[None])[:, None, :]
        x1, x2 = x[..., :half], x[..., half:]
        return torch.cat([x1 * rad.cos() - x2 * rad.sin(), x2 * rad.cos() + x1 * rad.sin()], -1)

    def W(name, l, sfx):
        return Pm(LY + name.replace("#", sfx))[l]

    def qkv(xn, l, sfx, pos):
        q = torch.einsum("td,ndh->tnh", xn, W("attn.q_einsum#.w", l, sfx))
        kv = W("attn.kv_einsum#.w", l, sfx)                                        # [2, 1, D, 256]
        k = torch.einsum("td,kdh->tkh", xn, kv[0])
        v = torch.einsum("td,kdh->tkh", xn, kv[1])
        return rope(q, pos) * HD ** -0.5, rope(k, pos), v                          # k/v [T, 1, 256]

    def attend(q, k, v, mask):                                                     # mask [Tq, Tk] bool
        lg = torch.einsum("tnh,skh->nts", q, k)
        lg = torch.where(mask[None], lg, torch.full_like(lg, -2.3819763e38))
        pr = lg.softmax(-1)
        return torch.einsum("nts,skh->tnh", pr, v), pr

    def mlp(xn, l, sfx, keep=None):
        g = W("mlp#.gating_einsum", l, sfx)
        h = F.gelu(xn @ g[0], approximate="tanh") * (xn @ g[1])
        if keep is not None:
            keep["mlp_h"] = h
        return h @ W("mlp#.linear", l, sfx)

    def ada(l, kind):
        return (W(f"{kind}_1.Dense_0.kernel", l, ""), W(f"{kind}_1.Dense_0.bias", l, ""))

    def time_cond(t):
        frac = torch.linspace(0.0, 1.0, DE // 2, device=dev)
        period = 4e-3 * (4.0 / 4e-3) ** frac
        si = t * (1.0 / period * 2 * math.pi)
        pe = torch.cat([si.sin(), si.cos()])
        h = F.silu(pe @ Pm("time_mlp_in.kernel") + Pm("time_mlp_in.bias"))
        return pe, h, F.silu(h @ Pm("time_mlp_out.kernel") + Pm("time_mlp_out.bias"))

    pos_p = torch.arange(Np, device=dev)
    pos_s = Np + torch.arange(HOR, device=dev)

    # the shown token window: the last WT prefix tokens (the tail of the text prompt: the
    # discretised state tokens and "Action: ")
    assert T >= WT, "prompt shorter than the token window"
    t0w = Np - WT
    tw = slice(t0w, Np)
    pieces = [sp.id_to_piece(i) for i in ids]
    state_start = next(i for i in range(T) if "".join(pieces[:i + 1]).endswith("State:"))
    state_end = next(i for i in range(state_start + 1, T) if ";" in pieces[i])
    # prefix key groups for the attention-mass split [state tokens | image | other text]
    col_state = torch.zeros(Np, dtype=torch.bool, device=dev)
    col_state[NIMG + state_start + 1:NIMG + state_end] = True
    col_img = torch.zeros(Np, dtype=torch.bool, device=dev)
    col_img[:NIMG] = True
    col_txt = ~(col_state | col_img)
    Wx, A = {}, {}

    def put(D, name, window, full, offset, desc):
        D[name] = ent(window, full, offset, desc)

    def norm_agg(x):  # [T, C] -> [T, 2]: mean(x^2) over the FULL C | rsqrt(mean+eps)
        ms = x.pow(2).mean(-1)
        return torch.stack([ms, torch.rsqrt(ms + 1e-6)], -1), ms

    def heads_first(x):  # [T, N, H] -> [N, T, H]
        return x.permute(1, 0, 2)

    with torch.no_grad():
        # prefix pass (fills the KV cache); prefix mask is all-True after trimming text padding
        mp = torch.ones(Np, Np, dtype=torch.bool, device=dev)
        x = prefix0
        Kc, Vc = [], []
        for l in range(NL):
            x_in = x
            xn, _, _ = rms(x, W("pre_attention_norm.scale", l, ""))
            q, k, v = qkv(xn, l, "", pos_p)
            enc, pr = attend(q, k, v, mp)
            ao = torch.einsum("tnh,nhd->td", enc, W("attn.attn_vec_einsum#.w", l, ""))
            r1 = x + ao
            xn2, _, _ = rms(r1, W("pre_ffw_norm.scale", l, ""))
            gg = W("mlp#.gating_einsum", l, "")
            gate, up = xn2 @ gg[0], xn2 @ gg[1]
            fo = mlp(xn2, l, "")
            x = r1 + fo
            Kc.append(k)
            Vc.append(v)

            # ---- layer l, k005b schema: window rows tw, channel / head-dim / FF windows ----
            Ln = f"L{l}"
            put(A, f"{Ln}.x_in", x_in[tw, :WC], x_in, [t0w, 0], f"layer {l} residual stream in [Np, 2048]")
            agg, ms = norm_agg(x_in)
            put(A, f"{Ln}.ln1_agg", agg[tw], ms, [t0w, 0], "RMSNorm agg over the full 2048: mean(x^2) | rsqrt(mean+1e-6)")
            put(A, f"{Ln}.ln1_out", xn[tw, :WC], xn, [t0w, 0], f"layer {l} pre_attention_norm out = RMS(x)(1+scale)")
            qu = q * math.sqrt(HD)                                                 # q after RoPE, before 1/sqrt(d)
            put(A, f"{Ln}.q", heads_first(qu[tw, :, :WD]), heads_first(qu), [0, t0w, 0], f"layer {l} Q after RoPE [head, token, dim]")
            put(A, f"{Ln}.k", heads_first(k[tw, :, :WD]), heads_first(k), [0, t0w, 0], f"layer {l} K after RoPE (1 shared kv head)")
            put(A, f"{Ln}.v", heads_first(v[tw, :, :WD]), heads_first(v), [0, t0w, 0], f"layer {l} V (1 shared kv head)")
            s = torch.einsum("tnh,skh->nts", q[tw], k)                            # [H, Wt, Np] = q.k / 16
            mx = s.max(-1).values
            inv = 1.0 / torch.exp(s - mx[..., None]).sum(-1)
            put(A, f"{Ln}.attn_scores", s[:, :, tw], s, [0, t0w, t0w], f"layer {l} q.k/sqrt(256) (bidirectional prefix: no mask)")
            put(A, f"{Ln}.attn_agg", torch.stack([mx, inv], -1), mx, [0, t0w, 0], f"softmax agg over all {Np} keys: max | 1/sum exp(s-max)")
            put(A, f"{Ln}.attn_probs", pr[:, tw, tw], pr[:, tw], [0, t0w, t0w], f"layer {l} attention probs (rows normalised over all {Np} keys)")
            put(A, f"{Ln}.attn_out", heads_first(enc[tw, :, :WD]), heads_first(enc), [0, t0w, 0], f"layer {l} per-head sum p.V")
            for nm, tns, d in (("o_out", ao, "attn_vec_einsum out"), ("attn_resid", r1, "x + attn (plain residual)"),
                               ("ln2_out", xn2, "pre_ffw_norm out"), ("mlp_out", fo, "GeGLU MLP out (linear)"),
                               ("mlp_resid", x, "x + MLP (layer out)")):
                put(A, f"{Ln}.{nm}", tns[tw, :WC], tns, [t0w, 0], f"layer {l} {d}")
            agg, ms = norm_agg(r1)
            put(A, f"{Ln}.ln2_agg", agg[tw], ms, [t0w, 0], "RMSNorm agg over the full 2048: mean(x^2) | rsqrt(mean+1e-6)")
            hh = F.gelu(gate, approximate="tanh") * up
            for nm, tns, d in (("mlp_gate", gate, "gating_einsum[0] out"), ("mlp_up", up, "gating_einsum[1] out"),
                               ("mlp_act", hh, "gelu_tanh(gate) * up")):
                put(A, f"{Ln}.{nm}", tns[tw, :WF], tns, [t0w, 0], f"layer {l} {d} [Np, 16384]")
        prefix_final, _, _ = rms(x, Pm(P + "llm.final_norm.scale"))                # not used by the expert
        print(f"[prefix] Np={Np} ({time.time() - t0:.0f}s)", flush=True)

        def suffix(x_t, t):
            pe, th, cond = time_cond(torch.tensor(float(t), device=dev))
            a_tok = x_t @ Pm("action_in_proj.kernel") + Pm("action_in_proj.bias")
            xs = a_tok
            rec = dict(x_t=x_t.clone(), t=t, pe=pe, th=th, cond=cond, a_tok=a_tok, layers=[])
            ms = torch.ones(HOR, Np + HOR, dtype=torch.bool, device=dev)
            for l in range(NL):
                xn, ga, (sa, ha, _) = rms(xs, dense=ada(l, "pre_attention_norm"), cond=cond)
                q, k, v = qkv(xn, l, "_1", pos_s)
                Kj, Vj = torch.cat([Kc[l], k]), torch.cat([Vc[l], v])
                enc, pr = attend(q, Kj, Vj, ms)
                lg = torch.einsum("tnh,skh->nts", q, Kj)
                ao = torch.einsum("tnh,nhd->td", enc, W("attn.attn_vec_einsum#.w", l, "_1"))
                r1 = xs + ao * ga
                xn2, gf, (sf, hf, _) = rms(r1, dense=ada(l, "pre_ffw_norm"), cond=cond)
                gg = W("mlp#.gating_einsum", l, "_1")
                gate, up = xn2 @ gg[0], xn2 @ gg[1]
                kp = {}
                fo = mlp(xn2, l, "_1", kp)
                out = r1 + fo * gf
                pm = pr.mean(0)                                                    # [32, Np+32] head-mean
                mass4 = torch.stack([pm[:, :Np][:, col_state].sum(-1), pm[:, :Np][:, col_img].sum(-1),
                                     pm[:, :Np][:, col_txt].sum(-1), pm[:, Np:].sum(-1)], -1)
                L = dict(x_in=xs, out=out, scale_a=sa, shift_a=ha, gate_a=ga, scale_f=sf, shift_f=hf, gate_f=gf,
                         xn=xn, q=q, k=k, v=v, probs=pr, lg=lg, enc=enc, ao=ao, r1=r1, xn2=xn2, gate=gate, up=up,
                         ff=fo, mlp_h=kp["mlp_h"], pm=pm, mass4=mass4)
                rec["layers"].append(L)
                xs = out
            fin, _, (fs, fh, fg) = rms(xs, dense=(Pm(P + "llm.final_norm_1.Dense_0.kernel"),
                                                  Pm(P + "llm.final_norm_1.Dense_0.bias")), cond=cond)
            v_t = fin @ Pm("action_out_proj.kernel") + Pm("action_out_proj.bias")
            rec.update(x_out=xs, final=fin, final_scale=fs, final_shift=fh, v=v_t)
            return rec

        g = torch.Generator().manual_seed(SEED)
        noise = torch.randn(HOR, ADIM, generator=g).to(dev)
        dt = -1.0 / NSTEP
        x_t, t = noise.clone(), 1.0
        steps = []
        for kstep in range(NSTEP):
            rec = suffix(x_t, t)
            rec["k"] = kstep
            rec["x_next"] = x_t + dt * rec["v"]
            steps.append(rec)
            x_t, t = rec["x_next"], t + dt
        action_n = x_t                                                             # [32, 32] normalised
        print(f"[euler] done |a|max={float(action_n.abs().max()):.3f} ({time.time() - t0:.0f}s)", flush=True)

        # NPART one-step particles at t = 1 (seeds 0..NPART-1; seed 0 = the decode's own noise):
        # x1 | v | x1 - v, and each layer's suffix attention mass
        parts = []
        for sd in range(NPART):
            n_s = torch.randn(HOR, ADIM, generator=torch.Generator().manual_seed(sd)).to(dev)
            r = suffix(n_s, 1.0)
            parts.append(dict(x1=n_s, v=r["v"], chunk=n_s - r["v"], mass=torch.stack([L["mass4"] for L in r["layers"]])))

        # joint (non-cached) forward of both streams at step 0: the two streams share one
        # attention per layer; prefix rows cannot see the suffix (make_attn_mask with ar=[0..0 | 1, 0..0])
        ar = torch.tensor([0] * Np + [1] + [0] * (HOR - 1), device=dev).cumsum(0)
        mj = ar[None, :] <= ar[:, None]
        cond0 = steps[0]["cond"]
        xp, xs = prefix0, steps[0]["a_tok"]
        posj = torch.arange(Np + HOR, device=dev)
        joint0 = {}
        for l in range(NL):
            xnp, _, _ = rms(xp, W("pre_attention_norm.scale", l, ""))
            xns, ga, _ = rms(xs, dense=ada(l, "pre_attention_norm"), cond=cond0)
            qp, kp_, vp = qkv(xnp, l, "", posj[:Np])
            qs, ks, vs = qkv(xns, l, "_1", posj[Np:])
            enc, pr = attend(torch.cat([qp, qs]), torch.cat([kp_, ks]), torch.cat([vp, vs]), mj)
            if l == 0:
                joint0 = dict(mask=mj, probs_h0=pr[0])
            xp = xp + torch.einsum("tnh,nhd->td", enc[:Np], W("attn.attn_vec_einsum#.w", l, ""))
            xs = xs + torch.einsum("tnh,nhd->td", enc[Np:], W("attn.attn_vec_einsum#.w", l, "_1")) * ga
            xnp2, _, _ = rms(xp, W("pre_ffw_norm.scale", l, ""))
            xns2, gf, _ = rms(xs, dense=ada(l, "pre_ffw_norm"), cond=cond0)
            xp = xp + mlp(xnp2, l, "")
            xs = xs + mlp(xns2, l, "_1") * gf
        finj, _, _ = rms(xs, dense=(Pm(P + "llm.final_norm_1.Dense_0.kernel"),
                                    Pm(P + "llm.final_norm_1.Dense_0.bias")), cond=cond0)
        v_joint = finj @ Pm("action_out_proj.kernel") + Pm("action_out_proj.bias")

    asserts = {
        "cache_vs_joint_v_maxabs(step0)": float((v_joint - steps[0]["v"]).abs().max()),
        "euler_identity_maxabs": max(float((s["x_next"] - s["x_t"] - dt * s["v"]).abs().max()) for s in steps),
        "gated_residual_identity_maxabs(L0,step0)": float(
            (steps[0]["layers"][0]["out"] - steps[0]["layers"][0]["x_in"]
             - steps[0]["layers"][0]["ao"] * steps[0]["layers"][0]["gate_a"]
             - steps[0]["layers"][0]["ff"] * steps[0]["layers"][0]["gate_f"]).abs().max()),
        "prefix_blind_to_suffix": float(joint0["probs_h0"][:Np, Np:].abs().max()),
    }
    action_n_np = action_n.cpu().numpy()
    act32 = (action_n_np + 1.0) / 2.0 * (aq99 - aq01 + 1e-6) + aq01
    action23 = act32[:, :23]

    # ---------------- official openpi (JAX) reference ---------------- #
    jax_ref = None
    if args.jax_ref:
        npz = "/tmp/pi0_ref_in.npz"
        tok = np.zeros(200, np.int32)
        tok[:T] = ids
        msk = np.zeros(200, bool)
        msk[:T] = True
        np.savez(npz, **{n: imgs224[n] for n, _, _ in CAMS}, tokens=tok, mask=msk,
                 state=np.pad(state_n, (0, 32 - 23)).astype(np.float32), noise=noise.cpu().numpy())
        env = dict(os.environ, PYTHONPATH=f"{JAXENV}:{OPENPI_SRC}:{PYDEPS}", JAX_PLATFORMS="cpu")
        r = subprocess.run([sys.executable, os.path.join(HERE, "ref_pi0_jax.py"), CKPT, npz, "/tmp/pi0_ref_out.npy"],
                           env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout[-2000:], r.stderr[-4000:])
            raise SystemExit("JAX reference failed")
        jax_ref = np.load("/tmp/pi0_ref_out.npy")
        asserts["vs_official_jax_action_maxabs"] = float(np.abs(jax_ref - action_n_np).max())
    print("[assert]", asserts, flush=True)
    if max(asserts.values()) > 2e-3:
        raise SystemExit("re-implementation disagrees with itself / the official openpi model")

    # ---------------- export (the shared VLA schema of public/k005b-*.json) ---------------- #
    def tw_(x, cw=WC):
        return x[tw, :cw]

    # embed: the WT window tokens' rows of the 257152-row table (gathered, `row_ids`)
    wids = ids[t0w - NIMG:]
    put(Wx, "embed", emb_table[torch.tensor(wids)][:, :WC], emb_table, [0, 0],
        "input_embedding [257152, 2048]: the rows of the window's tokens (gathered); embed_out = row x sqrt(2048)")
    Wx["embed"]["row_ids"] = wids
    put(A, "embed_out", prefix0[tw, :WC], prefix0, [t0w, 0], "prefix input = [image tokens | text rows x sqrt(2048)] [Np, 2048]")

    def put_layer_w(tag, l, sfx, Dm):
        qe = W("attn.q_einsum#.w", l, sfx).permute(0, 2, 1)                       # [8, 256, D]
        kv = W("attn.kv_einsum#.w", l, sfx).permute(0, 1, 3, 2)                   # [2, 1, 256, D]
        o = W("attn.attn_vec_einsum#.w", l, sfx).reshape(NH * HD, Dm).T           # [D, 2048] (out, in)
        gg = W("mlp#.gating_einsum", l, sfx)                                       # [2, D, F]
        li = W("mlp#.linear", l, sfx)                                              # [F, D]
        put(Wx, f"{tag}{l}.q_w", qe[:, :WD, :WC], qe, [0, 0, 0], f"q_einsum{sfx} [8, {Dm}, 256] as [head, dim, in]")
        put(Wx, f"{tag}{l}.k_w", kv[0][:, :WD, :WC], kv[0], [0, 0, 0], f"kv_einsum{sfx} K [1, 256, {Dm}]")
        put(Wx, f"{tag}{l}.v_w", kv[1][:, :WD, :WC], kv[1], [0, 0, 0], f"kv_einsum{sfx} V [1, 256, {Dm}]")
        put(Wx, f"{tag}{l}.o_w", o[:WC, :WC], o, [0, 0], f"attn_vec_einsum{sfx} [8, 256, {Dm}] as [out, in]")
        put(Wx, f"{tag}{l}.gate_w", gg[0].T[:WF, :WC], gg[0].T, [0, 0], f"gating_einsum{sfx}[0] as [F, in]")
        put(Wx, f"{tag}{l}.up_w", gg[1].T[:WF, :WC], gg[1].T, [0, 0], f"gating_einsum{sfx}[1] as [F, in]")
        put(Wx, f"{tag}{l}.down_w", li.T[:WC, :WF], li.T, [0, 0], f"mlp{sfx}.linear as [out, F]")

    for l in range(NL):
        put_layer_w("L", l, "", DP)
        put_layer_w("E", l, "_1", DE)
        for nm, key in (("ln1_w", "pre_attention_norm.scale"), ("ln2_w", "pre_ffw_norm.scale")):
            g1 = 1.0 + W(key, l, "")
            put(Wx, f"L{l}.{nm}", g1, g1, [0], f"1 + {key} (Gemma RMSNorm multiplies by 1 + scale)")

    # expert anatomy at Euler step 0 (t = 1, x_t = the noise)
    s0 = steps[0]
    put(A, "expert_in", s0["a_tok"][:, :WC], s0["a_tok"], [0, 0], "expert input = action_in_proj(x_t) [32, 1024]")
    for l, L in enumerate(s0["layers"]):
        En = f"E{l}"
        agg, ms = norm_agg(L["x_in"])
        put(A, f"{En}.ln1_agg", agg, ms, [0, 0], "adaRMSNorm agg over the full 1024: mean(x^2) | rsqrt(mean+1e-6)")
        put(A, f"{En}.ln1", L["xn"][:, :WC], L["xn"], [0, 0], f"expert layer {l} adaRMS(x) = RMS(x)(1+scale)+shift")
        adas = torch.stack([L[k] for k in ("scale_a", "shift_a", "gate_a", "scale_f", "shift_f", "gate_f")])
        put(A, f"{En}.ada", adas[:, :WC], adas, [0, 0],
            f"layer {l}: Dense_0(cond) chunks [scale, shift, gate] of pre_attention_norm_1 | pre_ffw_norm_1 [6, 1024]")
        qu = L["q"] * math.sqrt(HD)
        put(A, f"{En}.q", heads_first(qu[:, :, :WD]), heads_first(qu), [0, 0, 0], f"expert layer {l} Q after RoPE")
        put(A, f"{En}.k", heads_first(L["k"][:, :, :WD]), heads_first(L["k"]), [0, 0, 0], f"expert layer {l} K (1 kv head)")
        put(A, f"{En}.v", heads_first(L["v"][:, :, :WD]), heads_first(L["v"]), [0, 0, 0], f"expert layer {l} V (1 kv head)")
        lg, pr = L["lg"], L["probs"]
        put(A, f"{En}.scp", lg[:, :, tw], lg, [0, 0, t0w], f"action queries . prefix-window keys / 16 (keys = layer {l}'s prefix KV cache)")
        put(A, f"{En}.scs", lg[:, :, Np:], lg, [0, 0, Np], "action queries . action keys / 16")
        mx = lg.max(-1).values
        inv = 1.0 / torch.exp(lg - mx[..., None]).sum(-1)
        put(A, f"{En}.attn_agg", torch.stack([mx, inv], -1), mx, [0, 0, 0], f"softmax agg over all {Np + HOR} keys")
        put(A, f"{En}.pp", pr[:, :, tw], pr, [0, 0, t0w], f"probs on the prefix-window keys (rows over all {Np + HOR} keys)")
        put(A, f"{En}.ps", pr[:, :, Np:], pr, [0, 0, Np], "probs on the 32 action keys")
        put(A, f"{En}.vo", heads_first(L["enc"][:, :, :WD]), heads_first(L["enc"]), [0, 0, 0], "per-head sum p.V")
        put(A, f"{En}.attn", torch.cat([L["pm"][:, tw], L["pm"][:, Np:]], -1), L["pm"], [0, t0w],
            "head-mean attention of the 32 action queries: prefix window | 32 action keys")
        put(A, f"{En}.attn_mass", L["mass4"], L["mass4"], [0, 0], "attention mass on [state text tokens | image | other text | actions]")
        agg, ms = norm_agg(L["r1"])
        put(A, f"{En}.ln2_agg", agg, ms, [0, 0], "adaRMSNorm agg over the full 1024")
        for nm, k, d in (("o", "ao", "attn_vec_einsum_1 out (before the gate)"), ("res1", "r1", "x + attn * gate_a"),
                         ("ln2", "xn2", "pre_ffw adaRMS out"), ("mlp", "ff", "GeGLU MLP out"), ("out", "out", "x + mlp * gate_f")):
            put(A, f"{En}.{nm}", L[k][:, :WC], L[k], [0, 0], f"expert layer {l} {d}")
        for nm, k, d in (("gate", "gate", "gating_einsum_1[0] out"), ("up", "up", "gating_einsum_1[1] out"),
                         ("act", "mlp_h", "gelu_tanh(gate) * up")):
            put(A, f"{En}.{nm}", L[k][:, :WF], L[k], [0, 0], f"expert layer {l} {d} [32, 4096]")
    agg, ms = norm_agg(s0["x_out"])
    put(A, "expert_out_agg", agg, ms, [0, 0], "final adaRMSNorm agg over the full 1024")
    put(A, "expert_out", s0["final"][:, :WC], s0["final"], [0, 0], "final_norm_1 = RMS(x)(1+scale)+shift")
    fa = torch.stack([s0["final_scale"], s0["final_shift"]])
    put(A, "expert_final_ada", fa[:, :WC], fa, [0, 0], "final_norm_1.Dense_0(cond): [scale, shift] (its gate chunk is unused)")
    aw = Pm("action_out_proj.kernel").T
    put(Wx, "action_w", aw, aw, [0, 0], "action_out_proj [1024 -> 32] as [out, in]")
    put(Wx, "action_b", Pm("action_out_proj.bias"), Pm("action_out_proj.bias"), [0], "action_out_proj bias")
    put(A, "action_x1", noise, noise, [0, 0], f"x_1 = randn(32, 32), seed {SEED} (normalised action space)")
    put(A, "action_velocity", s0["v"], s0["v"], [0, 0], "v(x_1, t=1) = action_out_proj(final_norm_1) [32, 32]")
    c1 = noise - s0["v"]
    put(A, "action_chunk", c1, c1, [0, 0], "one Euler step over the whole interval: x_1 - v (K = 1)")
    put(A, "fm_time_emb", s0["pe"][:WC], s0["pe"], [0], "posemb_sincos(t=1, 1024, 4e-3, 4.0)")
    put(A, "fm_time_mlp_hidden", s0["th"][None, :WC], s0["th"][None], [0, 0], "swish(time_mlp_in(pe)) [1, 1024]")
    put(A, "fm_cond", s0["cond"][None, :WC], s0["cond"][None], [0, 0], "cond = swish(time_mlp_out(.)) [1, 1024]: the adaRMS input")
    for nm, key in (("fm_action_in_w", "action_in_proj"), ("fm_time_in_w", "time_mlp_in"), ("fm_time_out_w", "time_mlp_out")):
        w = Pm(key + ".kernel").T
        put(A, nm, w[:WC, :WC], w, [0, 0], f"{key} as [out, in] {list(w.shape)}")
    pt = torch.stack([torch.stack([p["x1"], p["v"], p["chunk"]]) for p in parts])
    put(A, "fm_particles", pt, pt, [0, 0, 0, 0], f"{NPART} seeds at t=1: x_1 | v | x_1 - v [K, 3, 32, 32]")
    pms = torch.stack([p["mass"] for p in parts])
    put(A, "fm_particles_mass", pms, pms, [0, 0, 0, 0], f"{NPART} seeds: per layer, per action query attention mass [K, 18, 32, 4]")
    mc = pt[:, 2].mean(0)
    put(A, "fm_mean_chunk", mc, mc, [0, 0], "mean of the particles' x_1 - v")

    fm_steps = []
    for s in steps:
        Tt = {"x": ent16(s["x_t"], "x_t [32, 32]"), "v": ent16(s["v"], "v [32, 32]"),
              "x_next": ent16(s["x_next"], "x_t + dt v (dt = -0.1)"),
              "expert_in": ent16(s["a_tok"][:, :WC], "action_in_proj(x_t)", s["a_tok"]),
              "expert_out": ent16(s["final"][:, :WC], "final_norm_1 out", s["final"]),
              "fm_time_emb": ent16(s["pe"][:WC], "posemb_sincos(t)", s["pe"]),
              "fm_time_mlp_hidden": ent16(s["th"][None, :WC], "swish(time_mlp_in(pe))", s["th"][None]),
              "fm_cond": ent16(s["cond"][None, :WC], "adaRMS cond", s["cond"][None]),
              "expert_final_ada": ent16(torch.stack([s["final_scale"], s["final_shift"]])[:, :WC], "final [scale, shift]")}
        for l, L in enumerate(s["layers"]):
            for nm, k in (("ln1", "xn"), ("o", "ao"), ("res1", "r1"), ("ln2", "xn2"), ("mlp", "ff"), ("out", "out")):
                Tt[f"E{l}.{nm}"] = ent16(L[k][:, :WC], f"expert layer {l} {nm}", L[k])
            Tt[f"E{l}.attn"] = ent16(torch.cat([L["pm"][:, tw], L["pm"][:, Np:]], -1), f"layer {l} head-mean attention", L["pm"])
            Tt[f"E{l}.mass"] = ent16(L["mass4"], f"layer {l} attention mass")
            adas = torch.stack([L[k] for k in ("scale_a", "shift_a", "gate_a", "scale_f", "shift_f", "gate_f")])
            Tt[f"E{l}.ada"] = ent16(adas[:, :WC], f"layer {l} [scale, shift, gate] x 2", adas)
        fm_steps.append({"k": s["k"], "t": round(s["t"], 6), "tensors": Tt})

    pchunk = pt[:, 2]
    n_params = {"total": Pm.size(), "siglip": Pm.size(P + "img."),
                "prefix_llm": sum(Pm.size(P + "llm.layers." + k) for k in
                                  ("attn.q_einsum.", "attn.kv_einsum.", "attn.attn_vec_einsum.", "mlp.",
                                   "pre_attention_norm.", "pre_ffw_norm.")) + Pm.size(P + "llm.embedder."),
                "expert": sum(Pm.size(P + "llm.layers." + k) for k in
                              ("attn.q_einsum_1", "attn.kv_einsum_1", "attn.attn_vec_einsum_1", "mlp_1",
                               "pre_attention_norm_1", "pre_ffw_norm_1")) + Pm.size(P + "llm.final_norm_1")}
    config = {
        "model_type": "pi05", "name": "pi0.5 (openpi-comet pi05-b1kpt12-cs32)", "checkpoint": CKPT, "step": None,
        "train_config": "pi05_b1k-pt12_cs32_bs64_lr2.5e-5_step50k",
        "variant": {
            "id": "pi0", "title": "pi0.5 b1k",
            "trunk_qk_norm": False, "trunk_causal": False, "trunk_mlp": "geglu", "trunk_heads": False,
            "embed_scale": math.sqrt(DP), "expert_bias": False, "expert_ada": "adarms_gated",
            "expert_mlp": "geglu", "expert_final_ada": True, "state_token": False, "time_into_tokens": False,
            "ada_rows": 3,
        },
        "n_layer": NL, "n_head": NH, "n_kv_head": 1, "head_dim": HD, "n_embd": DP, "intermediate": 16384,
        "vocab_size": int(emb_table.shape[0]), "rms_norm_eps": 1e-6,
        "expert_hidden": DE, "expert_heads": NH, "expert_kv_heads": 1, "expert_intermediate": 4096,
        "T_prefix": Np, "T_text": T, "n_state_tokens": 0, "text_start": NIMG, "n_suffix": HOR, "L_joint": Np + HOR,
        "n_visual_tokens": NIMG, "p_star": Np - 1, "boundary_column": Np - 1,
        "window": {"t0": t0w, "Wt": WT, "Wc": WC, "Wd": WD, "Wf": WF, "Wv": WT, "v0": 0, "embed_gathered": True},
        "window_token_ids": wids, "window_tokens": pieces[t0w - NIMG:],
        "state_token_range": [NIMG + state_start + 1, NIMG + state_end],
        "knack_names": [], "done_names": [],
        "action_dim": ADIM, "action_horizon": HOR,
        "action_dim_names": ACTION_DIM_NAMES + [f"pad_{i}" for i in range(23, ADIM)],
        "action_q01": aq01.tolist(), "action_q99": aq99.tolist(), "behavior_action_dim": 23,
        "expert_n_layer": NL, "n_layer_pairs": NL, "suffix_visible_prefix": Np, "suffix_visible_visual": NIMG,
        "n_params": n_params["total"], "n_params_trunk": n_params["prefix_llm"],
        "n_params_expert": n_params["expert"], "n_params_vision": n_params["siglip"],
        "sample": {"row": None, "episode": args.episode, "frame": args.frame, "goal_text": prompt,
                   "target_knack": None, "target_done": None, "task_index": fr["task"]["task_index"]},
        "pred_knack": None, "pred_done": None,
        "prompt": full_prompt, "state23": state23.tolist(), "state_tokens": disc.tolist(),
        "action_norm": action_n_np.tolist(), "action23": action23.tolist(),
        "fm": {"t": 1.0, "steps": NSTEP, "dt": dt, "t_list": [s["t"] for s in steps], "min_period": 4e-3,
               "max_period": 4.0, "seeds": list(range(NPART)), "n_particles": NPART,
               "particle_chunk_std": float(pchunk.std(0).mean()), "asserts": asserts},
        "notes": [
            "weights + activations are real: local openpi-comet pi05-b1kpt12-cs32 (pi0.5 fine-tuned on "
            "BEHAVIOR-1K), BEHAVIOR 2026-challenge-demos frame (same episode/frame as the k005b export)",
            "R1 Pro: state 23 (from the 61-dim proprio) enters as discretised text tokens; actions are 23 dims padded to 32",
            "tokenizer: the public PaliGemma sentencepiece vocab (~/.cache/openpi/big_vision); zero weight downloads",
            "compute in float32 torch, re-implementing openpi/models/{pi0,gemma,siglip}.py; asserts vs a joint "
            "forward (and --jax-ref: the official openpi JAX model)",
            f"noise seed {SEED}; Euler t = 1 (noise) -> 0, dt = -0.1",
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    Wx["config"] = config
    A["config"] = config
    fm_out = {"K": NSTEP, "checkpoint_step": None, "in_distribution": True, "steps": fm_steps}
    for fn, obj in (("pi0-weights.json", Wx), ("pi0-acts.json", A), ("pi0-fm.json", fm_out)):
        path = os.path.join(out_dir, fn)
        with open(path, "w") as f:
            json.dump(obj, f)
        print(f"[write] {path} {os.path.getsize(path) / 1e6:.1f} MB", flush=True)

    L0 = s0["layers"][0]
    check = {
        "W.L0.q_w[h0,d5,c7]": float(W("attn.q_einsum#.w", 0, "")[0, 7, 5]),
        "W.E0.gate_w[f3,c9]": float(W("mlp#.gating_einsum", 0, "_1")[0, 9, 3]),
        "A.E0.ada[gate_a,c7]": float(L0["gate_a"][7]),
        "A.s1.v[0,:5]": steps[1]["v"][0, :5].tolist(),
        "A.action_norm[0,:5]": action_n_np[0, :5].tolist(),
        "A.action23[0,:5]": action23[0, :5].tolist(),
        "jax_ref.action_norm[0,:5]": None if jax_ref is None else jax_ref[0, :5].tolist(),
        "asserts": asserts,
    }
    with open(os.path.join(out_dir, "pi0-check.json"), "w") as f:
        json.dump(check, f, indent=1)
    print("[check]", json.dumps(check), flush=True)
    print(f"[done] total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
