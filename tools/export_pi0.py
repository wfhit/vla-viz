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

Outputs (public/): pi0-weights.json, pi0-acts.json, pi0-fm.json (per-step,
float16), pi0-check.json (spot values from the live tensors).
"""
import argparse
import base64
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
WC = 64       # hidden-channel window [0, WC)
WD = 64       # head-dim window (head_dim 256)
WF = 64       # MLP hidden window
WS = 32       # channel window of the per-layer prefix stack
PR = 16       # image rows per view shipped for prefix walls (all text rows are shipped)
THUMB = 128
SEED = 0
NSTEP = 10
HOR, ADIM = 32, 32

# openpi-comet b1k_policy.extract_state_from_proprio (R1Pro PROPRIOCEPTION_INDICES)
STATE23 = [("base_qvel", 0, 3), ("trunk_qpos", 53, 57), ("arm_left_qpos", 3, 10),
           ("arm_right_qpos", 28, 35), ("gripper_left_width", 24, 26), ("gripper_right_width", 49, 51)]
ACTION_DIM_NAMES = (["base_vx", "base_vy", "base_yaw"] + [f"torso_{i}" for i in range(1, 5)]
                    + [f"L_arm_j{i}" for i in range(1, 8)] + [f"R_arm_j{i}" for i in range(1, 8)]
                    + ["L_gripper", "R_gripper"])


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
    from PIL import Image
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

    def siglip(img_u8, keep):
        x = torch.from_numpy(img_u8).to(dev).float() / 255.0 * 2.0 - 1.0          # [224,224,3]
        k = Pm(P + "img.embedding.kernel").permute(3, 2, 0, 1)                    # HWIO -> OIHW
        x = F.conv2d(x.permute(2, 0, 1)[None], k, Pm(P + "img.embedding.bias"), stride=14)
        x = x[0].flatten(1).T                                                      # [256, 1152] row-major
        keep["stem"] = x.clone()
        x = x + Pm(P + "img.pos_embedding")[0]
        keep["posemb"] = x.clone()
        outs = []
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
            outs.append(x)
        keep["blocks"] = outs
        enc = ln(x, P + "img.Transformer.encoder_norm")
        keep["enc"] = enc
        return enc @ Pm(P + "img.head.kernel") + Pm(P + "img.head.bias")          # [256, 2048]

    with torch.no_grad():
        vis = {}
        img_tok = []
        for name, view, _ in CAMS:
            vis[view] = {}
            img_tok.append(siglip(imgs224[name], vis[view]))
        img_tok = torch.cat(img_tok, 0)                                            # [768, 2048]
    print(f"[siglip] {tuple(img_tok.shape)} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- Gemma prefix + expert ---------------- #
    NL, NH, HD = 18, 8, 256
    DP, DE = 2048, 1024
    LY = P + "llm.layers."

    emb_table = Params(CKPT, "cpu")(P + "llm.embedder.input_embedding")
    txt_emb = (emb_table[torch.tensor(ids)] * math.sqrt(DP)).to(dev)              # [T, 2048]
    emb_rows = {"ids": ids, "rows": emb_table[torch.tensor(ids)][:, :WC].clone()}
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
        return pe, F.silu(h @ Pm("time_mlp_out.kernel") + Pm("time_mlp_out.bias"))

    pos_p = torch.arange(Np, device=dev)
    pos_s = Np + torch.arange(HOR, device=dev)

    with torch.no_grad():
        # prefix pass (fills the KV cache); prefix mask is all-True after trimming text padding
        mp = torch.ones(Np, Np, dtype=torch.bool, device=dev)
        x = prefix0
        Kc, Vc, pre_in, pre_mass, pre_txt_attn, pre_keep = [], [], [], [], {}, {}
        for l in range(NL):
            pre_in.append(x)
            xn, _, _ = rms(x, W("pre_attention_norm.scale", l, ""))
            q, k, v = qkv(xn, l, "", pos_p)
            enc, pr = attend(q, k, v, mp)
            ao = torch.einsum("tnh,nhd->td", enc, W("attn.attn_vec_einsum#.w", l, ""))
            r1 = x + ao
            xn2, _, _ = rms(r1, W("pre_ffw_norm.scale", l, ""))
            kp = {} if l == 0 else None
            fo = mlp(xn2, l, "", kp)
            if l == 0:
                pre_keep = dict(xn=xn, q=q, k=k, v=v, probs=pr[0], ao=ao, r1=r1, xn2=xn2, ff=fo, mlp_h=kp["mlp_h"], out=r1 + fo)
            x = r1 + fo
            Kc.append(k)
            Vc.append(v)
            pre_mass.append(pr.mean((0, 1)))
            if l in (0, NL - 1):
                pre_txt_attn[l] = pr[:, NIMG:].mean(0)                             # [T, Np] text queries
        pre_in.append(x)
        prefix_final, _, _ = rms(x, Pm(P + "llm.final_norm.scale"))                # not used by the expert
        print(f"[prefix] Np={Np} ({time.time() - t0:.0f}s)", flush=True)

        def suffix(x_t, t, keep_all=False):
            pe, cond = time_cond(torch.tensor(float(t), device=dev))
            a_tok = x_t @ Pm("action_in_proj.kernel") + Pm("action_in_proj.bias")
            xs = a_tok
            rec = dict(x_t=x_t.clone(), t=t, pe=pe, cond=cond, a_tok=a_tok, layers=[])
            ms = torch.ones(HOR, Np + HOR, dtype=torch.bool, device=dev)
            for l in range(NL):
                xn, ga, (sa, ha, _) = rms(xs, dense=ada(l, "pre_attention_norm"), cond=cond)
                q, k, v = qkv(xn, l, "_1", pos_s)
                enc, pr = attend(q, torch.cat([Kc[l], k]), torch.cat([Vc[l], v]), ms)
                ao = torch.einsum("tnh,nhd->td", enc, W("attn.attn_vec_einsum#.w", l, "_1"))
                r1 = xs + ao * ga
                xn2, gf, (sf, hf, _) = rms(r1, dense=ada(l, "pre_ffw_norm"), cond=cond)
                kp = {}
                fo = mlp(xn2, l, "_1", kp)
                out = r1 + fo * gf
                L = dict(x_in=xs, out=out, scale_a=sa, shift_a=ha, gate_a=ga, scale_f=sf, shift_f=hf, gate_f=gf,
                         mass=pr.mean((0, 1)), prefix_share=pr[..., :Np].sum(-1).mean(0))
                if keep_all or l == 0:
                    L.update(xn=xn, q=q, k=k, v=v, probs=pr, ao=ao, r1=r1, xn2=xn2, ff=fo, mlp_h=kp["mlp_h"])
                rec["layers"].append(L)
                xs = out
            fin, _, (fs, fh, fg) = rms(xs, dense=(Pm(P + "llm.final_norm_1.Dense_0.kernel"),
                                                  Pm(P + "llm.final_norm_1.Dense_0.bias")), cond=cond)
            v_t = fin @ Pm("action_out_proj.kernel") + Pm("action_out_proj.bias")
            rec.update(final=fin, final_scale=fs, final_shift=fh, v=v_t)
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

    # ---------------- export ---------------- #
    Wx, A = {}, {}

    def wput(name, full, window, desc):
        Wx[name] = ent(window, full, [0] * window.dim(), desc)

    def aput(name, full, window, desc):
        A[name] = ent(window, full, [0] * window.dim(), desc)

    def a2(name, x, desc, cw=WC):
        aput(name, x, x[:, :cw], desc)

    # vision
    for view in vis:
        a2(f"vit.{view}.patch", vis[view]["posemb"], f"{view}: conv 14x14 patch embed + learned pos [256, 1152]")
    hv = vis["head"]
    a2("vit.head.stem", hv["stem"], "head: conv 14x14 stride 14 output (before pos) [256, 1152]")
    for i in (0, NV - 1):
        a2(f"vit.head.B{i}", hv["blocks"][i], f"head view, SigLIP block {i} output [256, 1152]")
    a2("vit.head.enc", hv["enc"], "head view, encoder_norm output [256, 1152]")
    vnorm = torch.stack([torch.cat([vis[v]["blocks"][i].norm(dim=-1) for v in vis]) for i in range(NV)])
    aput("vit.block_norm", vnorm, vnorm, "per-patch residual norm after each of the 27 SigLIP blocks [27, 768]")
    A["img_tok"] = ent16(img_tok[:, :WS], "image tokens = head(encoder_norm(...)) 1152->2048 [768, 2048]", img_tok)
    a2("txt_emb", txt_emb, f"text embeddings x sqrt(2048) [{T}, 2048]")

    # prefix stream
    # prefix rows shipped for walls: the first PR patches of each view + every text token
    prow = [v * 256 + i for v in range(3) for i in range(PR)] + list(range(NIMG, Np))
    pix = torch.tensor(prow, device=dev)

    def ar(name, x, desc, cw=WC):  # prefix tensor [Np, ...] -> shipped rows, channel window
        A[name] = ent16(x[pix][:, :cw], desc, x)

    for l in range(NL + 1):
        ar(f"pre.L{l}.x_in" if l < NL else "pre.out", pre_in[l],
           f"prefix residual stream into layer {l} [Np, 2048]" if l < NL else "prefix stream after layer 17", WS)
    pn = torch.stack([x.norm(dim=-1) for x in pre_in])
    aput("pre.token_norm", pn, pn, "per-token prefix residual norm into each layer; last row = after layer 17 [19, Np]")
    pm = torch.stack(pre_mass)
    aput("pre.key_mass", pm, pm, "prefix self-attention: mean prob per key (heads, queries) per layer [18, Np]")
    for l, a in pre_txt_attn.items():
        A[f"pre.L{l}.txt_attn"] = ent16(a, f"layer {l}: text-query rows, mean over 8 heads [T, Np]")
    K0 = pre_keep
    ar("pre.L0.xn", K0["xn"], "RMSNorm(x)(1+scale) [Np, 2048]")
    A["pre.L0.q"] = ent16(K0["q"][pix][:, 0, :WD], "Q head 0 after RoPE, x 256^-0.5 [Np, 8, 256]", K0["q"])
    A["pre.L0.k"] = ent16(K0["k"][pix][:, 0, :WD], "K (1 shared KV head) after RoPE [Np, 1, 256]", K0["k"])
    A["pre.L0.v"] = ent16(K0["v"][pix][:, 0, :WD], "V (1 shared KV head) [Np, 1, 256]", K0["v"])
    A["pre.L0.probs"] = ent16(K0["probs"][pix], f"head 0: prefix queries over the {Np} prefix keys (bidirectional)",
                              K0["probs"])
    ar("pre.L0.ao", K0["ao"], "attn_vec_einsum output [Np, 2048]")
    ar("pre.L0.r1", K0["r1"], "x + attn (plain residual)")
    ar("pre.L0.xn2", K0["xn2"], "pre_ffw RMSNorm")
    ar("pre.L0.mlp_h", K0["mlp_h"], "GeGLU hidden gelu(xW0) * xW1 [Np, 16384]", WF)
    ar("pre.L0.ff", K0["ff"], "MLP output")
    ar("pre.L0.out", K0["out"], "prefix layer 0 out")

    # expert, step 0 (t = 1) layer 0 anatomy
    s0 = steps[0]
    L0 = s0["layers"][0]
    aput("x0", noise, noise, f"noise x_1 = randn(32, 32), seed {SEED}")
    aput("s0.pe", s0["pe"], s0["pe"][:WC], "posemb_sincos(t=1, 1024, 4e-3, 4.0)")
    aput("s0.cond", s0["cond"], s0["cond"][:WC], "adaRMS condition = swish(time_mlp_out(swish(time_mlp_in(pe))))")
    a2("s0.a_tok", s0["a_tok"], "action tokens = action_in_proj(x_t) 32->1024 [32, 1024]")
    for nm, desc in (("x_in", "expert layer 0 in"), ("xn", "adaRMS(x) = RMS(x)(1+scale)+shift"),
                     ("ao", "attn_vec_einsum_1 output"), ("r1", "x + attn * gate_a"),
                     ("xn2", "pre_ffw adaRMS"), ("ff", "GeGLU MLP output"), ("out", "x + mlp * gate_f")):
        a2(f"s0.L0.{nm}", L0[nm], desc)
    for nm in ("scale_a", "shift_a", "gate_a", "scale_f", "shift_f", "gate_f"):
        aput(f"s0.L0.{nm}", L0[nm], L0[nm][:WC], f"adaRMS {nm} = chunk(Dense_0(cond)) [1024]")
    aput("s0.L0.q", L0["q"], L0["q"][:, 0, :WD], "expert Q head 0 after RoPE [32, 8, 256]")
    aput("s0.L0.k", L0["k"], L0["k"][:, 0, :WD], "expert K [32, 1, 256]")
    aput("s0.L0.v", L0["v"], L0["v"][:, 0, :WD], "expert V [32, 1, 256]")
    A["s0.L0.probs"] = ent16(L0["probs"][0], f"head 0: 32 action queries over [{Np} prefix | 32 action] keys", L0["probs"])
    aput("s0.L0.mlp_h", L0["mlp_h"], L0["mlp_h"][:, :WF], "GeGLU hidden [32, 4096]")
    jm = joint0["mask"].float()
    aput("joint.mask", jm, jm[::8, ::8].contiguous(), "joint attention mask [Np+32]^2, every 8th row/col")

    # weights
    wput("vit.embedding", Pm(P + "img.embedding.kernel"),
         Pm(P + "img.embedding.kernel").reshape(-1, VW)[:WC, :WC], "conv kernel [14,14,3,1152] as [588, 1152]")
    wput("vit.head", Pm(P + "img.head.kernel"), Pm(P + "img.head.kernel")[:WC, :WC], "head Dense 1152->2048")
    wput("txt.embed", emb_table[:1], emb_rows["rows"], f"input_embedding [257152, 2048]: the {T} prompt rows")
    for sfx, Dm, tag in (("", DP, "pre"), ("_1", DE, "exp")):
        q = W("attn.q_einsum#.w", 0, sfx)
        kv = W("attn.kv_einsum#.w", 0, sfx)
        o = W("attn.attn_vec_einsum#.w", 0, sfx)
        gg = W("mlp#.gating_einsum", 0, sfx)
        li = W("mlp#.linear", 0, sfx)
        wput(f"{tag}.L0.wq", q, q[0, :WC, :WD], f"q_einsum{sfx} [8, {Dm}, 256], head 0")
        wput(f"{tag}.L0.wk", kv, kv[0, 0, :WC, :WD], f"kv_einsum{sfx} [2, 1, {Dm}, 256], K")
        wput(f"{tag}.L0.wv", kv, kv[1, 0, :WC, :WD], f"kv_einsum{sfx} [2, 1, {Dm}, 256], V")
        wput(f"{tag}.L0.wo", o, o[0, :WD, :WC], f"attn_vec_einsum{sfx} [8, 256, {Dm}], head 0")
        wput(f"{tag}.L0.w_gate", gg, gg[0, :WC, :WF], f"gating_einsum [2, {Dm}, {gg.shape[-1]}], gate half")
        wput(f"{tag}.L0.w_up", gg, gg[1, :WC, :WF], f"gating_einsum [2, {Dm}, {gg.shape[-1]}], up half")
        wput(f"{tag}.L0.w_down", li, li[:WF, :WC], f"linear [{li.shape[0]}, {Dm}]")
    ak = W("pre_attention_norm_1.Dense_0.kernel", 0, "")
    wput("exp.L0.ada_attn", ak, ak[:WC, :WC], "pre_attention_norm_1.Dense_0 [1024, 3072] -> scale|shift|gate")
    fk = W("pre_ffw_norm_1.Dense_0.kernel", 0, "")
    wput("exp.L0.ada_ffw", fk, fk[:WC, :WC], "pre_ffw_norm_1.Dense_0 [1024, 3072]")
    pns = Pm(LY + "pre_attention_norm.scale")
    wput("pre.L0.norm_scale", pns, pns[0, :WC], "prefix pre_attention_norm.scale (plain RMSNorm)")
    for nm in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        w = Pm(nm + ".kernel")
        wput(nm, w, w[:WC, :WC], f"{nm}.kernel {list(w.shape)}")

    # per-step, float16
    fm_steps = []
    for s in steps:
        Tt = {}
        Tt["x_t"] = ent16(s["x_t"], "x_t [32, 32] (normalised action space)")
        Tt["v"] = ent16(s["v"], "v = action_out_proj(final_norm_1(expert)) [32, 32]")
        Tt["x_next"] = ent16(s["x_next"], "x_t - 0.1 v")
        Tt["pe"] = ent16(s["pe"][:WC], "posemb_sincos(t) window", s["pe"])
        Tt["cond"] = ent16(s["cond"][:WC], "adaRMS cond window", s["cond"])
        Tt["a_tok"] = ent16(s["a_tok"][:, :WC], "action_in_proj(x_t) window", s["a_tok"])
        Tt["final"] = ent16(s["final"][:, :WC], "final_norm_1 output window", s["final"])
        Tt["final_scale"] = ent16(s["final_scale"][:WC], "final adaRMS scale", s["final_scale"])
        Tt["final_shift"] = ent16(s["final_shift"][:WC], "final adaRMS shift", s["final_shift"])
        for l, L in enumerate(s["layers"]):
            Tt[f"L{l}.out"] = ent16(L["out"][:, :WC], f"expert layer {l} out [32, 1024]", L["out"])
            for nm in ("scale_a", "shift_a", "gate_a", "gate_f"):
                Tt[f"L{l}.{nm}"] = ent16(L[nm][:WC], f"layer {l} adaRMS {nm}", L[nm])
            Tt[f"L{l}.mass"] = ent16(L["mass"], f"layer {l}: mean attention prob per key [Np+32]")
        en = torch.stack([s["a_tok"].norm(dim=-1)] + [L["out"].norm(dim=-1) for L in s["layers"]])
        Tt["layer_norm"] = ent16(en, "per-action-token residual norm: row 0 = input, row l+1 = after layer l [19, 32]")
        ps = torch.stack([L["prefix_share"] for L in s["layers"]])
        Tt["prefix_share"] = ent16(ps, "per layer, per action query: attention share on prefix keys [18, 32]")
        fm_steps.append({"k": s["k"], "t": round(s["t"], 6), "tensors": Tt,
                         "v_norm": float(s["v"].norm()), "v23_norm": float(s["v"][:, :23].norm())})

    thumbs = {}
    for name, view, _ in CAMS:
        im = Image.fromarray(imgs224[name]).resize((THUMB, THUMB), Image.BILINEAR)
        thumbs[view] = base64.b64encode(np.asarray(im, dtype=np.uint8).tobytes()).decode("ascii")

    pieces = [sp.id_to_piece(i) for i in ids]
    state_start = next(i for i in range(T) if "".join(pieces[:i + 1]).endswith("State:"))
    config = {
        "model": "pi0.5 (openpi-comet pi05-b1kpt12-cs32)", "checkpoint": CKPT,
        "train_config": "pi05_b1k-pt12_cs32_bs64_lr2.5e-5_step50k",
        "sample": {"episode": args.episode, "frame": args.frame, "task_index": fr["task"]["task_index"],
                   "task": prompt, "video_hw": fr["video_hw"], "image_hw": [224, 224]},
        "views": [v for _, v, _ in CAMS], "image_keys": [n for n, _, _ in CAMS],
        "camera_keys": [k for _, _, k in CAMS], "thumbs": thumbs, "thumb_size": THUMB,
        "prompt": full_prompt, "T": T, "token_ids": ids, "tokens": pieces, "state_token_start": state_start,
        "Np": Np, "n_img": NIMG, "prefix_rows": prow, "n_img_per_view": 256,
        "window": {"Wc": WC, "Wd": WD, "Wf": WF, "Ws": WS},
        "vit": {"n_block": NV, "width": VW, "heads": VH, "head_dim": VD, "mlp": 4304, "patch": 14, "grid": 16},
        "prefix": {"n_layer": NL, "width": DP, "heads": NH, "kv_heads": 1, "head_dim": HD, "mlp": 16384,
                   "vocab": int(emb_table.shape[0])},
        "expert": {"n_layer": NL, "width": DE, "heads": NH, "kv_heads": 1, "head_dim": HD, "mlp": 4096},
        "action_horizon": HOR, "action_dim": ADIM, "behavior_action_dim": 23,
        "action_dim_names": ACTION_DIM_NAMES, "state_groups": STATE23,
        "state23": state23.tolist(), "state_norm": state_n.tolist(), "state_tokens": disc.tolist(),
        "proprio61": proprio.tolist(),
        "K": NSTEP, "dt": dt, "t": [s["t"] for s in steps], "seed": SEED,
        "action_norm": action_n_np.tolist(), "action23": action23.tolist(),
        "n_params": {"total": Pm.size(), "siglip": Pm.size(P + "img."),
                     "prefix_llm": sum(Pm.size(P + "llm.layers." + k) for k in
                                       ("attn.q_einsum.", "attn.kv_einsum.", "attn.attn_vec_einsum.", "mlp.",
                                        "pre_attention_norm.", "pre_ffw_norm.")) + Pm.size(P + "llm.embedder."),
                     "expert": sum(Pm.size(P + "llm.layers." + k) for k in
                                   ("attn.q_einsum_1", "attn.kv_einsum_1", "attn.attn_vec_einsum_1", "mlp_1",
                                    "pre_attention_norm_1", "pre_ffw_norm_1")) + Pm.size(P + "llm.final_norm_1"),
                     "heads": sum(Pm.size(k) for k in ("action_in_proj", "action_out_proj", "time_mlp"))},
        "asserts": asserts,
        "notes": [
            "weights + activations are real: local openpi-comet pi05-b1kpt12-cs32 (pi0.5 fine-tuned on "
            "BEHAVIOR-1K, 12 tasks incl. turning_on_radio), BEHAVIOR 2026-challenge-demos frame",
            "R1 Pro embodiment: state 23 = base_qvel 3 + trunk 4 + arms 7+7 + gripper widths 2 extracted from "
            "the 61-dim proprio; it enters as discretised text tokens (pi0.5), actions are 23 dims padded to 32",
            "tokenizer: the public PaliGemma sentencepiece vocab (4 MB asset, ~/.cache/openpi/big_vision) - "
            "the only fetched file; zero weight downloads",
            "compute in float32 in torch (official runtime: bfloat16 JAX); text padding trimmed (masked, so "
            "equivalent)",
            f"noise seed {SEED}; Euler integrates t = 1 (noise) -> 0 with dt = -0.1 (openpi convention)",
            "2026 demos: base_qvel may be world-frame vs the 2025 training data's local frame (serve_comet note)",
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    Wx["config"] = config
    A["config"] = config
    for fn, obj in (("pi0-weights.json", Wx), ("pi0-acts.json", A), ("pi0-fm.json", {"K": NSTEP, "steps": fm_steps})):
        path = os.path.join(out_dir, fn)
        with open(path, "w") as f:
            json.dump(obj, f)
        print(f"[write] {path} {os.path.getsize(path) / 1e6:.1f} MB", flush=True)

    check = {
        "W.exp.L0.ada_attn[r3,c9]": float(W("pre_attention_norm_1.Dense_0.kernel", 0, "")[3, 9]),
        "W.pre.L0.wq[h0,r5,c7]": float(W("attn.q_einsum#.w", 0, "")[0, 5, 7]),
        "A.s0.L0.scale_a[c7]": float(L0["scale_a"][7]),
        "A.s0.L0.gate_a[c7]": float(L0["gate_a"][7]),
        "A.s5.L17.gate_f[c3]": float(steps[5]["layers"][17]["gate_f"][3]),
        "A.pre.L17.x_in[t5,c11]": float(pre_in[17][5, 11]),
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
