#!/usr/bin/env python3
"""Export REAL GR00T N1.7-3B weights + activations for the /groot page.

One BEHAVIOR demo frame (the same episode/frame as the k005b export) is pushed
through the local GR00T-N1.7-3B checkpoint:

  3 cameras (HEVC, 2026-challenge-demos) --eval transform--> Qwen3-VL processor
  -> Cosmos-Reason2 backbone (Qwen3-VL arch, LM truncated to 16 layers)
  -> vlln -> vl_self_attention (4 blocks) -> VL tokens
  state 61 (BEHAVIOR proprio) -> state_encoder[slot 23]
  noise 40x132 -> 4 Euler steps of {action_encoder[23] -> 32-block AlternateVLDiT
                                    -> action_decoder[23]}  (t = 0, .25, .5, .75)

The backbone runs in transformers' Qwen3VL (weights from the GR00T safetensors).
The action head is re-implemented here in plain torch so every intermediate is
visible; it is checked against Isaac-GR00T's own AlternateVLDiT / SelfAttention
Transformer (gr00t/model/modules/dit.py, needs diffusers) on the same inputs.

Compute: float32 from the bf16 checkpoint (deterministic; the official runtime
uses bf16). Facts and caveats are written into config.notes.

Outputs (public/): groot-weights.json, groot-acts.json, groot-fm.json (per-step,
float16), groot-check.json (spot values from the live torch tensors).
"""
import argparse
import base64
import importlib.util
import json
import math
import os
import re
import sys
import time

MODEL = "/home/frank/workspace/model/nvidia/GR00T-N1.7-3B"
QWEN = "/home/frank/workspace/model/qwen/Qwen3-VL-2B-Instruct"   # config/processor stand-in
DATA = "/home/frank/workspace/behavior_challenge_data/dataset"
GR00T_SRC = "/tmp/Isaac-GR00T"
PYDEPS = "/home/frank/vla-src/pydeps"

EMB_TAG = "sim_behavior_r1_pro"
CAMS = [("head", "observation.rgb.zed_link_camera_0"),
        ("left_wrist", "observation.rgb.left_realsense_link_camera_0"),
        ("right_wrist", "observation.rgb.right_realsense_link_camera_0")]
WC = 64       # hidden-channel window [0, WC)
WD = 48       # one full DiT head (head_dim 48)
WF = 64       # FF hidden window
THUMB = 128   # RGB thumbnails shipped for the image tiles
SEED = 0

# the 61-dim R1Pro proprioception (OmniGibson PROPRIOCEPTION_INDICES["R1Pro"])
STATE_GROUPS = [("base_qvel", 0, 3), ("arm_left_qpos", 3, 10), ("arm_left_qvel", 10, 17),
                ("eef_left_pos", 17, 20), ("eef_left_quat", 20, 24), ("gripper_left_qpos", 24, 26),
                ("gripper_left_qvel", 26, 28), ("arm_right_qpos", 28, 35), ("arm_right_qvel", 35, 42),
                ("eef_right_pos", 42, 45), ("eef_right_quat", 45, 49), ("gripper_right_qpos", 49, 51),
                ("gripper_right_qvel", 51, 53), ("trunk_qpos", 53, 57), ("trunk_qvel", 57, 61)]
ACTION_DIM_NAMES = (["base_vx", "base_vy", "base_yaw"] + [f"torso_{i}" for i in range(1, 5)]
                    + [f"L_arm_j{i}" for i in range(1, 8)] + ["L_gripper"]
                    + [f"R_arm_j{i}" for i in range(1, 8)] + ["R_gripper"])


def stats(t):
    import torch
    t = t.detach().float()
    f = t[torch.isfinite(t)]
    if f.numel() == 0:
        return dict(min=0.0, max=0.0, mean=0.0, std=0.0)
    return dict(min=float(f.min()), max=float(f.max()), mean=float(f.mean()),
                std=float(f.std()) if f.numel() > 1 else 0.0)


def ent(window, full, offset, desc):
    import torch
    w = window.detach().to(torch.float32).contiguous().cpu()
    return {"shape": list(w.shape), "dtype": "torch.float32",
            "data": base64.b64encode(w.numpy().astype("<f4").tobytes()).decode("ascii"),
            "full_shape": list(full.shape), "offset": list(offset), "stats": stats(full), "desc": desc}


def ent16(x, desc, full=None, offset=None):
    import torch
    w = x.detach().to(torch.float32).contiguous().cpu()
    full = w if full is None else full
    return {"shape": list(w.shape), "dtype": "torch.float16",
            "data": base64.b64encode(w.numpy().astype("<f2").tobytes()).decode("ascii"),
            "full_shape": list(full.shape), "offset": list(offset or [0] * w.dim()),
            "stats": stats(full), "desc": desc}


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------------------------------------------------ input --- #

def load_frame(episode, frame):
    import av
    import cv2
    import numpy as np
    import pandas as pd
    idx = os.path.join(DATA, "behavior-1k-knack", "index")
    demos = os.path.join(DATA, "2026-challenge-demos")
    eps = pd.read_parquet(os.path.join(idx, "episodes.parquet"))
    ep = eps[eps.episode_index == episode].iloc[0]
    assert frame < ep.length
    dfile = os.path.join(demos, "data", f"chunk-{int(ep.data_chunk):03d}", f"file-{int(ep.data_file):03d}.parquet")
    df = pd.read_parquet(dfile, columns=["observation.state", "episode_index", "frame_index", "task_index"])
    row = df.iloc[int(ep.data_row_start) + frame]
    assert int(row.episode_index) == episode and int(row.frame_index) == frame, "row pointer mismatch"
    state = np.asarray(row["observation.state"], dtype=np.float32)
    task_index = int(row.task_index)
    tasks = [json.loads(l) for l in open(os.path.join(demos, "meta", "tasks.jsonl"))]
    task = next(t for t in tasks if t["task_index"] == task_index)

    images, raw = {}, {}
    for view, key in CAMS:
        ci, fi = int(ep[f"videos/{key}/chunk_index"]), int(ep[f"videos/{key}/file_index"])
        t_target = float(ep[f"videos/{key}/from_timestamp"]) + frame / 30.0
        path = os.path.join(demos, "videos", key, f"chunk-{ci:03d}", f"file-{fi:03d}.mp4")
        c = av.open(path)
        s = c.streams.video[0]
        c.seek(int(max(0.0, t_target - 1.0) / s.time_base), stream=s, backward=True, any_frame=False)
        img = None
        for fr in c.decode(s):
            if fr.pts * s.time_base >= t_target - 0.5 / 30:
                img = fr.to_ndarray(format="rgb24")
                break
        c.close()
        assert img is not None, f"no frame decoded for {key}"
        raw[view] = img
        # Gr00tN1d7Processor eval transform (final_processor_config): SmallestMaxSize(256,
        # INTER_AREA) -> FractionalCenterCrop(0.95) -> SmallestMaxSize(256, INTER_AREA)
        h, w = img.shape[:2]
        sc = 256 / min(h, w)
        x = cv2.resize(img, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
        h, w = x.shape[:2]
        ch, cw = int(h * 0.95), int(w * 0.95)
        y0, x0 = (h - ch) // 2, (w - cw) // 2
        x = x[y0:y0 + ch, x0:x0 + cw]
        h, w = x.shape[:2]
        sc = 256 / min(h, w)
        x = cv2.resize(x, (round(w * sc), round(h * sc)), interpolation=cv2.INTER_AREA)
        images[view] = x
    return dict(state=state, task=task, images=images, raw=raw,
                video_hw={v: list(raw[v].shape[:2]) for v in raw})


# ------------------------------------------------------------- the model --- #

class Weights:
    def __init__(self, path, device):
        from safetensors import safe_open
        idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.files = {}
        self.map = idx
        self.path = path
        self.device = device
        self.cache = {}

    def __call__(self, key, dtype=None):
        import torch
        from safetensors import safe_open
        if key not in self.cache:
            fn = self.map[key]
            if fn not in self.files:
                self.files[fn] = safe_open(os.path.join(self.path, fn), framework="pt", device="cpu")
            self.cache[key] = self.files[fn].get_tensor(key).to(self.device, torch.float32)
        return self.cache[key]

    def keys(self, prefix):
        return [k for k in self.map if k.startswith(prefix)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episode", type=int, default=None, help="default: the k005b export's episode")
    ap.add_argument("--frame", type=int, default=None)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public"))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)
    sys.path.insert(0, PYDEPS)

    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    t0 = time.time()

    if args.episode is None or args.frame is None:
        s = json.load(open(os.path.join(out_dir, "k005b-acts.json")))["config"]["sample"]
        args.episode = s["episode"] if args.episode is None else args.episode
        args.frame = s["frame"] if args.frame is None else args.frame

    cfg = json.load(open(os.path.join(MODEL, "config.json")))
    emb_ids = json.load(open(os.path.join(MODEL, "embodiment_id.json")))
    slot = int(emb_ids[EMB_TAG])
    H_A, D_A = cfg["diffusion_model_cfg"]["num_attention_heads"], cfg["diffusion_model_cfg"]["attention_head_dim"]
    NB = cfg["diffusion_model_cfg"]["num_layers"]
    DIT = H_A * D_A
    HOR, ADIM, K = cfg["action_horizon"], cfg["max_action_dim"], cfg["num_inference_timesteps"]
    NBUCK = cfg["num_timestep_buckets"]
    SEL = cfg["select_layer"]
    NV = cfg["vl_self_attention_cfg"]["num_layers"]
    H_V, D_V = cfg["vl_self_attention_cfg"]["num_attention_heads"], cfg["vl_self_attention_cfg"]["attention_head_dim"]

    # ---------------- input ---------------- #
    fr = load_frame(args.episode, args.frame)
    language = re.sub(r"[^\w\s]", "", fr["task"]["task"].lower())  # formalize_language
    print(f"[sample] episode={args.episode} frame={args.frame} task='{language}' "
          f"video={fr['video_hw']}", flush=True)
    istats = json.load(open(os.path.join(DATA, "behavior-1k-knack", "index", "stats.json")))["observation.state"]
    q01, q99 = np.asarray(istats["q01"], np.float32), np.asarray(istats["q99"], np.float32)
    state61 = fr["state"]
    mask = ~np.isclose(q99, q01)
    sn = np.zeros_like(state61)
    sn[mask] = 2 * (state61[mask] - q01[mask]) / (q99[mask] - q01[mask]) - 1   # use_percentiles=True
    state = torch.zeros(1, 1, cfg["max_state_dim"])
    state[0, 0, :61] = torch.from_numpy(sn)

    proc = Qwen3VLProcessor.from_pretrained(QWEN)
    proc.tokenizer.padding_side = "left"
    frames = [torch.from_numpy(fr["images"][v]).permute(2, 0, 1).contiguous() for v, _ in CAMS]
    conv = [{"role": "user", "content": [*[{"type": "image", "image": f} for f in frames],
                                         {"type": "text", "text": language}]}]
    text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    vl = proc(text=[text], images=frames, return_tensors="pt", padding=True)
    input_ids = vl["input_ids"][0]
    S = input_ids.shape[0]
    image_token_id = 151655
    image_mask = (input_ids == image_token_id)
    print(f"[tokens] S={S} image tokens={int(image_mask.sum())} grid={vl['image_grid_thw'].tolist()}", flush=True)

    # ---------------- weights ---------------- #
    Wt = Weights(MODEL, dev)

    # ---------------- backbone (transformers Qwen3VL) ---------------- #
    qcfg = AutoConfig.from_pretrained(QWEN)
    n_lm_full = qcfg.text_config.num_hidden_layers
    qcfg.text_config.num_hidden_layers = SEL
    qcfg._attn_implementation = "eager"
    qcfg.text_config._attn_implementation = "eager"
    qcfg.vision_config._attn_implementation = "eager"
    with torch.device(dev):
        bb = Qwen3VLForConditionalGeneration(qcfg)
    sd = {k[len("backbone.model."):]: Wt(k) for k in Wt.keys("backbone.model.")}
    missing, unexpected = bb.load_state_dict(sd, strict=False)
    print(f"[backbone] missing={missing} unexpected={unexpected}", flush=True)
    if missing or unexpected:
        raise SystemExit("backbone weights do not match the 16-layer Qwen3-VL config")
    bb = bb.float().eval()

    lm_layers = bb.model.language_model.layers
    vblocks = bb.model.visual.blocks
    cap = {"lm_in": [None] * SEL, "lm_attn": [None] * SEL, "vit": [None] * len(vblocks)}
    hooks = []
    for i, L in enumerate(lm_layers):
        hooks.append(L.register_forward_pre_hook(
            lambda m, a, kw, i=i: cap["lm_in"].__setitem__(i, (a[0] if a else kw["hidden_states"])[0].detach().clone()),
            with_kwargs=True))
        hooks.append(L.self_attn.register_forward_hook(
            lambda m, a, o, i=i: cap["lm_attn"].__setitem__(i, o[1][0].detach().mean(0).clone())))
    for i, B in enumerate(vblocks):
        hooks.append(B.register_forward_hook(
            lambda m, a, o, i=i: cap["vit"].__setitem__(i, (o[0] if isinstance(o, tuple) else o).detach().clone())))
    hooks.append(bb.model.visual.patch_embed.register_forward_hook(
        lambda m, a, o: cap.__setitem__("patch", o.detach().clone())))
    hooks.append(bb.model.visual.merger.register_forward_hook(
        lambda m, a, o: cap.__setitem__("merger", o.detach().clone())))
    hooks.append(bb.model.language_model.embed_tokens.register_forward_hook(
        lambda m, a, o: cap.__setitem__("tok_emb", o[0].detach().clone())))
    with torch.no_grad():
        vin = {k: v.to(dev) for k, v in vl.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")}
        vin["pixel_values"] = vin["pixel_values"].float()
        out = bb(**vin, output_hidden_states=True)
        feats = out.hidden_states[-1][0].float()             # [S, 2048], includes the final RMSNorm
    for h in hooks:
        h.remove()
    assert all(x is not None for x in cap["lm_in"]) and all(x is not None for x in cap["vit"])
    print(f"[backbone] done S={S} feats={tuple(feats.shape)} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- action head (plain torch) ---------------- #
    P = "action_head."

    def lin(x, pre):
        b = Wt(pre + ".bias") if (pre + ".bias") in Wt.map else None
        return F.linear(x, Wt(pre + ".weight"), b)

    def ln(x, eps, pre=None):
        if pre is None:
            return F.layer_norm(x, x.shape[-1:], eps=eps)
        return F.layer_norm(x, x.shape[-1:], Wt(pre + ".weight"), Wt(pre + ".bias"), eps)

    def mha(xq, xkv, pre, H, D, kmask=None, keep=None):
        q = lin(xq, pre + ".to_q").view(-1, H, D).transpose(0, 1)
        k = lin(xkv, pre + ".to_k").view(-1, H, D).transpose(0, 1)
        v = lin(xkv, pre + ".to_v").view(-1, H, D).transpose(0, 1)
        sc = q @ k.transpose(-1, -2) / math.sqrt(D)
        if kmask is not None:
            sc = sc.masked_fill(~kmask[None, None, :], float("-inf"))
        pr = sc.softmax(-1)
        o = (pr @ v).transpose(0, 1).reshape(xq.shape[0], H * D)
        out = lin(o, pre + ".to_out.0")
        if keep is not None:
            keep.update(q=q, k=k, v=v, probs=pr, o=o)
        return out

    def ff(x, pre, keep=None):
        h = F.gelu(lin(x, pre + ".net.0.proj"), approximate="tanh")
        if keep is not None:
            keep["ff_h"] = h
        return lin(h, pre + ".net.2")

    def cat_lin(x, pre):  # CategorySpecificLinear at the embodiment slot
        return x @ Wt(pre + ".W")[slot] + Wt(pre + ".b")[slot]

    with torch.no_grad():
        vlln = ln(feats, 1e-5, P + "vlln")
        x = vlln
        vlsa = []
        for j in range(NV):
            pre = P + f"vl_self_attention.transformer_blocks.{j}"
            x = x + mha(ln(x, 1e-5, pre + ".norm1"), ln(x, 1e-5, pre + ".norm1"), pre + ".attn1", H_V, D_V)
            x = x + ff(ln(x, 1e-5, pre + ".norm3"), pre + ".ff")
            vlsa.append(x)
        vl_tok = x                                            # [S, 2048] the DiT's K/V source

        st = state[0].to(dev)                                 # [1, 132]
        st_h = F.relu(cat_lin(st, P + "state_encoder.layer1"))
        st_feat = cat_lin(st_h, P + "state_encoder.layer2")  # [1, 1536]

        g = torch.Generator().manual_seed(SEED)
        x_t = torch.randn(1, HOR, ADIM, generator=g)[0].to(dev)
        x0 = x_t.clone()
        txt_mask = (~image_mask).to(dev)
        img_mask = image_mask.to(dev)
        pos = Wt(P + "position_embedding.weight")[:HOR]

        def tsin(tvec, dim):  # MultiEmbodimentActionEncoder.pos_encoding
            half = dim // 2
            ex = -torch.arange(half, dtype=torch.float, device=dev) * (math.log(10000.0) / half)
            fq = tvec[:, None] * ex.exp()[None]
            return torch.cat([fq.sin(), fq.cos()], -1)

        def timesteps_proj(t):  # diffusers Timesteps(256, flip_sin_to_cos=True, shift=1)
            half = 128
            ex = -math.log(10000) * torch.arange(half, dtype=torch.float32, device=dev) / (half - 1)
            e = t * ex.exp()
            e = torch.cat([e.sin(), e.cos()])
            return torch.cat([e[half:], e[:half]])

        steps = []
        for k in range(K):
            t_cont = k / float(K)
            t_disc = int(t_cont * NBUCK)
            rec = {"k": k, "t_cont": t_cont, "t_disc": t_disc, "x_t": x_t.clone()}
            tv = torch.full((HOR,), float(t_disc), device=dev)
            a_emb = cat_lin(x_t, P + "action_encoder.W1")
            tau = tsin(tv, DIT)
            h2 = cat_lin(torch.cat([a_emb, tau], -1), P + "action_encoder.W2")
            h2s = h2 * torch.sigmoid(h2)
            a_feat = cat_lin(h2s, P + "action_encoder.W3")
            a_feat_pos = a_feat + pos
            sa = torch.cat([st_feat, a_feat_pos], 0)          # [41, 1536]
            rec.update(a_emb=a_emb, tau=tau[0], w2=h2s, a_feat=a_feat, a_feat_pos=a_feat_pos, sa_in=sa.clone())

            tp = timesteps_proj(torch.tensor(float(t_disc), device=dev))
            temb = lin(F.silu(lin(tp, P + "model.timestep_encoder.timestep_embedder.linear_1")),
                       P + "model.timestep_encoder.timestep_embedder.linear_2")
            rec.update(t_proj=tp, temb=temb)
            h = sa
            blocks = []
            for b in range(NB):
                pre = P + f"model.transformer_blocks.{b}"
                ss = lin(F.silu(temb), pre + ".norm1.linear")
                scale, shift = ss.chunk(2)
                n1 = F.layer_norm(h, h.shape[-1:], eps=1e-5) * (1 + scale) + shift
                keep = {}
                if b % 2 == 1:
                    kind, kv, km = "self", n1, None
                else:
                    kind = "text" if b % 4 == 0 else "image"
                    kv, km = vl_tok, (txt_mask if kind == "text" else img_mask)
                at = mha(n1, kv, pre + ".attn1", H_A, D_A, km, keep)
                r1 = h + at
                n3 = F.layer_norm(r1, r1.shape[-1:], eps=1e-5)
                fo = ff(n3, pre + ".ff", keep)
                h_out = r1 + fo
                blk = dict(kind=kind, scale=scale, shift=shift, out=h_out,
                           attn_mass=keep["probs"].mean((0, 1)))  # [n_keys] mean over heads+queries
                if b < 2 or b == NB - 1:
                    blk.update(x_in=h, n1=n1, attn=at, r1=r1, n3=n3, ff_out=fo, **keep)
                blocks.append(blk)
                h = h_out
            shift_o, scale_o = lin(F.silu(temb), P + "model.proj_out_1").chunk(2)
            no = F.layer_norm(h, h.shape[-1:], eps=1e-6) * (1 + scale_o) + shift_o
            dit_out = lin(no, P + "model.proj_out_2")         # [41, 1024]
            dh = F.relu(cat_lin(dit_out, P + "action_decoder.layer1"))
            pred = cat_lin(dh, P + "action_decoder.layer2")  # [41, 132]
            vel = pred[-HOR:]
            x_next = x_t + (1.0 / K) * vel
            rec.update(blocks=blocks, final_shift=shift_o, final_scale=scale_o, final_norm=no,
                       dit_out=dit_out, dec_h=dh, pred=pred, v=vel, x_next=x_next.clone())
            steps.append(rec)
            x_t = x_next
        action = x_t
    print(f"[action head] done |a|max={float(action.abs().max()):.3f} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- equivalence vs the official Isaac-GR00T modules ---------------- #
    asserts = {}
    dit_mod = load_module("gr00t_dit", os.path.join(GR00T_SRC, "gr00t/model/modules/dit.py"))
    with torch.no_grad():
        off = dit_mod.AlternateVLDiT(**{**cfg["diffusion_model_cfg"], "output_dim": cfg["hidden_size"]},
                                     cross_attention_dim=cfg["backbone_embedding_dim"],
                                     attend_text_every_n_blocks=2).to(dev).float().eval()
        osd = {k[len(P + "model."):]: Wt(k) for k in Wt.keys(P + "model.")}
        mi, un = off.load_state_dict(osd, strict=True), None
        sa0 = steps[2]["sa_in"][None]
        ref = off(hidden_states=sa0, encoder_hidden_states=vl_tok[None],
                  timestep=torch.tensor([steps[2]["t_disc"]], device=dev),
                  image_mask=img_mask[None], backbone_attention_mask=torch.ones_like(img_mask)[None])
        asserts["dit_vs_official_maxabs(step2)"] = float((ref[0] - steps[2]["dit_out"]).abs().max())
        vsa = dit_mod.SelfAttentionTransformer(**cfg["vl_self_attention_cfg"]).to(dev).float().eval()
        vsa.load_state_dict({k[len(P + "vl_self_attention."):]: Wt(k) for k in Wt.keys(P + "vl_self_attention.")}, strict=True)
        asserts["vlsa_vs_official_maxabs"] = float((vsa(vlln[None])[0] - vl_tok).abs().max())
        del off, vsa
    asserts["euler_identity_maxabs"] = max(
        float((s["x_next"] - s["x_t"] - s["v"] / K).abs().max()) for s in steps)
    asserts["dit_residual_identity_maxabs(b0,step0)"] = float(
        (steps[0]["blocks"][0]["out"] - steps[0]["blocks"][0]["x_in"] - steps[0]["blocks"][0]["attn"]
         - steps[0]["blocks"][0]["ff_out"]).abs().max())
    print("[assert]", asserts, flush=True)
    if max(asserts.values()) > 2e-3:
        raise SystemExit("re-implementation disagrees with the official Isaac-GR00T modules")

    # ---------------- export ---------------- #
    W, A = {}, {}

    def wput(name, full, window, offset, desc):
        W[name] = ent(window, full, offset, desc)

    def aput(name, full, window, offset, desc):
        A[name] = ent(window, full, offset, desc)

    def a2(name, x, desc, cw=WC):  # [T, C] -> all rows, channels [0, cw)
        aput(name, x, x[:, :cw], [0, 0], desc)

    # backbone
    vis = cap["merger"]                                      # [192, 2048]
    a2("bb.tok_emb", cap["tok_emb"], "input embeddings (text rows: embed_tokens; image rows are replaced by the merger output)")
    a2("bb.visual_tokens", vis, "vision merger output = the 64 visual tokens per view [192, 2048]")
    patch = cap["patch"]
    aput("vit.patch", patch, patch[:256, :WC], [0, 0], "patch_embed output, head view (256 patches of 16px, merge-window order)")
    vit_norm = torch.stack([v.norm(dim=-1) for v in cap["vit"]])   # [24, 768]
    aput("vit.block_norm", vit_norm, vit_norm, [0, 0], "per-patch residual norm after each of the 24 vision blocks")
    for i in (0, len(vblocks) - 1):
        aput(f"vit.B{i}.out", cap["vit"][i], cap["vit"][i][:256, :WC], [0, 0], f"vision block {i} output, head view")
    for i in range(SEL):
        a2(f"lm.L{i}.x_in", cap["lm_in"][i], f"LM layer {i} residual stream in [S, 2048]")
    lm_norm = torch.stack([x.norm(dim=-1) for x in cap["lm_in"]] + [feats.norm(dim=-1)])  # [17, S]
    aput("lm.token_norm", lm_norm, lm_norm, [0, 0], "per-token residual norm into each LM layer; last row = backbone output")
    a2("lm.out", feats, "backbone output = hidden_states[-1] (after the final RMSNorm)")
    for i in (0, SEL - 1):
        aput(f"lm.L{i}.attn", cap["lm_attn"][i], cap["lm_attn"][i], [0, 0], f"LM layer {i} causal attention, mean over 16 heads [S, S]")
    a2("vlln", vlln, "vlln LayerNorm(2048) of the backbone output")
    for j in range(NV):
        a2(f"vlsa.B{j}.out", vlsa[j], f"vl_self_attention block {j} output (block {NV - 1} = the DiT's K/V source)")
    vl_norm = vl_tok.norm(dim=-1)
    aput("vl.token_norm", vl_norm, vl_norm, [0], "per-token norm of the VL tokens the DiT reads")
    # the two K/V sources of the cross blocks: the same VL tokens split by image_mask
    vl_img, vl_txt = vl_tok[img_mask], vl_tok[txt_mask]
    a2("vl.image", vl_img, f"VL tokens where image_mask = 1 [{vl_img.shape[0]}, 2048]: K/V of the image cross blocks (b % 4 == 2)")
    a2("vl.text", vl_txt, f"VL tokens where image_mask = 0 [{vl_txt.shape[0]}, 2048]: K/V of the text cross blocks (b % 4 == 0)")

    # state + adapters
    aput("state.raw", torch.from_numpy(state61), torch.from_numpy(state61), [0], "BEHAVIOR observation.state (61, R1Pro proprio)")
    aput("state.norm", state[0, 0], state[0, 0], [0], "q01/q99 -> [-1,1] (BEHAVIOR index stats), zero-padded to 132")
    aput("state.h", st_h[0], st_h[0, :WC], [0], "state_encoder layer1 + ReLU [1024] (slot 23)")
    aput("state.feat", st_feat[0], st_feat[0, :WC], [0], "state token = state_encoder output [1536] (slot 23)")
    pos_full = Wt(P + "position_embedding.weight")
    wput("pos_emb", pos_full, pos_full[:HOR, :WC], [0, 0], "learned action position embedding [1024, 1536], rows 0..39 used")

    # DiT, step 0 (anatomy) in float32
    s0 = steps[0]
    aput("x0", x0, x0, [0, 0], f"initial noise randn(40, 132), seed {SEED}")
    for nm, desc in (("a_emb", "action_encoder W1(a) [40, 1536]"), ("w2", "swish(W2[a_emb | tau]) [40, 1536]"),
                     ("a_feat", "W3(...) [40, 1536]"), ("a_feat_pos", "action tokens = W3(...) + pos [40, 1536]"),
                     ("sa_in", "DiT input sequence = [state token | 40 action tokens] [41, 1536]")):
        a2(f"s0.{nm}", s0[nm], desc)
    aput("s0.tau", s0["tau"], s0["tau"][:WC], [0], "sincos(t_disc) [1536] concatenated to W1(a)")
    aput("s0.t_proj", s0["t_proj"], s0["t_proj"], [0], "Timesteps(256) of t_disc (flip_sin_to_cos)")
    aput("s0.temb", s0["temb"], s0["temb"][:WC], [0], "temb = TimestepEmbedding(Timesteps(t_disc)) [1536]")
    for b in (0, 1):
        B = s0["blocks"][b]
        pre = f"s0.D{b}"
        a2(pre + ".x_in", B["x_in"], f"DiT block {b} in")
        a2(pre + ".n1", B["n1"], f"AdaLN(x) = LN(x)(1+scale)+shift")
        aput(pre + ".scale", B["scale"], B["scale"][:WC], [0], "AdaLN scale = chunk0(Linear(SiLU(temb)))")
        aput(pre + ".shift", B["shift"], B["shift"][:WC], [0], "AdaLN shift = chunk1(Linear(SiLU(temb)))")
        aput(pre + ".q", B["q"], B["q"][0], [0, 0, 0], "Q head 0 [41, 48] (from the sa sequence)")
        aput(pre + ".k", B["k"], B["k"][0], [0, 0, 0], "K head 0 (cross: from the VL tokens; self: from sa)")
        aput(pre + ".v", B["v"], B["v"][0], [0, 0, 0], "V head 0")
        aput(pre + ".probs", B["probs"], B["probs"][0], [0, 0, 0], "attention probs head 0 (masked keys = 0)")
        a2(pre + ".attn", B["attn"], "to_out(attention)")
        a2(pre + ".r1", B["r1"], "residual after attention")
        a2(pre + ".n3", B["n3"], "LN (no affine) before FF")
        aput(pre + ".ff_h", B["ff_h"], B["ff_h"][:, :WF], [0, 0], "GELU(tanh) hidden [41, 6144]")
        a2(pre + ".ff_out", B["ff_out"], "FF output")
        a2(pre + ".out", B["out"], f"DiT block {b} out")
        wq = Wt(P + f"model.transformer_blocks.{b}.attn1.to_q.weight")
        wk = Wt(P + f"model.transformer_blocks.{b}.attn1.to_k.weight")
        wv = Wt(P + f"model.transformer_blocks.{b}.attn1.to_v.weight")
        wo = Wt(P + f"model.transformer_blocks.{b}.attn1.to_out.0.weight")
        wa = Wt(P + f"model.transformer_blocks.{b}.norm1.linear.weight")
        w1 = Wt(P + f"model.transformer_blocks.{b}.ff.net.0.proj.weight")
        w2 = Wt(P + f"model.transformer_blocks.{b}.ff.net.2.weight")
        wput(f"D{b}.to_q", wq, wq[:D_A, :WC], [0, 0], "to_q [1536, 1536], head-0 rows")
        wput(f"D{b}.to_k", wk, wk[:D_A, :WC], [0, 0], f"to_k [1536, {wk.shape[1]}], head-0 rows")
        wput(f"D{b}.to_v", wv, wv[:D_A, :WC], [0, 0], f"to_v [1536, {wv.shape[1]}], head-0 rows")
        wput(f"D{b}.to_out", wo, wo[:WC, :WC], [0, 0], "to_out [1536, 1536]")
        wput(f"D{b}.ada", wa, wa[:WC, :WC], [0, 0], "norm1.linear [3072, 1536]: rows 0..1535 -> scale, 1536.. -> shift")
        wput(f"D{b}.ff1", w1, w1[:WF, :WC], [0, 0], "ff.net.0.proj [6144, 1536]")
        wput(f"D{b}.ff2", w2, w2[:WC, :WF], [0, 0], "ff.net.2 [1536, 6144]")
    for nm, key in (("state_enc.L1", "state_encoder.layer1"), ("state_enc.L2", "state_encoder.layer2"),
                    ("action_enc.W1", "action_encoder.W1"), ("action_enc.W2", "action_encoder.W2"),
                    ("action_enc.W3", "action_encoder.W3"), ("action_dec.L1", "action_decoder.layer1"),
                    ("action_dec.L2", "action_decoder.layer2")):
        Wf = Wt(P + key + ".W")
        ws = Wf[slot]
        wput(nm, ws, ws[:WC, :WC] if ws.shape[1] >= WC else ws[:WC], [0, 0],
             f"{key}.W[slot {slot}] {list(ws.shape)} (CategorySpecificLinear, x @ W + b)")
    # the untrained-slot evidence: per-slot std of every adapter matrix (init = 0.02)
    slot_std = {}
    for key in ("state_encoder.layer1", "action_encoder.W1", "action_encoder.W2", "action_decoder.layer2"):
        Wf = Wt(P + key + ".W")
        slot_std[key] = [float(Wf[i].std()) for i in range(Wf.shape[0])]
    tw1 = Wt(P + "model.timestep_encoder.timestep_embedder.linear_1.weight")
    wput("time.lin1", tw1, tw1[:WC, :WC], [0, 0], "timestep_embedder.linear_1 [1536, 256]")
    po2 = Wt(P + "model.proj_out_2.weight")
    wput("proj_out_2", po2, po2[:WC, :WC], [0, 0], "proj_out_2 [1024, 1536]")

    # per-step, float16 (all K steps, all blocks)
    fm_steps = []
    for s in steps:
        T = {}
        T["x_t"] = ent16(s["x_t"], "x_t [40, 132]")
        T["v"] = ent16(s["v"], "velocity v = action_decoder(DiT)[-40:] [40, 132]")
        T["x_next"] = ent16(s["x_next"], "x_t + v/4")
        T["sa_in"] = ent16(s["sa_in"][:, :WC], "DiT input [41, 1536] window", s["sa_in"])
        T["temb"] = ent16(s["temb"][:WC], "temb [1536] window", s["temb"])
        T["tau"] = ent16(s["tau"][:WC], "sincos(t_disc) [1536] window", s["tau"])
        T["final_scale"] = ent16(s["final_scale"][:WC], "final AdaLN scale", s["final_scale"])
        T["final_shift"] = ent16(s["final_shift"][:WC], "final AdaLN shift", s["final_shift"])
        T["final_norm"] = ent16(s["final_norm"][:, :WC], "final AdaLN output", s["final_norm"])
        T["dit_out"] = ent16(s["dit_out"][:, :WC], "proj_out_2 output [41, 1024]", s["dit_out"])
        T["dec_h"] = ent16(s["dec_h"][:, :WC], "action_decoder layer1 + ReLU [41, 1024] (slot 23)", s["dec_h"])
        for b, B in enumerate(s["blocks"]):
            T[f"D{b}.out"] = ent16(B["out"][:, :WC], f"block {b} out [41, 1536]", B["out"])
            T[f"D{b}.scale"] = ent16(B["scale"][:WC], f"block {b} AdaLN scale", B["scale"])
            T[f"D{b}.shift"] = ent16(B["shift"][:WC], f"block {b} AdaLN shift", B["shift"])
            T[f"D{b}.attn_mass"] = ent16(B["attn_mass"], f"block {b} attention mass per key (mean heads, queries)")
        bn = torch.stack([s["sa_in"].norm(dim=-1)] + [B["out"].norm(dim=-1) for B in s["blocks"]])
        T["block_norm"] = ent16(bn, "per-token residual norm: row 0 = DiT input, row b+1 = after block b [33, 41]")
        fm_steps.append({"k": s["k"], "t": s["t_cont"], "t_disc": s["t_disc"], "tensors": T,
                         "v_norm": float(s["v"].norm()), "v23_norm": float(s["v"][:, :23].norm())})

    # thumbnails
    thumbs = {}
    for v, _ in CAMS:
        im = Image.fromarray(fr["images"][v]).resize((THUMB, THUMB), Image.BILINEAR)
        thumbs[v] = base64.b64encode(np.asarray(im, dtype=np.uint8).tobytes()).decode("ascii")

    def nparams(prefix):
        tot = 0
        for k in Wt.keys(prefix):
            fn = Wt.map[k]
            if fn not in Wt.files:
                from safetensors import safe_open
                Wt.files[fn] = safe_open(os.path.join(MODEL, fn), framework="pt", device="cpu")
            tot += math.prod(Wt.files[fn].get_slice(k).get_shape())
        return tot

    ids = input_ids.tolist()
    runs, i = [], 0
    while i < S:
        if ids[i] == image_token_id:
            j = i
            while j < S and ids[j] == image_token_id:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    config = {
        "model": "GR00T-N1.7-3B", "checkpoint": MODEL, "global_step": open(os.path.join(MODEL, "latest")).read().strip(),
        "sample": {"episode": args.episode, "frame": args.frame, "task_index": fr["task"]["task_index"],
                   "task": fr["task"]["task"], "language": language, "video_hw": fr["video_hw"],
                   "image_hw": list(fr["images"]["head"].shape[:2]),
                   "image_grid_thw": vl["image_grid_thw"].tolist()},
        "views": [v for v, _ in CAMS], "camera_keys": [k for _, k in CAMS],
        "thumbs": thumbs, "thumb_size": THUMB,
        "S": S, "token_ids": ids, "tokens": [proc.tokenizer.decode([t]) for t in ids],
        "image_runs": runs, "image_token_id": image_token_id,
        "window": {"Wc": WC, "Wd": WD, "Wf": WF},
        "vit": {"n_block": len(vblocks), "width": qcfg.vision_config.hidden_size,
                "heads": qcfg.vision_config.num_heads, "patch": qcfg.vision_config.patch_size,
                "temporal_patch": qcfg.vision_config.temporal_patch_size,
                "merge": qcfg.vision_config.spatial_merge_size,
                "deepstack": list(qcfg.vision_config.deepstack_visual_indexes),
                "n_patch": int(patch.shape[0])},
        "lm": {"n_layer": SEL, "n_layer_full": n_lm_full, "width": qcfg.text_config.hidden_size,
               "heads": qcfg.text_config.num_attention_heads, "kv_heads": qcfg.text_config.num_key_value_heads,
               "head_dim": qcfg.text_config.head_dim, "ffn": qcfg.text_config.intermediate_size},
        "vlsa": {"n_block": NV, "heads": H_V, "head_dim": D_V},
        "dit": {"n_block": NB, "width": DIT, "heads": H_A, "head_dim": D_A, "ffn": 4 * DIT,
                "out_dim": cfg["hidden_size"],
                "kinds": [s0["blocks"][b]["kind"] for b in range(NB)]},
        "action_horizon": HOR, "action_dim": ADIM, "state_dim": cfg["max_state_dim"],
        "behavior_state_dim": 61, "behavior_action_dim": 23, "action_dim_names": ACTION_DIM_NAMES,
        "state_groups": STATE_GROUPS,
        "K": K, "num_timestep_buckets": NBUCK, "seed": SEED,
        "t_disc": [s["t_disc"] for s in steps], "t_cont": [s["t_cont"] for s in steps],
        "embodiment": {"tag": EMB_TAG, "slot": slot, "ids": emb_ids, "slot_std": slot_std,
                       "trained": False},
        "n_params": {"backbone": nparams("backbone."), "vision": nparams("backbone.model.model.visual."),
                     "lm": nparams("backbone.model.model.language_model."),
                     "action_head": nparams("action_head."), "dit": nparams("action_head.model."),
                     "vlsa": nparams("action_head.vl_self_attention."),
                     "adapters": nparams("action_head.state_encoder.") + nparams("action_head.action_encoder.")
                     + nparams("action_head.action_decoder.")},
        "action_norm_head": [float(x) for x in action[:, :23].norm(dim=0)],
        "asserts": asserts,
        "notes": [
            "weights + activations are real: local GR00T-N1.7-3B safetensors, BEHAVIOR 2026-challenge-demos frame",
            "backbone config/processor: local Qwen3-VL-2B-Instruct (Cosmos-Reason2-2B is not local; same "
            "architecture, shapes match exactly); weights are GR00T's own backbone.* tensors",
            f"embodiment slot {slot} ({EMB_TAG}) is untrained (per-slot std = init 0.02): the state/action "
            "adapters are real-but-untrained weights -> toy; the decoded actions carry no meaning",
            "state = BEHAVIOR 61-dim proprio (q01/q99 -> [-1,1]); the sim template's world-pose keys are "
            "absent from the 2026 demos, so the 61 dims fill slots 0..60 and 61..131 are zero",
            "compute in float32 from bf16 weights (official runtime: bf16)",
            f"noise seed {SEED}; Euler integrates upward t = 0, .25, .5, .75 (t=0 is noise)",
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    W["config"] = config
    A["config"] = config
    for fn, obj in (("groot-weights.json", W), ("groot-acts.json", A),
                    ("groot-fm.json", {"K": K, "steps": fm_steps})):
        path = os.path.join(out_dir, fn)
        with open(path, "w") as f:
            json.dump(obj, f)
        print(f"[write] {path} {os.path.getsize(path) / 1e6:.1f} MB", flush=True)

    with torch.no_grad():
        check = {
            "W.D0.to_q[r5,c7]": float(Wt(P + "model.transformer_blocks.0.attn1.to_q.weight")[5, 7]),
            "W.D0.ada[r3,c9]": float(Wt(P + "model.transformer_blocks.0.norm1.linear.weight")[3, 9]),
            "A.s0.D0.scale[c7]": float(steps[0]["blocks"][0]["scale"][7]),
            "A.s0.D0.shift[c7]": float(steps[0]["blocks"][0]["shift"][7]),
            "A.s2.D17.scale[c3]": float(steps[2]["blocks"][17]["scale"][3]),
            "A.lm.L15.x_in[t5,c11]": float(cap["lm_in"][15][5, 11]),
            "A.vlsa.B3.out[t0,c2]": float(vl_tok[0, 2]),
            "A.s1.v[0,:5]": steps[1]["v"][0, :5].tolist(),
            "A.action[0,:5]": action[0, :5].tolist(),
            "asserts": asserts,
        }
    with open(os.path.join(out_dir, "groot-check.json"), "w") as f:
        json.dump(check, f, indent=1)
    print("[check]", json.dumps(check), flush=True)
    print(f"[done] total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
