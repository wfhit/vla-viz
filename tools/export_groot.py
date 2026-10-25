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

Outputs (public/), in the shared VLA schema of tools/export_k005b.py (the /groot page
runs on the shared VLA engine; config.variant carries GR00T's structural differences):
groot-weights.json, groot-acts.json (every layer; DiT at Euler step 0 = t 0),
groot-fm.json (every Euler step, float16), groot-check.json (spot values).

Trunk layers L0..L15 = the 16 kept Qwen3-VL LM layers, L16..L19 = the 4 vl_self_attention
blocks (vlln = the LayerNorm between them). Expert layers E0..E31 = the DiT blocks; only
head 0 of the 32 vlsa / DiT heads is exported (the page draws one head, "x32 heads").
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
WT = 32       # token window: the last WT backbone tokens (tail of the last image + the text)
WC = 64       # hidden-channel window [0, WC)
WD = 32       # head-dim window
WF = 64       # MLP hidden window
NPART = 8     # one-step particles (seeds) at t = 0
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

    def load(self, key):  # uncached
        import torch
        from safetensors import safe_open
        fn = self.map[key]
        if fn not in self.files:
            self.files[fn] = safe_open(os.path.join(self.path, fn), framework="pt", device="cpu")
        return self.files[fn].get_tensor(key).to(self.device, torch.float32)

    def __call__(self, key):
        if key not in self.cache:
            self.cache[key] = self.load(key)
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
    import transformers.models.qwen3_vl.modeling_qwen3_vl as mq
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
    NS = HOR + 1                                              # DiT tokens: [state | 40 actions]
    BA = 23                                                   # BEHAVIOR R1 Pro action dims (of 132)

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
    state = torch.zeros(cfg["max_state_dim"])
    state[:61] = torch.from_numpy(sn)
    state = state.to(dev)

    proc = Qwen3VLProcessor.from_pretrained(QWEN)
    proc.tokenizer.padding_side = "left"
    frames = [torch.from_numpy(fr["images"][v]).permute(2, 0, 1).contiguous() for v, _ in CAMS]
    conv = [{"role": "user", "content": [*[{"type": "image", "image": f} for f in frames],
                                         {"type": "text", "text": language}]}]
    text = proc.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)
    vl = proc(text=[text], images=frames, return_tensors="pt", padding=True)
    input_ids = vl["input_ids"][0]
    ids = input_ids.tolist()
    S = len(ids)
    image_token_id = 151655
    image_mask = (input_ids == image_token_id)
    pieces = [proc.tokenizer.decode([t]) for t in ids]
    assert S >= WT
    t0w = S - WT
    tw = slice(t0w, S)
    print(f"[tokens] S={S} image tokens={int(image_mask.sum())} window [{t0w}, {S}): "
          f"{''.join('I' if image_mask[i] else 't' for i in range(t0w, S))}", flush=True)
    assert image_mask[tw].any() and (~image_mask[tw]).any(), "window must straddle image and text tokens"

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
    sd = {k[len("backbone.model."):]: Wt.load(k) for k in Wt.keys("backbone.model.")}
    missing, unexpected = bb.load_state_dict(sd, strict=False)
    del sd
    print(f"[backbone] missing={missing} unexpected={unexpected}", flush=True)
    if missing or unexpected:
        raise SystemExit("backbone weights do not match the 16-layer Qwen3-VL config")
    bb = bb.float().eval()
    tc = qcfg.text_config
    H, KV, HD, C = tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.hidden_size
    FF = tc.intermediate_size
    EPS = tc.rms_norm_eps

    lm_layers = bb.model.language_model.layers
    cap = {"x_in": [None] * SEL, "out": [None] * SEL, "attn": [None] * SEL}
    hooks = []
    for i, L in enumerate(lm_layers):
        hooks.append(L.register_forward_pre_hook(
            lambda m, a, kw, i=i: cap["x_in"].__setitem__(i, (a[0] if a else kw["hidden_states"])[0].detach().clone()),
            with_kwargs=True))
        hooks.append(L.register_forward_hook(
            lambda m, a, o, i=i: cap["out"].__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0].detach().clone())))
    hooks.append(bb.model.language_model.embed_tokens.register_forward_hook(
        lambda m, a, o: cap.__setitem__("tok_emb", o[0].detach().clone())))
    orig_eager = mq.eager_attention_forward

    def eager_capture(module, query, key, value, attention_mask, scaling, dropout=0.0, **kw):
        out, w = orig_eager(module, query, key, value, attention_mask, scaling, dropout, **kw)
        if isinstance(module, mq.Qwen3VLTextAttention):
            cap["attn"][module.layer_idx] = dict(q=query[0].detach().clone(), k=key[0].detach().clone(),
                                                 v=value[0].detach().clone(), probs=w[0].detach().clone(),
                                                 out=out[0].detach().clone(), scaling=scaling)
        return out, w

    mq.eager_attention_forward = eager_capture
    try:
        with torch.no_grad():
            vin = {k: v.to(dev) for k, v in vl.items() if k in ("input_ids", "attention_mask", "pixel_values", "image_grid_thw", "mm_token_type_ids")}
            vin["pixel_values"] = vin["pixel_values"].float()
            out = bb(**vin, output_hidden_states=True)
            feats = out.hidden_states[-1][0].float()             # GR00T backbone_features
    finally:
        mq.eager_attention_forward = orig_eager
    for h in hooks:
        h.remove()
    assert all(x is not None for x in cap["x_in"]) and all(x is not None for x in cap["attn"])
    print(f"[backbone] done S={S} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- action head + trunk recompute (plain torch) ---------------- #
    P = "action_head."

    def lin(x, pre):
        b = Wt(pre + ".bias") if (pre + ".bias") in Wt.map else None
        return F.linear(x, Wt(pre + ".weight"), b)

    def ln(x, eps, pre=None):
        if pre is None:
            return F.layer_norm(x, x.shape[-1:], eps=eps)
        return F.layer_norm(x, x.shape[-1:], Wt(pre + ".weight"), Wt(pre + ".bias"), eps)

    def mha(xq, xkv, pre, Hh, D, kmask=None, keep=None):
        q = lin(xq, pre + ".to_q").view(-1, Hh, D).transpose(0, 1)
        k = lin(xkv, pre + ".to_k").view(-1, Hh, D).transpose(0, 1)
        v = lin(xkv, pre + ".to_v").view(-1, Hh, D).transpose(0, 1)
        sc = q @ k.transpose(-1, -2) / math.sqrt(D)
        if kmask is not None:
            sc = sc.masked_fill(~kmask[None, None, :], float("-inf"))
        pr = sc.softmax(-1)
        o = (pr @ v).transpose(0, 1).reshape(xq.shape[0], Hh * D)
        out = lin(o, pre + ".to_out.0")
        if keep is not None:
            keep.update(q=q, k=k, v=v, sc=sc, probs=pr, enc=pr @ v, ao=out)
        return out

    def ff(x, pre, keep=None):
        g = lin(x, pre + ".net.0.proj")
        h = F.gelu(g, approximate="tanh")
        if keep is not None:
            keep.update(gate=g, act=h)
        return lin(h, pre + ".net.2")

    def cat_lin(x, pre):  # CategorySpecificLinear at the embodiment slot
        return x @ Wt(pre + ".W")[slot] + Wt(pre + ".b")[slot]

    asserts = {}
    with torch.no_grad():
        # the 16 LM layers, recomputed from the captured x_in / q k v / softmax·V
        trunk = []
        for i, L in enumerate(lm_layers):
            ca = cap["attn"][i]
            x_in = cap["x_in"][i]
            xn = L.input_layernorm(x_in)
            enc = ca["out"]                                                    # [S, H, HD]
            ao = L.self_attn.o_proj(enc.reshape(S, H * HD))
            r1 = x_in + ao
            xn2 = L.post_attention_layernorm(r1)
            gate, up = L.mlp.gate_proj(xn2), L.mlp.up_proj(xn2)
            act = F.silu(gate) * up
            fo = L.mlp.down_proj(act)
            o = r1 + fo
            asserts[f"lm_recompute_maxabs"] = max(asserts.get("lm_recompute_maxabs", 0.0),
                                                  float((o - cap["out"][i]).abs().max()))
            kr = ca["k"].repeat_interleave(H // KV, dim=0)                     # [H, S, HD]
            sc = (ca["q"][:, tw] @ kr.transpose(-1, -2)) * ca["scaling"]      # [H, Wt, S]
            causal = torch.arange(S, device=dev)[None, :] > torch.arange(t0w, S, device=dev)[:, None]
            sc = sc.masked_fill(causal[None], float("-inf"))
            pr = sc.softmax(-1)
            asserts["lm_probs_vs_captured_maxabs"] = max(asserts.get("lm_probs_vs_captured_maxabs", 0.0),
                                                         float((pr - ca["probs"][:, tw]).abs().max()))
            trunk.append(dict(kind="lm", x_in=x_in, xn=xn, q=ca["q"], k=ca["k"], v=ca["v"], sc=sc, probs=pr,
                              enc=enc.transpose(0, 1), ao=ao, r1=r1, xn2=xn2, gate=gate, up=up, act=act, ff=fo, out=o))
        lm_out = cap["out"][SEL - 1]
        lnf = bb.model.language_model.norm(lm_out)
        asserts["backbone_features_eq_final_rmsnorm_maxabs"] = float((lnf - feats).abs().max())

        vlln = ln(feats, 1e-5, P + "vlln")
        x = vlln
        for j in range(NV):
            pre = P + f"vl_self_attention.transformer_blocks.{j}"
            keep = {}
            xn = ln(x, 1e-5, pre + ".norm1")
            ao = mha(xn, xn, pre + ".attn1", H_V, D_V, None, keep)
            r1 = x + ao
            xn2 = ln(r1, 1e-5, pre + ".norm3")
            fo = ff(xn2, pre + ".ff", keep)
            o = r1 + fo
            trunk.append(dict(kind="vlsa", pre=pre, x_in=x, xn=xn, q=keep["q"], k=keep["k"], v=keep["v"],
                              sc=keep["sc"][:, tw], probs=keep["probs"][:, tw], enc=keep["enc"], ao=ao, r1=r1,
                              xn2=xn2, gate=keep["gate"], act=keep["act"], ff=fo, out=o))
            x = o
        vl_tok = x                                            # [S, 2048] the DiT's cross-attention K/V source

        st_h = F.relu(cat_lin(state, P + "state_encoder.layer1"))
        st_feat = cat_lin(st_h, P + "state_encoder.layer2")  # [1536] = DiT token 0
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

        def dit(x_t, k):
            t_cont = k / float(K)
            t_disc = int(t_cont * NBUCK)
            rec = {"k": k, "t": t_cont, "t_disc": t_disc, "x_t": x_t.clone()}
            tv = torch.full((HOR,), float(t_disc), device=dev)
            a_emb = cat_lin(x_t, P + "action_encoder.W1")
            tau = tsin(tv, DIT)
            h2 = cat_lin(torch.cat([a_emb, tau], -1), P + "action_encoder.W2")
            h2s = h2 * torch.sigmoid(h2)
            a_feat = cat_lin(h2s, P + "action_encoder.W3")
            a_feat_pos = a_feat + pos
            sa = torch.cat([st_feat[None], a_feat_pos], 0)   # [41, 1536]
            tp = timesteps_proj(torch.tensor(float(t_disc), device=dev))
            th = F.silu(lin(tp, P + "model.timestep_encoder.timestep_embedder.linear_1"))
            temb = lin(th, P + "model.timestep_encoder.timestep_embedder.linear_2")
            rec.update(a_emb=a_emb, tau=tau[0], h2s=h2s, a_feat=a_feat, a_feat_pos=a_feat_pos, sa=sa,
                       tp=tp, th=th, temb=temb, layers=[])
            h = sa
            for b in range(NB):
                pre = P + f"model.transformer_blocks.{b}"
                scale, shift = lin(F.silu(temb), pre + ".norm1.linear").chunk(2)
                n1 = F.layer_norm(h, h.shape[-1:], eps=1e-5) * (1 + scale) + shift
                keep = {}
                if b % 2 == 1:
                    kind, kv, km = "self", n1, None
                else:
                    kind = "cross_text" if b % 4 == 0 else "cross_image"
                    kv, km = vl_tok, (txt_mask if kind == "cross_text" else img_mask)
                ao = mha(n1, kv, pre + ".attn1", H_A, D_A, km, keep)
                r1 = h + ao
                n3 = F.layer_norm(r1, r1.shape[-1:], eps=1e-5)
                fo = ff(n3, pre + ".ff", keep)
                o = r1 + fo
                pm = keep["probs"].mean(0)                    # [41, keys] head-mean
                if kind == "self":
                    mass4 = torch.stack([pm[:, 0], torch.zeros_like(pm[:, 0]), torch.zeros_like(pm[:, 0]),
                                         pm[:, 1:].sum(-1)], -1)
                    attn = torch.cat([torch.zeros(NS, WT, device=dev), pm], -1)
                else:
                    z = torch.zeros_like(pm[:, 0])
                    mass4 = torch.stack([z, pm[:, img_mask].sum(-1), pm[:, txt_mask].sum(-1), z], -1)
                    attn = torch.cat([pm[:, tw], torch.zeros(NS, NS, device=dev)], -1)
                rec["layers"].append(dict(kind=kind, x_in=h, scale=scale, shift=shift, n1=n1, r1=r1, n3=n3,
                                          ff=fo, out=o, mass4=mass4, attn=attn, **keep))
                h = o
            shift_o, scale_o = lin(F.silu(temb), P + "model.proj_out_1").chunk(2)
            no = F.layer_norm(h, h.shape[-1:], eps=1e-6) * (1 + scale_o) + shift_o
            dit_out = lin(no, P + "model.proj_out_2")         # [41, 1024]
            dh = F.relu(cat_lin(dit_out, P + "action_decoder.layer1"))
            pred = cat_lin(dh, P + "action_decoder.layer2")  # [41, 132]
            vel = pred[-HOR:]
            rec.update(x_out=h, final_scale=scale_o, final_shift=shift_o, final=no, dit_out=dit_out, dec_h=dh,
                       pred=pred, v=vel, x_next=x_t + vel / K)
            return rec

        g = torch.Generator().manual_seed(SEED)
        noise = torch.randn(1, HOR, ADIM, generator=g)[0].to(dev)
        x_t = noise.clone()
        steps = []
        for k in range(K):
            rec = dit(x_t, k)
            steps.append(rec)
            x_t = rec["x_next"]
        action = x_t
        parts = []
        for sd_ in range(NPART):
            n_s = torch.randn(1, HOR, ADIM, generator=torch.Generator().manual_seed(sd_))[0].to(dev)
            r = dit(n_s, 0)
            parts.append(dict(x0=n_s, v=r["v"], chunk=n_s + r["v"],
                              mass=torch.stack([L_["mass4"] for L_ in r["layers"]])))
    print(f"[action head] done |a|max={float(action.abs().max()):.3f} ({time.time() - t0:.0f}s)", flush=True)

    # ---------------- equivalence vs the official Isaac-GR00T modules ---------------- #
    dit_mod = load_module("gr00t_dit", os.path.join(GR00T_SRC, "gr00t/model/modules/dit.py"))
    with torch.no_grad():
        off = dit_mod.AlternateVLDiT(**{**cfg["diffusion_model_cfg"], "output_dim": cfg["hidden_size"]},
                                     cross_attention_dim=cfg["backbone_embedding_dim"],
                                     attend_text_every_n_blocks=2).to(dev).float().eval()
        off.load_state_dict({k[len(P + "model."):]: Wt(k) for k in Wt.keys(P + "model.")}, strict=True)
        ref = off(hidden_states=steps[2]["sa"][None], encoder_hidden_states=vl_tok[None],
                  timestep=torch.tensor([steps[2]["t_disc"]], device=dev),
                  image_mask=img_mask[None], backbone_attention_mask=torch.ones_like(img_mask)[None])
        asserts["dit_vs_official_maxabs(step2)"] = float((ref[0] - steps[2]["dit_out"]).abs().max())
        vsa = dit_mod.SelfAttentionTransformer(**cfg["vl_self_attention_cfg"]).to(dev).float().eval()
        vsa.load_state_dict({k[len(P + "vl_self_attention."):]: Wt(k) for k in Wt.keys(P + "vl_self_attention.")}, strict=True)
        asserts["vlsa_vs_official_maxabs"] = float((vsa(vlln[None])[0] - vl_tok).abs().max())
        del off, vsa
    asserts["euler_identity_maxabs"] = max(float((s["x_next"] - s["x_t"] - s["v"] / K).abs().max()) for s in steps)
    L00 = steps[0]["layers"][0]
    asserts["dit_residual_identity_maxabs(E0,step0)"] = float((L00["out"] - L00["x_in"] - L00["ao"] - L00["ff"]).abs().max())
    print("[assert]", asserts, flush=True)
    if max(asserts.values()) > 2e-3:
        raise SystemExit("re-implementation disagrees with transformers / the official Isaac-GR00T modules")

    # ---------------- export (the shared VLA schema of public/k005b-*.json) ---------------- #
    Wx, A = {}, {}

    def put(D, name, window, full, offset, desc):
        D[name] = ent(window, full, offset, desc)

    def rms_agg(x):  # [T, C] -> [T, 2]: mean(x^2) over the FULL C | rsqrt(mean+eps)
        ms = x.pow(2).mean(-1)
        return torch.stack([ms, torch.rsqrt(ms + EPS)], -1), ms

    def ln_agg(x, eps):  # [T, C] -> [T, 2]: mean over the FULL C | 1/sqrt(var+eps)
        mu = x.mean(-1)
        return torch.stack([mu, torch.rsqrt(x.var(-1, unbiased=False) + eps)], -1), mu

    def sm_agg(sc):  # softmax agg over the full key row: max | 1/sum exp(s-max)
        mx = sc.max(-1).values
        return torch.stack([mx, 1.0 / torch.exp(sc - mx[..., None]).sum(-1)], -1), mx

    def heads_w(w, nh, hd):  # Linear weight [nh*hd, in] -> [nh, hd, in]
        return w.view(nh, hd, w.shape[-1])

    # embed: the WT window tokens' rows of the embed table (gathered, `row_ids`)
    emb_table = bb.model.language_model.embed_tokens.weight
    wids = ids[t0w:]
    put(Wx, "embed", emb_table[torch.tensor(wids, device=dev)][:, :WC], emb_table, [0, 0],
        "embed_tokens [vocab, 2048]: the rows of the window's tokens (gathered); image rows are <|image_pad|>")
    Wx["embed"]["row_ids"] = wids
    x0_ = cap["x_in"][0]
    put(A, "embed_out", x0_[tw, :WC], x0_, [t0w, 0],
        "LM input = text rows of embed_tokens | image rows replaced by the vision merger output [S, 2048]")

    with torch.no_grad():
        for l, Lr in enumerate(trunk):
            Ln = f"L{l}"
            lm = Lr["kind"] == "lm"
            tag = f"LM layer {l}" if lm else f"vl_self_attention block {l - SEL}"
            put(A, f"{Ln}.x_in", Lr["x_in"][tw, :WC], Lr["x_in"], [t0w, 0], f"{tag} residual stream in [S, 2048]")
            if lm:
                agg, ms = rms_agg(Lr["x_in"])
                put(A, f"{Ln}.ln1_agg", agg[tw], ms, [t0w, 0], "RMSNorm agg over the full 2048: mean(x^2) | rsqrt(mean+1e-6)")
                agg, ms = rms_agg(Lr["r1"])
                put(A, f"{Ln}.ln2_agg", agg[tw], ms, [t0w, 0], "RMSNorm agg over the full 2048: mean(x^2) | rsqrt(mean+1e-6)")
            else:
                agg, mu = ln_agg(Lr["x_in"], 1e-5)
                put(A, f"{Ln}.ln1_agg", agg[tw], mu, [t0w, 0], "LayerNorm agg over the full 2048: mean | 1/sqrt(var+1e-5)")
                agg, mu = ln_agg(Lr["r1"], 1e-5)
                put(A, f"{Ln}.ln2_agg", agg[tw], mu, [t0w, 0], "LayerNorm agg over the full 2048: mean | 1/sqrt(var+1e-5)")
            nh = H if lm else 1
            nkv = KV if lm else 1
            q, k, v, enc = Lr["q"][:nh], Lr["k"][:nkv], Lr["v"][:nkv], Lr["enc"][:nh]   # [h, S, d]
            sc, pr = Lr["sc"][:nh], Lr["probs"][:nh]                                     # [h, Wt, S]
            qd = "Q (q_norm + RoPE)" if lm else "Q = to_q(LN(x)) + bias (no RoPE; head 0 of 32)"
            put(A, f"{Ln}.q", q[:, tw, :WD], q, [0, t0w, 0], f"{tag} {qd} [head, token, dim]")
            put(A, f"{Ln}.k", k[:, tw, :WD], k, [0, t0w, 0], f"{tag} K [kv_head, token, dim]")
            put(A, f"{Ln}.v", v[:, tw, :WD], v, [0, t0w, 0], f"{tag} V [kv_head, token, dim]")
            mdesc = "causal: masked=-inf" if lm else "bidirectional: no mask"
            put(A, f"{Ln}.attn_scores", sc[:, :, tw], sc, [0, t0w, t0w], f"{tag} q.k/sqrt(d) ({mdesc}) rows=query cols=key")
            agg, mx = sm_agg(sc)
            put(A, f"{Ln}.attn_agg", agg, mx, [0, t0w, 0], f"softmax agg over all {S} keys: max | 1/sum exp(s-max)")
            put(A, f"{Ln}.attn_probs", pr[:, :, tw], pr, [0, t0w, t0w], f"{tag} attention probs (rows over all {S} keys)")
            put(A, f"{Ln}.attn_out", enc[:, tw, :WD], enc, [0, t0w, 0], f"{tag} per-head sum p.V")
            od = "o_proj out" if lm else "to_out out (incl. bias)"
            md = "down_proj(silu(gate) * up)" if lm else "fc2(gelu_tanh(fc1)) (incl. biases)"
            for nm, tns, d in (("ln1_out", Lr["xn"], "norm1 out" if not lm else "input RMSNorm out"),
                               ("o_out", Lr["ao"], od), ("attn_resid", Lr["r1"], "x + attn"),
                               ("ln2_out", Lr["xn2"], "norm3 out" if not lm else "post-attn RMSNorm out"),
                               ("mlp_out", Lr["ff"], md), ("mlp_resid", Lr["out"], "x + MLP (layer out"
                                                         + (", before deepstack)" if lm and l < 3 else ")"))):
                put(A, f"{Ln}.{nm}", tns[tw, :WC], tns, [t0w, 0], f"{tag} {d}")
            put(A, f"{Ln}.mlp_gate", Lr["gate"][tw, :WF], Lr["gate"], [t0w, 0], f"{tag} " + ("gate_proj out" if lm else "fc1 out (pre-GELU)"))
            if lm:
                put(A, f"{Ln}.mlp_up", Lr["up"][tw, :WF], Lr["up"], [t0w, 0], f"{tag} up_proj out")
            put(A, f"{Ln}.mlp_act", Lr["act"][tw, :WF], Lr["act"], [t0w, 0], f"{tag} " + ("silu(gate) * up" if lm else "gelu_tanh(fc1)"))
            # weights
            if lm:
                La = lm_layers[l]
                at = La.self_attn
                qw, kw, vw = heads_w(at.q_proj.weight, H, HD), heads_w(at.k_proj.weight, KV, HD), heads_w(at.v_proj.weight, KV, HD)
                put(Wx, f"{Ln}.ln1_w", La.input_layernorm.weight, La.input_layernorm.weight, [0], "input RMSNorm γ")
                put(Wx, f"{Ln}.ln2_w", La.post_attention_layernorm.weight, La.post_attention_layernorm.weight, [0], "post-attn RMSNorm γ")
                put(Wx, f"{Ln}.q_norm", at.q_norm.weight, at.q_norm.weight, [0], "q_norm γ [head_dim]")
                put(Wx, f"{Ln}.k_norm", at.k_norm.weight, at.k_norm.weight, [0], "k_norm γ [head_dim]")
                ow, gw, uw, dw = at.o_proj.weight, La.mlp.gate_proj.weight, La.mlp.up_proj.weight, La.mlp.down_proj.weight
                put(Wx, f"{Ln}.up_w", uw[:WF, :WC], uw, [0, 0], "up_proj [FF, C]")
            else:
                pre = Lr["pre"]
                qw = heads_w(Wt(pre + ".attn1.to_q.weight"), H_V, D_V)[:1]
                kw = heads_w(Wt(pre + ".attn1.to_k.weight"), H_V, D_V)[:1]
                vw = heads_w(Wt(pre + ".attn1.to_v.weight"), H_V, D_V)[:1]
                for nm, key in (("q_b", "to_q"), ("k_b", "to_k"), ("v_b", "to_v")):
                    bfull = Wt(pre + f".attn1.{key}.bias").view(H_V, D_V)
                    put(Wx, f"{Ln}.{nm}", bfull[:1, :WD], bfull, [0, 0], f"{key} bias [head, head_dim]")
                for nm, key in (("ln1", "norm1"), ("ln2", "norm3")):
                    put(Wx, f"{Ln}.{nm}_w", Wt(pre + f".{key}.weight"), Wt(pre + f".{key}.weight"), [0], f"{key} LayerNorm γ")
                    put(Wx, f"{Ln}.{nm}_b", Wt(pre + f".{key}.bias"), Wt(pre + f".{key}.bias"), [0], f"{key} LayerNorm β")
                ow, gw, dw = Wt(pre + ".attn1.to_out.0.weight"), Wt(pre + ".ff.net.0.proj.weight"), Wt(pre + ".ff.net.2.weight")
            put(Wx, f"{Ln}.q_w", qw[:, :WD, :WC], qw, [0, 0, 0], "q weight [head, head_dim, C]")
            put(Wx, f"{Ln}.k_w", kw[:, :WD, :WC], kw, [0, 0, 0], "k weight [kv_head, head_dim, C]")
            put(Wx, f"{Ln}.v_w", vw[:, :WD, :WC], vw, [0, 0, 0], "v weight [kv_head, head_dim, C]")
            put(Wx, f"{Ln}.o_w", ow[:WC, :WC], ow, [0, 0], "output projection [C, H·head_dim]")
            put(Wx, f"{Ln}.gate_w", gw[:WF, :WC], gw, [0, 0], "gate_proj [FF, C]" if lm else "ff fc1 [FF, C]")
            put(Wx, f"{Ln}.down_w", dw[:WC, :WF], dw, [0, 0], "down_proj [C, FF]" if lm else "ff fc2 [C, FF]")

        agg, ms = rms_agg(lm_out)
        put(A, "lnf_agg", agg[tw], ms, [t0w, 0], "backbone final RMSNorm agg over the full 2048")
        nw = bb.model.language_model.norm.weight
        put(Wx, "lnf_w", nw, nw, [0], "backbone final RMSNorm γ")
        put(A, "lnf_out", lnf[tw, :WC], lnf, [t0w, 0], "backbone_features = final RMSNorm(LM layer 15 out) [S, 2048]")
        agg, mu = ln_agg(feats, 1e-5)
        put(A, "vlln_agg", agg[tw], mu, [t0w, 0], "vlln LayerNorm agg over the full 2048: mean | 1/sqrt(var+1e-5)")
        put(Wx, "vlln_w", Wt(P + "vlln.weight"), Wt(P + "vlln.weight"), [0], "vlln LayerNorm γ")
        put(Wx, "vlln_b", Wt(P + "vlln.bias"), Wt(P + "vlln.bias"), [0], "vlln LayerNorm β")
        put(A, "vlln_out", vlln[tw, :WC], vlln, [t0w, 0], "vlln(backbone_features) -> vl_self_attention")

        # expert = DiT anatomy at Euler step 0 (t = 0, x_t = the noise)
        s0 = steps[0]
        put(A, "expert_in", s0["sa"][:, :WC], s0["sa"], [0, 0], "DiT input [41, 1536] = [state token | 40 action tokens]")
        for b, Lr in enumerate(s0["layers"]):
            En = f"E{b}"
            pre = P + f"model.transformer_blocks.{b}"
            cross = Lr["kind"] != "self"
            tag = f"DiT block {b} ({Lr['kind']})"
            agg, mu = ln_agg(Lr["x_in"], 1e-5)
            put(A, f"{En}.ln1_agg", agg, mu, [0, 0], "AdaLayerNorm agg over the full 1536: mean | 1/sqrt(var+1e-5)")
            put(A, f"{En}.ln1", Lr["n1"][:, :WC], Lr["n1"], [0, 0], f"{tag} LN(x)(1+scale)+shift")
            ada = torch.stack([Lr["scale"], Lr["shift"]])
            put(A, f"{En}.ada", ada[:, :WC], ada, [0, 0], f"{tag} norm1.linear(SiLU(temb)) = [scale, shift] [2, 1536]")
            q, k, v = Lr["q"][:1], Lr["k"][:1], Lr["v"][:1]
            put(A, f"{En}.q", q[:, :, :WD], q, [0, 0, 0], f"{tag} Q = to_q(norm1) + bias (head 0 of 32)")
            ks = (slice(None), tw if cross else slice(None), slice(0, WD))
            koff = [0, t0w, 0] if cross else [0, 0, 0]
            kd = "over the VL tokens (to_k(vl_self_attention out))" if cross else "over the 41 DiT tokens"
            put(A, f"{En}.k", k[ks], k, koff, f"{tag} K {kd}")
            put(A, f"{En}.v", v[ks], v, koff, f"{tag} V {kd}")
            sc, pr = Lr["sc"][:1], Lr["probs"][:1]
            if cross:
                md = "text-only keys (image = -inf)" if Lr["kind"] == "cross_text" else "image-only keys (text = -inf)"
                put(A, f"{En}.scp", sc[:, :, tw], sc, [0, 0, t0w], f"{tag} q.k/sqrt(48) on the VL window keys, {md}")
                put(A, f"{En}.pp", pr[:, :, tw], pr, [0, 0, t0w], f"{tag} probs on the VL window keys (rows over all {S})")
            else:
                put(A, f"{En}.scs", sc, sc, [0, 0, 0], f"{tag} q.k/sqrt(48) over the 41 DiT tokens (no mask)")
                put(A, f"{En}.ps", pr, pr, [0, 0, 0], f"{tag} probs over the 41 DiT tokens")
            agg, mx = sm_agg(sc)
            put(A, f"{En}.attn_agg", agg, mx, [0, 0, 0], "softmax agg over the full key row: max | 1/sum exp(s-max)")
            enc = Lr["enc"][:1]
            put(A, f"{En}.vo", enc[:, :, :WD], enc, [0, 0, 0], f"{tag} per-head sum p.V")
            put(A, f"{En}.attn", Lr["attn"], Lr["attn"], [0, 0], "head-mean attention of the 41 DiT queries: VL window | 41 DiT keys")
            put(A, f"{En}.attn_mass", Lr["mass4"], Lr["mass4"], [0, 0], "attention mass on [state token | image | text | action tokens]")
            agg, mu = ln_agg(Lr["r1"], 1e-5)
            put(A, f"{En}.ln2_agg", agg, mu, [0, 0], "norm3 LayerNorm agg (no affine) over the full 1536")
            for nm, kk, d in (("o", "ao", "to_out out (incl. bias)"), ("res1", "r1", "x + attn"),
                              ("ln2", "n3", "norm3 = LN(x) (no affine, no time)"), ("mlp", "ff", "fc2(gelu_tanh(fc1))"),
                              ("out", "out", "x + MLP")):
                put(A, f"{En}.{nm}", Lr[kk][:, :WC], Lr[kk], [0, 0], f"{tag} {d}")
            for nm, kk, d in (("gate", "gate", "ff fc1 out (pre-GELU)"), ("act", "act", "gelu_tanh(fc1)")):
                put(A, f"{En}.{nm}", Lr[kk][:, :WF], Lr[kk], [0, 0], f"{tag} {d} [41, 6144]")
            kin = 2048 if cross else DIT
            for nm, key in (("q", "to_q"), ("k", "to_k"), ("v", "to_v")):
                wf = heads_w(Wt(pre + f".attn1.{key}.weight"), H_A, D_A)[:1]
                put(Wx, f"{En}.{nm}_w", wf[:, :WD, :WC], wf, [0, 0, 0], f"{key} [head, head_dim, {kin if nm != 'q' else DIT}]")
                bfull = Wt(pre + f".attn1.{key}.bias").view(H_A, D_A)
                put(Wx, f"{En}.{nm}_b", bfull[:1, :WD], bfull, [0, 0], f"{key} bias [head, head_dim]")
            for nm, key, sl in (("o_w", "attn1.to_out.0", (slice(0, WC), slice(0, WC))),
                                ("gate_w", "ff.net.0.proj", (slice(0, WF), slice(0, WC))),
                                ("down_w", "ff.net.2", (slice(0, WC), slice(0, WF)))):
                wf = Wt(pre + f".{key}.weight")
                put(Wx, f"{En}.{nm}", wf[sl], wf, [0, 0], f"{key} {list(wf.shape)} as [out, in]")
        agg, mu = ln_agg(s0["x_out"], 1e-6)
        put(A, "expert_out_agg", agg, mu, [0, 0], "final LayerNorm agg over the full 1536 (eps 1e-6)")
        put(A, "expert_out", s0["final"][:, :WC], s0["final"], [0, 0], "LN(x)(1+scale)+shift, [shift, scale] = proj_out_1(SiLU(temb))")
        fa = torch.stack([s0["final_scale"], s0["final_shift"]])
        put(A, "expert_final_ada", fa[:, :WC], fa, [0, 0], "proj_out_1(SiLU(temb)) chunks as [scale, shift] (source order: shift, scale)")
        po2 = Wt(P + "model.proj_out_2.weight")
        put(Wx, "out_proj_w", po2[:WC, :WC], po2, [0, 0], "proj_out_2 [1024, 1536] as [out, in]")
        put(A, "dit_out", s0["dit_out"][:, :WC], s0["dit_out"], [0, 0], "proj_out_2 out [41, 1024]")
        d1 = Wt(P + "action_decoder.layer1.W")[slot].T
        put(Wx, "dec_l1_w", d1[:WC, :WC], d1, [0, 0], f"action_decoder.layer1.W[slot {slot}] [1024, 1024] as [out, in]")
        put(A, "dec_h", s0["dec_h"][:, :WC], s0["dec_h"], [0, 0], "ReLU(action_decoder.layer1) [41, 1024]")
        d2 = Wt(P + "action_decoder.layer2.W")[slot].T
        put(Wx, "action_w", d2[:BA], d2, [0, 0], f"action_decoder.layer2.W[slot {slot}] [132, 1024] as [out, in] (rows 0..22)")
        d2b = Wt(P + "action_decoder.layer2.b")[slot]
        put(Wx, "action_b", d2b[:BA], d2b, [0], "action_decoder.layer2.b (rows 0..22)")
        put(A, "action_x1", noise[:, :BA], noise, [0, 0], f"x_0 = randn(40, 132), seed {SEED} (dims 0..22 shown)")
        put(A, "action_velocity", s0["v"][:, :BA], s0["v"], [0, 0], "v(x_0, t=0) = action_decoder(...)[-40:] [40, 132]")
        c1 = noise + s0["v"]
        put(A, "action_chunk", c1[:, :BA], c1, [0, 0], "one Euler step over the whole interval: x_0 + v (K = 1)")

        # suffix inputs (state / action / time encoders) at step 0
        put(A, "fm_state75", state[:61], state, [0], "BEHAVIOR 61-dim proprio, q01/q99 -> [-1,1], zero-padded to 132")
        w1 = Wt(P + "state_encoder.layer1.W")[slot].T
        put(A, "fm_state_proj_w", w1[:WC, :61], w1, [0, 0], f"state_encoder.layer1.W[slot {slot}] [1024, 132] as [out, in]")
        put(A, "fm_state_hidden", st_h[:WC], st_h, [0], "ReLU(state_encoder.layer1(state)) [1024]")
        w2 = Wt(P + "state_encoder.layer2.W")[slot].T
        put(A, "fm_state_l2_w", w2[:WC, :WC], w2, [0, 0], f"state_encoder.layer2.W[slot {slot}] [1536, 1024] as [out, in]")
        put(A, "fm_state_emb", st_feat[:WC], st_feat, [0], "state_encoder(state) = DiT token 0 [1536]")
        put(A, "fm_action_emb", s0["a_emb"][:, :WC], s0["a_emb"], [0, 0], "action_encoder.W1(x_t) [40, 1536]")
        put(A, "fm_time_emb", s0["tau"][:WC], s0["tau"], [0], f"sincos(t_disc={s0['t_disc']}, 1536) (same for all 40 rows)")
        put(A, "fm_time_mlp_hidden", s0["h2s"][:, :WC], s0["h2s"], [0, 0], "swish(W2(cat[a_emb, tau])) [40, 1536]")
        put(A, "fm_action_feat", s0["a_feat"][:, :WC], s0["a_feat"], [0, 0], "action_encoder.W3(.) [40, 1536]")
        put(A, "fm_pos_emb", pos[:, :WC], pos, [0, 0], "position_embedding.weight[:40] (learned) [40, 1536]")
        put(A, "fm_action_time", s0["a_feat_pos"][:, :WC], s0["a_feat_pos"], [0, 0], "W3(.) + pos = DiT tokens 1..40")
        for nm, key in (("fm_action_in_w", "W1"), ("fm_time_in_w", "W2"), ("fm_time_out_w", "W3")):
            w = Wt(P + f"action_encoder.{key}.W")[slot].T
            put(A, nm, w[:WC, :(BA if key == "W1" else WC)], w, [0, 0], f"action_encoder.{key}.W[slot {slot}] {list(w.shape)} as [out, in]")
        put(A, "fm_t_proj", s0["tp"][:WC], s0["tp"], [0], f"Timesteps(256)(t_disc={s0['t_disc']}) [256]")
        tw1 = Wt(P + "model.timestep_encoder.timestep_embedder.linear_1.weight")
        put(A, "fm_temb_w1", tw1[:WC, :WC], tw1, [0, 0], "timestep_embedder.linear_1 [1536, 256]")
        put(A, "fm_temb_hidden", s0["th"][None, :WC], s0["th"][None], [0, 0], "SiLU(linear_1(t_proj)) [1, 1536]")
        tw2 = Wt(P + "model.timestep_encoder.timestep_embedder.linear_2.weight")
        put(A, "fm_temb_w2", tw2[:WC, :WC], tw2, [0, 0], "timestep_embedder.linear_2 [1536, 1536]")
        put(A, "fm_cond", s0["temb"][None, :WC], s0["temb"][None], [0, 0], "temb = linear_2(.) [1, 1536]: the AdaLN input of every block")
        pt = torch.stack([torch.stack([p["x0"], p["v"], p["chunk"]]) for p in parts])[..., :BA]
        put(A, "fm_particles", pt, pt, [0, 0, 0, 0], f"{NPART} seeds at t=0: x_0 | v | x_0 + v [K, 3, 40, 23]")
        pms = torch.stack([p["mass"] for p in parts])
        put(A, "fm_particles_mass", pms, pms, [0, 0, 0, 0], f"{NPART} seeds: per block, per DiT query attention mass [K, 32, 41, 4]")
        mc = pt[:, 2].mean(0)
        put(A, "fm_mean_chunk", mc, mc, [0, 0], "mean of the particles' x_0 + v")

    fm_steps = []
    for s in steps:
        Tt = {"x": ent16(s["x_t"][:, :BA], "x_t [40, 132]", s["x_t"]), "v": ent16(s["v"][:, :BA], "v [40, 132]", s["v"]),
              "x_next": ent16(s["x_next"][:, :BA], "x_t + v/4", s["x_next"]),
              "expert_in": ent16(s["sa"][:, :WC], "DiT input", s["sa"]),
              "expert_out": ent16(s["final"][:, :WC], "final AdaLN out", s["final"]),
              "fm_time_emb": ent16(s["tau"][:WC], "sincos(t_disc)", s["tau"]),
              "fm_action_emb": ent16(s["a_emb"][:, :WC], "W1(x_t)", s["a_emb"]),
              "fm_time_mlp_hidden": ent16(s["h2s"][:, :WC], "swish(W2(cat[a_emb, tau]))", s["h2s"]),
              "fm_action_feat": ent16(s["a_feat"][:, :WC], "W3(.)", s["a_feat"]),
              "fm_action_time": ent16(s["a_feat_pos"][:, :WC], "W3(.) + pos", s["a_feat_pos"]),
              "fm_t_proj": ent16(s["tp"][:WC], "Timesteps(256)(t_disc)", s["tp"]),
              "fm_temb_hidden": ent16(s["th"][None, :WC], "SiLU(linear_1)", s["th"][None]),
              "fm_cond": ent16(s["temb"][None, :WC], "temb", s["temb"][None]),
              "expert_final_ada": ent16(torch.stack([s["final_scale"], s["final_shift"]])[:, :WC], "final [scale, shift]"),
              "dit_out": ent16(s["dit_out"][:, :WC], "proj_out_2 out", s["dit_out"]),
              "dec_h": ent16(s["dec_h"][:, :WC], "ReLU(decoder.layer1)", s["dec_h"])}
        for b, Lr in enumerate(s["layers"]):
            for nm, kk in (("ln1", "n1"), ("o", "ao"), ("res1", "r1"), ("ln2", "n3"), ("mlp", "ff"), ("out", "out")):
                Tt[f"E{b}.{nm}"] = ent16(Lr[kk][:, :WC], f"DiT block {b} {nm}", Lr[kk])
            Tt[f"E{b}.attn"] = ent16(Lr["attn"], f"block {b} head-mean attention")
            Tt[f"E{b}.mass"] = ent16(Lr["mass4"], f"block {b} attention mass")
            ada = torch.stack([Lr["scale"], Lr["shift"]])
            Tt[f"E{b}.ada"] = ent16(ada[:, :WC], f"block {b} [scale, shift]", ada)
        fm_steps.append({"k": s["k"], "t": s["t"], "t_disc": s["t_disc"], "tensors": Tt})

    def nparams(prefix):
        tot = 0
        from safetensors import safe_open
        for k in Wt.keys(prefix):
            fn = Wt.map[k]
            if fn not in Wt.files:
                Wt.files[fn] = safe_open(os.path.join(MODEL, fn), framework="pt", device="cpu")
            tot += math.prod(Wt.files[fn].get_slice(k).get_shape())
        return tot

    slot_std = {}
    for key in ("state_encoder.layer1", "action_encoder.W1", "action_encoder.W2", "action_decoder.layer2"):
        Wf = Wt(P + key + ".W")
        slot_std[key] = [float(Wf[i].std()) for i in range(Wf.shape[0])]
    kinds = [L_["kind"] for L_ in s0["layers"]]
    action_n = action.cpu().numpy()
    pchunk = pt[:, 2]
    config = {
        "model_type": "groot_n1d7", "name": "GR00T N1.7-3B (BEHAVIOR R1 Pro slot)", "checkpoint": MODEL,
        "step": open(os.path.join(MODEL, "latest")).read().strip(),
        "variant": {
            "id": "groot", "title": "GR00T N1.7",
            "trunk_qk_norm": True, "trunk_causal": True, "trunk_mlp": "swiglu", "trunk_heads": False,
            "embed_scale": 1, "expert_bias": True, "expert_ada": "adaln", "expert_mlp": "gelu",
            "expert_final_ada": True, "state_token": True, "time_into_tokens": True, "ada_rows": 2,
            "trunk_layer_kinds": ["lm"] * SEL + ["vlsa"] * NV, "expert_layer_kinds": kinds,
        },
        "n_layer": SEL + NV, "n_head": H, "n_kv_head": KV, "head_dim": HD, "n_embd": C, "intermediate": FF,
        "vocab_size": int(emb_table.shape[0]), "rms_norm_eps": EPS,
        "vlsa": {"n_layer": NV, "heads": H_V, "head_dim": D_V, "intermediate": int(Wt(trunk[SEL]["pre"] + ".ff.net.0.proj.weight").shape[0])},
        "expert_hidden": DIT, "expert_heads": H_A, "expert_kv_heads": H_A, "expert_head_dim": D_A,
        "expert_intermediate": int(Wt(P + "model.transformer_blocks.0.ff.net.0.proj.weight").shape[0]),
        "dit_out_dim": cfg["hidden_size"],
        "T_prefix": S, "T_text": S, "n_state_tokens": 0, "text_start": 0, "n_suffix": NS, "L_joint": S + NS,
        "n_visual_tokens": int(image_mask.sum()), "p_star": S - 1, "boundary_column": S - 1,
        "window": {"t0": t0w, "Wt": WT, "Wc": WC, "Wd": WD, "Wf": WF, "Wv": WT, "v0": 0, "embed_gathered": True},
        "window_token_ids": wids, "window_tokens": pieces[t0w:],
        "window_is_image": [bool(image_mask[i]) for i in range(t0w, S)],
        "knack_names": [], "done_names": [],
        "action_dim": BA, "action_horizon": HOR, "max_action_dim": ADIM, "max_state_dim": cfg["max_state_dim"],
        "action_dim_names": ACTION_DIM_NAMES, "behavior_action_dim": BA, "state_groups": STATE_GROUPS,
        "expert_n_layer": NB, "n_layer_pairs": NB, "suffix_visible_prefix": S, "suffix_visible_visual": int(image_mask.sum()),
        "n_params": nparams(""), "n_params_trunk": nparams("backbone.model.model.language_model."),
        "n_params_expert": nparams("action_head.model."), "n_params_vision": nparams("backbone.model.model.visual."),
        "n_params_vlsa": nparams("action_head.vl_self_attention."),
        "sample": {"row": None, "episode": args.episode, "frame": args.frame, "goal_text": language,
                   "target_knack": None, "target_done": None, "task_index": fr["task"]["task_index"]},
        "pred_knack": None, "pred_done": None, "prompt": text,
        "action_norm": action_n[:, :BA].tolist(),
        "embodiment": {"tag": EMB_TAG, "slot": slot, "trained": False, "slot_std": slot_std},
        "fm": {"t": 0.0, "steps": K, "dt": 1.0 / K, "t_list": [s["t"] for s in steps], "t_end": 1.0,
               "t_disc": [s["t_disc"] for s in steps], "num_timestep_buckets": NBUCK,
               "seeds": list(range(NPART)), "n_particles": NPART,
               "particle_chunk_std": float(pchunk.std(0).mean()), "asserts": asserts},
        "notes": [
            "weights + activations are real: local GR00T-N1.7-3B safetensors (global_step " +
            open(os.path.join(MODEL, "latest")).read().strip() + "), BEHAVIOR 2026-challenge-demos frame "
            "(same episode/frame as the k005b export)",
            "backbone config/processor: local Qwen3-VL-2B-Instruct (same architecture, shapes match exactly); "
            "weights are GR00T's own backbone.* tensors; the LM is truncated to select_layer=16 layers",
            "deepstack: vision blocks 5/11/17 are added to the image rows after LM layers 0/1/2 (outside the layer); "
            "L0..L2 mlp_resid is the layer output before that add",
            f"embodiment slot {slot} ({EMB_TAG}) is untrained (per-slot std = init 0.02): the state/action "
            "encoders and the action decoder are real-but-untrained weights; the decoded actions carry no meaning",
            "state = BEHAVIOR 61-dim proprio (q01/q99 -> [-1,1]) in slots 0..60 of 132; actions: dims 0..22 of 132 shown",
            "only head 0 of the 32 vl_self_attention / DiT heads is exported; to_out / ff biases are folded into the outputs",
            "compute in float32 from bf16 weights (official runtime: bf16); asserts vs transformers and the official "
            "Isaac-GR00T AlternateVLDiT / SelfAttentionTransformer",
            f"noise seed {SEED}; Euler integrates upward t = 0, .25, .5, .75 (t=0 is noise), x += v/4",
        ],
    }
    os.makedirs(out_dir, exist_ok=True)
    Wx["config"] = config
    A["config"] = config
    fm_out = {"K": K, "checkpoint_step": config["step"], "in_distribution": False, "steps": fm_steps}
    for fn, obj in (("groot-weights.json", Wx), ("groot-acts.json", A), ("groot-fm.json", fm_out)):
        path = os.path.join(out_dir, fn)
        with open(path, "w") as f:
            json.dump(obj, f)
        print(f"[write] {path} {os.path.getsize(path) / 1e6:.1f} MB", flush=True)

    check = {
        "W.E0.q_w[h0,d5,c7]": float(Wt(P + "model.transformer_blocks.0.attn1.to_q.weight")[5, 7]),
        "A.E0.ada[scale,c7]": float(s0["layers"][0]["scale"][7]),
        "A.L5.x_in[t=S-1,c11]": float(cap["x_in"][5][S - 1, 11]),
        "A.L19.out[t=S-1,c2]": float(vl_tok[S - 1, 2]),
        "A.s1.v[0,:5]": steps[1]["v"][0, :5].tolist(),
        "A.action_norm[0,:5]": action_n[0, :5].tolist(),
        "asserts": asserts,
    }
    with open(os.path.join(out_dir, "groot-check.json"), "w") as f:
        json.dump(check, f, indent=1)
    print("[check]", json.dumps(check), flush=True)
    print(f"[done] total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
