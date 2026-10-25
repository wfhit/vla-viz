#!/usr/bin/env python
"""Export REAL k005b weights + REAL activations for the vla-viz 3D view.

Loads the k005b fused VLA checkpoint on CPU (fp32), builds one real training row
exactly like the trainer (RowSet → collate_fused → fused_train_tensors), runs one
forward with hooks, and writes:

  public/k005b-weights.json  — real weight windows + full-tensor stats
  public/k005b-acts.json     — real activation windows + full-tensor stats
  public/k005b-check.json    — spot values used by the JS-vs-Python self test

Full tensors (~2.6B params; 16x1257^2 attention per layer) cannot ship to a
browser, so every big tensor is exported as a REAL contiguous sub-region (the
"window") plus `full_shape`, `offset` (window origin inside the full tensor) and
`stats` {min,max,mean,std} computed over the FULL tensor. Small tensors (norm γ,
knack/done heads, h*, action chunk) are exported whole.

Run with the behavior_challenge venv:
  /home/frank/behavior_challenge_ws/behavior_challenge/.venv/bin/python tools/export_k005b.py
"""
import argparse
import base64
import json
import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

BC_SRC = "/home/frank/behavior_challenge_ws/behavior_challenge/src"
DATA = "/home/frank/workspace/behavior_challenge_data"
DEFAULTS = dict(
    ckpt=f"{DATA}/outputs/k005b-task0-v3/checkpoints/step_130000/checkpoint.pt",
    sample_table="/home/frank/behavior_challenge_ws/outputs/train-data-task0-donehold-v3",
    dataset_root=f"{DATA}/dataset/behavior-1k-knack",
    vocab_cache=f"{DATA}/outputs/k005b-task0-v3-vocab-cache.json",
    processor="/home/frank/workspace/model/qwen/Qwen3-VL-2B-Instruct",
)

# window sizes (real sub-regions shipped to the browser)
WT = 32   # tokens, ending at the h* boundary token
WC = 64   # hidden channels [0, WC)
WD = 32   # head dims [0, WD)
WF = 64   # SwiGLU intermediate channels [0, WF)
WV = 64   # embed-table rows, starting at the h* token's id

# data/schema.py: base 3 + trunk 4 + L-arm 7 + L-grip 1 + R-arm 7 + R-grip 1 (dims 0-2 are
# velocities, 3-22 positions; fused_serve closes a gripper on negative values)
ACTION_DIM_NAMES = (["base_vx", "base_vy", "base_yaw"]
                    + [f"trunk_{i}" for i in range(1, 5)]
                    + [f"L_arm_j{i}" for i in range(1, 8)] + ["L_gripper"]
                    + [f"R_arm_j{i}" for i in range(1, 8)] + ["R_gripper"])


def stats(t):
    import torch
    t = t.detach().float()
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return dict(min=0.0, max=0.0, mean=0.0, std=0.0)
    return dict(min=float(finite.min()), max=float(finite.max()),
                mean=float(finite.mean()),
                std=float(finite.std()) if finite.numel() > 1 else 0.0)


FM_SEEDS = [None] + list(range(1001, 1008))   # None = the row's own x1 (batch action_noise)
FM_CF_STEPS = 10


def export_flow_matching(model, tensors, batch, put, expert, NL, T, H, HD, WC,
                         x1, vel, action, vis_cols, txt_cols, fba, orig_attn, tw):
    """The suffix inputs, per-layer AdaRMSNorm modulation, the training target, K=8 real
    noise particles (t=1 deploy forwards) and the 10-step counterfactual summary."""
    import torch
    import torch.nn.functional as F
    from model.one_step_actions import create_sinusoidal_pos_embedding

    P = model.projections
    x1, vel, action = x1.float(), vel.float(), action.float()
    s75 = tensors["action_state"][0].float()
    with torch.no_grad():
        pw = P.action_time_mlp_out.in_features
        te = create_sinusoidal_pos_embedding(torch.ones(1), pw, min_period=4e-3, max_period=4.0)[0]
        se = P.state_proj(s75)
        ae = P.action_in_proj(x1)
        hid = F.silu(P.action_time_mlp_in(torch.cat([ae, te[None].expand(ae.shape[0], pw)], -1)))
        at = P.action_time_mlp_out(hid)
    put("fm_state75", s75, s75, [0], "action_state: bounds99-normalised state slots, zero-padded to 75")
    put("fm_time_emb", te, te, [0], "sinusoidal time embedding of t=1 (sin|cos, 768)")
    put("fm_state_emb", se[:WC], se, [0], "state_proj(state75) = suffix token 0")
    put("fm_action_emb", ae[:, :WC], ae, [0, 0], "action_in_proj(x1) [16, 768]")
    put("fm_time_mlp_hidden", hid[:, :WC], hid, [0, 0], "silu(action_time_mlp_in(cat[action_emb, te])) [16, 768]")
    put("fm_action_time", at[:, :WC], at, [0, 0], "action_time_mlp_out(...) = suffix tokens 1..16")
    for nm, wt in (("fm_state_proj_w", P.state_proj.weight), ("fm_action_in_w", P.action_in_proj.weight),
                   ("fm_time_in_w", P.action_time_mlp_in.weight), ("fm_time_out_w", P.action_time_mlp_out.weight)):
        win = wt[:WC, :WC] if wt.shape[1] > 75 else wt[:WC]
        put(nm, win, wt, [0, 0], f"{nm[3:-2]} weight {list(wt.shape)}")

    # t's per-layer modulation: (1 + γ(te))·RMSNorm(x)·w + β(te), input + post-attn norms
    with torch.no_grad():
        for i in range(NL):
            lay = expert.layers[i]
            rows = []
            for nrm in (lay.input_layernorm, lay.post_attention_layernorm):
                rows += [nrm.gamma(te), nrm.beta(te)]
            ada = torch.stack(rows)               # [4, 768] γ_in | β_in | γ_post | β_post
            put(f"E{i}.ada", ada[:, :WC], ada, [0, 0],
                f"expert layer {i} AdaRMSNorm(t=1): rows γ_in, β_in, γ_post, β_post")

    tg = batch["action_target"][0].float()
    u = x1 - tg
    put("fm_target", tg, tg, [0, 0], "training label a (bounds99-normalised) [16, 23]")
    put("fm_u", u, u, [0, 0], "training target velocity u = x1 − a")
    mse = lambda a, b: float(((a - b) ** 2).mean())
    per_dim = ((action - tg) ** 2).mean(0)
    put("fm_err_dim", per_dim, per_dim, [0], "per-dim MSE(chunk, a) over the 16 steps")

    # K=8 real noise particles: each a full t=1 forward; capture prefix K/V + suffix mass
    cap = {"i": 0}

    def light_attn(q, k, v, mask):
        out = orig_attn(q, k, v, mask)
        i = cap["i"]
        kr = k[0].repeat_interleave(H // k.shape[2], dim=1)
        sq = torch.einsum("thd,shd->hts", q[0, T:], kr) / math.sqrt(HD)
        ps = torch.softmax(sq.masked_fill(~mask[0, T:][None], float("-inf")), -1).mean(0)
        cap["mass"].append(torch.stack([ps[:, :16].sum(-1), ps[:, :T][:, vis_cols].sum(-1),
                                        ps[:, :T][:, txt_cols].sum(-1), ps[:, T:].sum(-1)], -1))
        kv = torch.cat([k[0, :T], v[0, :T]], -1)
        if cap["ref"] is None:
            cap["kv"].append(kv.clone())
        else:
            cap["kvdiff"] = max(cap["kvdiff"], float((kv - cap["ref"][i]).abs().max()))
        cap["i"] = i + 1
        return out

    def run(xn, t=1.0):
        d = dict(tensors)
        d["noisy_action"] = xn[None]
        cap.update(i=0, mass=[])
        with torch.no_grad():
            return model(**d).action_velocity[0].float()

    fba.our_eager_attention_forward = light_attn
    cap.update(ref=None, kv=[], kvdiff=0.0)
    parts, masses = [], []
    t1 = time.time()
    try:
        for k, sd in enumerate(FM_SEEDS):
            xn = x1 if sd is None else torch.randn(x1.shape, generator=torch.Generator().manual_seed(sd))
            vk = run(xn)
            if k == 0:
                cap["ref"] = cap["kv"]
            parts.append(torch.stack([xn, vk, xn - vk]))
            masses.append(torch.stack(cap["mass"]))
    finally:
        fba.our_eager_attention_forward = orig_attn
    print(f"[fm] {len(FM_SEEDS)} particle forwards {time.time() - t1:.0f}s", flush=True)
    Pt = torch.stack(parts)           # [K, 3, 16, 23]  x1 | v | chunk
    Mt = torch.stack(masses)          # [K, NL, 17, 4]
    put("fm_particles", Pt, Pt, [0, 0, 0, 0], f"K={len(FM_SEEDS)} noise particles × [x1, v, chunk=x1−v] [16, 23]")
    put("fm_particles_mass", Mt, Mt, [0, 0, 0, 0],
        "per-particle per-layer suffix attention mass [state | visual | text | suffix]")
    chunks = Pt[:, 2]
    mean_chunk = chunks.mean(0)
    put("fm_mean_chunk", mean_chunk, mean_chunk, [0, 0], "mean over the K particle chunks")

    # K Euler steps x ← x − v/K, t = 1, 0.9, …, 0.1 (sample_action_chunk's loop, run as joint
    # forwards with flow_time = t_k — numerically the encode_prefix + predict_velocity path).
    # Per step: x_t, v, and every layer's suffix lane (the expert half of each layer block).
    # Steps ≥ 1 are only in-distribution for a checkpoint trained over t ∈ (0, 1].
    steps = {}
    scap = {}

    def lane(i, nm, x):
        x = x[0] if x.dim() == 3 else x
        scap[f"E{i}.{nm}"] = x.float()
        return x

    def void(_):
        return None

    def step_hooks(i, layer):
        hs = []
        ex = {}

        def pre(mod, a, kw):
            if kw.get("compute_qkv"):
                x = a[0]
                ex["x_in"] = x[0] if x.dim() == 3 else x
                if i == 0:
                    scap["expert_in"] = ex["x_in"].float()
        hs.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        hs.append(layer.input_layernorm.register_forward_hook(lambda m, a, o: void(lane(i, "ln1", o))))

        def o_hook(m, a, o):
            out = lane(i, "o", o)
            lane(i, "res1", out + ex["x_in"])
        hs.append(layer.self_attn.o_proj.register_forward_hook(o_hook))
        hs.append(layer.post_attention_layernorm.register_forward_hook(lambda m, a, o: void(lane(i, "ln2", o))))
        hs.append(layer.mlp.register_forward_hook(lambda m, a, o: void(lane(i, "mlp", o))))

        def post(mod, a, kw, o):
            if kw.get("output_atten"):
                x = o[0] if isinstance(o, (tuple, list)) else o
                lane(i, "out", x)
        hs.append(layer.register_forward_hook(post, with_kwargs=True))
        return hs

    def step_attn(q, k, v, mask):
        out = orig_attn(q, k, v, mask)
        i = scap["_i"]
        kr = k[0].repeat_interleave(H // k.shape[2], dim=1)
        sq = torch.einsum("thd,shd->hts", q[0, T:], kr) / math.sqrt(HD)
        ps = torch.softmax(sq.masked_fill(~mask[0, T:][None], float("-inf")), -1)
        pm = ps.mean(0)
        scap[f"E{i}.attn"] = torch.cat([pm[:, tw], pm[:, T:]], -1)
        scap[f"E{i}.mass"] = torch.stack([pm[:, :16].sum(-1), pm[:, :T][:, vis_cols].sum(-1),
                                          pm[:, :T][:, txt_cols].sum(-1), pm[:, T:].sum(-1)], -1)
        scap["_i"] = i + 1
        return out

    hs = [h for i in range(NL) for h in step_hooks(i, expert.layers[i])]

    def enorm(m, a, o):
        x = o[0] if isinstance(o, (tuple, list)) else o
        scap["expert_out"] = (x[0] if x.dim() == 3 else x).float()
    hs.append(expert.norm.register_forward_hook(enorm))
    fba.our_eager_attention_forward = step_attn
    cf = []
    t1 = time.time()
    try:
        xc = x1.clone()
        for s in range(FM_CF_STEPS):
            tk = 1.0 - s / FM_CF_STEPS
            d = dict(tensors)
            d["noisy_action"] = xc[None].to(tensors["noisy_action"].dtype)
            d["flow_time"] = torch.full_like(tensors["flow_time"], tk)
            scap.clear()
            scap["_i"] = 0
            with torch.no_grad():
                vc = model(**d).action_velocity[0].float()
            assert scap["_i"] == NL, scap["_i"]
            with torch.no_grad():  # the suffix input chain and the AdaRMSNorm γβ at this t_k
                tek = create_sinusoidal_pos_embedding(torch.full((1,), tk), pw, min_period=4e-3, max_period=4.0)[0]
                aek = P.action_in_proj(xc)
                hik = F.silu(P.action_time_mlp_in(torch.cat([aek, tek[None].expand(aek.shape[0], pw)], -1)))
                atk = P.action_time_mlp_out(hik)
                assert float((atk - scap["expert_in"][1:]).abs().max()) < 1e-3, "suffix tokens 1..16 mismatch"
                scap["fm_time_emb"] = tek
                scap["fm_action_emb"], scap["fm_time_mlp_hidden"], scap["fm_action_time"] = aek, hik, atk
                for i in range(NL):
                    lay = expert.layers[i]
                    scap[f"E{i}.ada"] = torch.stack([f(tek) for nrm in (lay.input_layernorm, lay.post_attention_layernorm)
                                                     for f in (nrm.gamma, nrm.beta)])
            if s == 0:  # step 0 (t = 1, x = x1) is the main forward exactly
                assert float((vc - vel).abs().max()) < 1e-3, "step 0 != main forward"
            xn = xc - vc / FM_CF_STEPS
            st = {"x": xc.clone(), "v": vc, "x_next": xn.clone()}
            for kname, val in scap.items():
                if not kname.startswith("_"):
                    wide = kname.split(".")[-1] in ("ln1", "o", "res1", "ln2", "mlp", "out", "ada") or kname in (
                        "expert_in", "expert_out", "fm_action_emb", "fm_time_mlp_hidden", "fm_action_time", "fm_time_emb")
                    st[kname] = val[..., :WC] if wide else val
            steps[s] = (tk, st)
            xc = xn
            cf.append({"t": tk, "vnorm": float(vc.norm()), "mse": mse(xc, tg)})
    finally:
        fba.our_eager_attention_forward = orig_attn
        for h in hs:
            h.remove()
    print(f"[fm] euler{FM_CF_STEPS} ({time.time() - t1:.0f}s) final mse {cf[-1]['mse']:.3f}", flush=True)

    asserts = {
        "prefix_kv_max_abs_diff_across_seeds": cap["kvdiff"],
        "particle0_v_matches_main": float((Pt[0, 1] - vel).abs().max()),
        "x1_minus_v_eq_chunk": float((x1 - vel - action).abs().max()),
        "mse_chunk_a": mse(action, tg), "mse_v_u": mse(vel, u),
    }
    assert asserts["prefix_kv_max_abs_diff_across_seeds"] < 1e-4, asserts
    assert asserts["particle0_v_matches_main"] < 1e-3, asserts
    assert abs(asserts["mse_chunk_a"] - asserts["mse_v_u"]) < 1e-5, asserts
    print("[fm] asserts", asserts, flush=True)
    return {
        "t": 1.0, "steps": 1, "min_period": 4e-3, "max_period": 4.0,
        "seeds": [0 if s is None else s for s in FM_SEEDS], "n_particles": len(FM_SEEDS),
        "mse": mse(action, tg), "mse_zero": float((tg ** 2).mean()),
        "mse_mean_chunk": mse(mean_chunk, tg),
        "mse_per_particle": [mse(c, tg) for c in chunks],
        "particle_chunk_std": float(chunks.std(0).mean()),
        "state_valid": int((s75 != 0).sum()),
        "cf_euler": cf, "asserts": asserts, "_steps": steps,
    }


def ent16(x, desc):
    """float16 entry (whole tensor) for the lazily loaded per-Euler-step file."""
    import torch
    w = x.detach().to(torch.float32).contiguous().cpu()
    return {
        "shape": list(w.shape), "dtype": "torch.float16",
        "data": base64.b64encode(w.numpy().astype("<f2").tobytes()).decode("ascii"),
        "stats": stats(w), "desc": desc,
    }


def ent(window, full_shape, offset, full_stats, desc):
    import torch
    w = window.detach().to(torch.float32).contiguous().cpu()
    return {
        "shape": list(w.shape),
        "dtype": "torch.float32",
        "data": base64.b64encode(w.numpy().astype("<f4").tobytes()).decode("ascii"),
        "full_shape": list(full_shape),
        "offset": list(offset),
        "stats": full_stats,
        "desc": desc,
    }


def main():
    ap = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        ap.add_argument("--" + k.replace("_", "-"), default=v)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public"))
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    out_dir = os.path.abspath(args.out)

    sys.path.insert(0, BC_SRC)
    os.chdir(BC_SRC)
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.set_num_threads(args.threads)
    from transformers import AutoProcessor
    from data.behavior_action_rows import RowSet
    from data.fused_batch import collate_fused
    from knacks.taxonomy import CANONICAL_KNACKS
    import model.fused_behavior_action as fba
    from training.fused_dual_head_sft import (SUPERSEDED_CKPT_KEYS, build_fused_model,
                                              fused_train_tensors)

    t0 = time.time()
    model = build_fused_model(260, 261, gradient_checkpointing=False)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = {k: v for k, v in ck["model"].items() if k not in SUPERSEDED_CKPT_KEYS}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[load] step={ck.get('step')} missing={missing} unexpected={unexpected} "
          f"({time.time() - t0:.0f}s)", flush=True)
    if missing or unexpected:
        raise SystemExit("checkpoint does not match the model exactly — refusing to export")
    ckpt_step = ck.get("step")
    del ck, state
    model = model.float().eval()

    rs = RowSet(args.sample_table, args.dataset_root, vocab_cache_path=args.vocab_cache)
    proc = AutoProcessor.from_pretrained(args.processor)
    proc.tokenizer.padding_side = "right"
    row = rs[args.row]
    batch = collate_fused([row], proc)
    image_token_id = getattr(proc, "image_token_id", None) or \
        proc.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    trunk = model.trunk.qwenvl.model
    expert = model.trunk.qwen_expert.model
    vcfg = model.trunk.config.vlm
    ecfg = model.trunk.config.expert
    NL = len(trunk.layers)
    H, KV, HD = vcfg.num_attention_heads, vcfg.num_key_value_heads, vcfg.head_dim
    C, FF = vcfg.hidden_size, vcfg.intermediate_size

    with torch.no_grad():
        tensors = fused_train_tensors(model, batch, image_token_id, "cpu")
    # step_130000 is the unstamped fixed-t=1 lineage: pin t=1, so noisy_action is the noise x1
    tensors["noisy_action"] = batch["action_noise"].to(tensors["text_embeds"].dtype)
    tensors["flow_time"] = torch.ones_like(tensors["flow_time"])
    text_ids = batch["text_input_ids"][0]
    LT = text_ids.shape[0]
    T = 16 + LT                         # trunk (prefix) stream length
    boundary = int(tensors["boundary_column"][0])
    p_star = 16 + boundary              # h* position in the prefix stream
    t0w = p_star - WT + 1
    assert t0w >= 16, "token window must sit inside the text"
    tw = slice(t0w, p_star + 1)
    print(f"[sample] row={args.row} episode={row.episode} frame={row.frame} T_prefix={T} "
          f"boundary={boundary} p*={p_star} window=[{t0w},{p_star}]", flush=True)

    # ---------------- capture ---------------- #
    acts = {}
    priv = {}
    ctx = {"i": 0}

    def put(name, window, full, offset, desc):
        acts[name] = ent(window, full.shape, offset, stats(full), desc)

    def rms_agg(x):  # x [T, C] → [Wt, 2]: mean(x²) over the FULL C | rsqrt(mean+eps)
        ms = x[tw].float().pow(2).mean(-1)
        return torch.stack([ms, torch.rsqrt(ms + vcfg.rms_norm_eps)], dim=-1)

    def put_agg(name, x):
        put(name, rms_agg(x), x.pow(2).mean(-1), [t0w, 0],
            "RMSNorm agg over the full C: mean(x²) | rsqrt(mean(x²)+eps); stats over mean(x²)[T]")

    def act2d(name, x, desc, cw=WC):  # x [T, D] trunk prefix
        put(name, x[tw, :cw], x, [t0w, 0], desc)

    hooks = []

    def trunk_hooks(i, layer):
        def pre(mod, a, kw):
            x = a[0][0]
            if kw.get("compute_qkv"):
                ctx["i"] = i
                act2d(f"L{i}.x_in", x, f"layer {i} residual stream in")
                put_agg(f"L{i}.ln1_agg", x)
            else:
                ctx["x_in"] = x
        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        hooks.append(layer.input_layernorm.register_forward_hook(
            lambda m, a, o: act2d(f"L{i}.ln1_out", o[0], f"layer {i} input RMSNorm out")))

        def o_hook(m, a, o):
            out = o[0]
            act2d(f"L{i}.o_out", out, f"layer {i} o_proj out")
            res = out + ctx["x_in"]
            act2d(f"L{i}.attn_resid", res, f"layer {i} attention residual")
            put_agg(f"L{i}.ln2_agg", res)
            ctx["attn_resid"] = res
        hooks.append(layer.self_attn.o_proj.register_forward_hook(o_hook))
        hooks.append(layer.post_attention_layernorm.register_forward_hook(
            lambda m, a, o: act2d(f"L{i}.ln2_out", o[0], f"layer {i} post-attn RMSNorm out")))

        def mlp_hook(m, a, o):
            xn = a[0][0]
            g = m.gate_proj(xn)
            u = m.up_proj(xn)
            act2d(f"L{i}.mlp_gate", g, f"layer {i} gate_proj out", WF)
            act2d(f"L{i}.mlp_up", u, f"layer {i} up_proj out", WF)
            act2d(f"L{i}.mlp_act", F.silu(g) * u, f"layer {i} silu(gate)·up", WF)
            act2d(f"L{i}.mlp_out", o[0], f"layer {i} down_proj out")
            act2d(f"L{i}.mlp_resid", o[0] + ctx["attn_resid"], f"layer {i} MLP residual")
        hooks.append(layer.mlp.register_forward_hook(mlp_hook))

    for i, layer in enumerate(trunk.layers):
        trunk_hooks(i, layer)

    def lnf_hook(m, a, o):
        act2d("lnf_out", o[0], "final trunk RMSNorm (ln_f) out")
        put_agg("lnf_agg", a[0][0])
        priv["lnf_full"] = o[0]
    hooks.append(trunk.norm.register_forward_hook(lnf_hook))
    def enorm_hook(m, a, o):
        priv["expert_out"] = o[0]
        x = a[0][0] if a[0].dim() == 3 else a[0]
        ms = x.float().pow(2).mean(-1)
        put("expert_out_agg", torch.stack([ms, torch.rsqrt(ms + ecfg.rms_norm_eps)], dim=-1), ms, [0, 0],
            "expert final RMSNorm agg over the full C: mean(x²) | rsqrt(mean(x²)+eps)")
    hooks.append(expert.norm.register_forward_hook(enorm_hook))

    # the action expert's own stream: suffix in, and each run layer's suffix output
    def expert_in_pre(mod, a, kw):
        if kw.get("compute_qkv"):
            x = a[0][0]
            put("expert_in", x[:, :WC], x, [0, 0],
                f"expert suffix input [state_proj(state75) | action_time_mlp(x1, t=1)] [{x.shape[0]}, {x.shape[1]}]")
    hooks.append(expert.layers[0].register_forward_pre_hook(expert_in_pre, with_kwargs=True))

    # each expert layer's sublayers (TwoPhaseDecoderLayer): AdaRMSNorm → [joint attn] → o_proj →
    # +x_in → AdaRMSNorm → SwiGLU MLP → +res1; windows are [n_suf, :WC]
    def e2d(name, x, desc):
        x = x[0] if x.dim() == 3 else x
        put(name, x[:, :WC], x, [0, 0], desc)
        return x

    def expert_hooks(i, layer):
        ectx = {}

        def eagg(name, x):  # [n_suf, C] → [n_suf, 2]: mean(x²) over the FULL 768 | rsqrt(mean+eps)
            ms = x.float().pow(2).mean(-1)
            put(name, torch.stack([ms, torch.rsqrt(ms + ecfg.rms_norm_eps)], dim=-1), ms, [0, 0],
                "AdaRMSNorm agg over the full C: mean(x²) | rsqrt(mean(x²)+eps); stats over mean(x²)")

        def pre(mod, a, kw):
            x = a[0]
            x = x[0] if x.dim() == 3 else x
            if kw.get("compute_qkv"):
                ectx["x_in"] = x
                eagg(f"E{i}.ln1_agg", x)
        hooks.append(layer.register_forward_pre_hook(pre, with_kwargs=True))
        def ln1_hook(m, a, o):
            e2d(f"E{i}.ln1", o, f"expert layer {i} input AdaRMSNorm(te) out")
        hooks.append(layer.input_layernorm.register_forward_hook(ln1_hook))

        def o_hook(m, a, o):
            out = e2d(f"E{i}.o", o, f"expert layer {i} o_proj out (suffix rows of the joint attention)")
            res = out + ectx["x_in"]
            e2d(f"E{i}.res1", res, f"expert layer {i} attention residual (x_in + o_proj)")
            eagg(f"E{i}.ln2_agg", res)
            ectx["res1"] = res
        hooks.append(layer.self_attn.o_proj.register_forward_hook(o_hook))
        def ln2_hook(m, a, o):
            e2d(f"E{i}.ln2", o, f"expert layer {i} post-attn AdaRMSNorm(te) out")
        hooks.append(layer.post_attention_layernorm.register_forward_hook(ln2_hook))

        def mlp_hook(m, a, o):
            xn = a[0]
            g, u = m.gate_proj(xn), m.up_proj(xn)
            e2d(f"E{i}.gate", g, f"expert layer {i} gate_proj out")
            e2d(f"E{i}.up", u, f"expert layer {i} up_proj out")
            e2d(f"E{i}.act", F.silu(g) * u, f"expert layer {i} silu(gate)·up")
            e2d(f"E{i}.mlp", o, f"expert layer {i} SwiGLU MLP out (down_proj)")
            ectx["mlp"] = o[0] if o.dim() == 3 else o
        hooks.append(layer.mlp.register_forward_hook(mlp_hook))

        def post(mod, a, kw, o):
            if kw.get("output_atten"):
                x = o[0] if isinstance(o, (tuple, list)) else o
                x = x[0] if x.dim() == 3 else x
                assert torch.allclose(x.float(), (ectx["res1"] + ectx["mlp"]).float(), atol=1e-3), \
                    f"expert layer {i}: out != res1 + mlp"
                put(f"E{i}.out", x[:, :WC], x, [0, 0], f"expert layer {i} suffix output (res1 + MLP)")
        hooks.append(layer.register_forward_hook(post, with_kwargs=True))

    for i in range(NL):
        expert_hooks(i, expert.layers[i])

    orig_attn = fba.our_eager_attention_forward
    scale = 1.0 / math.sqrt(HD)

    def attn_capture(q, k, v, mask):
        out = orig_attn(q, k, v, mask)
        i = ctx["i"]
        # q,k,v: [1, L, heads, D] post q/k-norm + RoPE; k/v carry 8 kv heads
        qp, kp, vp = q[0, :T], k[0, :T], v[0, :T]
        put(f"L{i}.q", qp[tw, :, :WD].permute(1, 0, 2), qp.permute(1, 0, 2), [0, t0w, 0],
            f"layer {i} Q (q_norm + RoPE) [head, token, dim]")
        put(f"L{i}.k", kp[tw, :, :WD].permute(1, 0, 2), kp.permute(1, 0, 2), [0, t0w, 0],
            f"layer {i} K (k_norm + RoPE) [kv_head, token, dim]")
        put(f"L{i}.v", vp[tw, :, :WD].permute(1, 0, 2), vp.permute(1, 0, 2), [0, t0w, 0],
            f"layer {i} V [kv_head, token, dim]")
        kr = k[0].repeat_interleave(H // k.shape[2], dim=1)  # [L, H, D] (q head h ↔ kv h//2)
        s = torch.einsum("thd,shd->hts", q[0, tw], kr) * scale  # [H, Wt, L] — full key range
        s = s.masked_fill(~mask[0, tw][None], float("-inf"))
        mx = s.max(-1).values
        p = torch.softmax(s, dim=-1)
        inv_sum = 1.0 / torch.exp(s - mx[..., None]).sum(-1)
        put(f"L{i}.attn_scores", s[:, :, tw], s, [0, t0w, t0w],
            f"layer {i} q·k/√{HD} (masked=-inf) rows=query cols=key; stats over [H, Wt, L_joint]")
        put(f"L{i}.attn_agg", torch.stack([mx, inv_sum], dim=-1), mx, [0, t0w, 0],
            "softmax agg over the FULL key row: max | 1/Σexp(s−max)")
        put(f"L{i}.attn_probs", p[:, :, tw], p, [0, t0w, t0w],
            f"layer {i} attention probs (each row normalised over all {mask.shape[-1]} keys)")
        o = out[0, :T].view(T, H, HD)
        put(f"L{i}.attn_out", o[tw, :, :WD].permute(1, 0, 2), o.permute(1, 0, 2), [0, t0w, 0],
            f"layer {i} per-head attention output Σp·V [head, token, dim]")
        priv[f"p_star_probs_{i}"] = p[:, -1].clone()
        # the expert's half of the SAME shared attention: suffix queries over all keys
        L = q.shape[1]
        sq = torch.einsum("thd,shd->hts", q[0, T:], kr) * scale      # [H, n_suf, L]
        sm = mask[0, T:]
        sq = sq.masked_fill(~sm[None], float("-inf"))
        ps = torch.softmax(sq, dim=-1)
        vr = v[0].repeat_interleave(H // v.shape[2], dim=1)
        o_suf = torch.einsum("hts,shd->thd", ps, vr)
        assert torch.allclose(o_suf, out[0, T:].view(L - T, H, HD), atol=1e-4), "suffix attention mismatch"
        # the suffix "tail" of every per-head block of this layer's joint attention: the
        # expert's 17 tokens are extra columns/rows of the SAME q, k, v, scores, probs, out
        ns = L - T
        put(f"E{i}.q", q[0, T:, :, :WD].permute(1, 0, 2), q[0].permute(1, 0, 2), [0, T, 0],
            f"layer pair {i}: expert Q of the suffix tokens (AdaRMSNorm → q_proj (+bias) → RoPE; the expert has no q_norm) [head, token, dim]")
        put(f"E{i}.k", k[0, T:, :, :WD].permute(1, 0, 2), k[0].permute(1, 0, 2), [0, T, 0],
            f"layer pair {i}: expert K of the suffix tokens (k_proj (+bias) → RoPE; 2 kv heads tiled to {k.shape[2]}) [kv_head, token, dim]")
        put(f"E{i}.v", v[0, T:, :, :WD].permute(1, 0, 2), v[0].permute(1, 0, 2), [0, T, 0],
            f"layer pair {i}: expert V of the suffix tokens (tiled) [kv_head, token, dim]")
        put(f"E{i}.scp", sq[:, :, tw], sq, [0, T, t0w],
            f"layer pair {i}: suffix queries · prefix-window keys / √{HD} (masked=-inf) [head, suffix q, key]")
        put(f"E{i}.scs", sq[:, :, T:], sq, [0, T, T],
            f"layer pair {i}: suffix queries · suffix keys / √{HD} [head, suffix q, suffix key]")
        smx = sq.max(-1).values
        sinv = 1.0 / torch.exp(sq - smx[..., None]).sum(-1)
        put(f"E{i}.attn_agg", torch.stack([smx, sinv], dim=-1), smx, [0, T, 0],
            f"layer pair {i}: suffix softmax agg over the FULL key row ({L} keys): max | 1/Σexp(s−max)")
        put(f"E{i}.pp", ps[:, :, tw], ps, [0, T, t0w],
            f"layer pair {i}: suffix attention probs on the prefix-window keys (rows normalised over all {L} keys)")
        put(f"E{i}.ps", ps[:, :, T:], ps, [0, T, T],
            f"layer pair {i}: suffix attention probs on the {ns} suffix keys")
        put(f"E{i}.vo", o_suf[:, :, :WD].permute(1, 0, 2), o_suf.permute(1, 0, 2), [0, T, 0],
            f"layer pair {i}: per-head attention output of the suffix tokens Σp·V [head, token, dim]")
        pm = ps.mean(0)                                                # [n_suf, L] head-mean
        put(f"E{i}.attn", torch.cat([pm[:, tw], pm[:, T:]], dim=-1), pm, [0, t0w],
            f"layer pair {i}: expert suffix queries' head-mean attention; cols = prefix window "
            f"[{t0w},{p_star}] then the {L - T} suffix keys; stats over [n_suf, L_joint]")
        mass = torch.stack([pm[:, :16].sum(-1), pm[:, :T][:, vis_cols].sum(-1),
                            pm[:, :T][:, txt_cols].sum(-1), pm[:, T:].sum(-1)], dim=-1)
        put(f"E{i}.attn_mass", mass, mass, [0, 0],
            f"layer pair {i}: suffix attention mass on [state tokens | visual | text | suffix]")
        if i == 0:
            priv["suffix_visible"] = int(sm[0, :T].sum())
            priv["suffix_visible_vis"] = int(sm[0, :T][vis_cols].sum())
        return out

    is_vis = torch.zeros(T, dtype=torch.bool)
    is_vis[16:] = tensors["visual_pos_mask"][0].bool().cpu()
    vis_cols = is_vis
    txt_cols = ~is_vis
    txt_cols[:16] = False
    fba.our_eager_attention_forward = attn_capture
    t1 = time.time()
    with torch.no_grad():
        out = model(**tensors)
    fba.our_eager_attention_forward = orig_attn
    for h in hooks:
        h.remove()
    print(f"[forward] {time.time() - t1:.0f}s", flush=True)

    lnf_full = priv["lnf_full"]
    expert_out = priv["expert_out"][0] if priv["expert_out"].dim() == 3 else priv["expert_out"]
    bf = model.behavior_fields
    h_star = lnf_full[p_star]
    knack = out.behavior_fields.knack[0]
    done = out.behavior_fields.done[0]
    vel = out.action_velocity[0]
    x1 = tensors["noisy_action"][0]
    action = x1 - vel
    with torch.no_grad():  # our captured h* must reproduce the model's own head outputs
        assert torch.allclose(bf.knack_head(h_star), knack, atol=1e-4)
        assert torch.allclose(bf.done_head(h_star), done, atol=1e-4)

    acts["embed_out"] = acts["L0.x_in"]
    kp_ = torch.softmax(knack, -1)
    dp_ = torch.softmax(done, -1)
    put("h_star", h_star, h_star, [0], f"h* = ln_f(trunk)[p*={p_star}] (generation boundary)")
    put("knack_logits", knack, knack, [0], "knack_head(h*) logits over CANONICAL_KNACKS")
    put("knack_probs", kp_, kp_, [0], "softmax(knack logits)")
    put("done_logits", done, done, [0], "done_head(h*) logits [NOT_DONE, DONE]")
    put("done_probs", dp_, dp_, [0], "softmax(done logits)")
    for nm, lg in (("knack", knack), ("done", done)):
        mx_ = lg.max()
        agg = torch.stack([mx_, 1.0 / torch.exp(lg - mx_).sum()])
        put(f"{nm}_agg", agg, agg, [0], f"{nm} softmax agg: max | 1/Σexp(logit−max)")
    put("expert_out", expert_out[:, :WC], expert_out, [0, 0],
        f"action expert final RMSNorm out [{expert_out.shape[0]} suffix tokens, {ecfg.hidden_size}]")
    put("action_x1", x1, x1, [0, 0], "x1 = the t=1 noise (suffix input)")
    put("action_velocity", vel, vel, [0, 0], "action_out_proj(expert suffix tail) [16, 23]")
    put("action_chunk", action, action, [0, 0], "decoded action = x1 − velocity [16, 23]")

    # ---------------- flow matching (docs/design/flow-matching-viz.md §2) ---------------- #
    fm = export_flow_matching(model, tensors, batch, put, expert, NL, T, H, HD, WC,
                              x1, vel, action, vis_cols, txt_cols, fba, orig_attn, tw)

    # ---------------- weights ---------------- #
    W = {}

    def wput(name, window, full, offset, desc):
        W[name] = ent(window, full.shape, offset, stats(full), desc)

    with torch.no_grad():
        emb = trunk.embed_tokens.weight
        tok_star = int(text_ids[boundary])
        v0 = max(0, min(tok_star, emb.shape[0] - WV))
        wput("embed", emb[v0:v0 + WV, :WC], emb, [v0, 0], "embed_tokens [vocab, C] (tied)")
        for i, layer in enumerate(trunk.layers):
            a = layer.self_attn
            g1, g2 = layer.input_layernorm.weight, layer.post_attention_layernorm.weight
            wput(f"L{i}.ln1_w", g1, g1, [0], "input RMSNorm γ")
            wput(f"L{i}.ln2_w", g2, g2, [0], "post-attn RMSNorm γ")
            qw = a.q_proj.weight.view(H, HD, C)
            kw = a.k_proj.weight.view(KV, HD, C)
            vw = a.v_proj.weight.view(KV, HD, C)
            wput(f"L{i}.q_w", qw[:, :WD, :WC], qw, [0, 0, 0], "q_proj [head, head_dim, C]")
            wput(f"L{i}.k_w", kw[:, :WD, :WC], kw, [0, 0, 0], "k_proj [kv_head, head_dim, C]")
            wput(f"L{i}.v_w", vw[:, :WD, :WC], vw, [0, 0, 0], "v_proj [kv_head, head_dim, C]")
            wput(f"L{i}.q_norm", a.q_norm.weight, a.q_norm.weight, [0], "q_norm γ [head_dim]")
            wput(f"L{i}.k_norm", a.k_norm.weight, a.k_norm.weight, [0], "k_norm γ [head_dim]")
            wput(f"L{i}.o_w", a.o_proj.weight[:WC, :WC], a.o_proj.weight, [0, 0], "o_proj [C, H·head_dim]")
            m = layer.mlp
            wput(f"L{i}.gate_w", m.gate_proj.weight[:WF, :WC], m.gate_proj.weight, [0, 0], "gate_proj [FF, C]")
            wput(f"L{i}.up_w", m.up_proj.weight[:WF, :WC], m.up_proj.weight, [0, 0], "up_proj [FF, C]")
            wput(f"L{i}.down_w", m.down_proj.weight[:WC, :WF], m.down_proj.weight, [0, 0], "down_proj [C, FF]")
        EH, EKV, EHD, EC = ecfg.num_attention_heads, ecfg.num_key_value_heads, ecfg.head_dim, ecfg.hidden_size
        for i in range(NL):
            layer = expert.layers[i]
            a = layer.self_attn
            for nm, ln in (("ln1", layer.input_layernorm), ("ln2", layer.post_attention_layernorm)):
                wput(f"E{i}.{nm}_w", ln.weight, ln.weight, [0], f"expert AdaRMSNorm weight w ({nm})")
            for nm, lin, nh in (("q", a.q_proj, EH), ("k", a.k_proj, EKV), ("v", a.v_proj, EKV)):
                w_ = lin.weight.view(nh, EHD, EC)
                b_ = lin.bias.view(nh, EHD)
                wput(f"E{i}.{nm}_w", w_[:, :WD, :WC], w_, [0, 0, 0], f"expert {nm}_proj [head, head_dim, C]")
                wput(f"E{i}.{nm}_b", b_[:, :WD], b_, [0, 0], f"expert {nm}_proj bias [head, head_dim]")
            wput(f"E{i}.o_w", a.o_proj.weight[:WC, :WC], a.o_proj.weight, [0, 0], "expert o_proj [C, H·head_dim]")
            m = layer.mlp
            wput(f"E{i}.gate_w", m.gate_proj.weight[:WF, :WC], m.gate_proj.weight, [0, 0], "expert gate_proj [FF, C]")
            wput(f"E{i}.up_w", m.up_proj.weight[:WF, :WC], m.up_proj.weight, [0, 0], "expert up_proj [FF, C]")
            wput(f"E{i}.down_w", m.down_proj.weight[:WC, :WF], m.down_proj.weight, [0, 0], "expert down_proj [C, FF]")
        wput("expert_norm_w", expert.norm.weight, expert.norm.weight, [0], "expert final RMSNorm weight")
        wput("lnf_w", trunk.norm.weight, trunk.norm.weight, [0], "final RMSNorm γ (ln_f)")
        wput("knack_w", bf.knack_head.weight, bf.knack_head.weight, [0, 0], "knack_head [17, C]")
        wput("knack_b", bf.knack_head.bias, bf.knack_head.bias, [0], "knack_head bias")
        wput("done_w", bf.done_head.weight, bf.done_head.weight, [0, 0], "done_head [2, C]")
        wput("done_b", bf.done_head.bias, bf.done_head.bias, [0], "done_head bias")
        aop = model.projections.action_out_proj
        wput("action_w", aop.weight, aop.weight, [0, 0], f"action_out_proj [23, {ecfg.hidden_size}]")
        wput("action_b", aop.bias, aop.bias, [0], "action_out_proj bias")

    # ---------------- config / sample metadata ---------------- #
    from data.schema import ACTION_DIM
    from vla.prepare_input import load_norm_stats
    norm_stats = load_norm_stats()["action.r1pro"]   # the serve path's bounds99 stats
    n_params = sum(p.numel() for p in model.parameters())
    n_trunk = sum(p.numel() for p in trunk.parameters())
    n_expert = sum(p.numel() for p in expert.parameters())
    n_vision = sum(p.numel() for n, p in model.named_parameters() if "visual" in n)
    win_ids = text_ids[boundary - WT + 1: boundary + 1].tolist()
    tgt = row.behavior_target if isinstance(row.behavior_target, dict) else {}
    config = {
        "model_type": "k005b",
        "name": "k005b fused VLA (Qwen3-VL-2B trunk + action expert)",
        "checkpoint": args.ckpt.replace("/home/frank/workspace/behavior_challenge_data/", ""),
        "step": ckpt_step,
        "n_layer": NL, "n_head": H, "n_kv_head": KV, "head_dim": HD,
        "n_embd": C, "intermediate": FF, "vocab_size": int(emb.shape[0]),
        "rms_norm_eps": vcfg.rms_norm_eps, "rope_max_wavelength": 10000.0,
        "expert_hidden": ecfg.hidden_size, "expert_heads": ecfg.num_attention_heads,
        "expert_kv_heads": ecfg.num_key_value_heads, "expert_intermediate": ecfg.intermediate_size,
        "T_prefix": T, "T_text": LT, "n_state_tokens": 16, "n_suffix": int(expert_out.shape[0]),
        "L_joint": T + int(expert_out.shape[0]),
        "n_visual_tokens": int(tensors["visual_pos_mask"][0].sum()),
        "block_size": T, "B": 1,
        "p_star": p_star, "boundary_column": boundary,
        "window": {"t0": t0w, "Wt": WT, "Wc": WC, "Wd": WD, "Wf": WF, "Wv": WV, "v0": v0},
        "window_token_ids": win_ids,
        "window_tokens": [proc.tokenizer.decode([t]) for t in win_ids],
        "knack_names": list(CANONICAL_KNACKS), "done_names": ["NOT_DONE", "DONE"],
        # the expert checkpoint carries more layers than the trunk; the forward runs
        # len(trunk.layers) layer-pairs (QwenvlWithExpertModel.forward), the rest are unused
        "expert_n_layer": len(expert.layers), "n_layer_pairs": NL,
        # §3.2 mask: suffix queries see the suffix + valid non-answer prompt columns,
        # never the 16 behavior-state columns
        "suffix_visible_prefix": priv["suffix_visible"], "suffix_visible_visual": priv["suffix_visible_vis"],
        "action_dim": int(vel.shape[-1]), "action_horizon": int(vel.shape[0]),
        "action_dim_names": ACTION_DIM_NAMES,
        "action_q01": [float(x) for x in norm_stats["q01"][:ACTION_DIM]],
        "action_q99": [float(x) for x in norm_stats["q99"][:ACTION_DIM]],
        "n_params": n_params, "n_params_trunk": n_trunk, "n_params_expert": n_expert,
        "n_params_vision": n_vision,
        "sample": {
            "row": args.row, "episode": int(row.episode), "frame": int(row.frame),
            "goal_text": str(getattr(row, "goal_text", ""))[:300],
            "target_knack": tgt.get("knack"), "target_done": tgt.get("done"),
            "pixel_values": list(batch["pixel_values"].shape),
            "image_grid_thw": batch["image_grid_thw"].tolist(),
        },
        "pred_knack": CANONICAL_KNACKS[int(knack.argmax())],
        "pred_done": bool(int(done.argmax()) == 1),
        "fm": fm,
    }
    steps_ = fm.pop("_steps")

    os.makedirs(out_dir, exist_ok=True)
    W["config"] = config
    acts["config"] = config
    for fn, obj in (("k005b-weights.json", W), ("k005b-acts.json", acts)):
        path = os.path.join(out_dir, fn)
        with open(path, "w") as f:
            json.dump(obj, f)
        print(f"[write] {path} {os.path.getsize(path) / 1e6:.1f} MB", flush=True)
    fm_out = {"K": FM_CF_STEPS, "checkpoint_step": ckpt_step,
              "in_distribution": False,  # this lineage only trained t = 1; steps ≥ 1 are a preview
              "steps": [{"t": tk, "tensors": {nm: ent16(x, nm) for nm, x in st.items()}}
                        for _, (tk, st) in sorted(steps_.items())]}
    path = os.path.join(out_dir, "k005b-fm.json")
    with open(path, "w") as f:
        json.dump(fm_out, f)
    print(f"[write] {path} {os.path.getsize(path) / 1e6:.1f} MB", flush=True)

    # spot values for the JS-vs-Python self test — read straight from the live
    # torch tensors (NOT from the exported windows) so the comparison is end-to-end
    q5 = None
    with torch.no_grad():
        l5 = trunk.layers[5].self_attn
        check = {
            "W.L5.q_w[h3,d7,c11]": float(l5.q_proj.weight[3 * HD + 7, 11]),
            "W.L20.down_w[c9,f40]": float(trunk.layers[20].mlp.down_proj.weight[9, 40]),
            "W.embed[v0+2,c3]": float(emb[v0 + 2, 3]),
            "A.L13.attn_probs[h0,p*,p*]": float(priv["p_star_probs_13"][0, p_star]),
            "A.h_star[c17]": float(h_star[17]),
            "A.knack_logits": knack.tolist(),
            "A.done_logits": done.tolist(),
            "A.action_chunk[0,:5]": action[0, :5].tolist(),
            "FM.assert": fm["asserts"],
        }

    def dec(e):
        return np.frombuffer(base64.b64decode(e["data"]), dtype="<f4").reshape(e["shape"])
    check["A.L27.mlp_resid[t=p*,c5] (window)"] = float(dec(acts["L27.mlp_resid"])[WT - 1, 5])
    check["A.L5.q[h3,t=p*,d7] (window)"] = float(dec(acts["L5.q"])[3, WT - 1, 7])
    with open(os.path.join(out_dir, "k005b-check.json"), "w") as f:
        json.dump(check, f, indent=1)
    print("[check]", json.dumps(check), flush=True)
    print(f"[done] pred knack={config['pred_knack']} done={config['pred_done']} "
          f"target knack={config['sample']['target_knack']} done={config['sample']['target_done']} "
          f"total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
