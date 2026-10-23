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


def stats(t):
    import torch
    t = t.detach().float()
    finite = t[torch.isfinite(t)]
    if finite.numel() == 0:
        return dict(min=0.0, max=0.0, mean=0.0, std=0.0)
    return dict(min=float(finite.min()), max=float(finite.max()),
                mean=float(finite.mean()),
                std=float(finite.std()) if finite.numel() > 1 else 0.0)


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
    hooks.append(expert.norm.register_forward_hook(
        lambda m, a, o: priv.__setitem__("expert_out", o[0])))

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
        return out

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
    x1 = tensors["x1"][0]
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
        wput("lnf_w", trunk.norm.weight, trunk.norm.weight, [0], "final RMSNorm γ (ln_f)")
        wput("knack_w", bf.knack_head.weight, bf.knack_head.weight, [0, 0], "knack_head [17, C]")
        wput("knack_b", bf.knack_head.bias, bf.knack_head.bias, [0], "knack_head bias")
        wput("done_w", bf.done_head.weight, bf.done_head.weight, [0, 0], "done_head [2, C]")
        wput("done_b", bf.done_head.bias, bf.done_head.bias, [0], "done_head bias")
        aop = model.projections.action_out_proj
        wput("action_w", aop.weight, aop.weight, [0, 0], f"action_out_proj [23, {ecfg.hidden_size}]")
        wput("action_b", aop.bias, aop.bias, [0], "action_out_proj bias")

    # ---------------- config / sample metadata ---------------- #
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
        "action_dim": int(vel.shape[-1]), "action_horizon": int(vel.shape[0]),
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
    }

    os.makedirs(out_dir, exist_ok=True)
    W["config"] = config
    acts["config"] = config
    for fn, obj in (("k005b-weights.json", W), ("k005b-acts.json", acts)):
        path = os.path.join(out_dir, fn)
        with open(path, "w") as f:
            json.dump(obj, f)
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
