"""Path-continuity trace (public/<model>-trace.json), shared by the three exporters.

The walkthrough follows ONE real object per chapter (the example token through L0 RMSNorm and
head-0 attention; one action number through the first Euler step) and shows every intermediate
number, such that the number a step outputs is exactly the number the next step takes in.

Every builder recomputes its chain in float64 from the full tensors and asserts it against the
model's own captured outputs, so the ledger the page shows is the model's real arithmetic.
A dot product Σ w_i·x_i is shown as: the term that carries the previous link's number, the
largest |w_i·x_i| terms, and "the other n−k terms" as one remainder — sum = the real output.
"""
import json
import math

import numpy as np


def _np(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().double().numpy()
    return np.asarray(x, np.float64)


def _close(a, b, what, tol=2e-3):
    a, b = _np(a), _np(b)
    err = float(np.abs(a - b).max() / (np.abs(b).max() + 1e-6))
    assert err < tol, f"trace: {what} rel err {err:.2e}"
    return err


def terms(w, x, must=(), k=3):
    """Σ_i w_i·x_i as {n, terms: [[i, w_i, x_i, w_i·x_i], …], rest, sum}."""
    w, x = _np(w), _np(x)
    p = w * x
    top = [int(i) for i in np.argsort(-np.abs(p))[:k + len(must)]]
    idx = list(dict.fromkeys([int(i) for i in must] + top))[:len(must) + k]
    tot = float(p.sum())
    return {"n": int(p.size), "terms": [[i, float(w[i]), float(x[i]), float(p[i])] for i in idx],
            "rest": tot - float(p[idx].sum()), "sum": tot}


def linear(name, w_row, x, b=None, must=(), ref=None, k=3, out=""):
    """One output of a linear layer: y = Σ_i W[r, i]·x_i (+ b)."""
    t = terms(w_row, x, must, k)
    y = t["sum"] + (float(b) if b is not None else 0.0)
    if ref is not None:
        _close(y, ref, name)
    return {"k": "lin", "name": name, "out": out, **t, "b": None if b is None else float(b), "y": y}


def rms(x, gain, eps, c, y_ref, gain_label="g", must_extra=()):
    """y_c = x_c · rsqrt(mean(x²) + eps) · gain_c; asserts the whole y against the model."""
    x, g = _np(x), _np(gain)
    sq = terms(x, x, [c, *must_extra])
    mean = sq["sum"] / x.size
    r = 1.0 / math.sqrt(mean + eps)
    _close(x * r * g, y_ref, f"rms {gain_label}")
    return {"c": int(c), "C": int(x.size), "x_c": float(x[c]), "sumsq": sq, "mean": mean, "eps": eps,
            "rsqrt": r, "gain": gain_label, "g_c": float(g[c]), "y_c": float(x[c] * r * g[c])}


def rope_pair(pre, post, d):
    """Solve RoPE's rotation of the pair (i, i + D/2) from the vector before / after it."""
    pre, post = _np(pre), _np(post)
    half = pre.size // 2
    a, b, a2, b2 = pre[:half], pre[half:], post[:half], post[half:]
    n2 = a * a + b * b
    c = (a * a2 + b * b2) / n2
    s = (a * b2 - b * a2) / n2
    ok = n2 > 1e-6 * n2.max()
    assert np.abs(c[ok] ** 2 + s[ok] ** 2 - 1).max() < 5e-3, "rope: rotation does not preserve norm"
    i = d % half
    return {"d": int(d), "pair": [int(i), int(i + half)], "pre": [float(a[i]), float(b[i])],
            "cos": float(c[i]), "sin": float(s[i]), "angle": math.atan2(s[i], c[i]),
            "post": [float(a2[i]), float(b2[i])]}


def qk_path(name, y, W, bias, d, c, gain, eps, post):
    """y (normed C) → raw = W·y (+b) → [per-head RMSNorm] → RoPE → post; shows element d."""
    W, y = _np(W), _np(y)
    raw = W @ y + (_np(bias) if bias is not None else 0.0)
    half = raw.size // 2
    i = d % half
    out = {"name": name,
           "raw": [linear(f"{name}_proj", W[r], y, None if bias is None else _np(bias)[r], [c], raw[r], out=f"{name}_raw[{r}]")
                   for r in (i, i + half)]}
    if gain is not None:
        g = _np(gain)
        ms = float((raw ** 2).mean())
        r_ = 1.0 / math.sqrt(ms + eps)
        n = raw * r_ * g
        out["norm"] = {"mean": ms, "eps": eps, "rsqrt": r_, "g": [float(g[i]), float(g[i + half])],
                       "y": [float(n[i]), float(n[i + half])]}
    else:
        n = raw
    out["rope"] = rope_pair(n, post, d)
    return out


def attention(*, p, y_all, x_p, Wq, bq, Wk, bk, Wv, bv, qn, kn, eps, q, K, V, mask_row, probs_row,
              attn_out, o_cat, Wo_row, o_ref, resid_ref, c, d, head, D, scale, win):
    """Query token p, head `head` (kv head of K/V), output dim d, residual channel c."""
    q, K, V = _np(q), _np(K), _np(V)
    s = K @ q * scale
    s = np.where(_np(mask_row) > 0.5, s, -np.inf)
    m = float(s.max())
    e = np.exp(s - m)
    Z = float(e.sum())
    pr = e / Z
    _close(pr, probs_row, "attention probs")
    t0, t1 = win
    j = int(t0 + np.argmax(pr[t0:t1]))
    jm = int(np.argmax(pr))
    top = [int(i) for i in np.argsort(-pr)[:3]]
    od = terms(pr, V[:, d], [j], k=3)
    _close(od["sum"], _np(attn_out)[d], "Σp·v")
    _close(_np(o_cat)[head * D + d], _np(attn_out)[d], "o_cat slot")
    oo = linear("o_proj", Wo_row, o_cat, None, [head * D + d], o_ref, out=f"o_out[{c}]")
    res = float(x_p) + oo["y"]
    _close(res, resid_ref, "residual")
    return {
        "p": int(p), "head": int(head), "d": int(d), "c": int(c), "D": int(D), "j": j,
        "q": qk_path("q", y_all[p], Wq, bq, d, c, qn, eps, q),
        "k": qk_path("k", y_all[j], Wk, bk, d, c, kn, eps, K[j]),
        "k_y_c": float(_np(y_all[j])[c]),
        "dot": terms(q, K[j], [d % (D // 2), d % (D // 2) + D // 2], k=2),
        "scale": scale, "s_j": float(s[j]), "max": m, "max_at": jm, "Z": Z, "n_keys": int(np.isfinite(s).sum()),
        "p_j": float(pr[j]), "top": [[i, float(s[i]), float(pr[i])] for i in top],
        "v_j": float(V[j, d]), "o": od, "o_slot": head * D + d, "o_proj": oo,
        "x_c": float(x_p), "resid": res,
    }


def mlp2(names, x, W1, b1, W2, b2, out_idx, must, act, ref, xnames=None):
    """y[out] = W2[out]·act(W1·x + b1) + b2 shown through the hidden unit k* that contributes most.
    act ∈ {'silu', 'swish', 'relu'}; `must` = indices of x to show (or fn(W1[k*], x) → indices);
    xnames = [[offset, name], …] labels the segments of a concatenated input."""
    x, W1, W2 = _np(x), _np(W1), _np(W2)
    b1 = _np(b1) if b1 is not None else np.zeros(W1.shape[0])
    h_pre = W1 @ x + b1
    h = np.maximum(h_pre, 0) if act == "relu" else h_pre / (1 + np.exp(-h_pre))
    contrib = W2[out_idx] * h
    ks = int(np.argmax(np.abs(contrib)))
    must = must(W1[ks], x) if callable(must) else must
    l1 = linear(names[0], W1[ks], x, b1[ks], must, h_pre[ks], out=f"{names[0]}[{ks}]")
    if xnames:
        l1["xnames"] = xnames
    return [l1, {"k": "act", "name": act, "x": float(h_pre[ks]), "y": float(h[ks]), "out": f"{names[1]}[{ks}]"},
            linear(names[2], W2[out_idx], h, None if b2 is None else _np(b2)[out_idx], [ks], ref,
                   out=f"{names[2]}[{out_idx}]")]


def top_in(lo, hi):
    """`must` fn for mlp2: the largest |w·x| term among input indices [lo, hi)."""
    return lambda w, x: [lo + int(np.argmax(np.abs(_np(w)[lo:hi] * _np(x)[lo:hi])))]


def add(name, a, b, y, out):
    assert abs(float(a) + float(b) - float(y)) < 1e-3 * (1 + abs(float(y))), name
    return {"k": "add", "name": name, "a": float(a), "b": float(b), "y": float(y), "out": out}


def norm_mod(name, x, eps, c, mult, add, y_ref, kind="rms"):
    """y_c = norm(x)_c · mult_c + add_c, norm = RMS (x·rsqrt(mean x²+eps)) or LayerNorm."""
    x = _np(x)
    mu = float(x.mean()) if kind == "ln" else 0.0
    ms = float(((x - mu) ** 2).mean())
    r = 1.0 / math.sqrt(ms + eps)
    mult = _np(mult) if mult is not None else np.ones_like(x)
    add = _np(add) if add is not None else np.zeros_like(x)
    _close((x - mu) * r * mult + add, y_ref, name)
    return {"k": "norm", "name": name, "kind": kind, "x": float(x[c]), "mu": mu, "ms": ms, "eps": eps, "rsqrt": r,
            "mult": float(mult[c]), "add": float(add[c]), "y": float((x[c] - mu) * r * mult[c] + add[c]),
            "out": f"{name}[{c}]"}


def euler(*, step, t, dt, i, j, c, x, enc, layers, dec, v, x_next, tok):
    assert abs(float(x) + dt * float(v) - float(x_next)) < 1e-4 * (1 + abs(float(x_next))), "euler identity"
    return {"step": step, "t": t, "dt": dt, "i": int(i), "tok": int(tok), "j": int(j), "c": int(c), "x": float(x),
            "enc": enc, "layers": layer_chain(layers), "dec": dec, "v": float(v), "x_next": float(x_next)}


def layer_chain(rows):
    """rows: per layer (in_c, res1_c, out_c); asserts out_i = in_{i+1} (the residual stream)."""
    rows = [[float(a), float(b), float(o)] for a, b, o in rows]
    for i in range(len(rows) - 1):
        assert abs(rows[i][2] - rows[i + 1][0]) < 1e-3 * (1 + abs(rows[i][2])), f"layer {i} out != layer {i + 1} in"
    return rows


def write(path, trace):
    def clean(o):
        if isinstance(o, dict):
            return {k: clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (float, np.floating)):
            return float(f"{float(o):.7g}")
        if isinstance(o, np.integer):
            return int(o)
        return o
    with open(path, "w") as fh:
        json.dump(clean(trace), fh, separators=(",", ":"))
    print(f"[trace] wrote {path}", flush=True)
