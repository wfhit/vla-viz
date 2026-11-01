"""Path-continuity check: each link's output must equal the next link's input (relative 2e-3)."""
import base64, json, os, struct, sys

def close(a, b, tol=2e-3):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def term_x(link, i):
    for t in link["terms"]:
        if t[0] == i:
            return t[2]
    raise KeyError(f"{link.get('name')}: term {i} missing")

def chain(links, x0, name):
    """links consume the previous output; return the final output."""
    y, idx = x0, None
    for L in links:
        if L["k"] == "lin":
            i = L["terms"][0][0]
            assert close(term_x(L, i), y), (name, L["name"], term_x(L, i), y)
            s = sum(t[3] for t in L["terms"]) + L["rest"]
            assert close(s, L["sum"]), (name, L["name"], "sum", s, L["sum"])
            y = L["y"]
        elif L["k"] == "act":
            assert close(L["x"], y), (name, L["name"]); y = L["y"]
        elif L["k"] == "add":
            assert close(L["a"], y) and close(L["a"] + L["b"], L["y"]), (name, L["name"]); y = L["y"]
        elif L["k"] == "norm":
            assert close(L["x"], y), (name, L["name"]); y = L["y"]
    return y

def check(path):
    t = json.load(open(path))
    r, a, e = t["rms"], t["attn"], t["euler"]
    assert close(t["embed"]["x_c"], r["x_c"]), "embed -> rms"
    assert close(r["sumsq"]["terms"][0][2], r["x_c"])
    assert close(r["y_c"], r["x_c"] * r["rsqrt"] * r["g_c"]), "rms y"
    for L in a["q"]["raw"]:
        assert close(term_x(L, a["c"]), r["y_c"]), "rms -> q"
    assert close(a["x_c"], r["x_c"]), "resid uses embed x"
    assert close(a["o"]["terms"][0][2], a["v_j"]), "v_j -> o"
    assert close(term_x(a["o_proj"], a["o_slot"]), a["o"]["sum"]), "o -> o_proj"
    assert close(a["resid"], a["x_c"] + a["o_proj"]["y"]), "resid"
    y = chain(e["enc"], e["x"], "enc")
    for i, (xin, r1, out) in enumerate(e["layers"]):
        assert close(xin, y), ("layer", i, xin, y); y = out
    v = chain(e["dec"], y, "dec")
    assert close(v, e["v"]) and close(e["x_next"], e["x"] + e["dt"] * e["v"]), "euler"
    # the traced element must be the one the page draws: <prefix>-fm.json step tensors at [i][j] (float16)
    fm = path.replace("-trace.json", "-fm.json")
    if os.path.exists(fm):
        st = json.load(open(fm))["steps"][e["step"]]["tensors"]
        for k, want in (("x", e["x"]), ("x_next", e["x_next"])):
            T = st[k]; raw = base64.b64decode(T["data"])
            got = struct.unpack_from("<e", raw, 2 * (e["i"] * T["shape"][1] + e["j"]))[0]
            assert close(got, want, 2e-3), (k, "fm", got, want)
    print(f"[ok] {t['model']}: embed->rms->q/k->softmax->o->o_proj->resid; euler {len(e['enc'])}+{len(e['layers'])}L+{len(e['dec'])} links")

for p in sys.argv[1:]:
    check(p)
