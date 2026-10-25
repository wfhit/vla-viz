import { base64ToArrayBuffer } from "@/src/utils/data";
import { fetchTensorFile, IRealTex, ITensorEntry, makeTex, texFromF16Entry } from "@/src/vla/VlaModel";

// tools/export_groot.py → public/groot-{acts,weights,fm}.json (real GR00T-N1.7-3B weights,
// real activations on one BEHAVIOR 2026-challenge-demos frame)
export interface IGrootConfig {
    model: string;
    global_step: string;
    sample: { episode: number; frame: number; task: string; language: string; image_hw: number[] };
    views: string[];
    thumbs: Record<string, string>; // 128² RGB, base64
    thumb_size: number;
    S: number;
    token_ids: number[];
    tokens: string[];
    image_runs: [number, number][];
    window: { Wc: number; Wd: number; Wf: number };
    vit: { n_block: number; width: number; heads: number; patch: number; merge: number; deepstack: number[]; n_patch: number };
    lm: { n_layer: number; n_layer_full: number; width: number; heads: number; kv_heads: number; head_dim: number; ffn: number };
    vlsa: { n_block: number; heads: number; head_dim: number };
    dit: { n_block: number; width: number; heads: number; head_dim: number; ffn: number; out_dim: number; kinds: ('text' | 'self' | 'image')[] };
    action_horizon: number;
    action_dim: number;
    state_dim: number;
    behavior_state_dim: number;
    behavior_action_dim: number;
    action_dim_names: string[];
    state_groups: [string, number, number][];
    K: number;
    num_timestep_buckets: number;
    t_disc: number[];
    t_cont: number[];
    embodiment: { tag: string; slot: number; ids: Record<string, number> };
    n_params: Record<string, number>;
    asserts: Record<string, number>;
    notes: string[];
}

export interface IGrootStep {
    k: number;
    t: number;
    t_disc: number;
    v_norm: number;
    v23_norm: number;
    tex: Record<string, IRealTex>;
}

export interface IGrootModel {
    cfg: IGrootConfig;
    acts: Record<string, IRealTex>;
    w: Record<string, IRealTex>;
    steps: IGrootStep[];
    tokIds: IRealTex;
    // thumbnails downsampled for drawing as quads: [view][y][x] → RGB 0..1
    thumbs: { view: string; n: number; rgb: Float32Array }[];
}

function texAll(gl: WebGL2RenderingContext, ts: Record<string, ITensorEntry>) {
    let out: Record<string, IRealTex> = {};
    for (let k in ts) out[k] = makeTex(gl, ts[k]);
    return out;
}

export function downsampleThumb(b64: string, size: number, n: number) {
    let px = new Uint8Array(base64ToArrayBuffer(b64));
    let f = size / n;
    let rgb = new Float32Array(n * n * 3);
    for (let y = 0; y < n; y++) for (let x = 0; x < n; x++) for (let c = 0; c < 3; c++) {
        let s = 0, cnt = 0;
        for (let yy = Math.floor(y * f); yy < Math.floor((y + 1) * f); yy++)
            for (let xx = Math.floor(x * f); xx < Math.floor((x + 1) * f); xx++) { s += px[(yy * size + xx) * 3 + c]; cnt++; }
        rgb[(y * n + x) * 3 + c] = s / cnt / 255;
    }
    return rgb;
}

export const thumbRes = 32;

export async function loadGrootModel(gl: WebGL2RenderingContext): Promise<IGrootModel> {
    let [acts, weights, fmResp] = await Promise.all([
        fetchTensorFile<IGrootConfig>('/groot-acts.json'),
        fetchTensorFile<IGrootConfig>('/groot-weights.json'),
        fetch('/groot-fm.json').then(r => { if (!r.ok) throw new Error(`groot-fm.json: ${r.status}`); return r.json(); }),
    ]);
    let cfg = acts.config;
    let steps: IGrootStep[] = fmResp.steps.map((s: any) => {
        let tex: Record<string, IRealTex> = {};
        for (let k in s.tensors) tex[k] = texFromF16Entry(gl, `step${s.k}.${k}`, s.tensors[k]);
        return { k: s.k, t: s.t, t_disc: s.t_disc, v_norm: s.v_norm, v23_norm: s.v23_norm, tex };
    });
    let ids = new Float32Array(cfg.token_ids);
    let mean = ids.reduce((a, b) => a + b, 0) / ids.length;
    let std = Math.sqrt(ids.reduce((a, b) => a + (b - mean) ** 2, 0) / ids.length);
    let tokIds = makeTex(gl, { name: 'input_ids', shape: [ids.length], data: ids, fullShape: [ids.length], offset: [0],
        stats: { min: Math.min(...ids), max: Math.max(...ids), mean, std },
        desc: 'Qwen3-VL token ids of the prompt (image runs = <|image_pad|>)' });
    let thumbs = cfg.views.map(view => ({ view, n: thumbRes, rgb: downsampleThumb(cfg.thumbs[view], cfg.thumb_size, thumbRes) }));
    return { cfg, acts: texAll(gl, acts.tensors), w: texAll(gl, weights.tensors), steps, tokIds, thumbs };
}

// the real (un-windowed) value of a 1D / 2D window entry, for the commentary
export function texVal(t: IRealTex | undefined, r: number, c: number = 0) {
    if (!t?.localBuffer) return NaN;
    return t.localBuffer[r * t.width + c];
}

export function texNorm(t: IRealTex | undefined) {
    if (!t?.localBuffer) return NaN;
    let s = 0;
    for (let v of t.localBuffer) s += v * v;
    return Math.sqrt(s);
}
