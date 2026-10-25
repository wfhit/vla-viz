import { fetchTensorFile, IRealTex, ITensorEntry, makeTex, texFromF16Entry } from "@/src/vla/VlaModel";
import { downsampleThumb, thumbRes } from "@/src/groot/GrootModel";

// tools/export_pi0.py → public/pi0-{acts,weights,fm}.json (real openpi-comet pi05-b1kpt12-cs32
// weights, real activations on one BEHAVIOR 2026-challenge-demos frame; the action output is
// checked against the official openpi JAX sample_actions)
export interface IPi0Config {
    model: string;
    checkpoint: string;
    train_config: string;
    sample: { episode: number; frame: number; task: string };
    views: string[];
    thumbs: Record<string, string>; // 128² RGB, base64
    thumb_size: number;
    prompt: string;
    T: number;
    token_ids: number[];
    tokens: string[];
    state_token_start: number;
    Np: number;
    n_img: number;
    n_img_per_view: number;
    prefix_rows: number[]; // which prefix tokens the row-subset prefix tensors carry
    window: { Wc: number; Wd: number; Wf: number; Ws: number };
    vit: { n_block: number; width: number; heads: number; head_dim: number; mlp: number; patch: number; grid: number };
    prefix: { n_layer: number; width: number; heads: number; kv_heads: number; head_dim: number; mlp: number; vocab: number };
    expert: { n_layer: number; width: number; heads: number; kv_heads: number; head_dim: number; mlp: number };
    action_horizon: number;
    action_dim: number;
    behavior_action_dim: number;
    action_dim_names: string[];
    state_groups: [string, number, number][];
    state23: number[];
    state_norm: number[];
    state_tokens: number[];
    K: number;
    dt: number;
    t: number[];
    seed: number;
    action_norm: number[][];
    action23: number[][];
    n_params: Record<string, number>;
    asserts: Record<string, number>;
    notes: string[];
}

export interface IPi0Step {
    k: number;
    t: number;
    v_norm: number;
    v23_norm: number;
    tex: Record<string, IRealTex>;
}

export interface IPi0Model {
    cfg: IPi0Config;
    acts: Record<string, IRealTex>;
    w: Record<string, IRealTex>;
    steps: IPi0Step[];
    tokIds: IRealTex;
    state23: IRealTex;
    action23: IRealTex;
    thumbs: { view: string; n: number; rgb: Float32Array }[];
}

function texAll(gl: WebGL2RenderingContext, ts: Record<string, ITensorEntry>) {
    let out: Record<string, IRealTex> = {};
    for (let k in ts) out[k] = makeTex(gl, ts[k]);
    return out;
}

function texOf(gl: WebGL2RenderingContext, name: string, shape: number[], data: Float32Array, desc: string) {
    let mean = data.reduce((a, b) => a + b, 0) / data.length;
    let std = Math.sqrt(data.reduce((a, b) => a + (b - mean) ** 2, 0) / data.length);
    let min = Infinity, max = -Infinity;
    for (let v of data) { min = Math.min(min, v); max = Math.max(max, v); }
    return makeTex(gl, { name, shape, data, fullShape: shape, offset: shape.map(() => 0), stats: { min, max, mean, std }, desc });
}

export async function loadPi0Model(gl: WebGL2RenderingContext): Promise<IPi0Model> {
    let [acts, weights, fmResp] = await Promise.all([
        fetchTensorFile<IPi0Config>('/pi0-acts.json'),
        fetchTensorFile<IPi0Config>('/pi0-weights.json'),
        fetch('/pi0-fm.json').then(r => { if (!r.ok) throw new Error(`pi0-fm.json: ${r.status}`); return r.json(); }),
    ]);
    let cfg = acts.config;
    let steps: IPi0Step[] = fmResp.steps.map((s: any) => {
        let tex: Record<string, IRealTex> = {};
        for (let k in s.tensors) tex[k] = texFromF16Entry(gl, `step${s.k}.${k}`, s.tensors[k]);
        return { k: s.k, t: s.t, v_norm: s.v_norm, v23_norm: s.v23_norm, tex };
    });
    let tokIds = texOf(gl, 'input_ids', [cfg.T], new Float32Array(cfg.token_ids), 'PaliGemma token ids of the prompt (task + 23 discretised state tokens)');
    let state23 = texOf(gl, 'state23', [23], new Float32Array(cfg.state23), 'R1 Pro state (23 of the 61 proprio dims)');
    let H = cfg.action_horizon, D = cfg.behavior_action_dim;
    let action23 = texOf(gl, 'action23', [H, D], new Float32Array(cfg.action23.flat()), 'action chunk, unnormalised (q01/q99), 23 R1 Pro dims');
    let thumbs = cfg.views.map(view => ({ view, n: thumbRes, rgb: downsampleThumb(cfg.thumbs[view], cfg.thumb_size, thumbRes) }));
    return { cfg, acts: texAll(gl, acts.tensors), w: texAll(gl, weights.tensors), steps, tokIds, state23, action23, thumbs };
}
