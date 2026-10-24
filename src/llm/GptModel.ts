// The REAL k005b model link: every texture below is filled from tools/export_k005b.py
// output (public/k005b-weights.json + public/k005b-acts.json) — real checkpoint
// weights and real activations from one real training row. Nothing is computed or
// synthesised here; the browser only uploads the exported windows to the GPU.

import { createBufferTex, IBufferTex, writeToBufferTex } from "@/src/utils/renderPhases";
import { base64ToArrayBuffer } from "@/src/utils/data";

// Geometry of the 3D scene, in cells. Each block shows a REAL contiguous window of
// its tensor (sizes from config.window); `full` carries the real model dims.
export interface IModelShape {
    B: number;
    T: number;          // token window (real tokens t0 .. t0+T-1 of the trunk stream)
    C: number;          // hidden-channel window
    nHeads: number;     // real: 16 query heads
    nKvHeads: number;   // real: 8 kv heads (GQA)
    A: number;          // head-dim window
    F: number;          // SwiGLU intermediate window
    nBlocks: number;    // real: 28 trunk layers
    vocabSize: number;  // embed-table row window
    nKnack: number;     // real: 17 knack classes
    nDone: number;      // real: 2
    actH: number;       // real: 16 action steps
    actD: number;       // real: 23 action dims
    nSuffix: number;    // real: 17 expert suffix tokens
    full: IK005bConfig | null;
}

export interface IK005bWindow { t0: number; Wt: number; Wc: number; Wd: number; Wf: number; Wv: number; v0: number; }

export interface IK005bConfig {
    model_type: string;
    name: string;
    checkpoint: string;
    step: number;
    n_layer: number;
    n_head: number;
    n_kv_head: number;
    head_dim: number;
    n_embd: number;
    intermediate: number;
    vocab_size: number;
    rms_norm_eps: number;
    expert_hidden: number;
    expert_heads: number;
    expert_kv_heads: number;
    expert_intermediate: number;
    T_prefix: number;
    T_text: number;
    n_state_tokens: number;
    n_suffix: number;
    L_joint: number;
    n_visual_tokens: number;
    p_star: number;
    boundary_column: number;
    window: IK005bWindow;
    window_token_ids: number[];
    window_tokens: string[];
    knack_names: string[];
    done_names: string[];
    action_dim: number;
    action_horizon: number;
    action_dim_names: string[];
    action_q01: number[];   // bounds99 stats (robot units) the serve path un-normalizes with
    action_q99: number[];
    expert_n_layer: number; // layers in the expert checkpoint
    n_layer_pairs: number;  // layer-pairs the forward actually runs (= trunk layers)
    suffix_visible_prefix: number; // prefix columns the §3.2 mask lets suffix queries see
    suffix_visible_visual: number;
    n_params: number;
    n_params_trunk: number;
    n_params_expert: number;
    n_params_vision: number;
    sample: {
        row: number; episode: number; frame: number; goal_text: string;
        target_knack: string | null; target_done: boolean | null;
        pixel_values: number[]; image_grid_thw: number[][];
    };
    pred_knack: string;
    pred_done: boolean;
}

export interface ITensorStats { min: number; max: number; mean: number; std: number; }

export interface ITensorEntry {
    name: string;
    shape: number[];        // window shape
    data: Float32Array;
    fullShape: number[];
    offset: number[];       // window origin inside the full tensor
    stats: ITensorStats;    // over the FULL tensor
    desc: string;
}

export interface ITensorMeta {
    name: string;
    desc: string;
    fullShape: number[];
    // index prefix (e.g. [head]) + texture-row/col origin inside the full tensor
    prefix: number[];
    rowOffset: number;
    colOffset: number;
    is1d: boolean;
    stats: ITensorStats;
    absScale: number;       // colour scale derived from the window's own values
}

// A GPU texture holding one real window. rows = tensor dim -2, cols = dim -1.
export interface IRealTex extends IBufferTex {
    localBuffer: Float32Array;
    meta: ITensorMeta;
}

export interface IRmsNormLink {
    agg: IRealTex;      // [Wt, 2]: mean(x²) over full C | rsqrt(mean+eps)
    weight: IRealTex;   // γ [C]
    output: IRealTex;
}

export interface IHeadLink {
    qWeight: IRealTex;
    qNorm: IRealTex;
    q: IRealTex;
    attnScores: IRealTex;
    attnAgg: IRealTex;
    attnProbs: IRealTex;
    attnOut: IRealTex;
}

export interface IKvGroupLink {
    kWeight: IRealTex;
    vWeight: IRealTex;
    kNorm: IRealTex;
    k: IRealTex;
    v: IRealTex;
}

export interface IBlockLayerLink {
    xIn: IRealTex;
    ln_1: IRmsNormLink;
    heads: IHeadLink[];
    kv: IKvGroupLink[];
    oWeight: IRealTex;
    oOut: IRealTex;
    attnResid: IRealTex;
    ln_2: IRmsNormLink;
    gateWeight: IRealTex;
    upWeight: IRealTex;
    downWeight: IRealTex;
    gate: IRealTex;
    up: IRealTex;
    act: IRealTex;
    mlpOut: IRealTex;
    mlpResid: IRealTex;
}

// One layer-pair of the action expert: its suffix output and the suffix queries'
// share of the SAME joint attention the trunk layer runs.
export interface IExpertLayerLink {
    out: IRealTex;      // [n_suf, Wc] expert residual stream after layer i
    attn: IRealTex;     // [n_suf, Wt + n_suf] head-mean probs: prefix window keys | suffix keys
    mass: Float32Array; // [n_suf, 4] attention mass on [state tokens | visual | text | suffix]
}

export interface IGptModelLink {
    gl: WebGL2RenderingContext;
    config: IK005bConfig;
    shape: IModelShape;
    inputTokens: IRealTex;
    embedWeight: IRealTex;
    embedOut: IRealTex;
    blocks: IBlockLayerLink[];
    ln_f: IRmsNormLink;
    hStar: IRealTex;
    knackWeight: IRealTex;
    knackBias: IRealTex;
    knackLogits: IRealTex;
    knackAgg: IRealTex;
    knackProbs: IRealTex;
    doneWeight: IRealTex;
    doneBias: IRealTex;
    doneLogits: IRealTex;
    doneAgg: IRealTex;
    doneProbs: IRealTex;
    expertIn: IRealTex;
    expertLayers: IExpertLayerLink[];
    expertOut: IRealTex;
    actionWeight: IRealTex;
    actionBias: IRealTex;
    actionX1: IRealTex;
    actionVelocity: IRealTex;
    actionChunk: IRealTex;
}

export interface IK005bData {
    config: IK005bConfig;
    weights: Record<string, ITensorEntry>;
    acts: Record<string, ITensorEntry>;
}

async function fetchTensorFile(url: string) {
    let resp = await fetch(url);
    if (!resp.ok) {
        throw new Error(`failed to fetch ${url}: ${resp.status}`);
    }
    let json = await resp.json();
    let out: Record<string, ITensorEntry> = {};
    for (let k in json) {
        let e = json[k];
        if (!e || !e.shape || !e.data) {
            continue;
        }
        if (e.dtype !== 'torch.float32') {
            throw new Error(`tensor ${k}: unsupported dtype ${e.dtype}`);
        }
        out[k] = {
            name: k,
            shape: e.shape,
            data: new Float32Array(base64ToArrayBuffer(e.data)),
            fullShape: e.full_shape,
            offset: e.offset,
            stats: e.stats,
            desc: e.desc,
        };
    }
    return { config: json.config as IK005bConfig, tensors: out };
}

export async function fetchK005bData(): Promise<IK005bData> {
    let [w, a] = await Promise.all([fetchTensorFile('k005b-weights.json'), fetchTensorFile('k005b-acts.json')]);
    return { config: w.config, weights: w.tensors, acts: a.tensors };
}

export function shapeFromConfig(cfg: IK005bConfig): IModelShape {
    let w = cfg.window;
    return {
        B: 1,
        T: w.Wt,
        C: w.Wc,
        nHeads: cfg.n_head,
        nKvHeads: cfg.n_kv_head,
        A: w.Wd,
        F: w.Wf,
        nBlocks: cfg.n_layer,
        vocabSize: w.Wv,
        nKnack: cfg.knack_names.length,
        nDone: cfg.done_names.length,
        actH: cfg.action_horizon,
        actD: cfg.action_dim,
        nSuffix: cfg.n_suffix,
        full: cfg,
    };
}

function windowAbsScale(data: Float32Array) {
    let vals: number[] = [];
    for (let i = 0; i < data.length; i++) {
        let v = data[i];
        if (isFinite(v)) vals.push(Math.abs(v));
    }
    if (vals.length === 0) return 1;
    vals.sort((a, b) => a - b);
    let p95 = vals[Math.min(vals.length - 1, Math.floor(vals.length * 0.95))];
    return p95 > 0 ? 1 / p95 : 1;
}

function makeTex(gl: WebGL2RenderingContext, e: ITensorEntry, sub?: number): IRealTex {
    // ND window → 2D texture. A leading [head] axis is split off via `sub`.
    let shape = e.shape;
    let off = e.offset;
    let prefix: number[] = [];
    let data = e.data;
    if (sub !== undefined) {
        let per = shape.slice(1).reduce((a, b) => a * b, 1);
        data = e.data.subarray(sub * per, (sub + 1) * per);
        prefix = [off[0] + sub];
        shape = shape.slice(1);
        off = off.slice(1);
    }
    let is1d = shape.length === 1;
    let rows = is1d ? shape[0] : shape[0];
    let cols = is1d ? 1 : shape[1];
    if (shape.length > 2) {
        throw new Error(`tensor ${e.name}: unexpected rank ${e.shape.length}`);
    }
    let tex = createBufferTex(gl, cols, rows, 1) as IRealTex;
    let local = new Float32Array(data);
    // -inf (masked attention) can't be sampled meaningfully; keep it in the CPU copy
    // (hover shows "masked") but upload a large negative so the cell renders as "off"
    let upload = new Float32Array(local.length);
    for (let i = 0; i < local.length; i++) upload[i] = isFinite(local[i]) ? local[i] : -1e9;
    writeToBufferTex(gl, tex, upload);
    tex.localBuffer = local;
    tex.meta = {
        name: sub !== undefined ? `${e.name}[${prefix[0]}]` : e.name,
        desc: e.desc,
        fullShape: e.fullShape,
        prefix,
        rowOffset: off[0] ?? 0,
        colOffset: is1d ? 0 : off[1] ?? 0,
        is1d,
        stats: e.stats,
        absScale: windowAbsScale(local),
    };
    return tex;
}

function windowStats(a: Float32Array): ITensorStats {
    let min = Infinity, max = -Infinity, sum = 0, sum2 = 0;
    for (let v of a) { min = Math.min(min, v); max = Math.max(max, v); sum += v; sum2 += v * v; }
    let mean = sum / a.length;
    return { min, max, mean, std: Math.sqrt(Math.max(0, sum2 / a.length - mean * mean)) };
}

// Column of the (windowed) embed table holding window token t's row, or -1 if
// that token's row lies outside the exported embed rows [v0, v0+Wv).
export function embedColForToken(m: IGptModelLink, t: number): number {
    let id = m.config.window_token_ids[t];
    let c = id - m.config.window.v0;
    return id !== undefined && c >= 0 && c < m.config.window.Wv ? c : -1;
}

function tokenTex(gl: WebGL2RenderingContext, cfg: IK005bConfig): IRealTex {
    let ids = new Float32Array(cfg.window_token_ids);
    let entry: ITensorEntry = {
        name: 'text_input_ids',
        shape: [ids.length],
        data: ids,
        fullShape: [cfg.T_text],
        offset: [cfg.window.t0 - cfg.n_state_tokens],
        stats: windowStats(ids),
        desc: 'real prompt token ids (window ends at the h* boundary token; stats over the window)',
    };
    return makeTex(gl, entry);
}

export function createK005bGpuModel(gl: WebGL2RenderingContext, data: IK005bData): IGptModelLink {
    let { config, weights: W, acts: A } = data;
    let shape = shapeFromConfig(config);
    let get = (src: Record<string, ITensorEntry>, k: string) => {
        let e = src[k];
        if (!e) throw new Error(`k005b export is missing tensor ${k}`);
        return e;
    };
    let w = (k: string, sub?: number) => makeTex(gl, get(W, k), sub);
    let a = (k: string, sub?: number) => makeTex(gl, get(A, k), sub);

    let H = config.n_head, KV = config.n_kv_head;
    let blocks: IBlockLayerLink[] = [];
    for (let i = 0; i < config.n_layer; i++) {
        let qNorm = w(`L${i}.q_norm`);
        let kNorm = w(`L${i}.k_norm`);
        let heads: IHeadLink[] = [];
        for (let h = 0; h < H; h++) {
            heads.push({
                qWeight: w(`L${i}.q_w`, h),
                qNorm,
                q: a(`L${i}.q`, h),
                attnScores: a(`L${i}.attn_scores`, h),
                attnAgg: a(`L${i}.attn_agg`, h),
                attnProbs: a(`L${i}.attn_probs`, h),
                attnOut: a(`L${i}.attn_out`, h),
            });
        }
        let kv: IKvGroupLink[] = [];
        for (let g = 0; g < KV; g++) {
            kv.push({
                kWeight: w(`L${i}.k_w`, g),
                vWeight: w(`L${i}.v_w`, g),
                kNorm,
                k: a(`L${i}.k`, g),
                v: a(`L${i}.v`, g),
            });
        }
        blocks.push({
            xIn: a(`L${i}.x_in`),
            ln_1: { agg: a(`L${i}.ln1_agg`), weight: w(`L${i}.ln1_w`), output: a(`L${i}.ln1_out`) },
            heads,
            kv,
            oWeight: w(`L${i}.o_w`),
            oOut: a(`L${i}.o_out`),
            attnResid: a(`L${i}.attn_resid`),
            ln_2: { agg: a(`L${i}.ln2_agg`), weight: w(`L${i}.ln2_w`), output: a(`L${i}.ln2_out`) },
            gateWeight: w(`L${i}.gate_w`),
            upWeight: w(`L${i}.up_w`),
            downWeight: w(`L${i}.down_w`),
            gate: a(`L${i}.mlp_gate`),
            up: a(`L${i}.mlp_up`),
            act: a(`L${i}.mlp_act`),
            mlpOut: a(`L${i}.mlp_out`),
            mlpResid: a(`L${i}.mlp_resid`),
        });
    }

    return {
        gl,
        config,
        shape,
        inputTokens: tokenTex(gl, config),
        embedWeight: w('embed'),
        embedOut: a('embed_out'),
        blocks,
        ln_f: { agg: a('lnf_agg'), weight: w('lnf_w'), output: a('lnf_out') },
        hStar: a('h_star'),
        knackWeight: w('knack_w'),
        knackBias: w('knack_b'),
        knackLogits: a('knack_logits'),
        knackAgg: a('knack_agg'),
        knackProbs: a('knack_probs'),
        doneWeight: w('done_w'),
        doneBias: w('done_b'),
        doneLogits: a('done_logits'),
        doneAgg: a('done_agg'),
        doneProbs: a('done_probs'),
        expertIn: a('expert_in'),
        expertLayers: Array.from({ length: config.n_layer_pairs }, (_, i) => ({
            out: a(`E${i}.out`),
            attn: a(`E${i}.attn`),
            mass: get(A, `E${i}.attn_mass`).data,
        })),
        expertOut: a('expert_out'),
        actionWeight: w('action_w'),
        actionBias: w('action_b'),
        actionX1: a('action_x1'),
        actionVelocity: a('action_velocity'),
        actionChunk: a('action_chunk'),
    };
}

// Real tensor index of a texel (row/col inside the window texture).
export function texelToRealIndex(meta: ITensorMeta, texX: number, texY: number): number[] {
    if (meta.is1d) {
        return [...meta.prefix, texY + meta.rowOffset];
    }
    return [...meta.prefix, texY + meta.rowOffset, texX + meta.colOffset];
}
