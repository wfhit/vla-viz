// The REAL k005b model link: every texture below is filled from tools/export_k005b.py
// output (public/k005b-weights.json + public/k005b-acts.json) — real checkpoint
// weights and real activations from one real training row. Nothing is computed or
// synthesised here; the browser only uploads the exported windows to the GPU.

import { createBufferTex, IBufferTex, writeToBufferTex } from "@/src/utils/bufferTex";
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
    fm: IK005bFmConfig;
}

// flow matching facts measured by the exporter (tools/export_k005b.py export_flow_matching)
export interface IK005bFmConfig {
    t: number; steps: number; min_period: number; max_period: number;
    seeds: number[]; n_particles: number;
    mse: number; mse_zero: number; mse_mean_chunk: number; mse_per_particle: number[];
    particle_chunk_std: number; state_valid: number;
    cf_euler: { t: number; vnorm: number; mse: number }[];
    asserts: Record<string, number>;
}

interface ITensorStats { min: number; max: number; mean: number; std: number; }

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

// One expert (DiT) query head: q_proj (+bias, no q_norm) and its row of the joint attention,
// split at the prefix | suffix key boundary (prefix keys = the trunk's cached K/V of the same layer)
export interface IExpertHeadLink {
    qWeight: IRealTex;  // [Wd, Wc]
    qBias: IRealTex;    // [Wd]
    q: IRealTex;        // [n_suf, Wd]
    scoresP: IRealTex;  // [n_suf, Wt] scores vs the prefix-window keys
    scoresS: IRealTex;  // [n_suf, n_suf] scores vs the suffix keys
    attnAgg: IRealTex;  // [n_suf, 2] softmax agg over the FULL key row: max | 1/Σexp
    probsP: IRealTex;   // [n_suf, Wt]
    probsS: IRealTex;   // [n_suf, n_suf]
    attnOut: IRealTex;  // [n_suf, Wd]
}

// one of the expert's 2 kv heads (k/v come from the shared attention, which tiles them to the
// trunk's 8 groups; the tile copy 4g is exactly expert kv head g)
export interface IExpertKvLink {
    kWeight: IRealTex;
    kBias: IRealTex;
    vWeight: IRealTex;
    vBias: IRealTex;
    k: IRealTex;        // [n_suf, Wd]
    v: IRealTex;        // [n_suf, Wd]
}

// One layer of the action expert, in the same shape as a trunk layer (IBlockLayerLink).
export interface IExpertLayerLink {
    ln_1: IRmsNormLink;  // AdaRMSNorm: agg, base weight w, output (after the γβ(t) modulation)
    heads: IExpertHeadLink[];
    kv: IExpertKvLink[];
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
    attn: IRealTex;      // [n_suf, Wt + n_suf] head-mean probs: prefix window keys | suffix keys
    mass: Float32Array;  // [n_suf, 4] attention mass on [state tokens | visual | text | suffix]
}

// One Euler step of the K-step decode (k005b-fm.json): x_t, v, x_next and every layer's
// suffix lane at t_k. Lazily loaded; float16 on disk.
interface IFmStep {
    t: number;
    tex: Record<string, IRealTex>;
}

export interface IFmStepsLink {
    K: number;
    checkpointStep: number;
    inDistribution: boolean;
    steps: IFmStep[];
}

export interface IVlaModelLink {
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
    expertNorm: IRmsNormLink;
    expertOut: IRealTex;
    actionWeight: IRealTex;
    actionBias: IRealTex;
    actionX1: IRealTex;
    actionVelocity: IRealTex;
    actionChunk: IRealTex;
    fm: IFmLink;
    fmSteps?: IFmStepsLink;
}

// the suffix inputs and flow-matching tensors (all real, see export_flow_matching)
export interface IFmLink {
    state75: IRealTex;      // [75]
    timeEmb: IRealTex;      // [768] te of t=1
    stateEmb: IRealTex;     // [64 of 768] suffix token 0
    actionEmb: IRealTex;    // [16, 64 of 768]
    timeHidden: IRealTex;   // [16, 64 of 768]
    actionTime: IRealTex;   // [16, 64 of 768] suffix tokens 1..16
    stateProjW: IRealTex;   // [64 of 768, 75]
    actionInW: IRealTex;    // [64 of 768, 23]
    timeInW: IRealTex;      // [64 of 768, 64 of 1536]
    timeOutW: IRealTex;     // [64 of 768, 64 of 768]
    ada: IRealTex[];        // per layer [4, 64 of 768]: γ_in, β_in, γ_post, β_post
    target: IRealTex;       // [16, 23] label a
    u: IRealTex;            // [16, 23] x1 − a
    errDim: Float32Array;   // [23]
    particles: Float32Array; // [K, 3, 16, 23] x1 | v | chunk
    particlesMass: Float32Array; // [K, NL, 17, 4]
    meanChunk: IRealTex;    // [16, 23]
}

export interface IK005bData {
    config: IK005bConfig;
    weights: Record<string, ITensorEntry>;
    acts: Record<string, ITensorEntry>;
}

export async function fetchTensorFile<C = IK005bConfig>(url: string) {
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
        if (e.dtype !== 'torch.float32' && e.dtype !== 'torch.float16') {
            throw new Error(`tensor ${k}: unsupported dtype ${e.dtype}`);
        }
        out[k] = {
            name: k,
            shape: e.shape,
            data: e.dtype === 'torch.float16' ? decodeF16(e.data) : new Float32Array(base64ToArrayBuffer(e.data)),
            fullShape: e.full_shape,
            offset: e.offset,
            stats: e.stats,
            desc: e.desc,
        };
    }
    return { config: json.config as C, tensors: out };
}

function halfToFloat(h: number) {
    let s = (h & 0x8000) ? -1 : 1, e = (h >> 10) & 0x1f, f = h & 0x3ff;
    if (e === 0) return s * Math.pow(2, -14) * (f / 1024);
    if (e === 31) return f ? NaN : s * Infinity;
    return s * Math.pow(2, e - 15) * (1 + f / 1024);
}

export function decodeF16(b64: string) {
    let u16 = new Uint16Array(base64ToArrayBuffer(b64));
    let out = new Float32Array(u16.length);
    for (let i = 0; i < u16.length; i++) out[i] = halfToFloat(u16[i]);
    return out;
}

// the per-Euler-step capture (tools/export_k005b.py export_flow_matching), whole small tensors
// one float16 whole-tensor entry of a per-step capture (`ent16` in the exporters) → texture.
// `full_shape` / `offset` are present when the entry is a window of a larger tensor.
export function texFromF16Entry(gl: WebGL2RenderingContext, name: string, e: any): IRealTex {
    if (e.dtype !== 'torch.float16') throw new Error(`${name}: unsupported dtype ${e.dtype}`);
    return makeTex(gl, {
        name, shape: e.shape, data: decodeF16(e.data),
        fullShape: e.full_shape ?? e.shape, offset: e.offset ?? e.shape.map(() => 0), stats: e.stats, desc: e.desc,
    });
}

export async function fetchK005bFm(gl: WebGL2RenderingContext): Promise<IFmStepsLink> {
    let resp = await fetch('k005b-fm.json');
    if (!resp.ok) throw new Error(`failed to fetch k005b-fm.json: ${resp.status}`);
    let json = await resp.json();
    let steps: IFmStep[] = [];
    for (let st of json.steps) {
        let tex: Record<string, IRealTex> = {};
        for (let k in st.tensors) tex[k] = texFromF16Entry(gl, `${k} @t=${(+st.t).toFixed(1)}`, st.tensors[k]);
        steps.push({ t: st.t, tex });
    }
    return { K: json.K, checkpointStep: json.checkpoint_step, inDistribution: !!json.in_distribution, steps };
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

export function makeTex(gl: WebGL2RenderingContext, e: ITensorEntry, sub?: number): IRealTex {
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
export function embedColForToken(m: IVlaModelLink, t: number): number {
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

export function createK005bGpuModel(gl: WebGL2RenderingContext, data: IK005bData): IVlaModelLink {
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
            ln_1: { agg: a(`E${i}.ln1_agg`), weight: w(`E${i}.ln1_w`), output: a(`E${i}.ln1`) },
            heads: Array.from({ length: config.expert_heads }, (_, h) => ({
                qWeight: w(`E${i}.q_w`, h),
                qBias: w(`E${i}.q_b`, h),
                q: a(`E${i}.q`, h),
                scoresP: a(`E${i}.scp`, h),
                scoresS: a(`E${i}.scs`, h),
                attnAgg: a(`E${i}.attn_agg`, h),
                probsP: a(`E${i}.pp`, h),
                probsS: a(`E${i}.ps`, h),
                attnOut: a(`E${i}.vo`, h),
            })),
            kv: Array.from({ length: config.expert_kv_heads }, (_, g) => ({
                kWeight: w(`E${i}.k_w`, g),
                kBias: w(`E${i}.k_b`, g),
                vWeight: w(`E${i}.v_w`, g),
                vBias: w(`E${i}.v_b`, g),
                k: a(`E${i}.k`, g * (KV / config.expert_kv_heads)),
                v: a(`E${i}.v`, g * (KV / config.expert_kv_heads)),
            })),
            oWeight: w(`E${i}.o_w`),
            oOut: a(`E${i}.o`),
            attnResid: a(`E${i}.res1`),
            ln_2: { agg: a(`E${i}.ln2_agg`), weight: w(`E${i}.ln2_w`), output: a(`E${i}.ln2`) },
            gateWeight: w(`E${i}.gate_w`),
            upWeight: w(`E${i}.up_w`),
            downWeight: w(`E${i}.down_w`),
            gate: a(`E${i}.gate`),
            up: a(`E${i}.up`),
            act: a(`E${i}.act`),
            mlpOut: a(`E${i}.mlp`),
            mlpResid: a(`E${i}.out`),
            attn: a(`E${i}.attn`),
            mass: get(A, `E${i}.attn_mass`).data,
        })),
        expertNorm: { agg: a('expert_out_agg'), weight: w('expert_norm_w'), output: a('expert_out') },
        expertOut: a('expert_out'),
        actionWeight: w('action_w'),
        actionBias: w('action_b'),
        actionX1: a('action_x1'),
        actionVelocity: a('action_velocity'),
        actionChunk: a('action_chunk'),
        fm: {
            state75: a('fm_state75'),
            timeEmb: a('fm_time_emb'),
            stateEmb: a('fm_state_emb'),
            actionEmb: a('fm_action_emb'),
            timeHidden: a('fm_time_mlp_hidden'),
            actionTime: a('fm_action_time'),
            stateProjW: a('fm_state_proj_w'),
            actionInW: a('fm_action_in_w'),
            timeInW: a('fm_time_in_w'),
            timeOutW: a('fm_time_out_w'),
            ada: Array.from({ length: config.n_layer_pairs }, (_, i) => a(`E${i}.ada`)),
            target: a('fm_target'),
            u: a('fm_u'),
            errDim: get(A, 'fm_err_dim').data,
            particles: get(A, 'fm_particles').data,
            particlesMass: get(A, 'fm_particles_mass').data,
            meanChunk: a('fm_mean_chunk'),
        },
    };
}

// Real tensor index of a texel (row/col inside the window texture).
export function texelToRealIndex(meta: ITensorMeta, texX: number, texY: number): number[] {
    if (meta.is1d) {
        return [...meta.prefix, texY + meta.rowOffset];
    }
    return [...meta.prefix, texY + meta.rowOffset, texX + meta.colOffset];
}
