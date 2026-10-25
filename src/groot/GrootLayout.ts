import { Vec3 } from "@/src/utils/vector";
import { IRealTex } from "@/src/vla/VlaModel";
import { IBlkAccessDefArgs, IBlkDef, ISceneLayout, makeBlk } from "@/src/vla/VlaModelLayout";
import { DimStyle } from "@/src/vla/walkthrough/WalkthroughTools";
import { IGrootModel } from "./GrootModel";

// /groot scene. Every tensor is a horizontal plate (tokens / rows along x, channels along z,
// one cell thick) and data flows DOWN the scene:
//
//   System 2 (x ≈ 0):  token ids + 3 camera images -> ViT -> visual tokens -> LM L0..L15
//                      (L16..L27 pruned: ghosts) -> vlln -> vl_self_attention x4
//   System 1 (x ≈ X1): state / noisy-action / time front ends -> sa sequence -> 32 DiT blocks
//                      (image / text K/V pillars on the left, temb time bus + per-block γβ on
//                      the right) -> final AdaLN -> proj_out_2 -> action decoder -> v -> Euler
//   D0 / D1 anatomy panels to the right of System 1.

// 'skip' = a residual add: named on the destination instead of a line running past the sublayers
export type GEdgeKind = 'data' | 'weight' | 'time' | 'text' | 'image' | 'euler' | 'skip';

// src / dst may be a group (e.g. the γ/β rod pair): one arrow between the groups' outer boxes,
// recorded for every member
export interface IGEdge {
    src: IBlkDef[];
    dst: IBlkDef[];
    kind: GEdgeKind;
    // 'down': leave src's bottom (straight down, or down then across into dst's side)
    // 'side': leave src's side (straight across, or across then down into dst's top)
    // 'bus':  a tap off the temb time bus (the bus line is drawn once by the scene)
    route: 'side' | 'down' | 'bus';
}

export interface IGAnat {
    b: number;
    kind: 'text' | 'self' | 'image';
    ctx: IBlkDef | null; // cross blocks: the K/V source tokens (a copy of the VL pillar's tensor)
    xIn: IBlkDef; ada: IBlkDef; scale: IBlkDef; shift: IBlkDef; n1: IBlkDef;
    wq: IBlkDef; q: IBlkDef; wk: IBlkDef; k: IBlkDef; wv: IBlkDef; v: IBlkDef; probs: IBlkDef;
    wo: IBlkDef; attn: IBlkDef; r1: IBlkDef; n3: IBlkDef; ff1: IBlkDef; ffh: IBlkDef; ff2: IBlkDef;
    ffOut: IBlkDef; out: IBlkDef;
    cubes: IBlkDef[];
}

export interface IGDit {
    b: number;
    kind: 'text' | 'self' | 'image';
    out: IBlkDef;
    scale: IBlkDef;
    shift: IBlkDef;
}

export interface IGrootLayout extends ISceneLayout {
    ids: IBlkDef;
    vitPatch: IBlkDef; vitB0: IBlkDef; vitB23: IBlkDef; visTok: IBlkDef;
    vitGhostY: [number, number]; // y range of the 22 vision blocks drawn as ghosts between B0 and B23
    lm: IBlkDef[];
    attn0: IBlkDef; attn15: IBlkDef;
    lmGhostY: number[]; // y of the 12 pruned LM layers (L16..L27)
    lmOut: IBlkDef; vlln: IBlkDef; vlsa: IBlkDef[];
    vlImage: IBlkDef; vlText: IBlkDef;

    stateRaw: IBlkDef; stateNorm: IBlkDef; stateL1: IBlkDef; stateH: IBlkDef; stateL2: IBlkDef; stateFeat: IBlkDef;
    xt: IBlkDef; aW1: IBlkDef; aEmb: IBlkDef; tau: IBlkDef; aW2: IBlkDef; w2: IBlkDef; aW3: IBlkDef; aFeat: IBlkDef;
    posEmb: IBlkDef; aFeatPos: IBlkDef; saIn: IBlkDef;
    tProj: IBlkDef; timeLin1: IBlkDef; temb: IBlkDef;
    dit: IGDit[];
    timeBusX: number;
    finScale: IBlkDef; finShift: IBlkDef; finNorm: IBlkDef; projOut2: IBlkDef; ditOut: IBlkDef;
    decL1: IBlkDef; decH: IBlkDef; decL2: IBlkDef; v: IBlkDef;
    track: IBlkDef[]; // x_t0 .. x_t4 (x_t4 = the action chunk)
    anat: IGAnat[];

    // the embodiment-slot weights (slot 23 of 32; untrained -> toy)
    slotted: IBlkDef[];

    system2: IBlkDef[];
    frontEnd: IBlkDef[];
    stepOnly0: IBlkDef[]; // front-end tensors exported for Euler step 0 only
    ditStack: IBlkDef[];
    decode: IBlkDef[];

    edges: IGEdge[];
    srcOk: Set<IBlkDef>;
    sinkOk: Set<IBlkDef>;
    step: number; // Euler step whose tensors are bound to the step-dependent blocks
    // steps k > 0: the action-encoder internals (exported for step 0 only) are hidden and
    // x_t -> sa sequence is drawn as one collapsed edge
    collapsedFront: boolean;
}

const cell = 1.5;

export function genGrootLayout(m: IGrootModel | null): IGrootLayout {
    let cfg = m?.cfg;
    let S = cfg?.S ?? 216;
    let nImg = cfg ? cfg.image_runs.reduce((a, r) => a + r[1] - r[0], 0) : 192;
    let nTxt = S - nImg;
    let H = cfg?.action_horizon ?? 40;
    let AD = cfg?.action_dim ?? 132;
    let kinds = cfg?.dit.kinds ?? Array.from({ length: 32 }, (_, i) => (['text', 'self', 'image', 'self'] as const)[i % 4]);
    let a = m?.acts ?? {}, w = m?.w ?? {}, st = m?.steps[0]?.tex ?? {};

    let cubes: IBlkDef[] = [];
    let edges: IGEdge[] = [];

    // ---- access helpers (window textures: rows = dim -2, cols = last dim) ----
    // Blocks are walls, as in /vla: tokens along x, channels along y (down), z one cell thick.
    let sc = (t: IRealTex) => t.meta.absScale;
    // [T, C]: T along x, C along y
    let slab = (t?: IRealTex): IBlkAccessDefArgs | undefined => t && { src: t, x: [0, 1, 0], y: [1, 0, 0], scale: sc(t) };
    // W[r, c]: c along x, r along y  (also [T, C] pillars: C along x, T along y)
    let wmat = (t?: IRealTex): IBlkAccessDefArgs | undefined => t && { src: t, x: [1, 0, 0], y: [0, 1, 0], scale: sc(t) };
    // 1D along x / along y
    let vecX = (t?: IRealTex): IBlkAccessDefArgs | undefined => t && { src: t, x: [0], y: [1, 0, 0], scale: sc(t) };
    let vecY = (t?: IRealTex): IBlkAccessDefArgs | undefined => t && { src: t, x: [0], y: [0, 1, 0], scale: sc(t) };

    // display windows (channels shown): repeated stacks / single plates / anatomy panels
    let WS = 16, WF = 32, WA = 64;
    let hS = WS * cell, hF = WF * cell, hA = WA * cell;
    let g = 24; // vertical gap between consecutive rows (arrow + label)

    function P(name: string, x: number, y: number, cx: number, cy: number, acc: IBlkAccessDefArgs | undefined,
            t: 'i' | 'w' | 'a' = 'i', dimX = DimStyle.T, dimY = DimStyle.C): IBlkDef {
        let b = makeBlk({ t, xL: x, zB: 0, y, cx, cy, cz: 1, dimX, dimY, name, access: acc }, cell);
        b.idx = cubes.length;
        cubes.push(b);
        return b;
    }
    let arr = (b: IBlkDef | IBlkDef[]) => Array.isArray(b) ? b : [b];
    let E = (src: IBlkDef | IBlkDef[], dst: IBlkDef | IBlkDef[], kind: GEdgeKind = 'data', route: 'side' | 'down' | 'bus' = 'down') =>
        edges.push({ src: arr(src), dst: arr(dst), kind, route });
    let wid = (cx: number) => cx * cell;
    let below = (b: IBlkDef, gap = g) => b.y + b.dy + gap;

    // =================== System 2 ===================
    let xv = -520;
    let vitPatch = P('vit.patch_embed', xv, 60, 256, WF, slab(a['vit.patch']));
    let vitB0 = P('vit.block0', xv, below(vitPatch), 256, WF, slab(a['vit.B0.out']));
    let vitB23 = P('vit.block23', xv, below(vitB0, 60), 256, WF, slab(a['vit.B23.out']));
    let yEmb = below(vitB23);
    let visTok = P('visual_tokens', xv + wid(32), yEmb, nImg, WS, slab(a['bb.visual_tokens']));
    E(vitPatch, vitB0); E(vitB0, vitB23); E(vitB23, visTok);
    let ids = P('input_ids', 0, yEmb - 30, S, 1, vecX(m?.tokIds), 'i', DimStyle.TokenIdx, DimStyle.None);

    let pitchLM = hS + 16;
    let lm: IBlkDef[] = [];
    for (let i = 0; i < 16; i++) {
        lm.push(P(i === 0 ? 'lm.L0 in (embeddings)' : `lm.L${i} in`, 0, yEmb + i * pitchLM, S, WS, slab(a[`lm.L${i}.x_in`])));
    }
    E(ids, lm[0], 'data', 'down');
    E(visTok, lm[0], 'data', 'side');
    let attn0 = P('lm.L0 attention (head mean)', wid(S) + 40, lm[0].y, S, S, wmat(a['lm.L0.attn']), 'a', DimStyle.T, DimStyle.T);
    let attn15 = P('lm.L15 attention (head mean)', wid(S) + 40, lm[15].y, S, S, wmat(a['lm.L15.attn']), 'a', DimStyle.T, DimStyle.T);
    // the attention maps are side readouts of L0 / L15 (aggregates: no outgoing edge)
    E(lm[0], attn0, 'data', 'side');
    for (let i = 0; i < 15; i++) E(lm[i], lm[i + 1]);
    let lmGhostY = Array.from({ length: 12 }, (_, j) => below(lm[15], 10) + j * 8);
    let yOut = lmGhostY[11] + 4 + 30;
    let lmOut = P('lm.out (final norm)', 0, yOut, S, WS, slab(a['lm.out']));
    E(lm[15], attn15, 'data', 'side');
    E(lm[15], lmOut);
    let vlln = P('vlln (LayerNorm)', 0, below(lmOut, 16), S, WS, slab(a['vlln']));
    E(lmOut, vlln);
    let vlsa: IBlkDef[] = [];
    for (let i = 0; i < 4; i++) {
        vlsa.push(P(`vl_self_attention.B${i}`, 0, below(i === 0 ? vlln : vlsa[i - 1], 16), S, WS, slab(a[`vlsa.B${i}.out`])));
        E(i === 0 ? vlln : vlsa[i - 1], vlsa[i]);
    }

    // =================== System 1 ===================
    let X1 = 1100; // the state chain starts right of the L15 attention map
    let y1 = yOut;

    // state (61 BEHAVIOR dims -> 132 padded) -> state_encoder[slot 23] -> 1 token
    let xs = X1 - 380, xsw = xs + wid(132) + 40;
    let stateRaw = P('state (61, R1 Pro)', xs, y1, 61, 1, vecX(a['state.raw']), 'i', DimStyle.C, DimStyle.None);
    let stateNorm = P('state normalized (132)', xs, y1 + 30, 132, 1, vecX(a['state.norm']), 'i', DimStyle.C, DimStyle.None);
    let stateH = P('state hidden (ReLU)', xs, y1 + 60, WA, 1, vecX(a['state.h']), 'i', DimStyle.C, DimStyle.None);
    let stateL1 = P('state_enc.L1 [slot 23]', xsw, stateH.y, WF, WF, wmat(w['state_enc.L1']), 'w', DimStyle.C, DimStyle.C);
    let stateFeat = P('state token', xs, below(stateL1, 16), WA, 1, vecX(a['state.feat']), 'i', DimStyle.C, DimStyle.None);
    let stateL2 = P('state_enc.L2 [slot 23]', xsw, stateFeat.y, WF, WF, wmat(w['state_enc.L2']), 'w', DimStyle.C, DimStyle.C);
    E(stateRaw, stateNorm); E(stateNorm, stateH); E(stateL1, stateH, 'weight', 'side'); E(stateH, stateFeat); E(stateL2, stateFeat, 'weight', 'side');

    // noisy action chunk -> action_encoder[slot 23](a, tau) + pos
    let xa = X1;
    let xt = P('x_t (noisy actions 40x132)', xa, y1, H, AD, slab(a['x0']), 'i', DimStyle.T, DimStyle.C);
    let xw = xa + wid(H) + 70; // weight column (room for the name tags)
    let aEmb = P('W1(a)', xa, below(xt), H, WF, slab(a['s0.a_emb']));
    let aW1 = P('action_enc.W1 [slot 23]', xw, aEmb.y, WF, WF, wmat(w['action_enc.W1']), 'w', DimStyle.C, DimStyle.C);
    let w2 = P('swish(W2[a|tau])', xa, below(aEmb), H, WF, slab(a['s0.w2']));
    let aW2 = P('action_enc.W2 [slot 23]', xw, w2.y, WF, WF, wmat(w['action_enc.W2']), 'w', DimStyle.C, DimStyle.C);
    let tau = P('tau = sincos(t)', xa - 70, w2.y, 1, WF, vecY(st['tau']), 'i', DimStyle.None, DimStyle.C);
    let aFeat = P('W3(...)', xa, below(w2), H, WF, slab(a['s0.a_feat']));
    let aW3 = P('action_enc.W3 [slot 23]', xw, aFeat.y, WF, WF, wmat(w['action_enc.W3']), 'w', DimStyle.C, DimStyle.C);
    let aFeatPos = P('action tokens (+pos)', xa, below(aFeat), H, WF, slab(a['s0.a_feat_pos']));
    let posEmb = P('pos_emb', xw, aFeatPos.y, H, WF, slab(w['pos_emb']), 'w');
    E(xt, aEmb); E(aW1, aEmb, 'weight', 'side');
    E(aEmb, w2); E(tau, w2, 'time', 'side'); E(aW2, w2, 'weight', 'side');
    E(w2, aFeat); E(aW3, aFeat, 'weight', 'side');
    E(aFeat, aFeatPos); E(posEmb, aFeatPos, 'weight', 'side');

    // time t -> Timesteps(256) -> TimestepEmbedding -> temb
    let xT = X1 + 300;
    let tProj = P('Timesteps(t) 256', xT, y1, 256, 1, vecX(a['s0.t_proj']), 'i', DimStyle.C, DimStyle.None);
    let temb = P('temb', xT, y1 + 60, WA, 1, vecX(st['temb']), 'i', DimStyle.C, DimStyle.None);
    let timeLin1 = P('time.linear_1', xT - wid(WF) - 12, temb.y, WF, WF, wmat(w['time.lin1']), 'w', DimStyle.C, DimStyle.C);
    E(tProj, temb); E(timeLin1, temb, 'weight', 'side');

    let ySa = below(aFeatPos, 40);
    let saIn = P('sa sequence [state | 40 actions]', X1, ySa, H + 1, WS, slab(st['sa_in']));
    E(stateFeat, saIn); E(aFeatPos, saIn);

    // 32 DiT blocks; a γβ rod pair on the right of each, fed by the temb time bus
    let pitchD = hS + 12;
    let yD = below(saIn);
    let dit: IGDit[] = [];
    let xRod = X1 + wid(H + 1) + 14;
    for (let b = 0; b < 32; b++) {
        let y = yD + b * pitchD;
        let kind = kinds[b];
        let out = P(`DiT block ${b} (${kind === 'self' ? 'self-attn' : kind === 'text' ? 'cross: text' : 'cross: image'})`,
            X1, y, H + 1, WS, slab(st[`D${b}.out`]));
        let scale = P(`D${b} AdaLN scale`, xRod, y, 1, WS, vecY(st[`D${b}.scale`]), 'a', DimStyle.None, DimStyle.C);
        let shift = P(`D${b} AdaLN shift`, xRod + 5.5, y, 1, WS, vecY(st[`D${b}.shift`]), 'a', DimStyle.None, DimStyle.C);
        E(b === 0 ? saIn : dit[b - 1].out, out);
        E([scale, shift], out, 'time', 'side');
        dit.push({ b, kind, out, scale, shift });
    }
    // the bus itself is drawn by the scene; its taps are the temb -> rod-pair edges
    let timeBusX = xRod + 28;

    // K/V pillars, one column left of the stack: the 24 text tokens on top of the 192 image
    // tokens (channels across), both squeezed to the stack's height
    let hStack = 31 * pitchD + hS;
    let pillarAt = (name: string, y: number, n: number, t?: IRealTex) => {
        let c = makeBlk({ t: 'i', xL: X1 - 90, zB: 0, y, cx: WS, cy: n, cz: 1, dimX: DimStyle.C, dimY: DimStyle.T, name, access: wmat(t) }, cell);
        c.dy = hStack * n / S;
        c.idx = cubes.length;
        cubes.push(c);
        return c;
    };
    let vlText = pillarAt('VL text tokens (K/V)', yD, nTxt, a['vl.text']);
    let vlImage = pillarAt('VL image tokens (K/V)', yD + vlText.dy, nImg, a['vl.image']);
    E(vlsa[3], vlText); E(vlsa[3], vlImage);
    for (let d of dit) {
        if (d.kind === 'image') E(vlImage, d.out, 'image', 'side');
        if (d.kind === 'text') E(vlText, d.out, 'text', 'side');
    }

    // final AdaLN -> proj_out_2 -> action decoder[slot 23] -> v -> Euler
    let yF = yD + 32 * pitchD + 12;
    let finScale = P('final AdaLN scale', xRod, yF, 1, WS, vecY(st['final_scale']), 'a', DimStyle.None, DimStyle.C);
    let finShift = P('final AdaLN shift', xRod + 5.5, yF, 1, WS, vecY(st['final_shift']), 'a', DimStyle.None, DimStyle.C);
    let finNorm = P('final AdaLN(x)', X1, yF, H + 1, WS, slab(st['final_norm']));
    E(dit[31].out, finNorm);
    E([finScale, finShift], finNorm, 'time', 'side');
    let xdw = X1 + wid(H + 1) + 120; // room for the decoder name tags
    let ditOut = P('DiT output (1024)', X1, below(finNorm), H + 1, WF, slab(st['dit_out']));
    let projOut2 = P('proj_out_2', xdw, ditOut.y, WF, WF, wmat(w['proj_out_2']), 'w', DimStyle.C, DimStyle.C);
    E(finNorm, ditOut); E(projOut2, ditOut, 'weight', 'side');
    let decH = P('decoder hidden (ReLU)', X1, below(ditOut), H + 1, WF, slab(st['dec_h']));
    let decL1 = P('action_dec.L1 [slot 23]', xdw, decH.y, WF, WF, wmat(w['action_dec.L1']), 'w', DimStyle.C, DimStyle.C);
    E(ditOut, decH); E(decL1, decH, 'weight', 'side');
    let v = P('v (velocity 40x132)', X1, below(decH), H, AD, slab(st['v']));
    let decL2 = P('action_dec.L2 [slot 23]', xdw, v.y, WF, WF, wmat(w['action_dec.L2']), 'w', DimStyle.C, DimStyle.C);
    E(decH, v); E(decL2, v, 'weight', 'side');

    // Euler track, below-left of the decode column: x_t0 -> ... -> x_t4 = the action chunk. Step k's v
    // drops into x_t(k+1) (the scene draws that one edge); each track arrow = "+ v/4"
    let yE = below(v, 30);
    let track: IBlkDef[] = [];
    for (let j = 0; j <= 4; j++) {
        let tex = j < 4 ? m?.steps[j]?.tex['x_t'] : m?.steps[3]?.tex['x_next'];
        track.push(P(j < 4 ? `x_t${j} (t=${(j / 4).toFixed(2)})` : 'action chunk (t=1)', X1 - 600 + j * 110, yE, H, AD, slab(tex)));
        if (j > 0) E(track[j - 1], track[j], 'euler', 'side');
    }

    // =================== anatomy panels: D0 (cross: text) and D1 (self) at step 0 ===================
    // Each panel is a zoom-in of one stack block at step 0 (64-channel window): its "x in" is the
    // stack's input to that block and its "x out" the block's plate (labelled, not wired back).
    // Columns: weights + AdaLN on the left, the data path in the middle, K / V on the right.
    let anat: IGAnat[] = [];
    let anatX = [X1 + 620, X1 + 620 + 1500];
    let hq = 48 * cell, hp = (H + 1) * cell, gA = 30;
    // row tops: x in, AdaLN, QKV, probs, attn, x+attn, LN, GELU, FF out, x out
    let rowH = [hA, hA, hq, hp, hA, hA, hA, hA, hA, hA];
    let rowY: number[] = [];
    rowH.reduce((y, h) => (rowY.push(y), y + h + gA), 0);
    for (let b = 0; b < 2; b++) {
        let ax = anatX[b];
        let y0 = ySa;
        let p = `s0.D${b}.`;
        let kind = kinds[b];
        let nK = kind === 'self' ? H + 1 : S;
        let mk = (n: string, row: number, cx: number, cy: number, acc: IBlkAccessDefArgs | undefined, t: 'i' | 'w' | 'a' = 'i', dx = 0) =>
            P(`D${b} ${n}`, ax + dx, y0 + rowY[row], cx, cy, acc, t, t === 'w' ? DimStyle.C : DimStyle.T, DimStyle.C);
        let dW = -wid(WA) - 40; // weight column
        let kx = Math.max(wid(H + 1), wid(nK)) + 40; // K column (right of the widest data plate: probs)
        let vx = kx + wid(nK) + 12 + wid(WA) + 40; // V column (right of to_k)
        let xIn = mk('x in', 0, H + 1, WA, slab(a[p + 'x_in']));
        let ctx = kind === 'self' ? null : mk('K/V source: VL tokens', 1, S, WA, slab(a['vlsa.B3.out']), 'i', kx);
        let n1 = mk('AdaLN(x)', 1, H + 1, WA, slab(a[p + 'n1']));
        let ada = mk('norm1.linear', 1, WA, WA, wmat(w[`D${b}.ada`]), 'w', dW);
        let scale = mk('scale', 1, 1, WA, vecY(a[p + 'scale']), 'a', -24);
        let shift = mk('shift', 1, 1, WA, vecY(a[p + 'shift']), 'a', -18.5);
        let q = mk('Q head0', 2, H + 1, 48, slab(a[p + 'q']));
        let wq = mk('to_q', 2, WA, WA, wmat(w[`D${b}.to_q`]), 'w', dW);
        let k = mk('K head0', 2, nK, 48, slab(a[p + 'k']), 'i', kx);
        let wk = mk('to_k', 2, WA, WA, wmat(w[`D${b}.to_k`]), 'w', kx + wid(nK) + 12);
        let vv = mk('V head0', 2, nK, 48, slab(a[p + 'v']), 'i', vx);
        let wv = mk('to_v', 2, WA, WA, wmat(w[`D${b}.to_v`]), 'w', vx + wid(nK) + 12);
        let probs = mk('softmax(QK^T) head0', 3, nK, H + 1, wmat(a[p + 'probs']), 'a');
        let attn = mk('attention out', 4, H + 1, WA, slab(a[p + 'attn']));
        let wo = mk('to_out', 4, WA, WA, wmat(w[`D${b}.to_out`]), 'w', dW);
        let r1 = mk('x + attn', 5, H + 1, WA, slab(a[p + 'r1']));
        let n3 = mk('LayerNorm (no affine)', 6, H + 1, WA, slab(a[p + 'n3']));
        let ffh = mk('GELU hidden', 7, H + 1, WA, slab(a[p + 'ff_h']));
        let ff1 = mk('ff.proj (1536->6144)', 7, WA, WA, wmat(w[`D${b}.ff1`]), 'w', dW);
        let ffOut = mk('FF out', 8, H + 1, WA, slab(a[p + 'ff_out']));
        let ff2 = mk('ff.out (6144->1536)', 8, WA, WA, wmat(w[`D${b}.ff2`]), 'w', dW);
        let out = mk('x out', 9, H + 1, WA, slab(a[p + 'out']));
        let kvKind: GEdgeKind = kind === 'self' ? 'data' : kind;
        E(temb, ada, 'time', 'side'); E(ada, [scale, shift], 'weight', 'side'); E([scale, shift], n1, 'time', 'side');
        E(xIn, n1);
        E(n1, q); E(wq, q, 'weight', 'side');
        if (ctx) { E(ctx, k, kvKind); E(ctx, vv, kvKind, 'side'); }
        else { E(n1, k, 'data', 'side'); E(n1, vv, 'data', 'side'); }
        E(wk, k, 'weight', 'side'); E(wv, vv, 'weight', 'side');
        E(q, probs); E(k, probs);
        E(probs, attn); E(vv, attn); E(wo, attn, 'weight', 'side');
        E(attn, r1); E(xIn, r1, 'skip'); E(r1, n3); E(n3, ffh); E(ff1, ffh, 'weight', 'side');
        E(ffh, ffOut); E(ff2, ffOut, 'weight', 'side'); E(ffOut, out); E(r1, out, 'skip');
        let list = [...(ctx ? [ctx] : []), xIn, ada, scale, shift, n1, wq, q, wk, k, wv, vv, probs, wo, attn, r1, n3, ff1, ffh, ff2, ffOut, out];
        anat.push({ b, kind, ctx, xIn, ada, scale, shift, n1, wq, q, wk, k, wv, v: vv, probs, wo, attn, r1, n3, ff1, ffh, ff2, ffOut, out, cubes: list });
    }

    // stack plates and γβ rods are tagged by the scene (block number + kind), not one name each
    for (let d of dit) d.out.noTag = d.scale.noTag = d.shift.noTag = true;
    finScale.noTag = finShift.noTag = true;
    for (let p of anat) p.scale.noTag = p.shift.noTag = true;

    let system2 = [ids, vitPatch, vitB0, vitB23, visTok, ...lm, attn0, attn15, lmOut, vlln, ...vlsa];
    let frontEnd = [stateRaw, stateNorm, stateL1, stateH, stateL2, stateFeat, xt, aW1, aEmb, tau, aW2, w2, aW3, aFeat, posEmb, aFeatPos, tProj, timeLin1, temb, saIn];
    let stepOnly0 = [aEmb, w2, aFeat, aFeatPos, tProj];
    let ditStack = [vlText, vlImage, ...dit.flatMap(d => [d.out, d.scale, d.shift])];
    // temb feeds every γβ pair through the bus
    for (let d of dit) E(temb, [d.scale, d.shift], 'time', 'bus');
    E(temb, [finScale, finShift], 'time', 'bus');
    let decode = [finScale, finShift, finNorm, projOut2, ditOut, decL1, decH, decL2, v, ...track];

    return {
        cell, height: yE + AD * cell + 100, margin: 12, cubes, labels: [],
        ids, vitPatch, vitB0, vitB23, visTok, vitGhostY: [vitB0.y + vitB0.dy + 10, vitB23.y - 14], lm, attn0, attn15, lmGhostY, lmOut, vlln, vlsa, vlImage, vlText,
        stateRaw, stateNorm, stateL1, stateH, stateL2, stateFeat, xt, aW1, aEmb, tau, aW2, w2, aW3, aFeat, posEmb, aFeatPos, saIn,
        tProj, timeLin1, temb, dit, timeBusX, finScale, finShift, finNorm, projOut2, ditOut, decL1, decH, decL2, v, track, anat,
        slotted: [stateL1, stateL2, aW1, aW2, aW3, decL1, decL2],
        system2, frontEnd, stepOnly0, ditStack, decode,
        edges,
        // sources by construction: token ids, the ViT patch embedding (fed by the 3 camera images
        // drawn beside it), the BEHAVIOR state, the noise x_t (= track[0]), time t (t_proj, tau);
        // the anatomy panels' inputs are labelled copies of stack tensors
        srcOk: new Set([ids, vitPatch, stateRaw, xt, track[0], tProj, tau,
            ...anat.flatMap(p => p.ctx ? [p.xIn, p.ctx] : [p.xIn])]),
        // sinks: the action chunk; the panels' "x out" are the D0 / D1 plates of the stack
        sinkOk: new Set([track[4], ...anat.map(p => p.out)]),
        step: 0,
        collapsedFront: false,
    };
}

// bind Euler step k's tensors to the step-dependent blocks
export function applyGrootStep(L: IGrootLayout, m: IGrootModel | null, k: number) {
    L.step = k;
    L.collapsedFront = k > 0;
    if (k > 0) for (let b of [...L.stepOnly0, L.tau, L.aW1, L.aW2, L.aW3, L.posEmb]) b.opacity = 0;
    let s = m?.steps[k]?.tex;
    if (!s) return;
    let set = (b: IBlkDef, t?: IRealTex) => {
        if (!t || !b.access) return;
        b.access.src = t;
        b.access.scale = t.meta.absScale;
    };
    set(L.xt, s['x_t']);
    set(L.tau, s['tau']);
    set(L.temb, s['temb']);
    set(L.saIn, s['sa_in']);
    for (let d of L.dit) {
        set(d.out, s[`D${d.b}.out`]);
        set(d.scale, s[`D${d.b}.scale`]);
        set(d.shift, s[`D${d.b}.shift`]);
    }
    set(L.finScale, s['final_scale']);
    set(L.finShift, s['final_shift']);
    set(L.finNorm, s['final_norm']);
    set(L.ditOut, s['dit_out']);
    set(L.decH, s['dec_h']);
    set(L.v, s['v']);
}

export function blkCenter(b: IBlkDef) {
    return new Vec3(b.x + b.dx / 2, b.y + b.dy / 2, b.z + b.dz / 2);
}
