import { Vec3 } from "@/src/utils/vector";
import { IRealTex } from "@/src/vla/VlaModel";
import { IBlkAccessDefArgs, IBlkDef, ISceneLayout, makeBlk } from "@/src/vla/VlaModelLayout";
import { DimStyle } from "@/src/vla/walkthrough/WalkthroughTools";
import { IPi0Model } from "./Pi0Model";

// /pi0 scene (π0.5). Every tensor is a wall (tokens along x, channels along y, one cell thick);
// data flows DOWN the scene:
//
//   left:   3 camera frames -> SigLIP (head view internals; wrist views' patch embeds) -> 768
//           image tokens; prompt ids (task + 23 discretised state tokens) -> text embeddings
//   middle: the prefix tower (PaliGemma Gemma-2B, 18 layers) — its plates carry a row subset
//   right:  the action-expert tower (Gemma-300M, 18 layers), each plate level with the prefix
//           layer it attends to jointly (K/V edges), adaRMS scale/shift/gate rods fed by the
//           cond time bus; final adaRMS -> action_out_proj -> v -> 10-step Euler track
//   far right: the layer-0 anatomy, the two streams side by side

// 'kv'  = prefix keys / values read by the expert's joint attention
// 'skip'= a residual add, named on the destination
export type PEdgeKind = 'data' | 'weight' | 'time' | 'text' | 'image' | 'kv' | 'euler' | 'skip';

export interface IPEdge {
    src: IBlkDef[];
    dst: IBlkDef[];
    kind: PEdgeKind;
    route: 'side' | 'down' | 'bus';
}

export interface IPExp {
    l: number;
    out: IBlkDef;
    scale: IBlkDef;
    shift: IBlkDef;
    gate: IBlkDef;
}

// the layer-0 anatomy: one column per stream
export interface IPAnatStream {
    xIn: IBlkDef; norm: IBlkDef; normW: IBlkDef; q: IBlkDef; wq: IBlkDef; k: IBlkDef; wk: IBlkDef; v: IBlkDef; wv: IBlkDef;
    probs: IBlkDef; ao: IBlkDef; wo: IBlkDef; r1: IBlkDef; norm2: IBlkDef; mlpH: IBlkDef; wGate: IBlkDef; wUp: IBlkDef;
    ff: IBlkDef; wDown: IBlkDef; out: IBlkDef;
    cubes: IBlkDef[];
}

export interface IPAnat {
    pre: IPAnatStream;
    exp: IPAnatStream & { adaA: IBlkDef; adaF: IBlkDef; scaleA: IBlkDef; shiftA: IBlkDef; gateA: IBlkDef; scaleF: IBlkDef; shiftF: IBlkDef; gateF: IBlkDef };
    mask: IBlkDef;
    cubes: IBlkDef[];
}

export interface IPi0Layout extends ISceneLayout {
    patch: IBlkDef[]; // head, left wrist, right wrist
    stem: IBlkDef; vitB0: IBlkDef; vitB26: IBlkDef; vitEnc: IBlkDef; vitHead: IBlkDef; imgTok: IBlkDef;
    vitGhostY: [number, number];
    state: IBlkDef; ids: IBlkDef; txtW: IBlkDef; txtEmb: IBlkDef;
    pre: IBlkDef[]; preOut: IBlkDef;
    xt: IBlkDef; inProj: IBlkDef; aTok: IBlkDef;
    pe: IBlkDef; tIn: IBlkDef; tOut: IBlkDef; cond: IBlkDef;
    exp: IPExp[];
    busX: number;
    finScale: IBlkDef; finShift: IBlkDef; final: IBlkDef; outProj: IBlkDef; v: IBlkDef;
    track: IBlkDef[]; // x_t at t = 1.0 .. 0.1, then x at t = 0
    act23: IBlkDef;
    anat: IPAnat;

    vision: IBlkDef[];
    text: IBlkDef[];
    prefixStack: IBlkDef[];
    frontEnd: IBlkDef[];
    expStack: IBlkDef[];
    decode: IBlkDef[];

    edges: IPEdge[];
    srcOk: Set<IBlkDef>;
    sinkOk: Set<IBlkDef>;
    step: number;
}

const cell = 1.5;

export function genPi0Layout(m: IPi0Model | null): IPi0Layout {
    let cfg = m?.cfg;
    let T = cfg?.T ?? 117;
    let nImg = cfg?.n_img ?? 768;
    let Np = cfg?.Np ?? 885;
    let nRow = cfg?.prefix_rows.length ?? 165; // prefix tensors carry 16 patches per view + all text rows
    let nPv = (nRow - T) / 3;
    let H = cfg?.action_horizon ?? 32, AD = cfg?.action_dim ?? 32, D23 = cfg?.behavior_action_dim ?? 23;
    let NL = cfg?.prefix.n_layer ?? 18;
    let K = cfg?.K ?? 10;
    let a = m?.acts ?? {}, w = m?.w ?? {}, st = m?.steps[0]?.tex ?? {};

    let cubes: IBlkDef[] = [];
    let edges: IPEdge[] = [];

    let sc = (t: IRealTex) => t.meta.absScale;
    let slab = (t?: IRealTex): IBlkAccessDefArgs | undefined => t && { src: t, x: [0, 1, 0], y: [1, 0, 0], scale: sc(t) };
    let wmat = (t?: IRealTex): IBlkAccessDefArgs | undefined => t && { src: t, x: [1, 0, 0], y: [0, 1, 0], scale: sc(t) };
    let vecX = (t?: IRealTex): IBlkAccessDefArgs | undefined => t && { src: t, x: [0], y: [1, 0, 0], scale: sc(t) };
    let vecY = (t?: IRealTex): IBlkAccessDefArgs | undefined => t && { src: t, x: [0], y: [0, 1, 0], scale: sc(t) };

    // channels shown: repeated stacks / single plates / anatomy
    let WS = 16, WF = 32, WA = 64;
    let hS = WS * cell, hA = WA * cell;
    let g = 24;

    function P(name: string, x: number, y: number, cx: number, cy: number, acc: IBlkAccessDefArgs | undefined,
            t: 'i' | 'w' | 'a' = 'i', dimX = DimStyle.T, dimY = DimStyle.C): IBlkDef {
        let b = makeBlk({ t, xL: x, zB: 0, y, cx, cy, cz: 1, dimX, dimY, name, access: acc }, cell);
        b.idx = cubes.length;
        cubes.push(b);
        return b;
    }
    let arr = (b: IBlkDef | IBlkDef[]) => Array.isArray(b) ? b : [b];
    let E = (src: IBlkDef | IBlkDef[], dst: IBlkDef | IBlkDef[], kind: PEdgeKind = 'data', route: 'side' | 'down' | 'bus' = 'down') =>
        edges.push({ src: arr(src), dst: arr(dst), kind, route });
    let wid = (cx: number) => cx * cell;
    let below = (b: IBlkDef, gap = g) => b.y + b.dy + gap;

    // =================== vision: SigLIP So400m/14, one tower shared by the 3 views ===================
    let nP = cfg?.n_img_per_view ?? 256;
    let xv0 = -1300, dv = wid(nP) + 36;
    let views = ['head', 'left_wrist', 'right_wrist'];
    let yP = 80;
    let patch = views.map((v, i) => P(`${v} patch embed (+pos)`, xv0 + i * dv, yP, nP, WF, slab(a[`vit.${v}.patch`])));
    let stem = patch[0]; // the head view's internals are drawn below its patch plate
    let vitB0 = P('SigLIP block 0', xv0, below(stem), nP, WF, slab(a['vit.head.B0']));
    let vitB26 = P('SigLIP block 26', xv0, below(vitB0, 70), nP, WF, slab(a['vit.head.B26']));
    let vitEnc = P('encoder_norm', xv0, below(vitB26), nP, WF, slab(a['vit.head.enc']));
    E(stem, vitB0); E(vitB0, vitB26); E(vitB26, vitEnc);
    let yImg = below(vitEnc, 40);
    let imgTok = P('image tokens (768 = 3 x 256)', xv0, yImg, nImg, WS, slab(a['img_tok']));
    let vitHead = P('img head 1152->2048', xv0 - wid(WF) - 40, imgTok.y - 20, WF, WF, wmat(w['vit.head']), 'w', DimStyle.C, DimStyle.C);
    E(vitEnc, imgTok); E(vitHead, imgTok, 'weight', 'side');
    // the wrist views run the same 27 blocks; only their patch embeddings are exported
    E(patch[1], imgTok); E(patch[2], imgTok);

    // =================== text: task + discretised state, one prompt ===================
    let X0 = 0;
    let yPre = below(imgTok, 60);
    let xTxt = X0 + wid(3 * nPv);
    let txtEmb = P('text embeddings x sqrt(2048)', xTxt, yPre - 30 - hS, T, WS, slab(a['txt_emb']));
    let ids = P('prompt token ids', xTxt, txtEmb.y - 40, T, 1, vecX(m?.tokIds), 'i', DimStyle.TokenIdx, DimStyle.None);
    let txtW = P('embedder (117 vocab rows)', xTxt + wid(T) + 30, txtEmb.y - 10, WF, WF, wmat(w['txt.embed']), 'w', DimStyle.C, DimStyle.C);
    let state = P('state (23 of 61, R1 Pro)', xTxt + wid(T) + 30, ids.y - 50, 23, 1, vecX(m?.state23), 'i', DimStyle.C, DimStyle.None);
    E(state, ids, 'text', 'side');
    E(ids, txtEmb); E(txtW, txtEmb, 'weight', 'side');

    // =================== prefix tower ===================
    let pitch = hS + 16;
    let pre: IBlkDef[] = [];
    for (let l = 0; l < NL; l++)
        pre.push(P(`prefix L${l} in`, X0, yPre + l * pitch, nRow, WS, slab(a[`pre.L${l}.x_in`])));
    let preOut = P('prefix out', X0, yPre + NL * pitch, nRow, WS, slab(a['pre.out']));
    E(imgTok, pre[0], 'image', 'side');
    E(txtEmb, pre[0]);
    for (let l = 0; l < NL; l++) E(pre[l], l + 1 < NL ? pre[l + 1] : preOut);

    // =================== action expert ===================
    let X1 = X0 + wid(nRow) + 300;
    let aTok = P('action tokens', X1, yPre - 34 - hS, H, WS, slab(st['a_tok']));
    let xt = P('x_t (noise at t=1)', X1, aTok.y - 34 - wid(AD), H, AD, slab(st['x_t']), 'i', DimStyle.T, DimStyle.C);
    let inProj = P('action_in_proj 32->1024', X1 - wid(WF) - 40, aTok.y - 20, WF, AD, wmat(w['action_in_proj']), 'w', DimStyle.C, DimStyle.C);
    E(xt, aTok); E(inProj, aTok, 'weight', 'side');

    // time: t -> sincos -> time_mlp_in -> swish -> time_mlp_out -> swish = cond
    let xT = X1 + wid(H) + 200;
    let pe = P('sincos(t)', xT, xt.y, WA, 1, vecX(st['pe']), 'i', DimStyle.C, DimStyle.None);
    let cond = P('cond (adaRMS condition)', xT, xt.y + 70, WA, 1, vecX(st['cond']), 'i', DimStyle.C, DimStyle.None);
    let tIn = P('time_mlp_in', xT + wid(WA) + 30, cond.y - 24, WF, WF, wmat(w['time_mlp_in']), 'w', DimStyle.C, DimStyle.C);
    let tOut = P('time_mlp_out', xT + wid(WA) + 30 + wid(WF) + 50, cond.y - 24, WF, WF, wmat(w['time_mlp_out']), 'w', DimStyle.C, DimStyle.C);
    E(pe, cond); E([tIn, tOut], cond, 'weight', 'side');

    // 18 layers, each level with the prefix layer whose K/V it reads
    let xRod = X1 + wid(H) + 14;
    let exp: IPExp[] = [];
    for (let l = 0; l < NL; l++) {
        let y = yPre + (l + 1) * pitch;
        let out = P(`expert L${l} out`, X1, y, H, WS, slab(st[`L${l}.out`]));
        let scale = P(`L${l} adaRMS scale`, xRod, y, 1, WS, vecY(st[`L${l}.scale_a`]), 'a', DimStyle.None, DimStyle.C);
        let shift = P(`L${l} adaRMS shift`, xRod + 5.5, y, 1, WS, vecY(st[`L${l}.shift_a`]), 'a', DimStyle.None, DimStyle.C);
        let gate = P(`L${l} adaRMS gate`, xRod + 11, y, 1, WS, vecY(st[`L${l}.gate_a`]), 'a', DimStyle.None, DimStyle.C);
        E(l === 0 ? aTok : exp[l - 1].out, out);
        E([scale, shift, gate], out, 'time', 'side');
        E(pre[l], out, 'kv', 'side');
        exp.push({ l, out, scale, shift, gate });
    }
    let busX = xRod + 34;
    let yF = yPre + (NL + 1) * pitch + 12;
    let final = P('final adaRMS(x)', X1, yF, H, WS, slab(st['final']));
    let finScale = P('final scale', xRod, yF, 1, WS, vecY(st['final_scale']), 'a', DimStyle.None, DimStyle.C);
    let finShift = P('final shift', xRod + 5.5, yF, 1, WS, vecY(st['final_shift']), 'a', DimStyle.None, DimStyle.C);
    E(exp[NL - 1].out, final); E([finScale, finShift], final, 'time', 'side');
    let v = P('v (velocity 32x32)', X1, below(final, 30), H, AD, slab(st['v']));
    let outProj = P('action_out_proj 1024->32', X1 + wid(H) + 170, v.y, WF, AD, wmat(w['action_out_proj']), 'w', DimStyle.C, DimStyle.C);
    E(final, v); E(outProj, v, 'weight', 'side');
    for (let d of exp) E(cond, [d.scale, d.shift, d.gate], 'time', 'bus');
    E(cond, [finScale, finShift], 'time', 'bus');

    // Euler track: x(t=1) -> ... -> x(t=0); step k's v lands in cell k+1 (drawn by the scene)
    let yE = below(v, 50);
    let track: IBlkDef[] = [];
    let tStep = cfg?.t ?? Array.from({ length: K }, (_, i) => 1 - i / K);
    for (let j = 0; j <= K; j++) {
        let tex = j < K ? m?.steps[j]?.tex['x_t'] : m?.steps[K - 1]?.tex['x_next'];
        track.push(P(j < K ? `x (t=${tStep[j].toFixed(1)})` : 'action chunk (t=0)', X1 - 1000 + j * 100, yE, H, AD, slab(tex)));
        if (j > 0) E(track[j - 1], track[j], 'euler', 'side');
    }
    let act23 = P('actions (unnormalised, 23 dims)', track[K].x, below(track[K], 40), H, D23, slab(m?.action23));
    E(track[K], act23);

    // =================== layer-0 anatomy (step 0) ===================
    let XA = busX + 420;
    let y0 = yPre + 40;
    let nKe = Np + H;
    // rows: x in, norm, QKV, probs, attn out, x+attn, norm2, GeGLU hidden, MLP out, x out
    let rowH = [hA, hA, hA, wid(nRow), hA, hA, hA, hA, hA, hA];
    let rowY: number[] = [];
    rowH.reduce((y, h) => (rowY.push(y), y + h + 34), 0);
    let mk = (pfx: string, ax: number, n: string, row: number, cx: number, cy: number, acc: IBlkAccessDefArgs | undefined,
            t: 'i' | 'w' | 'a' = 'i', dx = 0) =>
        P(`${pfx} ${n}`, ax + dx, y0 + rowY[row], cx, cy, acc, t, t === 'w' ? DimStyle.C : DimStyle.T, DimStyle.C);

    function stream(pfx: 'prefix' | 'expert', ax: number, n: number, p: string, wp: string, xInTex?: IRealTex) {
        let mkS = (nm: string, row: number, cx: number, cy: number, acc: IBlkAccessDefArgs | undefined, t: 'i' | 'w' | 'a' = 'i', dx = 0) =>
            mk(pfx, ax, nm, row, cx, cy, acc, t, dx);
        let dW = -wid(WA) - 50;
        let kx = wid(n) + 40, wkx = kx + wid(n) + 12, vx = wkx + wid(WA) + 50, wvx = vx + wid(n) + 12;
        let xIn = mkS('x in', 0, n, pfx === 'prefix' ? WF : WA, slab(xInTex));
        let norm = mkS(pfx === 'prefix' ? 'RMSNorm(x)(1+w)' : 'adaRMS(x)', 1, n, WA, slab(a[p + 'xn']));
        let normW = pfx === 'prefix'
            ? mkS('norm weight', 1, 1, WA, vecY(w['pre.L0.norm_scale']), 'w', -30)
            : mkS('ada Dense (attn)', 1, WA, WA, wmat(w['exp.L0.ada_attn']), 'w', dW);
        let q = mkS('Q head0', 2, n, WA, slab(a[p + 'q']));
        let wq = mkS('W_q', 2, WA, WA, wmat(w[wp + 'wq']), 'w', dW);
        let k = mkS('K', 2, n, WA, slab(a[p + 'k']), 'i', kx);
        let wk = mkS('W_k', 2, WA, WA, wmat(w[wp + 'wk']), 'w', wkx);
        let vv = mkS('V', 2, n, WA, slab(a[p + 'v']), 'i', vx);
        let wv = mkS('W_v', 2, WA, WA, wmat(w[wp + 'wv']), 'w', wvx);
        let nK = pfx === 'prefix' ? Np : nKe;
        let probs = mkS(pfx === 'prefix' ? 'softmax head0 (prefix keys only)' : 'softmax head0 (prefix + action keys)', 3, nK, n, wmat(a[p + 'probs']), 'a');
        let ao = mkS('attention out', 4, n, WA, slab(a[p + 'ao']));
        let wo = mkS('W_o', 4, WA, WA, wmat(w[wp + 'wo']), 'w', dW);
        let r1 = mkS(pfx === 'prefix' ? 'x + attn' : 'x + gate_a * attn', 5, n, WA, slab(a[p + 'r1']));
        let norm2 = mkS(pfx === 'prefix' ? 'RMSNorm (ffw)' : 'adaRMS (ffw)', 6, n, WA, slab(a[p + 'xn2']));
        let mlpH = mkS('GeGLU gelu(xWg)*(xWu)', 7, n, WA, slab(a[p + 'mlp_h']));
        let wGate = mkS('W_gate', 7, WA, WA, wmat(w[wp + 'w_gate']), 'w', dW);
        let wUp = mkS('W_up', 7, WA, WA, wmat(w[wp + 'w_up']), 'w', dW - wid(WA) - 30);
        let ff = mkS('MLP out', 8, n, WA, slab(a[p + 'ff']));
        let wDown = mkS('W_down', 8, WA, WA, wmat(w[wp + 'w_down']), 'w', dW);
        let out = mkS(pfx === 'prefix' ? 'x out' : 'x out (+ gate_f * mlp)', 9, n, WA, slab(a[p + 'out']));
        E(xIn, norm); if (pfx === 'prefix') E(normW, norm, 'weight', 'side');
        E(norm, q); E(wq, q, 'weight', 'side');
        E(norm, k, 'data', 'side'); E(norm, vv, 'data', 'side');
        E(wk, k, 'weight', 'side'); E(wv, vv, 'weight', 'side');
        E(q, probs); E(k, probs);
        E(probs, ao); E(vv, ao); E(wo, ao, 'weight', 'side');
        E(ao, r1); E(xIn, r1, 'skip'); E(r1, norm2);
        E(norm2, mlpH); E([wGate, wUp], mlpH, 'weight', 'side');
        E(mlpH, ff); E(wDown, ff, 'weight', 'side');
        E(ff, out); E(r1, out, 'skip');
        let cubesS = [xIn, norm, normW, q, wq, k, wk, vv, wv, probs, ao, wo, r1, norm2, mlpH, wGate, wUp, ff, wDown, out];
        return { xIn, norm, normW, q, wq, k, wk, v: vv, wv, probs, ao, wo, r1, norm2, mlpH, wGate, wUp, ff, wDown, out, cubes: cubesS };
    }
    let preS = stream('prefix', XA, nRow, 'pre.L0.', 'pre.L0.', a['pre.L0.x_in']);
    let XB = XA + Math.max(wid(Np), wid(nRow) * 3 + wid(WA) * 2 + 200) + 400;
    let expS0 = stream('expert', XB, H, 's0.L0.', 'exp.L0.', a['s0.L0.x_in']);
    // the expert's adaRMS: cond -> Dense -> (scale, shift, gate) for attention and for the MLP
    let rod = (nm: string, row: number, dx: number, key: string) => mk('expert', XB, nm, row, 1, WA, vecY(a[key]), 'a', dx);
    let scaleA = rod('scale_a', 1, -24, 's0.L0.scale_a'), shiftA = rod('shift_a', 1, -18.5, 's0.L0.shift_a');
    let gateA = rod('gate_a', 5, -24, 's0.L0.gate_a');
    let scaleF = rod('scale_f', 6, -24, 's0.L0.scale_f'), shiftF = rod('shift_f', 6, -18.5, 's0.L0.shift_f');
    let gateF = rod('gate_f', 9, -24, 's0.L0.gate_f');
    let adaA = expS0.normW;
    let adaF = mk('expert', XB, 'ada Dense (ffw)', 6, WA, WA, wmat(w['exp.L0.ada_ffw']), 'w', -wid(WA) - 50);
    E(cond, [adaA, adaF], 'time', 'side');
    E(adaA, [scaleA, shiftA], 'weight', 'side'); E([scaleA, shiftA], expS0.norm, 'time', 'side');
    E(adaA, gateA, 'weight', 'side'); E(gateA, expS0.r1, 'time', 'side');
    E(adaF, [scaleF, shiftF], 'weight', 'side'); E([scaleF, shiftF], expS0.norm2, 'time', 'side');
    E(adaF, gateF, 'weight', 'side'); E(gateF, expS0.out, 'time', 'side');
    // joint attention: the expert's queries also read the prefix's K and V (the prefix never reads the expert's)
    E(preS.k, expS0.probs, 'kv', 'side'); E(preS.v, expS0.ao, 'kv', 'side');
    let mask = mk('joint', XB + wid(nKe) + 60, 'attention mask (every 8th)', 3, 115, 115, wmat(a['joint.mask']), 'a');
    E(mask, [preS.probs, expS0.probs], 'data', 'side');
    let expS = { ...expS0, adaA, adaF, scaleA, shiftA, gateA, scaleF, shiftF, gateF };
    expS.cubes = [...expS0.cubes, adaF, scaleA, shiftA, gateA, scaleF, shiftF, gateF];
    let anat: IPAnat = { pre: preS, exp: expS, mask, cubes: [...preS.cubes, ...expS.cubes, mask] };

    for (let d of exp) d.out.noTag = d.scale.noTag = d.shift.noTag = d.gate.noTag = true;
    for (let p of pre) p.noTag = true;
    finScale.noTag = finShift.noTag = true;
    for (let b of [scaleA, shiftA, gateA, scaleF, shiftF, gateF]) b.noTag = true;

    let vision = [...patch, vitB0, vitB26, vitEnc, vitHead, imgTok];
    let text = [state, ids, txtW, txtEmb];
    let prefixStack = [...pre, preOut];
    let frontEnd = [xt, inProj, aTok, pe, tIn, tOut, cond];
    let expStack = exp.flatMap(d => [d.out, d.scale, d.shift, d.gate]);
    let decode = [final, finScale, finShift, v, outProj, ...track, act23];

    return {
        cell, height: below(act23) + 100, margin: 12, cubes, labels: [],
        patch, stem, vitB0, vitB26, vitEnc, vitHead, imgTok, vitGhostY: [below(vitB0, 10), vitB26.y - 14],
        state, ids, txtW, txtEmb, pre, preOut, xt, inProj, aTok, pe, tIn, tOut, cond, exp, busX,
        finScale, finShift, final, outProj, v, track, act23, anat,
        vision, text, prefixStack, frontEnd, expStack, decode,
        edges,
        // sources: the 3 camera patch embeds (fed by the thumbnails drawn above them), the robot
        // state, the noise, time t (sincos), the joint mask, and the anatomy's "x in" copies
        srcOk: new Set([...patch, state, xt, track[0], pe, mask, preS.xIn, expS.xIn]),
        // sinks: the unnormalised action chunk; the prefix tower's last output (not read by the
        // action path: only the per-layer K/V matter); the anatomy's "x out" copies
        sinkOk: new Set([act23, preOut, preS.out, expS.out]),
        step: 0,
    };
}

// bind Euler step k's tensors to the step-dependent blocks
export function applyPi0Step(L: IPi0Layout, m: IPi0Model | null, k: number) {
    L.step = k;
    let s = m?.steps[k]?.tex;
    if (!s) return;
    let set = (b: IBlkDef, t?: IRealTex) => {
        if (!t || !b.access) return;
        b.access.src = t;
        b.access.scale = t.meta.absScale;
    };
    set(L.xt, s['x_t']);
    set(L.aTok, s['a_tok']);
    set(L.pe, s['pe']);
    set(L.cond, s['cond']);
    for (let d of L.exp) {
        set(d.out, s[`L${d.l}.out`]);
        set(d.scale, s[`L${d.l}.scale_a`]);
        set(d.shift, s[`L${d.l}.shift_a`]);
        set(d.gate, s[`L${d.l}.gate_a`]);
    }
    set(L.final, s['final']);
    set(L.finScale, s['final_scale']);
    set(L.finShift, s['final_shift']);
    set(L.v, s['v']);
}

export function blkCenter(b: IBlkDef) {
    return new Vec3(b.x + b.dx / 2, b.y + b.dy / 2, b.z + b.dz / 2);
}
