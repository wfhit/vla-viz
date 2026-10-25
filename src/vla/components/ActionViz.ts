// 3D visuals of the action path, all driven by REAL exported tensors:
//  - joint-attention links: each trunk layer i's K/V groups → expert layer i's attention
//    block, brightness = the real share of suffix attention that lands on the prefix
//  - the 16 × 23 action chunk as a bar field (bar height = real normalized value), with
//    the one-step Euler decode animated: x(τ) = x1 − τ·v, τ: 0 → 1 (t = 1 → 0)
import { IProgramState, vlaLayout } from "../Program";
import { IVlaModelLayout } from "../VlaModelLayout";
import { fmDir, IVlaModelLink } from "../VlaModel";
import { addLine } from "../render/lineRender";
import { addPrimitiveRestart, addVert } from "../render/triRender";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { RenderPhase } from "../render/sharedRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { clamp } from "@/src/utils/data";
import { actionGroups, unnormalizeAction } from "../actionDims";
import { recordArrowEdge } from "./Arrow";
import { IBlkDef } from "../VlaModelLayout";

export interface IActionVizState {
    links: number;       // opacity of the K/V → joint-attention arrows (active layer only)
    activeLayer: number; // highlighted layer pair (-1 = none)
    massBars: number;    // attention-mass bars under the active layer's joint attention
    field: number;       // opacity of the bar field
    fieldMode: 'decode' | 'velocity'; // bars = x1 − τ·v, or the velocity plate v itself
    decode: number;      // Euler progress τ ∈ [0, 1]; 1 = the decoded action chunk
    showVelocity: number; // opacity of the per-component −v segments
    target: number;      // training label a as ghost frames + error segments
    errBars: number;     // per-dim MSE bars along the front edge
    robotUnits: number;  // dim labels show step-0 values in robot units
    particles: number;   // particle-axis mode (replaces the bar field)
    particleTau: number; // 0..1: each particle moves x1_k → x1_k − v_k
    particleStep: number; // which of the 16 steps the particle axes show
    massStep: number;    // Euler pass whose attention mass the bars / links use (-1 = the joint forward)
    quadLabel: number;   // "prefix cannot see suffix" in the empty quadrant of the active layer
    eulerPass: number;   // decode chapter: the running pass (shown as the step indicator), -1 = none
    normCmp: number;     // opacity of the AdaRMSNorm before/after chart beside a gamma/beta block
    normLayer: number;   // which layer pair the chart reads
    normWhich: 1 | 2;    // 1 = AdaRMSNorm1 (gamma1/beta1), 2 = AdaRMSNorm2 (gamma2/beta2)
    normTok: number;     // which suffix token's 64 channels are charted
}

export function defaultActionViz(): IActionVizState {
    return {
        links: 0, activeLayer: -1, massBars: 0, field: 1, fieldMode: 'decode', decode: 1, showVelocity: 0,
        target: 0, errBars: 0, robotUnits: 0, particles: 0, particleTau: 1, particleStep: 0,
        massStep: -1, quadLabel: 0, eulerPass: -1, normCmp: 0, normLayer: 0, normWhich: 1, normTok: 1,
    };
}

const kColor = Vec4.fromHexColor('#f59e0b');
const vColor = Vec4.fromHexColor('#d946ef');
const posColor = Vec4.fromHexColor('#2563eb');
const negColor = Vec4.fromHexColor('#dc2626');
const errColor = Vec4.fromHexColor('#e11d48');
const particleColors = ['#0ea5e9', '#22c55e', '#eab308', '#f97316', '#a855f7', '#ec4899', '#14b8a6', '#64748b'].map(c => Vec4.fromHexColor(c));

function groupColorOf(d: number) {
    for (let g of actionGroups) {
        if (d >= g.start && d < g.end) return Vec4.fromHexColor(g.color);
    }
    return new Vec4(0.7, 0.7, 0.7, 1);
}

export function drawActionViz(state: IProgramState, isVisible?: (cubes: IBlkDef[]) => boolean) {
    let layout = vlaLayout(state);
    let m = state.vlaModel;
    if (!layout || !m || !state.render) return;
    let viz = state.display.actionViz ?? defaultActionViz();

    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    drawJointLinks(state, layout, m, viz, isVisible);
    if (viz.massBars > 0) drawMassBars(state, layout, m, viz);
    if (viz.quadLabel > 0) drawQuadLabel(state, layout, viz);
    if (viz.normCmp > 0) drawNormCompare(state, layout, m, viz);
    drawStripTitle(state, layout, m, viz);
    if (viz.particles > 0) {
        drawParticles(state, layout, m, viz);
    } else if (viz.field > 0) {
        drawBarField(state, layout, m, viz);
    }
}

function sceneText(state: IProgramState, text: string, x: number, y: number, z: number, size: number, color: Vec4) {
    let prev = state.render.sharedRender.activePhase;
    state.render.sharedRender.activePhase = RenderPhase.Overlay;
    drawText(state.render.modelFontBuf, text, 0, 0, { color, size, mtx: Mat4f.fromTranslation(new Vec3(x, y, z)) });
    state.render.sharedRender.activePhase = prev;
}

// the top-right quadrant of the joint score matrix (prefix queries x suffix keys) is empty:
// the mask keeps the prefix from seeing the suffix
function drawQuadLabel(state: IProgramState, layout: IVlaModelLayout, viz: IActionVizState) {
    let i = viz.activeLayer;
    if (i < 0) return;
    // pinned above one joint score row of head 0: which keys each part of the row reads
    let eh = layout.expertLayers[i].heads[0];
    let p = eh.attnMtxP, s = eh.attnMtx;
    if (p.opacity <= 0 || s.opacity <= 0) return;
    let col = new Vec4(0.25, 0.25, 0.3, 1).mul(viz.quadLabel);
    let z = p.z + p.dz / 2;
    sceneText(state, 'keys: prefix (trunk K, same layer)', p.x, p.y - 5, z, 2.2, col);
    sceneText(state, 'keys: suffix', s.x, s.y - 5, z, 2.2, col);
    sceneText(state, 'rows: 17 suffix queries', p.x, p.y + p.dy + 2, z, 2.2, col);
}

// Beside a gamma/beta block: per channel (rows aligned with the block's channel rows), the
// plain RMSNorm(x)*w value (grey) and the modulated (1+g)*n+b value (orange) of one token.
// The modulated value is the captured AdaRMSNorm output; the plain value is recovered
// exactly per channel as n = (out - b) / (1 + g), since the modulation is element-wise.
function drawNormCompare(state: IProgramState, layout: IVlaModelLayout, m: IVlaModelLink, viz: IActionVizState) {
    let i = viz.normLayer;
    let e = layout.expertLayers[i], el = m.expertLayers?.[i], ada = m.fm?.ada[i];
    if (!e || !el || !ada) return;
    let blk = viz.normWhich === 1 ? e.ada : e.ada2;
    let out = (viz.normWhich === 1 ? el.ln_1 : el.ln_2).output.localBuffer;
    if (!blk || blk.opacity <= 0) return;
    // rows per norm: [g, b] (k005b) or [scale, shift, gate] (pi0); the gate acts at the residual.
    // GR00T modulates only norm1 (AdaLayerNorm [scale, shift]); its norm3 is a plain LayerNorm
    let R = layout.variant.ada_rows, gated = R === 3;
    let adaLn = layout.variant.expert_ada === 'adaln';
    let wc = ada.localBuffer.length / ((e.ada2 ? 2 : 1) * R);
    let g0 = (viz.normWhich - 1) * R;
    let nTok = out.length / wc, t = Math.min(viz.normTok, nTok - 1);
    let a = viz.normCmp;
    let cell = layout.cell;
    let scale = 2.2, lim = 22;
    let x0 = blk.x + blk.dx + lim + 6, z = blk.z + blk.dz + 0.5;
    let grey = new Vec4(0.45, 0.45, 0.5, 1).mul(a), orange = Vec4.fromHexColor('#ea580c').mul(a);
    let tri = state.render.triRender;
    let prev = state.render.sharedRender.activePhase;
    state.render.sharedRender.activePhase = RenderPhase.Overlay;
    let n = new Vec3(0, 0, 1);
    let bar = (y0: number, y1: number, v: number, col: Vec4) => {
        let len = clamp(v * scale, -lim, lim);
        let xa = Math.min(x0, x0 + len), xb = Math.max(x0, x0 + len) + 0.15;
        quad(tri, new Vec3(xa, y0, z), new Vec3(xb, y0, z), new Vec3(xa, y1, z), new Vec3(xb, y1, z), col, n);
    };
    for (let c = 0; c < Math.min(wc, blk.cy); c++) {
        let gm = ada.localBuffer[g0 * wc + c], bt = ada.localBuffer[(g0 + 1) * wc + c];
        let post = out[t * wc + c];
        let pre = Math.abs(1 + gm) > 1e-3 ? (post - bt) / (1 + gm) : NaN;
        let y = blk.y + c * cell;
        if (!isNaN(pre)) bar(y + 0.1, y + cell * 0.45, pre, grey);
        bar(y + cell * 0.5, y + cell * 0.9, post, orange);
    }
    state.render.sharedRender.activePhase = prev;
    let txt = new Vec4(0.1, 0.1, 0.15, 1).mul(a);
    let ty = blk.y - 13;
    let nm = adaLn ? 'AdaLayerNorm' : `${gated ? 'adaRMS' : 'AdaRMSNorm'}${viz.normWhich}`;
    sceneText(state, `${nm} ${adaLn ? 'block' : 'layer'} ${i}, ${adaLn ? 'DiT' : 'suffix'} token ${t}, ch 0-${wc - 1}`, x0 - lim, ty, z, 2.4, txt);
    sceneText(state, adaLn ? 'grey  : LayerNorm(x)  (before t-modulation)' : gated ? 'grey  : RMSNorm(x)  (before t-modulation)' : 'grey  : RMSNorm(x)*w  (before t-modulation)', x0 - lim, ty + 3.5, z, 2.2, grey);
    sceneText(state, gated || adaLn ? 'orange: (1+scale(t))*n + shift(t)  (after)' : 'orange: (1+g(t))*n + b(t)  (after)', x0 - lim, ty + 6.5, z, 2.2, orange);
    let lr = state.render.lineRender;
    addLine(lr, 0.6, txt, new Vec3(x0, blk.y, z), new Vec3(x0, blk.y + blk.dy, z), undefined, undefined, 0);
}

// above the Euler strip: what it is, the pre-stamp warning, and the 10-cell step indicator
function drawStripTitle(state: IProgramState, layout: IVlaModelLayout, m: IVlaModelLink, viz: IActionVizState) {
    let S = layout.strip;
    if (!S.length || S[0].opacity <= 0) return;
    let fs = m.fmSteps;
    let K = S.length - 1;
    let x = S[0].x, z = S[0].z + S[0].dz / 2, a = S[0].opacity;
    let y = S[0].y - 24;
    let FD = fmDir(layout.variant);
    let td = Number.isInteger(10 / K) ? 1 : 2;
    sceneText(state, `Euler strip: x(t=${FD.tNoise}) noise -> x(t=${FD.tAct}) action,  K=${K} DiT passes,  x <- x ${FD.sign} v/${K}`, x, y, z, 4, new Vec4(0.1, 0.1, 0.15, 1).mul(a));
    if (fs && !fs.inDistribution) {
        sceneText(state, `preview: pre-stamp weights (step_${fs.checkpointStep}, trained at t=1 only; t<1 is out of distribution)`,
            x, y + 6, z, 3, Vec4.fromHexColor('#b45309').mul(a));
    }
    if (viz.eulerPass >= 0) {
        let k = viz.eulerPass;
        let cells = Array.from({ length: K }, (_, j) => j < k ? '[#]' : j === k ? '[>]' : '[ ]').join('');
        let t = fs?.steps[Math.min(k, K - 1)]?.t ?? FD.tOf(k, K);
        let txt = k >= K ? `${cells}  done: x(t=${FD.tAct}) = action chunk` : `${cells}  pass ${k + 1}/${K}   t = ${t.toFixed(td)}`;
        sceneText(state, txt, x, y + 12, z, 4, Vec4.fromHexColor('#a16207').mul(a));
    }
}

// mean over the 17 suffix queries of layer i's attention mass [state | visual | text | suffix]
export function layerMass(m: IVlaModelLink, i: number, step: number = -1) {
    let fs = step >= 0 ? m.fmSteps?.steps[step]?.tex[`E${i}.mass`] : undefined;
    let ms = fs ? fs.localBuffer : m.expertLayers[i].mass, n = m.config.n_suffix, acc = [0, 0, 0, 0];
    for (let q = 0; q < n; q++) for (let k = 0; k < 4; k++) acc[k] += ms[q * 4 + k] / n;
    return acc;
}

// a straight 3D arrow (line + two crossed head fins) between two block faces
function arrow3(state: IProgramState, a: Vec3, b: Vec3, thick: number, color: Vec4) {
    let dir = b.sub(a).normalize();
    let head = Math.min(4, b.sub(a).len() * 0.3);
    let base = b.sub(dir.mul(head));
    addLine(state.render.lineRender, thick, color, a, base, undefined, undefined, 0);
    let up = Math.abs(dir.y) > 0.9 ? new Vec3(1, 0, 0) : new Vec3(0, 1, 0);
    let p1 = Vec3.cross(dir, up).normalize().mul(head * 0.45);
    let p2 = Vec3.cross(dir, p1).normalize().mul(head * 0.45);
    let tri = state.render.triRender;
    for (let p of [p1, p2]) {
        addVert(tri, base.add(p), color, undefined);
        addVert(tri, base.sub(p), color, undefined);
        addVert(tri, b, color, undefined);
        addPrimitiveRestart(tri);
    }
}

// trunk layer i's K and V groups feed expert layer i's joint attention (the suffix Q attends
// over them). Every visible layer pair shows its coupling faintly; the active pair (walkthrough
// focus, or the layer the data pass is in) is bright, width ∝ the real prefix share of the
// suffix attention mass
function drawJointLinks(state: IProgramState, layout: IVlaModelLayout, m: IVlaModelLink, viz: IActionVizState,
    isVisible?: (cubes: IBlkDef[]) => boolean) {
    // GR00T: no layer pairing; its cross blocks read the VL tokens over the VL bus (Arrow.ts)
    if (layout.crossDit) return;
    let flow = layout.flow;
    for (let i = 0; i < layout.expertLayers.length; i++) {
        let e = layout.expertLayers[i];
        let block = layout.blocks[i];
        let bright = (viz.activeLayer === i ? viz.links : 0);
        if (flow && e.heads.some(h => flow!.active.has(h.attnMtxP) || flow!.active.has(h.vOutBlock))) bright = 1;
        let visible = !isVisible || isVisible([...e.heads.map(h => h.attnMtxP), ...block.kvGroups.map(g => g.kBlock)]);
        let faint = flow && e.heads.every(h => flow!.pending.has(h.attnMtxP)) ? 0.08 : 0.22;
        let mass = layerMass(m, i, viz.massStep);
        let thick = bright > 0 ? 0.6 + 2 * (mass[1] + mass[2]) : 0.5;
        // the coupling: the trunk's K of a kv group → the suffix-prefix score quadrant scP of
        // each query head in that group (Q_suffix · K_prefix^T); the trunk's V → the suffix's
        // softmax·V output of those heads
        for (let h = 0; h < block.heads.length; h++) {
            let g = block.kvGroups[block.heads[h].kvGroup];
            let eh = e.heads[h];
            for (let [src, dst, col] of [[g.kBlock, eh.attnMtxP, kColor], [g.vBlock, eh.vOutBlock, vColor]] as [IBlkDef, IBlkDef, Vec4][]) {
                if (src.opacity <= 0 || dst.opacity <= 0) continue;
                recordArrowEdge(src, dst);
                if (!visible) continue;
                let start = new Vec3(src.x + src.dx / 2, src.y + src.dy + 1, src.z + src.dz / 2);
                let end = new Vec3(dst.x + dst.dx / 2, dst.y - 1, dst.z + dst.dz / 2);
                arrow3(state, start, end, thick, col.mul(Math.max(bright, faint) * Math.min(src.opacity, dst.opacity)));
            }
        }
    }
}

function drawMassBars(state: IProgramState, layout: IVlaModelLayout, m: IVlaModelLink, viz: IActionVizState) {
    let i = viz.activeLayer;
    if (i < 0) return;
    let cfg = m.config;
    let e = layout.expertLayers[i];
    // anchor: right of the frontmost head's suffix-suffix score block
    let attn = e.heads[0].attnMtx;
    let mass = layerMass(m, i, viz.massStep);
    let nVis = cfg.suffix_visible_visual, nTxt = cfg.suffix_visible_prefix - cfg.suffix_visible_visual;
    let nAns = cfg.T_text - cfg.suffix_visible_prefix;
    // GR00T: a cross block attends to [image | text] VL tokens, a self block to [state | action] tokens
    let rows: [string, number | null, Vec4][] = layout.crossDit ? [
        [`state token (self)`, mass[0], new Vec4(0.6, 0.6, 0.6, 1)],
        [`image ${nVis} VL tokens (cross)`, mass[1], Vec4.fromHexColor('#0ea5e9')],
        [`text ${nTxt} VL tokens (cross)`, mass[2], Vec4.fromHexColor('#22c55e')],
        [`action ${cfg.n_suffix - 1} tokens (self)`, mass[3], Vec4.fromHexColor('#f59e0b')],
    ] : [
        [`state  ${cfg.n_state_tokens} cols  MASKED`, null, new Vec4(0.6, 0.6, 0.6, 1)],
        [`visual ${nVis} cols`, mass[1], Vec4.fromHexColor('#0ea5e9')],
        [`text   ${nTxt} cols`, mass[2], Vec4.fromHexColor('#22c55e')],
        [`answer ${nAns} cols  MASKED`, null, new Vec4(0.6, 0.6, 0.6, 1)],
        [`suffix ${cfg.n_suffix} cols (self)`, mass[3], Vec4.fromHexColor('#f59e0b')],
    ];
    let x0 = attn.x + 16, y = attn.y + attn.dy + 4, z = attn.z + attn.dz / 2 + 0.5, maxLen = 30;
    let fb = state.render.modelFontBuf;
    let tri = state.render.triRender;
    let a = viz.massBars;
    let txt = new Vec4(0.1, 0.1, 0.15, 1).mul(a);
    let prev = state.render.sharedRender.activePhase;
    for (let [label, v, col] of rows) {
        let pct = v === null ? '' : `  ${(v * 100).toFixed(1)}%`;
        state.render.sharedRender.activePhase = RenderPhase.Overlay;
        drawText(fb, label + pct, 0, 0, { color: txt, size: 2, mtx: Mat4f.fromTranslation(new Vec3(x0, y, z)) });
        state.render.sharedRender.activePhase = RenderPhase.Opaque;
        let len = v === null ? maxLen : Math.max(0.3, v * maxLen);
        if (v === null) {
            // a crossed-out empty bar: the mask removes these columns entirely
            let lr = state.render.lineRender;
            addLine(lr, 1, col.mul(a), new Vec3(x0, y + 2.4, z), new Vec3(x0 + len, y + 3.8, z), undefined, undefined, 0);
            addLine(lr, 1, col.mul(a), new Vec3(x0, y + 3.8, z), new Vec3(x0 + len, y + 2.4, z), undefined, undefined, 0);
        } else {
            quad(tri, new Vec3(x0, y + 2.4, z), new Vec3(x0 + len, y + 2.4, z), new Vec3(x0, y + 3.8, z), new Vec3(x0 + len, y + 3.8, z), col.mul(a), new Vec3(0, 0, 1));
        }
        y += 5;
    }
    state.render.sharedRender.activePhase = prev;
}

function fieldGeom(layout: IVlaModelLayout) {
    let f = layout.actionField;
    let barW = f.pitch * 0.72;
    let xOf = (d: number) => f.x + d * f.pitch;
    // step 0 nearest the default camera (+z), later steps recede
    let zOf = (s: number) => f.z + (f.nSteps - 1 - s) * f.pitch;
    return { f, barW, xOf, zOf, zFront: zOf(0) + barW + 2 };
}

function divergingColor(v: number) {
    let k = Math.min(1, Math.abs(v) / 1.5);
    let base = v >= 0 ? posColor : negColor;
    let grey = new Vec4(0.85, 0.85, 0.88, 1);
    return Vec4.lerp(grey, base, 0.25 + 0.75 * k);
}

function drawBarField(state: IProgramState, layout: IVlaModelLayout, m: IVlaModelLink, viz: IActionVizState) {
    let { f, barW, xOf, zOf, zFront } = fieldGeom(layout);
    let tri = state.render.triRender;
    let { nSteps, nDims, heightScale } = f;
    let x1 = m.actionX1.localBuffer;
    let vel = m.actionVelocity.localBuffer;
    let tgt = m.fm.target.localBuffer;
    let tau = Math.max(0, Math.min(1, viz.decode));
    let alpha = viz.field;
    let isVel = viz.fieldMode === 'velocity';
    let FD = fmDir(layout.variant), sg = FD.fwd ? 1 : -1;

    // base plane
    let plane = new Vec4(0.55, 0.6, 0.68, 1).mul(0.35 * alpha);
    quad(tri, new Vec3(xOf(0) - 1, f.y, zOf(nSteps - 1) - 1), new Vec3(xOf(nDims) - f.pitch + barW + 1, f.y, zOf(nSteps - 1) - 1),
        new Vec3(xOf(0) - 1, f.y, zOf(0) + barW + 1), new Vec3(xOf(nDims) - f.pitch + barW + 1, f.y, zOf(0) + barW + 1), plane, new Vec3(0, -1, 0));

    // ±1 guide lines (bounds99: the 1%/99% quantiles) along the front edge
    let lr = state.render.lineRender;
    let guide = new Vec4(0.6, 0.6, 0.65, 1).mul(0.6 * alpha);
    for (let lv of [-1, 1]) {
        let y = f.y - lv * heightScale;
        addLine(lr, 1, guide, new Vec3(xOf(0) - 2, y, zOf(0) + barW + 1), new Vec3(xOf(nDims - 1) + barW + 2, y, zOf(0) + barW + 1), undefined, undefined, 0);
    }

    for (let s = 0; s < nSteps; s++) {
        for (let d = 0; d < nDims; d++) {
            let i = s * nDims + d;
            let val = isVel ? vel[i] : x1[i] + sg * tau * vel[i];
            let h = val * heightScale;
            let base = isVel ? divergingColor(val) : groupColorOf(d);
            let x0 = xOf(d), z0 = zOf(s);
            box(tri, x0, x0 + barW, f.y - Math.max(h, 0), f.y - Math.min(h, 0), z0, z0 + barW, base, alpha, h >= 0);
            let cx = x0 + barW / 2, cz = z0 + barW / 2;
            if (viz.showVelocity > 0 && !isVel) {
                // this component's Euler step: from x1 to x1 − v
                let yA = f.y - x1[i] * heightScale, yB = f.y - (x1[i] + sg * vel[i]) * heightScale;
                addLine(lr, 1.5, new Vec4(1, 1, 1, 1).mul(0.85 * viz.showVelocity * alpha), new Vec3(cx, yA, cz), new Vec3(cx, yB, cz), undefined, undefined, 0);
            }
            if (viz.target > 0 && !isVel) {
                // label a as a ghost frame at its height; the error segment joins it to the bar top
                let yT = f.y - tgt[i] * heightScale;
                let gc = new Vec4(0.15, 0.15, 0.2, 1).mul(0.8 * viz.target * alpha);
                addLine(lr, 0.8, gc, new Vec3(x0 - 0.3, yT, z0 - 0.3), new Vec3(x0 + barW + 0.3, yT, z0 - 0.3), undefined, undefined, 0);
                addLine(lr, 0.8, gc, new Vec3(x0 - 0.3, yT, z0 + barW + 0.3), new Vec3(x0 + barW + 0.3, yT, z0 + barW + 0.3), undefined, undefined, 0);
                addLine(lr, 0.8, gc, new Vec3(x0 - 0.3, yT, z0 - 0.3), new Vec3(x0 - 0.3, yT, z0 + barW + 0.3), undefined, undefined, 0);
                addLine(lr, 0.8, gc, new Vec3(x0 + barW + 0.3, yT, z0 - 0.3), new Vec3(x0 + barW + 0.3, yT, z0 + barW + 0.3), undefined, undefined, 0);
                addLine(lr, 1.2, errColor.mul(viz.target * alpha), new Vec3(cx, f.y - val * heightScale, cz), new Vec3(cx, yT, cz), undefined, undefined, 0);
            }
        }
    }

    if (viz.errBars > 0 && !isVel) {
        // per-dim MSE over the 16 steps, as short red bars in front of the field
        let err = m.fm.errDim;
        let zE = zFront + 2;
        for (let d = 0; d < nDims; d++) {
            let h = err[d] * heightScale;
            box(tri, xOf(d), xOf(d) + barW, f.y + heightScale * 1.2 - h, f.y + heightScale * 1.2, zE, zE + 1, errColor, viz.errBars * alpha, true);
        }
    }

    // labels (font atlas is Latin-only: dims by their schema names)
    let cfg = m.config;
    let names = cfg.action_dim_names;
    let fb = state.render.modelFontBuf;
    state.render.sharedRender.activePhase = RenderPhase.Overlay;
    let txt = new Vec4(0.1, 0.1, 0.15, 1).mul(alpha);
    let yLab = f.y + heightScale * 2.2;
    let chunk = m.actionChunk.localBuffer;
    for (let d = 0; d < nDims; d++) {
        // vertical text: rotate about z so the name runs downward under its bar column
        let mtx = Mat4f.fromTranslation(new Vec3(xOf(d) + barW / 2 + 0.8, yLab, zFront)).mul(Mat4f.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2));
        let opts: IFontOpts = { color: groupColorOf(d).mul(0.7).add(new Vec4(0, 0, 0, 0.3)).mul(alpha), size: 2.2, mtx };
        let extra = viz.robotUnits > 0 ? `  ${fmtNum(unnormalizeAction(cfg, d, chunk[d]))}` : '';
        drawText(fb, `${d} ${names[d]}${extra}`, 0, 0, opts);
    }
    for (let s of [0, 5, 10, 15]) {
        if (s >= nSteps) continue;
        let opts: IFontOpts = { color: txt, size: 2.4, mtx: Mat4f.fromTranslation(new Vec3(xOf(nDims - 1) + barW + 3, f.y - 1.5, zOf(s) + barW / 2)) };
        drawText(fb, `step ${s}`, 0, 0, opts);
    }
    let tNow = FD.fwd ? tau : 1 - tau;
    let N = FD.noise;
    let title = isVel
        ? `velocity v = ${FD.fwd ? 'action_decoder(DiT)' : 'action_out_proj(expert)'}  ${nSteps} x ${nDims}   blue +, red -`
        : FD.fwd
            ? `action chunk ${nSteps} x ${nDims}  |  x(t) = x0 + t v,  t = ${tNow.toFixed(2)}${tau >= 1 ? '  (decoded: x0 + v)' : '  [interp only]'}`
            : `action chunk ${nSteps} x ${nDims}  |  x(t) = ${N} - (1 - t) v,  t = ${tNow.toFixed(2)}${tau >= 1 ? `  (decoded: ${N} - v)` : '  [interp only]'}`;
    let topY = f.y - heightScale * 2.6;
    let titleOpts: IFontOpts = { color: txt, size: 3.2, mtx: Mat4f.fromTranslation(new Vec3(xOf(0), topY, zFront)) };
    drawText(fb, title, 0, -3.2, titleOpts);
    if (viz.target > 0 && !isVel && cfg.fm) {
        let o: IFontOpts = { color: errColor.mul(viz.target * alpha), size: 2.8, mtx: Mat4f.fromTranslation(new Vec3(xOf(0), topY + 2, zFront)) };
        drawText(fb, `frames = label a,  red = error.  MSE(x1 - v, a) = ${cfg.fm.mse.toFixed(3)}   predict-zero MSE = ${cfg.fm.mse_zero.toFixed(3)}`, 0, 0, o);
    }
    if (viz.errBars > 0 && !isVel) {
        let o: IFontOpts = { color: errColor.mul(viz.errBars * alpha), size: 2.2, mtx: Mat4f.fromTranslation(new Vec3(xOf(nDims - 1) + barW + 3, f.y + heightScale * 1.2 - 2, zFront + 2)) };
        drawText(fb, 'per-dim MSE', 0, 0, o);
    }
    if (viz.robotUnits > 0) {
        let o: IFontOpts = { color: txt, size: 2.4, mtx: Mat4f.fromTranslation(new Vec3(xOf(0), yLab + 30, zFront)) };
        drawText(fb, 'labels: dim name + step-0 value in robot units (bounds99 inverse)', 0, 0, o);
    }
    let w = measureText(fb, '+1', { color: txt, size: 2.2 });
    for (let lv of [-1, 1]) {
        let opts: IFontOpts = { color: txt, size: 2.2, mtx: Mat4f.fromTranslation(new Vec3(xOf(0) - 3 - w, f.y - lv * heightScale - 1.1, zFront)) };
        drawText(fb, lv > 0 ? '+1' : '-1', 0, 0, opts);
    }
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
}

function fmtNum(v: number) {
    return (v >= 0 ? '+' : '-') + Math.abs(v).toFixed(2);
}

// K noise particles on 23 vertical axes (one per dim, step viz.particleStep): each particle
// starts at its own real x1_k and moves to its own real x1_k − v_k; the mean of the K results
// and the label a are marked at the end
function drawParticles(state: IProgramState, layout: IVlaModelLayout, m: IVlaModelLink, viz: IActionVizState) {
    let { f, barW, xOf, zFront } = fieldGeom(layout);
    let cfg = m.config;
    let K = cfg.fm.n_particles, H = f.nSteps, D = f.nDims;
    let P = m.fm.particles;
    let s = viz.particleStep;
    let tau = Math.max(0, Math.min(1, viz.particleTau));
    let a = viz.particles;
    let tri = state.render.triRender, lr = state.render.lineRender;
    let hs = f.heightScale;
    let FD = fmDir(layout.variant), sg = FD.fwd ? 1 : -1, N = FD.noise;
    let z = zFront - 2;
    let at = (k: number, which: number, d: number) => P[((k * 3 + which) * H + s) * D + d];
    let guide = new Vec4(0.6, 0.6, 0.65, 1).mul(0.6 * a);
    for (let lv of [-2, -1, 0, 1, 2]) {
        addLine(lr, lv === 0 ? 1.2 : 0.6, guide, new Vec3(xOf(0) - 2, f.y - lv * hs, z), new Vec3(xOf(D - 1) + barW + 2, f.y - lv * hs, z), undefined, undefined, 0);
    }
    let tgt = m.fm.target.localBuffer;
    for (let d = 0; d < D; d++) {
        let cx = xOf(d) + barW / 2;
        addLine(lr, 0.6, groupColorOf(d).mul(0.6 * a), new Vec3(cx, f.y - 2.6 * hs, z), new Vec3(cx, f.y + 2.6 * hs, z), undefined, undefined, 0);
        let mean = 0;
        for (let k = 0; k < K; k++) {
            let px = cx + (k - (K - 1) / 2) * 0.42;
            let y0 = f.y - at(k, 0, d) * hs;
            let yc = f.y - (at(k, 0, d) + sg * tau * at(k, 1, d)) * hs;
            let col = particleColors[k % particleColors.length];
            addLine(lr, 0.7, col.mul(0.55 * a), new Vec3(px, y0, z + 0.2), new Vec3(px, yc, z + 0.2), undefined, undefined, 0);
            box(tri, px - 0.25, px + 0.25, yc - 0.25, yc + 0.25, z, z + 0.5, col, a, true);
            mean += at(k, 2, d) / K;
        }
        if (tau >= 1) {
            let ym = f.y - mean * hs, yt = f.y - tgt[s * D + d] * hs;
            box(tri, cx - 1.8, cx + 1.8, ym - 0.2, ym + 0.2, z + 0.6, z + 0.9, new Vec4(0.1, 0.1, 0.12, 1), a, true);
            box(tri, cx - 1.8, cx + 1.8, yt - 0.2, yt + 0.2, z + 0.6, z + 0.9, errColor, a, true);
        }
    }
    let fb = state.render.modelFontBuf;
    state.render.sharedRender.activePhase = RenderPhase.Overlay;
    let txt = new Vec4(0.1, 0.1, 0.15, 1).mul(a);
    for (let d = 0; d < D; d++) {
        let mtx = Mat4f.fromTranslation(new Vec3(xOf(d) + barW / 2 + 0.8, f.y + 2.8 * hs, z)).mul(Mat4f.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2));
        drawText(fb, `${d} ${cfg.action_dim_names[d]}`, 0, 0, { color: groupColorOf(d).mul(0.7).add(new Vec4(0, 0, 0, 0.3)).mul(a), size: 2.2, mtx });
    }
    for (let lv of [-2, -1, 0, 1, 2]) {
        drawText(fb, lv > 0 ? `+${lv}` : `${lv}`, 0, 0, { color: txt, size: 2.2, mtx: Mat4f.fromTranslation(new Vec3(xOf(0) - 7, f.y - lv * hs - 1.1, z)) });
    }
    let fm = cfg.fm;
    let lines = [
        `${K} noise particles, step ${s}: ${N}_k -> ${N}_k ${FD.sign} v_k (one real t=${FD.tNoise} forward each)`,
        tau >= 1 ? `black = mean of ${K} (MSE ${fm.mse_mean_chunk.toFixed(3)}), red = label a, single MSE ${Math.min(...fm.mse_per_particle).toFixed(2)}..${Math.max(...fm.mse_per_particle).toFixed(2)}` : `tau = ${tau.toFixed(2)}`,
    ];
    lines.forEach((l, j) => drawText(fb, l, 0, 0, { color: txt, size: 2.4, mtx: Mat4f.fromTranslation(new Vec3(xOf(0), f.y - 3.2 * hs + j * 3.8, z)) }));
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
}

function quad(tri: any, a: Vec3, b: Vec3, c: Vec3, d: Vec3, color: Vec4, n: Vec3) {
    addVert(tri, a, color, n);
    addVert(tri, b, color, n);
    addVert(tri, c, color, n);
    addVert(tri, d, color, n);
    addPrimitiveRestart(tri);
}

// axis-aligned bar; layout y grows downward, so "up" = smaller y
function box(tri: any, x0: number, x1: number, yTop: number, yBot: number, z0: number, z1: number, base: Vec4, alpha: number, positive: boolean) {
    if (yBot - yTop < 0.05) yBot = yTop + 0.05;
    let shade = (k: number) => new Vec4(base.x * k, base.y * k, base.z * k, 1).mul(alpha);
    let top = shade(positive ? 1.1 : 0.75);
    let side = shade(positive ? 0.85 : 0.6);
    let front = shade(positive ? 0.95 : 0.68);
    // top (y = yTop), bottom cap
    quad(tri, new Vec3(x0, yTop, z0), new Vec3(x1, yTop, z0), new Vec3(x0, yTop, z1), new Vec3(x1, yTop, z1), top, new Vec3(0, -1, 0));
    quad(tri, new Vec3(x0, yBot, z0), new Vec3(x1, yBot, z0), new Vec3(x0, yBot, z1), new Vec3(x1, yBot, z1), top, new Vec3(0, 1, 0));
    // front / back (z)
    quad(tri, new Vec3(x0, yTop, z1), new Vec3(x1, yTop, z1), new Vec3(x0, yBot, z1), new Vec3(x1, yBot, z1), front, new Vec3(0, 0, 1));
    quad(tri, new Vec3(x0, yTop, z0), new Vec3(x1, yTop, z0), new Vec3(x0, yBot, z0), new Vec3(x1, yBot, z0), front, new Vec3(0, 0, -1));
    // left / right (x)
    quad(tri, new Vec3(x0, yTop, z0), new Vec3(x0, yTop, z1), new Vec3(x0, yBot, z0), new Vec3(x0, yBot, z1), side, new Vec3(-1, 0, 0));
    quad(tri, new Vec3(x1, yTop, z0), new Vec3(x1, yTop, z1), new Vec3(x1, yBot, z0), new Vec3(x1, yBot, z1), side, new Vec3(1, 0, 0));
}
