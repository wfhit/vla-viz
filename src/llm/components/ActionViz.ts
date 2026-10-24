// 3D visuals of the action path, all driven by REAL exported tensors:
//  - joint-attention links: each trunk layer i's K/V groups → expert layer i's attention
//    block, brightness = the real share of suffix attention that lands on the prefix
//  - the 16 × 23 action chunk as a bar field (bar height = real normalized value), with
//    the one-step Euler decode animated: x(τ) = x1 − τ·v, τ: 0 → 1 (t = 1 → 0)
import { IProgramState } from "../Program";
import { IGptModelLayout } from "../GptModelLayout";
import { IGptModelLink } from "../GptModel";
import { addLine } from "../render/lineRender";
import { addPrimitiveRestart, addVert } from "../render/triRender";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { RenderPhase } from "../render/sharedRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { actionGroups, unnormalizeAction } from "../actionDims";
import { drawArrow, recordArrowEdge } from "./Arrow";
import { IBlkDef } from "../GptModelLayout";

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
}

export function defaultActionViz(): IActionVizState {
    return {
        links: 0, activeLayer: -1, massBars: 0, field: 1, fieldMode: 'decode', decode: 1, showVelocity: 0,
        target: 0, errBars: 0, robotUnits: 0, particles: 0, particleTau: 1, particleStep: 0,
    };
}

const kColor = Vec4.fromHexColor('#f59e0b');
const vColor = Vec4.fromHexColor('#d946ef');
const expertArrowColor = Vec4.fromHexColor('#d97706');
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

export function drawActionViz(state: IProgramState) {
    let layout = state.layout;
    let m = state.jsGptModel;
    if (!layout || !m || !state.render) return;
    let viz = state.display.actionViz ?? defaultActionViz();

    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    drawExpertChain(state, layout, viz);
    drawJointLinks(state, layout, m, viz);
    if (viz.massBars > 0) drawMassBars(state, layout, m, viz);
    if (viz.particles > 0) {
        drawParticles(state, layout, m, viz);
    } else if (viz.field > 0) {
        drawBarField(state, layout, m, viz);
    }
}

function drawExpertChain(state: IProgramState, layout: IGptModelLayout, viz: IActionVizState) {
    let render = state.render;
    let n = new Vec3(0, 0, 1);
    let w = 4;
    let pad = 2;
    let arrow = (src: IBlkDef, dst: IBlkDef, x: number) => {
        let op = Math.min(src.opacity, dst.opacity);
        if (op <= 0) return;
        recordArrowEdge(src, dst);
        drawArrow(render, new Vec3(x, src.y + src.dy + pad, 0), new Vec3(x, dst.y - pad, 0), w, n, expertArrowColor.mul(op), true);
    };

    // a jump between columns gets no arrow across the scene; both ends are labelled (as the
    // trunk's wrap) and the edge is recorded so every block stays traceable
    let wrap = (src: IBlkDef, dst: IBlkDef, outText: string, inText: string) => {
        let op = Math.min(src.opacity, dst.opacity);
        if (op <= 0) return;
        recordArrowEdge(src, dst);
        let fb = render.modelFontBuf, size = 6;
        let color = new Vec4(0.55, 0.35, 0.15, 1).mul(op);
        let put = (blk: IBlkDef, text: string, y: number) => {
            let opts = { color, size, mtx: Mat4f.fromTranslation(new Vec3(blk.x + blk.dx / 2, y, blk.z + blk.dz / 2)) };
            let tw = measureText(fb, text, opts);
            let prev = render.sharedRender.activePhase;
            render.sharedRender.activePhase = RenderPhase.Overlay;
            drawText(fb, text, -tw / 2, 0, opts);
            render.sharedRender.activePhase = prev;
        };
        put(src, outText, src.y + src.dy + 3 * size);
        put(dst, inText, dst.y - size);
    };

    let els = layout.expertLayers;
    if (els.length === 0) return;
    let xIn = layout.expertIn.x + layout.expertIn.dx / 2;
    arrow(layout.expertIn, els[0].attn, xIn);
    for (let i = 0; i < els.length; i++) {
        let { attn, out } = els[i];
        let x = out.x + out.dx / 2;
        arrow(attn, out, x);
        let next = els[i + 1];
        if (!next) continue;
        if (Math.abs(next.attn.x - attn.x) < 1) {
            arrow(out, next.attn, x);
        } else {
            wrap(out, next.attn, `continues: expert ${i + 1} (next column) ->`, `<- from expert ${i}`);
        }
    }
    let nLast = els.length - 1;
    let last = els[nLast].out;
    let fin = layout.expertOut;
    if (Math.abs(fin.x - last.x) < 1) {
        arrow(last, fin, last.x + last.dx / 2);
    } else {
        wrap(last, fin, `continues: Expert Suffix Out (below heads) ->`, `<- from expert ${nLast}`);
    }
}

// mean over the 17 suffix queries of layer i's attention mass [state | visual | text | suffix]
export function layerMass(m: IGptModelLink, i: number) {
    let ms = m.expertLayers[i].mass, n = m.config.n_suffix, acc = [0, 0, 0, 0];
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
// over them); width ∝ the real share of suffix attention that lands on the prefix
function drawJointLinks(state: IProgramState, layout: IGptModelLayout, m: IGptModelLink, viz: IActionVizState) {
    if (viz.links <= 0 || viz.activeLayer < 0) return;
    let i = viz.activeLayer;
    let { attn } = layout.expertLayers[i];
    let block = layout.blocks[i];
    if (attn.opacity <= 0) return;
    let mass = layerMass(m, i);
    let pre = mass[1] + mass[2];
    let thick = 1 + 3 * pre;
    let end = new Vec3(attn.x - 2, attn.y + attn.dy / 2, attn.z + attn.dz / 2);
    for (let g of block.kvGroups) {
        for (let [src, col] of [[g.kBlock, kColor], [g.vBlock, vColor]] as [IBlkDef, Vec4][]) {
            if (src.opacity <= 0) continue;
            recordArrowEdge(src, attn);
            let start = new Vec3(src.x + src.dx + 2, src.y + src.dy / 2, src.z + src.dz / 2);
            arrow3(state, start, end, thick, col.mul(viz.links));
        }
    }
}

function drawMassBars(state: IProgramState, layout: IGptModelLayout, m: IGptModelLink, viz: IActionVizState) {
    let i = viz.activeLayer;
    if (i < 0) return;
    let cfg = m.config;
    let { attn } = layout.expertLayers[i];
    let mass = layerMass(m, i);
    let nVis = cfg.suffix_visible_visual, nTxt = cfg.suffix_visible_prefix - cfg.suffix_visible_visual;
    let nAns = cfg.T_text - cfg.suffix_visible_prefix;
    let rows: [string, number | null, Vec4][] = [
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

function fieldGeom(layout: IGptModelLayout) {
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

function drawBarField(state: IProgramState, layout: IGptModelLayout, m: IGptModelLink, viz: IActionVizState) {
    let { f, barW, xOf, zOf, zFront } = fieldGeom(layout);
    let tri = state.render.triRender;
    let { nSteps, nDims, heightScale } = f;
    let x1 = m.actionX1.localBuffer;
    let vel = m.actionVelocity.localBuffer;
    let tgt = m.fm.target.localBuffer;
    let tau = Math.max(0, Math.min(1, viz.decode));
    let alpha = viz.field;
    let isVel = viz.fieldMode === 'velocity';

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
            let val = isVel ? vel[i] : x1[i] - tau * vel[i];
            let h = val * heightScale;
            let base = isVel ? divergingColor(val) : groupColorOf(d);
            let x0 = xOf(d), z0 = zOf(s);
            box(tri, x0, x0 + barW, f.y - Math.max(h, 0), f.y - Math.min(h, 0), z0, z0 + barW, base, alpha, h >= 0);
            let cx = x0 + barW / 2, cz = z0 + barW / 2;
            if (viz.showVelocity > 0 && !isVel) {
                // this component's Euler step: from x1 to x1 − v
                let yA = f.y - x1[i] * heightScale, yB = f.y - (x1[i] - vel[i]) * heightScale;
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
    let tNow = 1 - tau;
    let title = isVel
        ? `velocity v = action_out_proj(expert)  ${nSteps} x ${nDims}   blue +, red -`
        : `action chunk ${nSteps} x ${nDims}  |  x(t) = x1 - (1 - t) v,  t = ${tNow.toFixed(2)}${tau >= 1 ? '  (decoded: x1 - v)' : '  [interp only]'}`;
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
function drawParticles(state: IProgramState, layout: IGptModelLayout, m: IGptModelLink, viz: IActionVizState) {
    let { f, barW, xOf, zFront } = fieldGeom(layout);
    let cfg = m.config;
    let K = cfg.fm.n_particles, H = f.nSteps, D = f.nDims;
    let P = m.fm.particles;
    let s = viz.particleStep;
    let tau = Math.max(0, Math.min(1, viz.particleTau));
    let a = viz.particles;
    let tri = state.render.triRender, lr = state.render.lineRender;
    let hs = f.heightScale;
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
            let yc = f.y - (at(k, 0, d) - tau * at(k, 1, d)) * hs;
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
        `${K} noise particles, step ${s}: x1_k -> x1_k - v_k (one real t=1 forward each)`,
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
