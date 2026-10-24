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
import { actionGroups } from "../actionDims";
import { drawArrow } from "./Arrow";

export interface IActionVizState {
    links: number;      // opacity of the joint-attention links
    linkSweep: number;  // 0..1: layers revealed top → bottom
    activeLayer: number; // highlighted layer pair (-1 = none)
    field: number;      // opacity of the bar field
    decode: number;     // Euler progress τ ∈ [0, 1]; 1 = the decoded action chunk
    showVelocity: number; // opacity of the per-component velocity segments
}

export function defaultActionViz(): IActionVizState {
    return { links: 0.18, linkSweep: 1, activeLayer: -1, field: 1, decode: 1, showVelocity: 0 };
}

const linkColor = Vec4.fromHexColor('#f59e0b');
const expertArrowColor = Vec4.fromHexColor('#d97706');

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
    if (viz.field > 0) {
        drawBarField(state, layout, m, viz);
    }
}

function drawExpertChain(state: IProgramState, layout: IGptModelLayout, viz: IActionVizState) {
    let render = state.render;
    let n = new Vec3(0, 0, 1);
    let w = 4;
    let pad = 2;
    let col = expertArrowColor;
    let vArrow = (ax: number, ay: number, by: number, z: number) =>
        drawArrow(render, new Vec3(ax, ay + pad, z), new Vec3(ax, by - pad, z), w, n, col, true);

    let els = layout.expertLayers;
    if (els.length === 0) return;
    let first = els[0].attn;
    vArrow(layout.expertIn.x + layout.expertIn.dx / 2, layout.expertIn.y + layout.expertIn.dy, first.y, 0);
    for (let i = 0; i < els.length; i++) {
        let { attn, out } = els[i];
        let x = out.x + out.dx / 2;
        vArrow(x, attn.y + attn.dy, out.y, 0);
        let next = els[i + 1];
        // only within a column — the column wrap is marked by the trunk's wrap labels
        if (next && Math.abs(next.attn.x - attn.x) < 1) {
            vArrow(x, out.y + out.dy, next.attn.y, 0);
        }
    }
    let last = els[els.length - 1].out;
    let fin = layout.expertOut;
    if (Math.abs(fin.x - last.x) < 1) {
        vArrow(last.x + last.dx / 2, last.y + last.dy, fin.y, 0);
    }
}

function drawJointLinks(state: IProgramState, layout: IGptModelLayout, m: IGptModelLink, viz: IActionVizState) {
    if (viz.links <= 0) return;
    let lr = state.render.lineRender;
    let nL = layout.expertLayers.length;
    let nSuf = m.config.n_suffix;
    for (let i = 0; i < nL; i++) {
        let reveal = Math.max(0, Math.min(1, viz.linkSweep * nL - i));
        if (reveal <= 0) continue;
        let { attn } = layout.expertLayers[i];
        let block = layout.blocks[i];
        let mass = m.expertLayers[i].mass;
        // mean share of the suffix queries' attention on the prefix (visual + text keys)
        let pre = 0;
        for (let q = 0; q < nSuf; q++) pre += mass[q * 4 + 1] + mass[q * 4 + 2];
        pre /= nSuf;
        let active = viz.activeLayer === i ? 1 : 0;
        let a = viz.links * reveal * (0.25 + 0.75 * pre) * (viz.activeLayer >= 0 && !active ? 0.35 : 1);
        let thick = 1.5 + 3 * pre + 2 * active;
        let end = new Vec3(attn.x - 2, attn.y + attn.dy / 2, attn.z + attn.dz / 2);
        for (let g of block.kvGroups) {
            for (let src of [g.kBlock, g.vBlock]) {
                let start = new Vec3(src.x + src.dx + 2, src.y + src.dy / 2, src.z + src.dz / 2);
                addLine(lr, thick, linkColor.mul(a), start, end, undefined, undefined, 0);
            }
        }
    }
}

function drawBarField(state: IProgramState, layout: IGptModelLayout, m: IGptModelLink, viz: IActionVizState) {
    let f = layout.actionField;
    let tri = state.render.triRender;
    let { nSteps, nDims, pitch, heightScale } = f;
    let x1 = m.actionX1.localBuffer;
    let vel = m.actionVelocity.localBuffer;
    let tau = Math.max(0, Math.min(1, viz.decode));
    let barW = pitch * 0.72;
    let alpha = viz.field;

    let xOf = (d: number) => f.x + d * pitch;
    // step 0 nearest the default camera (+z), later steps recede
    let zOf = (s: number) => f.z + (nSteps - 1 - s) * pitch;

    // base plane
    let plane = new Vec4(0.55, 0.6, 0.68, 1).mul(0.35 * alpha);
    quad(tri, new Vec3(xOf(0) - 1, f.y, zOf(nSteps - 1) - 1), new Vec3(xOf(nDims) - pitch + barW + 1, f.y, zOf(nSteps - 1) - 1),
        new Vec3(xOf(0) - 1, f.y, zOf(0) + barW + 1), new Vec3(xOf(nDims) - pitch + barW + 1, f.y, zOf(0) + barW + 1), plane, new Vec3(0, -1, 0));

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
            let val = x1[i] - tau * vel[i];
            let h = val * heightScale;
            let base = groupColorOf(d);
            let x0 = xOf(d), z0 = zOf(s);
            box(tri, x0, x0 + barW, f.y - Math.max(h, 0), f.y - Math.min(h, 0), z0, z0 + barW, base, alpha, h >= 0);
            if (viz.showVelocity > 0) {
                // the Euler step of this component: from x1 to x1 − v
                let yA = f.y - x1[i] * heightScale, yB = f.y - (x1[i] - vel[i]) * heightScale;
                let cx = x0 + barW / 2, cz = z0 + barW / 2;
                addLine(lr, 1.5, new Vec4(1, 1, 1, 1).mul(0.85 * viz.showVelocity * alpha), new Vec3(cx, yA, cz), new Vec3(cx, yB, cz), undefined, undefined, 0);
            }
        }
    }

    // labels (font atlas is Latin-only: dims by their schema names)
    let names = m.config.action_dim_names;
    let fb = state.render.modelFontBuf;
    state.render.sharedRender.activePhase = RenderPhase.Overlay;
    let txt = new Vec4(0.1, 0.1, 0.15, 1).mul(alpha);
    let zFront = zOf(0) + barW + 2;
    let yLab = f.y + heightScale * 2.2;
    for (let d = 0; d < nDims; d++) {
        // vertical text: rotate about z so the name runs downward under its bar column
        let mtx = Mat4f.fromTranslation(new Vec3(xOf(d) + barW / 2 + 0.8, yLab, zFront)).mul(Mat4f.fromAxisAngle(new Vec3(0, 0, 1), Math.PI / 2));
        let opts: IFontOpts = { color: groupColorOf(d).mul(0.7).add(new Vec4(0, 0, 0, 0.3)).mul(alpha), size: 2.2, mtx };
        drawText(fb, `${d} ${names[d]}`, 0, 0, opts);
    }
    for (let s of [0, 5, 10, 15]) {
        if (s >= nSteps) continue;
        let opts: IFontOpts = { color: txt, size: 2.4, mtx: Mat4f.fromTranslation(new Vec3(xOf(nDims - 1) + barW + 3, f.y - 1.5, zOf(s) + barW / 2)) };
        drawText(fb, `step ${s}`, 0, 0, opts);
    }
    let tNow = 1 - tau;
    let title = `action chunk ${nSteps} x ${nDims}  |  x(t) = x1 - (1 - t) v,  t = ${tNow.toFixed(2)}${tau >= 1 ? '  (decoded: x1 - v)' : ''}`;
    let topY = f.y - heightScale * 2.6;
    let titleOpts: IFontOpts = { color: txt, size: 3.2, mtx: Mat4f.fromTranslation(new Vec3(xOf(0), topY, zFront)) };
    drawText(fb, title, 0, -3.2, titleOpts);
    let w = measureText(fb, '+1', { color: txt, size: 2.2 });
    for (let lv of [-1, 1]) {
        let opts: IFontOpts = { color: txt, size: 2.2, mtx: Mat4f.fromTranslation(new Vec3(xOf(0) - 3 - w, f.y - lv * heightScale - 1.1, zFront)) };
        drawText(fb, lv > 0 ? '+1' : '-1', 0, 0, opts);
    }
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
