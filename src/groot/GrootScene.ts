import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IProgramState } from "@/src/vla/Program";
import { auditEdges, CornerMode, drawArrow, recordArrowEdge, resetArrowEdges } from "@/src/vla/components/Arrow";
import { drawText, measureText } from "@/src/vla/render/fontRender";
import { addLine } from "@/src/vla/render/lineRender";
import { IRenderState } from "@/src/vla/render/modelRender";
import { RenderPhase } from "@/src/vla/render/sharedRender";
import { addQuad } from "@/src/vla/render/triRender";
import { IBlkDef } from "@/src/vla/VlaModelLayout";
import { IGrootModel } from "./GrootModel";
import { GEdgeKind, IGEdge, IGrootLayout } from "./GrootLayout";

// Scene drawing for /groot: the layout's edges as arrows (each recorded for the orphan audit),
// the temb time bus, the camera thumbnails feeding the ViT, ghosts for the parts that exist in
// the model but carry no exported tensor, and the ASCII labels.

const edgeColor: Record<GEdgeKind, Vec4> = {
    data: Vec4.fromHexColor('#33aa33'),
    weight: Vec4.fromHexColor('#3333aa'),
    time: Vec4.fromHexColor('#0891b2'),
    text: Vec4.fromHexColor('#db2777'),
    image: Vec4.fromHexColor('#7c3aed'),
    euler: Vec4.fromHexColor('#ca8a04'),
    skip: Vec4.fromHexColor('#2f6f2f'),
};

interface IBox { x: number; y: number; z: number; dx: number; dy: number; dz: number; }

const pad = 2;
// blocks are one-cell-thick walls (z in [0, 1.5]); arrows and labels run just in front of them
const ZA = 2;

function unionBox(bs: IBlkDef[]): IBox {
    let x0 = Math.min(...bs.map(b => b.x)), x1 = Math.max(...bs.map(b => b.x + b.dx));
    let y0 = Math.min(...bs.map(b => b.y)), y1 = Math.max(...bs.map(b => b.y + b.dy));
    let z0 = Math.min(...bs.map(b => b.z)), z1 = Math.max(...bs.map(b => b.z + b.dz));
    return { x: x0, y: y0, z: z0, dx: x1 - x0, dy: y1 - y0, dz: z1 - z0 };
}

function sceneText(r: IRenderState, text: string, x: number, y: number, z: number, size: number, color: Vec4, align: 'l' | 'c' | 'r' = 'l') {
    let opts = { color, size: size * 1.8, mtx: Mat4f.fromTranslation(new Vec3(x, y, z)) };
    let w = align === 'l' ? 0 : measureText(r.modelFontBuf, text, opts);
    let prev = r.sharedRender.activePhase;
    r.sharedRender.activePhase = RenderPhase.Overlay;
    drawText(r.modelFontBuf, text, align === 'c' ? -w / 2 : align === 'r' ? -w : 0, 0, opts);
    r.sharedRender.activePhase = prev;
}

// outline of an x-y rectangle at depth z
function wireRect(r: IRenderState, x: number, y: number, dx: number, dy: number, z: number, color: Vec4, thick = 0.8) {
    let lr = r.lineRender;
    let p = [new Vec3(x, y, z), new Vec3(x + dx, y, z), new Vec3(x + dx, y + dy, z), new Vec3(x, y + dy, z)];
    for (let i = 0; i < 4; i++) addLine(lr, thick, color, p[i], p[(i + 1) % 4], undefined, undefined, 0);
}

export function drawGrootScene(state: IProgramState, L: IGrootLayout, m: IGrootModel | null) {
    let r = state.render;
    resetArrowEdges();
    r.sharedRender.activePhase = RenderPhase.Opaque;

    let edges: IGEdge[] = [...L.edges];
    // step k's velocity lands in x_t(k+1) of the Euler track
    edges.push({ src: [L.v], dst: [L.track[Math.min(L.step + 1, 4)]], kind: 'euler', route: 'side' });
    if (L.collapsedFront) edges.push({ src: [L.xt], dst: [L.saIn], kind: 'data', route: 'down' });

    for (let e of edges) drawEdge(r, L, e);

    drawTimeBus(r, L);
    drawThumbs(r, L, m);
    drawGhosts(r, L);
    drawLabels(r, L, m);

    if (typeof window !== 'undefined') {
        (window as any).__arrowAudit = () => auditEdges(L.cubes, L.srcOk, L.sinkOk);
    }
}

function drawEdge(r: IRenderState, L: IGrootLayout, e: IGEdge) {
    let sv = e.src.filter(b => b.opacity > 0), dv = e.dst.filter(b => b.opacity > 0);
    if (!sv.length || !dv.length) return;
    for (let a of sv) for (let b of dv) recordArrowEdge(a, b);
    let opacity = Math.min(Math.max(...sv.map(b => b.opacity)), Math.max(...dv.map(b => b.opacity)));
    let s = unionBox(sv), d = unionBox(dv);
    let color = edgeColor[e.kind].mul(opacity);
    let w = e.kind === 'data' ? 6 : e.kind === 'euler' ? 5 : 3;
    let n = new Vec3(0, 0, 1);

    if (e.kind === 'skip') {
        // the residual add is named on its destination, not drawn as a line past the sublayers
        let srcName = e.src[0].name.replace(/^D\d+ /, '');
        sceneText(r, `+ skip: ${srcName}`, d.x + d.dx + 3, d.y + 4, ZA, 3.5, edgeColor.skip.mul(opacity));
        return;
    }

    let scx = s.x + s.dx / 2, scy = s.y + s.dy / 2, dcx = d.x + d.dx / 2, dcy = d.y + d.dy / 2;
    let right = dcx > scx;
    let below = d.y >= s.y + s.dy - 0.01;

    if (e.route === 'bus') {
        drawArrow(r, new Vec3(L.timeBusX, dcy, ZA), new Vec3(d.x + d.dx + pad, dcy, ZA), w, n, color);
        return;
    }

    if (e.route === 'down') {
        let ox0 = Math.max(s.x, d.x), ox1 = Math.min(s.x + s.dx, d.x + d.dx);
        if (below && ox1 - ox0 >= w) {
            let x = (ox0 + ox1) / 2;
            drawArrow(r, new Vec3(x, s.y + s.dy + pad, ZA), new Vec3(x, d.y - pad, ZA), w, n, color);
            return;
        }
        // down, then across into dst's near side
        let start = new Vec3(scx, s.y + s.dy + pad, ZA);
        let end = new Vec3(right ? d.x - pad : d.x + d.dx + pad, dcy, ZA);
        let mid0 = new Vec3(start.x, end.y - w / 2, ZA);
        let mid1 = new Vec3(start.x + (right ? w / 2 : -w / 2), end.y, ZA);
        drawArrow(r, start, mid0, w, n, color, false, CornerMode.None, new Vec3(0, 1, 0));
        drawArrow(r, mid1, end, w, n, color, true, right ? CornerMode.Right : CornerMode.Left);
        return;
    }

    // side
    let yOverlap = (dcy >= s.y && dcy <= s.y + s.dy) || (scy >= d.y && scy <= d.y + d.dy);
    let sx = right ? s.x + s.dx + pad : s.x - pad;
    let ex = right ? d.x - pad : d.x + d.dx + pad;
    if (yOverlap) {
        let y = dcy >= s.y && dcy <= s.y + s.dy ? dcy : scy;
        drawArrow(r, new Vec3(sx, y, ZA), new Vec3(ex, y, ZA), w, n, color);
    } else if (e.kind === 'text' || e.kind === 'image') {
        // K/V pillar -> block: straight from the pillar's nearest point
        let y = Math.min(Math.max(dcy, s.y), s.y + s.dy);
        drawArrow(r, new Vec3(sx, y, ZA), new Vec3(ex, dcy, ZA), w, n, color);
    } else {
        // across, then down into dst's top
        let start = new Vec3(sx, scy, ZA);
        let end = new Vec3(dcx, below ? d.y - pad : d.y + d.dy + pad, ZA);
        let sgn = below ? 1 : -1;
        let mid0 = new Vec3(end.x + (right ? -w / 2 : w / 2), start.y, ZA);
        let mid1 = new Vec3(end.x, start.y + sgn * w / 2, ZA);
        drawArrow(r, start, mid0, w, n, color, false);
        drawArrow(r, mid1, end, w, n, color, true, right === below ? CornerMode.Left : CornerMode.Right);
    }
}

// temb -> down -> across -> the bus beside the γβ rods; each rod pair has its own tap arrow
function drawTimeBus(r: IRenderState, L: IGrootLayout) {
    let rods = [...L.dit.map(d => d.scale), L.finScale].filter(b => b.opacity > 0);
    if (L.temb.opacity === 0 || !rods.length) return;
    let o = L.temb.opacity;
    let color = edgeColor.time.mul(o);
    let w = 3, n = new Vec3(0, 0, 1);
    let tx = L.temb.x + L.temb.dx / 2;
    let yTurn = L.dit[0].out.y - 12;
    let yEnd = Math.max(...rods.map(b => b.y + b.dy / 2)) + w / 2;
    drawArrow(r, new Vec3(tx, L.temb.y + L.temb.dy + pad, ZA), new Vec3(tx, yTurn + w / 2, ZA), w, n, color, false);
    drawArrow(r, new Vec3(tx + w / 2, yTurn, ZA), new Vec3(L.timeBusX - w / 2, yTurn, ZA), w, n, color, false);
    drawArrow(r, new Vec3(L.timeBusX, yTurn - w / 2, ZA), new Vec3(L.timeBusX, yEnd, ZA), w, n, color, false);
    sceneText(r, 'temb time bus (t -> every block\'s AdaLN)', L.timeBusX + 4, yTurn - 2, ZA, 4, color);
}

function drawThumbs(r: IRenderState, L: IGrootLayout, m: IGrootModel | null) {
    let p = L.vitPatch;
    if (!m || p.opacity === 0) return;
    let size = 40, gap = 16, n = m.thumbs.length;
    let x0 = p.x + p.dx / 2 - (n * size + (n - 1) * gap) / 2;
    let y0 = p.y - 20 - size;
    let z = p.z + p.dz;
    let prev = r.sharedRender.activePhase;
    r.sharedRender.activePhase = RenderPhase.Opaque;
    m.thumbs.forEach((t, i) => {
        let px = size / t.n;
        let xi = x0 + i * (size + gap);
        for (let y = 0; y < t.n; y++) for (let x = 0; x < t.n; x++) {
            let k = (y * t.n + x) * 3;
            let col = new Vec4(t.rgb[k], t.rgb[k + 1], t.rgb[k + 2], p.opacity);
            addQuad(r.triRender, new Vec3(xi + x * px, y0 + y * px, z), new Vec3(xi + (x + 1) * px, y0 + (y + 1) * px, z), col);
        }
        sceneText(r, t.view, xi + size / 2, y0 - 3, z, 4, new Vec4(0.8, 0.83, 0.9, p.opacity), 'c');
    });
    r.sharedRender.activePhase = prev;
    // the 3 frames are the ViT's input: one arrow from the image row into the patch embedding
    let cx = p.x + p.dx / 2;
    drawArrow(r, new Vec3(cx, y0 + size + pad, z), new Vec3(cx, p.y - pad, z), 6, new Vec3(0, 0, 1), edgeColor.data.mul(p.opacity));
    sceneText(r, '3 cameras (BEHAVIOR demo frame), 256 patches each after 2x2 merge', cx, y0 - 10, z, 4.5, new Vec4(0.8, 0.83, 0.9, p.opacity), 'c');
}

function drawGhosts(r: IRenderState, L: IGrootLayout) {
    let grey = new Vec4(0.55, 0.55, 0.6, 1);
    // ViT blocks 1..22: same shape as block 0 / 23, not exported
    let v = L.vitB0;
    if (v.opacity > 0 && L.vitB23.opacity > 0) {
        let [ya, yb] = L.vitGhostY;
        for (let j = 0; j < 6; j++) {
            let y = ya + (yb - ya - 3) * j / 5;
            wireRect(r, v.x, y, v.dx, 3, ZA, grey.mul(0.6 * v.opacity));
        }
        sceneText(r, 'vit blocks 1-22 (same shape, not exported)', v.x + v.dx + 4, (ya + yb) / 2, v.z + v.dz, 4, grey.mul(v.opacity));
    }
    // LM layers 16..27 exist in the checkpoint but the policy reads layer 16's input: pruned
    let l = L.lm[15];
    if (l.opacity > 0) {
        let red = new Vec4(0.75, 0.35, 0.35, 1).mul(0.6 * l.opacity);
        for (let y of L.lmGhostY) wireRect(r, l.x, y, l.dx, 4, ZA, red);
        let ym = (L.lmGhostY[0] + L.lmGhostY[L.lmGhostY.length - 1]) / 2;
        sceneText(r, 'L16-L27 pruned (select_layer=16): not run', l.x - 4, ym, l.z + l.dz, 5, red.mul(1.5), 'r');
    }
    // embodiment slots: each slotted weight is slot 23 of a 32-slot bank
    for (let b of L.slotted) {
        if (b.opacity === 0) continue;
        for (let j = 1; j <= 3; j++) wireRect(r, b.x + 3 * j, b.y - 3 * j, b.dx, b.dy, b.z - 2 * j, grey.mul(0.5 * b.opacity), 0.6);
    }
    let s = L.stateL1;
    if (s.opacity > 0) {
        let s2 = L.stateL2;
        sceneText(r, 'adapters: slot 23 of 32 = sim_behavior_r1_pro', L.stateRaw.x, s2.y + s2.dy + 16, s.z + s.dz, 4, new Vec4(0.7, 0.4, 0.1, s.opacity));
    }
}

// which blocks get a name tag: everything except the 32 stack plates / rods (tagged by number)
function drawLabels(r: IRenderState, L: IGrootLayout, m: IGrootModel | null) {
    let txt = new Vec4(0.82, 0.85, 0.92, 1);
    // block names are tagged by drawBlockInfo; this adds the scene-specific annotations
    let kindTag = { text: 'cross: text', self: 'self-attn', image: 'cross: image' };
    for (let d of L.dit) {
        if (d.out.opacity === 0) continue;
        let col = d.kind === 'self' ? txt : edgeColor[d.kind];
        sceneText(r, `D${d.b} ${kindTag[d.kind]}`, L.timeBusX + 5, d.out.y + 4, ZA, 6, col.mul(d.out.opacity));
    }
    if (L.finNorm.opacity > 0) sceneText(r, 'final AdaLN (scale, shift)', L.timeBusX + 5, L.finNorm.y + 4, ZA, 6, txt);
    for (let a of L.anat) {
        if (a.out.opacity === 0) continue;
        let o = a.out.opacity;
        sceneText(r, `= sa sequence into block ${a.b} (step 0)`, a.xIn.x + a.xIn.dx + 4, a.xIn.y + 3, ZA, 3.5, txt.mul(o));
        sceneText(r, `= DiT block ${a.b} plate in the stack`, a.out.x + a.out.dx + 4, a.out.y + 3, ZA, 3.5, txt.mul(o));
        if (a.ctx) sceneText(r, '= vl_self_attention.B3 out, all 216 tokens; image keys masked', a.ctx.x, a.ctx.y - 8, a.ctx.z + a.ctx.dz, 3.5, edgeColor.text.mul(o));
        sceneText(r, `DiT block ${a.b} anatomy (${a.kind === 'self' ? 'self-attention' : 'cross-attention to text'})`,
            a.xIn.x, a.xIn.y - 45, a.xIn.z + a.xIn.dz, 7, txt.mul(o));
    }
    // Euler track: every arrow is x += v * dt with dt = 1/4
    for (let j = 1; j < L.track.length; j++) {
        let a = L.track[j - 1], b = L.track[j];
        if (a.opacity === 0 || b.opacity === 0) continue;
        sceneText(r, '+ v/4', (a.x + a.dx + b.x) / 2, a.y - 6, ZA, 4, edgeColor.euler.mul(Math.min(a.opacity, b.opacity)), 'c');
    }
    if (L.collapsedFront && L.saIn.opacity > 0) {
        let x = L.xt.x + L.xt.dx + 6;
        sceneText(r, `action_encoder(x_t${L.step}, tau) + pos  (internals shown at step 0)`, x, (L.xt.y + L.saIn.y) / 2, ZA, 4, txt);
    }
    if (m && L.ids.opacity > 0) {
        sceneText(r, `"${m.cfg.sample.task}"`, L.ids.x, L.ids.y - 8, L.ids.z + L.ids.dz, 5, txt);
    }
}
