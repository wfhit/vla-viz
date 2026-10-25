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
import { IPi0Model } from "./Pi0Model";
import { IPEdge, IPi0Layout, PEdgeKind } from "./Pi0Layout";

// Scene drawing for /pi0: the layout's edges as arrows (each recorded for the orphan audit),
// the cond time bus, the camera thumbnails feeding SigLIP, ghosts for the SigLIP blocks that
// carry no exported tensor, and the ASCII labels.

export const pi0EdgeColor: Record<PEdgeKind, Vec4> = {
    data: Vec4.fromHexColor('#33aa33'),
    weight: Vec4.fromHexColor('#3333aa'),
    time: Vec4.fromHexColor('#0891b2'),
    text: Vec4.fromHexColor('#db2777'),
    image: Vec4.fromHexColor('#7c3aed'),
    kv: Vec4.fromHexColor('#c026d3'),
    euler: Vec4.fromHexColor('#ca8a04'),
    skip: Vec4.fromHexColor('#2f6f2f'),
};
const edgeColor = pi0EdgeColor;

interface IBox { x: number; y: number; z: number; dx: number; dy: number; dz: number; }

const pad = 2;
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

function wireRect(r: IRenderState, x: number, y: number, dx: number, dy: number, z: number, color: Vec4, thick = 0.8) {
    let lr = r.lineRender;
    let p = [new Vec3(x, y, z), new Vec3(x + dx, y, z), new Vec3(x + dx, y + dy, z), new Vec3(x, y + dy, z)];
    for (let i = 0; i < 4; i++) addLine(lr, thick, color, p[i], p[(i + 1) % 4], undefined, undefined, 0);
}

export function drawPi0Scene(state: IProgramState, L: IPi0Layout, m: IPi0Model | null) {
    let r = state.render;
    resetArrowEdges();
    r.sharedRender.activePhase = RenderPhase.Opaque;

    let edges: IPEdge[] = [...L.edges];
    // step k's velocity lands in cell k+1 of the Euler track
    edges.push({ src: [L.v], dst: [L.track[Math.min(L.step + 1, L.track.length - 1)]], kind: 'euler', route: 'side' });
    for (let e of edges) drawEdge(r, L, e);

    drawTimeBus(r, L);
    drawThumbs(r, L, m);
    drawGhosts(r, L);
    drawLabels(r, L, m);

    if (typeof window !== 'undefined') {
        (window as any).__arrowAudit = () => auditEdges(L.cubes, L.srcOk, L.sinkOk);
    }
}

function drawEdge(r: IRenderState, L: IPi0Layout, e: IPEdge) {
    let sv = e.src.filter(b => b.opacity > 0), dv = e.dst.filter(b => b.opacity > 0);
    if (!sv.length || !dv.length) return;
    for (let a of sv) for (let b of dv) recordArrowEdge(a, b);
    let opacity = Math.min(Math.max(...sv.map(b => b.opacity)), Math.max(...dv.map(b => b.opacity)));
    let s = unionBox(sv), d = unionBox(dv);
    let color = edgeColor[e.kind].mul(opacity);
    let w = e.kind === 'data' || e.kind === 'image' ? 6 : e.kind === 'euler' ? 5 : 3;
    let n = new Vec3(0, 0, 1);

    if (e.kind === 'skip') {
        let srcName = e.src[0].name.replace(/^(prefix|expert) /, '');
        sceneText(r, `+ skip: ${srcName}`, d.x + d.dx + 3, d.y + 4, ZA, 3.5, edgeColor.skip.mul(opacity));
        return;
    }

    let scx = s.x + s.dx / 2, scy = s.y + s.dy / 2, dcx = d.x + d.dx / 2, dcy = d.y + d.dy / 2;
    let right = dcx > scx;
    let below = d.y >= s.y + s.dy - 0.01;

    if (e.route === 'bus') {
        drawArrow(r, new Vec3(L.busX, dcy, ZA), new Vec3(d.x + d.dx + pad, dcy, ZA), w, n, color);
        return;
    }

    if (e.route === 'down') {
        let ox0 = Math.max(s.x, d.x), ox1 = Math.min(s.x + s.dx, d.x + d.dx);
        if (below && ox1 - ox0 >= w) {
            let x = (ox0 + ox1) / 2;
            drawArrow(r, new Vec3(x, s.y + s.dy + pad, ZA), new Vec3(x, d.y - pad, ZA), w, n, color);
            return;
        }
        let start = new Vec3(scx, s.y + s.dy + pad, ZA);
        let end = new Vec3(right ? d.x - pad : d.x + d.dx + pad, dcy, ZA);
        let mid0 = new Vec3(start.x, end.y - w / 2, ZA);
        let mid1 = new Vec3(start.x + (right ? w / 2 : -w / 2), end.y, ZA);
        drawArrow(r, start, mid0, w, n, color, false, CornerMode.None, new Vec3(0, 1, 0));
        drawArrow(r, mid1, end, w, n, color, true, right ? CornerMode.Right : CornerMode.Left);
        return;
    }

    let yOverlap = (dcy >= s.y && dcy <= s.y + s.dy) || (scy >= d.y && scy <= d.y + d.dy);
    let sx = right ? s.x + s.dx + pad : s.x - pad;
    let ex = right ? d.x - pad : d.x + d.dx + pad;
    if (yOverlap) {
        let y = dcy >= s.y && dcy <= s.y + s.dy ? dcy : scy;
        drawArrow(r, new Vec3(sx, y, ZA), new Vec3(ex, y, ZA), w, n, color);
    } else if (e.kind === 'kv' || e.kind === 'image') {
        // K/V read by the expert: straight from src's nearest point
        let y = Math.min(Math.max(dcy, s.y), s.y + s.dy);
        drawArrow(r, new Vec3(sx, y, ZA), new Vec3(ex, dcy, ZA), w, n, color);
    } else {
        let start = new Vec3(sx, scy, ZA);
        let end = new Vec3(dcx, below ? d.y - pad : d.y + d.dy + pad, ZA);
        let sgn = below ? 1 : -1;
        let mid0 = new Vec3(end.x + (right ? -w / 2 : w / 2), start.y, ZA);
        let mid1 = new Vec3(end.x, start.y + sgn * w / 2, ZA);
        drawArrow(r, start, mid0, w, n, color, false);
        drawArrow(r, mid1, end, w, n, color, true, right === below ? CornerMode.Left : CornerMode.Right);
    }
}

// cond -> down -> left -> the bus beside the adaRMS rods; each rod group has its own tap arrow
function drawTimeBus(r: IRenderState, L: IPi0Layout) {
    let rods = [...L.exp.map(d => d.scale), L.finScale].filter(b => b.opacity > 0);
    if (L.cond.opacity === 0 || !rods.length) return;
    let color = edgeColor.time.mul(L.cond.opacity);
    let w = 3, n = new Vec3(0, 0, 1);
    let cx = L.cond.x + L.cond.dx / 2;
    let yTurn = L.exp[0].out.y - 10;
    let yEnd = Math.max(...rods.map(b => b.y + b.dy / 2)) + w / 2;
    drawArrow(r, new Vec3(cx, L.cond.y + L.cond.dy + pad, ZA), new Vec3(cx, yTurn + w / 2, ZA), w, n, color, false);
    drawArrow(r, new Vec3(cx - w / 2, yTurn, ZA), new Vec3(L.busX + w / 2, yTurn, ZA), w, n, color, false);
    drawArrow(r, new Vec3(L.busX, yTurn - w / 2, ZA), new Vec3(L.busX, yEnd, ZA), w, n, color, false);
    sceneText(r, 'cond time bus (t -> every layer\'s adaRMS)', L.busX + 4, yTurn - 2, ZA, 4, color);
}

function drawThumbs(r: IRenderState, L: IPi0Layout, m: IPi0Model | null) {
    if (!m) return;
    let size = 48;
    let prev = r.sharedRender.activePhase;
    r.sharedRender.activePhase = RenderPhase.Opaque;
    m.thumbs.forEach((t, i) => {
        let p = L.patch[i];
        if (!p || p.opacity === 0) return;
        let xi = p.x + p.dx / 2 - size / 2, y0 = p.y - 24 - size, z = p.z + p.dz;
        let px = size / t.n;
        for (let y = 0; y < t.n; y++) for (let x = 0; x < t.n; x++) {
            let k = (y * t.n + x) * 3;
            let col = new Vec4(t.rgb[k], t.rgb[k + 1], t.rgb[k + 2], p.opacity);
            addQuad(r.triRender, new Vec3(xi + x * px, y0 + y * px, z), new Vec3(xi + (x + 1) * px, y0 + (y + 1) * px, z), col);
        }
        sceneText(r, `${t.view} (224x224, 16x16 patches of 14px)`, xi + size / 2, y0 - 4, z, 4, new Vec4(0.8, 0.83, 0.9, p.opacity), 'c');
        let cx = p.x + p.dx / 2;
        drawArrow(r, new Vec3(cx, y0 + size + pad, z), new Vec3(cx, p.y - pad, z), 6, new Vec3(0, 0, 1), edgeColor.data.mul(p.opacity));
    });
    r.sharedRender.activePhase = prev;
}

function drawGhosts(r: IRenderState, L: IPi0Layout) {
    let grey = new Vec4(0.55, 0.55, 0.6, 1);
    let v = L.vitB0;
    if (v.opacity > 0 && L.vitB26.opacity > 0) {
        let [ya, yb] = L.vitGhostY;
        for (let j = 0; j < 6; j++) wireRect(r, v.x, ya + (yb - ya - 3) * j / 5, v.dx, 3, ZA, grey.mul(0.6 * v.opacity));
        sceneText(r, 'SigLIP blocks 1-25 (same shape, not exported)', v.x + v.dx + 4, (ya + yb) / 2, ZA, 4, grey.mul(v.opacity));
    }
    for (let i = 1; i < 3; i++) {
        let p = L.patch[i];
        if (p.opacity === 0) continue;
        sceneText(r, 'same 27 SigLIP blocks + head', p.x + p.dx / 2, p.y + p.dy + 40, ZA, 4, grey.mul(1.3 * p.opacity), 'c');
    }
}

function drawLabels(r: IRenderState, L: IPi0Layout, m: IPi0Model | null) {
    let txt = new Vec4(0.82, 0.85, 0.92, 1);
    let cfg = m?.cfg;
    let st = m?.steps[L.step]?.tex;
    let share = st?.['prefix_share']?.localBuffer;
    let H = cfg?.action_horizon ?? 32;
    for (let d of L.exp) {
        if (d.out.opacity === 0) continue;
        let s = '';
        if (share) {
            let t = 0;
            for (let i = 0; i < H; i++) t += share[d.l * H + i];
            s = `  prefix ${(t / H * 100).toFixed(0)}%`;
        }
        sceneText(r, `L${d.l}${s}`, L.busX + 6, d.out.y + 4, ZA, 6, txt.mul(d.out.opacity));
    }
    for (let p of L.pre) {
        if (p.opacity === 0) continue;
        sceneText(r, p.name.replace(' in', ''), p.x - 4, p.y + 4, ZA, 6, txt.mul(p.opacity), 'r');
    }
    if (L.final.opacity > 0) sceneText(r, 'final adaRMS (scale, shift)', L.busX + 6, L.final.y + 4, ZA, 6, txt.mul(L.final.opacity));
    let p0 = L.pre[0];
    if (p0.opacity > 0 && cfg) {
        let nPv = (cfg.prefix_rows.length - cfg.T) / 3;
        sceneText(r, `prefix rows shown: ${nPv} of 256 patches per view + all ${cfg.T} text tokens (of ${cfg.Np})`,
            p0.x, p0.y - 6, ZA, 3.5, txt.mul(p0.opacity * 0.9));
    }
    // K/V band label
    let e0 = L.exp[0].out;
    if (e0.opacity > 0 && p0.opacity > 0) {
        sceneText(r, 'joint attention: layer l reads prefix K/V of layer l', (p0.x + p0.dx + e0.x) / 2, p0.y - 18, ZA, 4, edgeColor.kv.mul(e0.opacity), 'c');
    }
    // Euler track: x += v * dt, dt = -0.1
    for (let j = 1; j < L.track.length; j++) {
        let a = L.track[j - 1], b = L.track[j];
        if (a.opacity === 0 || b.opacity === 0) continue;
        sceneText(r, '-0.1v', (a.x + a.dx + b.x) / 2, a.y - 6, ZA, 3.5, edgeColor.euler.mul(Math.min(a.opacity, b.opacity)), 'c');
    }
    if (L.act23.opacity > 0) sceneText(r, 'unnormalise (q01/q99), keep 23 R1 Pro dims', L.act23.x + L.act23.dx + 6, L.act23.y + L.act23.dy / 2, ZA, 4, txt.mul(L.act23.opacity));
    if (cfg && L.ids.opacity > 0) {
        sceneText(r, `"${cfg.sample.task}"`, L.ids.x, L.ids.y - 24, ZA, 4.5, txt.mul(L.ids.opacity));
        // the state segment of the prompt
        let toks = cfg.tokens;
        let s0 = cfg.state_token_start, s1 = toks.length - 1;
        while (s1 > s0 && !/;/.test(toks[s1])) s1--;
        let c = L.ids.dx / cfg.T;
        let x0 = L.ids.x + s0 * c, x1 = L.ids.x + s1 * c;
        wireRect(r, x0, L.ids.y - 2, x1 - x0, L.ids.dy + 4, ZA, edgeColor.text.mul(L.ids.opacity), 0.6);
        sceneText(r, 'state as text', (x0 + x1) / 2, L.ids.y + L.ids.dy + 8, ZA, 3.5, edgeColor.text.mul(L.ids.opacity), 'c');
    }
    let A = L.anat;
    if (A.exp.out.opacity > 0 || A.pre.out.opacity > 0) {
        let o = Math.max(A.exp.out.opacity, A.pre.out.opacity);
        sceneText(r, 'layer 0, prefix stream (PaliGemma 2B weights)', A.pre.xIn.x, A.pre.xIn.y - 40, ZA, 7, txt.mul(o));
        sceneText(r, 'layer 0, action-expert stream (300M weights), Euler step 0', A.exp.xIn.x - 150, A.exp.xIn.y - 40, ZA, 7, txt.mul(o));
        sceneText(r, '= prefix L0 in (row subset)', A.pre.xIn.x + A.pre.xIn.dx + 4, A.pre.xIn.y + 3, ZA, 3.5, txt.mul(o));
        sceneText(r, '= prefix L1 in', A.pre.out.x + A.pre.out.dx + 4, A.pre.out.y + 12, ZA, 3.5, txt.mul(o));
        sceneText(r, '= action tokens', A.exp.xIn.x + A.exp.xIn.dx + 4, A.exp.xIn.y + 3, ZA, 3.5, txt.mul(o));
        sceneText(r, '= expert L0 out', A.exp.out.x + A.exp.out.dx + 4, A.exp.out.y + 12, ZA, 3.5, txt.mul(o));
        let pr = A.exp.probs;
        let np = cfg?.Np ?? 885;
        let xs = pr.x + pr.dx * np / (np + H);
        wireRect(r, xs, pr.y - 2, pr.x + pr.dx - xs, pr.dy + 4, ZA, edgeColor.kv.mul(o), 0.6);
        sceneText(r, `${np} prefix keys | ${H} action keys`, xs, pr.y + pr.dy + 8, ZA, 4, edgeColor.kv.mul(o), 'c');
    }
}
