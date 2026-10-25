
/* Arrow Design:

This is a flat, rectangular, ribbon-like pathway with lines down the edges.

It has a defined width. The ribbon might have some motion animation when I write the shader for it.

The sides will have lines to border the transparent ribbon.

We define a start, end, width, and "normal". The normal is perpendicular to the direction of the arrow,
and the normal lies within all parts of the ribbon.

The ribbon can bend, subject to the normal. The ribbon always starts in-plane. We'll have to do some
curve generation to make the ribbon bend. (this probably requires a 3rd degree bezier curve)

The arrow head is fairly flat, and is a triangle. We define the depth of the arrow head, and the
width. Not sure if the width should be a constant, or proportional to the width/length of the arrow.

Arrows will also have padding between blocks.

Some special arrows bend at the start in the plane of the ribbon, where they diverge from a vertical arrow.

This should be able to be implemented just with lines & tris. Depth buf will be interesting.
*/

import { IBlkDef, IVlaModelLayout } from "../VlaModelLayout";
import { drawLineSegs, makeLineOpts } from "../render/lineRender";
import { IRenderState } from "../render/modelRender";
import { addPrimitiveRestart, addQuad, addVert } from "../render/triRender";
import { bezierCurveBuild } from "@/src/utils/bezier";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { drawText, measureText } from "../render/fontRender";
import { RenderPhase } from "../render/sharedRender";

// Every drawn arrow records its (src, dst) block pair here; auditArrows() lists data blocks
// that are visible but have no incoming / outgoing arrow (the "orphan" self-check).
interface IArrowEdge { src: IBlkDef; dst: IBlkDef; }
let arrowEdges: IArrowEdge[] = [];

export function recordArrowEdge(src: IBlkDef, dst: IBlkDef) {
    arrowEdges.push({ src, dst });
}

export function resetArrowEdges() {
    arrowEdges = [];
}

// visible data blocks with no incoming / outgoing arrow, except the declared sources / sinks
export function auditEdges(cubes: IBlkDef[], srcOk: Set<IBlkDef>, sinkOk: Set<IBlkDef>) {
    let hasIn = new Set<IBlkDef>(), hasOut = new Set<IBlkDef>();
    for (let e of arrowEdges) { hasOut.add(e.src); hasIn.add(e.dst); }
    let visible = cubes.filter(c => c.opacity > 0 && c.t === 'i');
    let noIn = visible.filter(c => !hasIn.has(c) && !srcOk.has(c) && c.name);
    let noOut = visible.filter(c => !hasOut.has(c) && !sinkOk.has(c) && c.name);
    return { noIn: noIn.map(c => c.name), noOut: noOut.map(c => c.name), edges: arrowEdges.length };
}

export function auditArrows(layout: IVlaModelLayout) {
    // sources by construction: the token ids, the FM inputs, te's per-layer AdaNorm γβ, and
    // the bottom x1 which is the same tensor as the FM input x1 (a skip, not a new source)
    let srcOk = new Set<IBlkDef>([layout.idxObj, layout.fm.state75, layout.fm.x1, layout.fm.timeEmb,
        layout.actionX1, layout.strip[0], ...layout.expertLayers.flatMap(e => [e.ada, e.ada2])]);
    if (layout.expertNorm.ada) srcOk.add(layout.expertNorm.ada);
    let sinkOk = new Set<IBlkDef>([layout.logitsSoftmax, layout.doneProbs, layout.actionChunk, layout.strip[layout.strip.length - 1]]);
    // without a final trunk norm (pi0) the last trunk layer's output is computed but never read:
    // the expert only reads the trunk's per-layer K/V
    if (layout.ln_f.lnResid.absent) sinkOk.add(layout.blocks[layout.blocks.length - 1].mlpResidual);
    return auditEdges(layout.cubes, srcOk, sinkOk);
}

// `isVisible` lets the caller skip a layer's detailed arrows when the layer is off-screen
// (28 layers × ~100 arrows otherwise dominate the per-frame geometry upload)
export function drawAllArrows(state: IRenderState, layout: IVlaModelLayout, isVisible?: (cubes: IBlkDef[]) => boolean) {

    let pad = 2.0;
    let weightColor = Vec4.fromHexColor('#3333aa');
    let dataColor = Vec4.fromHexColor('#33aa33');
    let expertColor = Vec4.fromHexColor('#d97706');
    let timeColor = Vec4.fromHexColor('#0891b2');
    let flowColor = Vec4.fromHexColor('#facc15');
    let ditSet = new Set<IBlkDef>(layout.ditCubes);

    // a [trunk | DiT tail] pair (or a stacked group) drawn as ONE arrow endpoint: the arrow meets
    // the pair's outer box, so it never runs through the tail
    let members = new Map<IBlkDef, IBlkDef[]>();
    function U(...bs: IBlkDef[]): IBlkDef {
        let vis = bs.filter(b => b.opacity > 0);
        let use = vis.length ? vis : bs;
        let x0 = Math.min(...use.map(b => b.x)), x1 = Math.max(...use.map(b => b.x + b.dx));
        let y0 = Math.min(...use.map(b => b.y)), y1 = Math.max(...use.map(b => b.y + b.dy));
        let z0 = Math.min(...use.map(b => b.z)), z1 = Math.max(...use.map(b => b.z + b.dz));
        let u = { ...bs[0], x: x0, y: y0, z: z0, dx: x1 - x0, dy: y1 - y0, dz: z1 - z0,
            opacity: Math.max(0, ...bs.map(b => b.opacity)), name: '' };
        members.set(u, bs);
        return u;
    }
    let mem = (b: IBlkDef) => (members.get(b) ?? [b]).filter(c => c.opacity > 0);

    // off-screen layers: record the edges (audit) but upload no geometry
    let dry = false;
    function rec(src: IBlkDef, dst: IBlkDef) {
        let s = mem(src), d = mem(dst);
        if (s.length === d.length) s.forEach((b, i) => recordArrowEdge(b, d[i]));
        else for (let a of s) for (let b of d) recordArrowEdge(a, b);
    }
    function drawA(st: IRenderState, start: Vec3, end: Vec3, width: number, normal: Vec3, color: Vec4, drawHead: boolean = true, drawCorner: CornerMode = CornerMode.None, endDir?: Vec3) {
        if (!dry) drawArrow(st, start, end, width, normal, color, drawHead, drawCorner, endDir);
    }

    resetArrowEdges();
    drawVertArrow(layout.idxObj, layout.residual0);
    drawHorizArrow(layout.tokEmbedObj, layout.residual0);

    // flow-matching suffix inputs → the DiT suffix input (the x_t chain runs right → left)
    let fm = layout.fm;
    drawVertArrow(fm.state75, fm.stateProjW, 4);
    drawArrowBetween(fm.stateProjW, BlockPos.Left, fm.stateEmb, BlockPos.Right, 4);
    drawStraightDown(fm.stateEmb, layout.expertIn, 4);
    drawArrowBetween(fm.x1, BlockPos.Left, fm.actionInW, BlockPos.Right, 4);
    drawArrowBetween(fm.actionInW, BlockPos.Left, fm.actionEmb, BlockPos.Right, 4);
    drawArrowBetween(fm.actionEmb, BlockPos.Left, fm.timeInW, BlockPos.Right, 4);
    drawVertArrow(fm.timeEmb, fm.timeInW, 4);
    drawArrowBetween(fm.timeInW, BlockPos.Left, fm.timeHidden, BlockPos.Right, 4);
    drawArrowBetween(fm.timeHidden, BlockPos.Left, fm.timeOutW, BlockPos.Right, 4);
    drawArrowBetween(fm.timeOutW, BlockPos.Left, fm.actionTime, BlockPos.Right, 4);
    drawArrowBetween(fm.actionTime, BlockPos.Left, layout.expertIn, BlockPos.Right, 4);
    if (!layout.variant.time_into_tokens) {
        // pi0: the action tokens are action_in_proj(x_t) alone; the time MLP ends in cond
        drawArrowBetween(fm.actionInW, BlockPos.Left, layout.expertIn, BlockPos.Right, 4);
        drawArrowBetween(fm.timeOutW, BlockPos.Left, fm.cond, BlockPos.Right, 4);
    }

    drawEulerStrip();

    type ILn = IVlaModelLayout['ln_f'];
    type ITrunkLayer = IVlaModelLayout['blocks'][number];
    type IExpertLayer = IVlaModelLayout['expertLayers'][number];
    let lnUnit = (ln: ILn) => U(ln.lnAgg1, ln.lnAgg2, ln.lnResid);

    // RMSNorm / AdaRMSNorm group: the stream enters on top; w (and γβ(t)) feed the output
    function drawLn(prev: IBlkDef, ln: ILn) {
        drawArrowBetween(prev, BlockPos.Bot, lnUnit(ln), BlockPos.Top);
        drawVertArrow(ln.lnAgg2, ln.lnResid, 2);
        drawHorizArrow(ln.lnSigma, ln.lnResid);
        if (ln.ada) drawArrowBetween(ln.ada, BlockPos.Left, ln.lnResid, BlockPos.Right, 2);
    }

    // one transformer layer — trunk or expert, the same drawing; the expert's score row is
    // [prefix keys | suffix keys] (its prefix K/V come from the trunk: see ActionViz drawJointLinks)
    function drawLayer(L: ITrunkLayer | IExpertLayer, prev: IBlkDef) {
        dry = !!isVisible && !isVisible(L.cubes);

        drawLn(prev, L.ln1);
        let lnR = L.ln1.lnResid;
        let gated = layout.variant.ada_rows === 3 && !!L.ln1.ada;

        // K/V are shared by the query heads of a kv group: one arrow per group
        for (let grp of L.kvGroups) {
            drawArrowBetween(lnR, BlockPos.Left, grp.kBlock, BlockPos.Right);
            drawArrowBetween(lnR, BlockPos.Left, grp.vBlock, BlockPos.Right);
            drawHorizArrow(grp.kWeightBlock, grp.kBlock);
            drawHorizArrow(grp.vWeightBlock, grp.vBlock);
        }

        let oB = L.attnOut;
        for (let head of L.heads) {
            let hp = (head as any).attnMtxP as IBlkDef | undefined;
            let hsp = (head as any).attnMtxSmP as IBlkDef | undefined;
            let attU = hp ? U(hp, head.attnMtx) : head.attnMtx;
            let smU = hsp ? U(hsp, head.attnMtxSm) : head.attnMtxSm;
            drawArrowBetween(lnR, BlockPos.Left, head.qBlock, BlockPos.Right);
            // biases / q_norm·k_norm γ act after the projection (no arrow into W)
            drawHorizArrow(head.qWeightBlock, head.qBlock);

            drawArrowBotToSide(head.qBlock, attU, 0);
            drawArrowBotToSide(head.kBlock, head.attnMtx, 0);
            drawArrowBotToSide(head.vBlock, head.vOutBlock, 0);

            drawArrowBetween(attU, BlockPos.Left, head.attnMtxAgg2, BlockPos.Right);
            drawArrowBetween(head.attnMtxAgg1, BlockPos.Left, smU, BlockPos.Right);
            drawArrowBetween(smU, BlockPos.Bot, head.vOutBlock, BlockPos.Left);

            drawArrowBetween(head.vOutBlock, BlockPos.Bot, oB, BlockPos.Top);
        }

        let r1 = L.attnResidual;
        drawHorizArrow(oB, r1);
        drawHorizArrow(L.projWeight, L.attnOut);
        // ⊕ = o_proj out + the layer input: the skip is a label, not a line past the sublayers
        drawSkip(prev, r1, gated ? '+ skip: x + gate1 * attn' : '+ skip: layer input');
        if (gated) drawGate(L.ln1.ada!, r1);

        drawLn(r1, L.ln2);
        let ln2R = L.ln2.lnResid;

        drawArrowBetween(ln2R, BlockPos.Bot, L.mlpFc, BlockPos.Right);
        drawArrowOverTo(ln2R, L.mlpUp);

        // SwiGLU: gate & up branches meet in silu(gate)·up
        drawVertArrow(L.mlpFcWeight, L.mlpFc, 12);
        drawVertArrow(L.mlpUpWeight, L.mlpUp, 12);
        drawVertArrow(L.mlpFc, L.mlpAct, 12);
        drawArrowBetween(L.mlpUp, BlockPos.Bot, L.mlpAct, BlockPos.Left);
        drawHorizArrow(L.mlpProjWeight, L.mlpResult);
        drawHorizArrow(L.mlpResult, L.mlpResidual);
        drawArrowBetween(L.mlpAct, BlockPos.Right, L.mlpResult, BlockPos.Top);
        drawSkip(r1, L.mlpResidual, gated ? '+ skip: x + gate2 * mlp' : '+ skip: attn residual');
        if (gated) drawGate(L.ln2.ada!, L.mlpResidual);
        return L.mlpResidual;
    }

    let prev: IBlkDef = layout.residual0;
    for (let i = 0; i < layout.blocks.length; i++) {
        prev = drawLayer(layout.blocks[i], prev);
    }
    dry = false;
    drawLn(prev, layout.ln_f);

    let eprev: IBlkDef = layout.expertIn;
    for (let i = 0; i < layout.expertLayers.length; i++) {
        eprev = drawLayer(layout.expertLayers[i], eprev);
    }
    dry = false;
    drawLn(eprev, layout.expertNorm);

    drawTimeBus();

    // h* (the boundary column of ln_f) → knack head (below) and done head (left)
    drawStraightDown(layout.ln_f.lnResid, layout.hStar, 2, layout.hStar.x + layout.hStar.dx / 2);
    drawVertArrow(layout.hStar, layout.logits, 2);
    drawHorizArrow(layout.lmHeadWeight, layout.logits, 2);
    drawVertArrow(layout.logits, layout.logitsAgg2, 2);
    drawVertArrow(layout.logitsAgg1, layout.logitsSoftmax, 2);

    drawArrowBetween(layout.hStar, BlockPos.Left, layout.doneLogits, BlockPos.Right, 2);
    drawHorizArrow(layout.doneWeight, layout.doneLogits, 2);
    drawVertArrow(layout.doneLogits, layout.doneAgg2, 2);
    drawVertArrow(layout.doneAgg1, layout.doneProbs, 2);

    // DiT final norm (tokens 1..16) → action_out_proj → v; one-step decode x1 − v below
    let v = layout.actionVelocity;
    drawStraightDown(layout.expertOut, v, 4, v.x + v.dx / 2);
    drawArrowBetween(layout.actionWeight, BlockPos.Left, v, BlockPos.Right, 4);
    drawVertArrow(v, layout.actionChunk, 4);
    drawArrowBetween(layout.actionX1, BlockPos.Left, layout.actionChunk, BlockPos.Right, 4);

    drawEulerReturn();

    // the gate row of a gated AdaNorm (pi0) scales the sublayer output in its residual add:
    // down out of the modulation block, into the ⊕ block's left face above the sublayer arrow
    function drawGate(src: IBlkDef, dst: IBlkDef) {
        let opacity = Math.min(src.opacity, dst.opacity);
        if (opacity === 0) return;
        rec(src, dst);
        let w = 1.6;
        let normal = new Vec3(0, 0, 1);
        let col = arrowColor(src, dst, opacity);
        let start = blockPos(src, BlockPos.Bot);
        start.x = src.x + src.dx * 5 / 6;
        let yEnd = dst.y + dst.dy * 0.25;
        drawA(state, start, new Vec3(start.x, yEnd - w / 2, start.z), w, normal, col, false, CornerMode.None, new Vec3(0, 1, 0));
        drawA(state, new Vec3(start.x + w / 2, yEnd, start.z), new Vec3(dst.x - pad, yEnd, start.z), w, normal, col, true, CornerMode.Right);
    }

    // te(t) (pi0: cond) → every expert layer's modulation slots: one bus per column, just right of them
    function drawTimeBus() {
        let te = layout.fm.busSrc;
        let els = layout.expertLayers;
        type IBusEnd = { ada: IBlkDef, ada2: IBlkDef | null, attnResidual: IBlkDef };
        let all: IBusEnd[] = [...els];
        if (layout.expertNorm.ada) all.push({ ada: layout.expertNorm.ada, ada2: null, attnResidual: layout.expertNorm.lnResid });
        let cols: IBusEnd[][] = [];
        for (let e of all) {
            let last = cols[cols.length - 1];
            if (last && Math.abs(last[0].ada.x - e.ada.x) < 1) last.push(e);
            else cols.push([e]);
        }
        let w = 1.6;
        let normal = new Vec3(0, 0, 1);
        for (let c = 0; c < cols.length; c++) {
            let col = cols[c];
            let ends = col.flatMap(e => [e.ada, e.ada2]).filter((b): b is IBlkDef => !!b && b.opacity > 0 && te.opacity > 0);
            if (ends.length === 0) continue;
            let res = col[0].attnResidual;
            let busX = (ends[0].x + ends[0].dx + res.x) / 2;
            let z = ends[0].z + ends[0].dz / 2;
            let op = Math.min(te.opacity, Math.max(...ends.map(b => b.opacity)));
            let lit = layout.flow && ends.some(b => layout.flow!.active.has(b));
            let col4 = (lit ? flowColor : timeColor).mul(op);
            let yEnd = ends[ends.length - 1].y + ends[ends.length - 1].dy / 2;
            let yStart: number;
            if (c === 0) {
                // from te: up out of te, across above the input cluster, down the bus
                let yTop = layout.fm.topY - layout.margin;
                let tx = te.x + te.dx / 2;
                yStart = yTop;
                drawA(state, new Vec3(tx, te.y - pad, z), new Vec3(tx, yTop + w / 2, z), w, normal, col4, false);
                drawA(state, new Vec3(tx + w / 2, yTop, z), new Vec3(busX - w / 2, yTop, z), w, normal, col4, false, CornerMode.Right);
            } else {
                yStart = ends[0].y - 4 * layout.margin;
                drawWrapLabel({ ...ends[0], x: busX - ends[0].dx / 2, y: yStart + 4, dy: 0 }, 'te(t) bus (from te)', -1);
            }
            drawA(state, new Vec3(busX, yStart - w / 2, z), new Vec3(busX, yEnd, z), w, normal, col4, false);
            for (let b of ends) {
                rec(te, b);
                let ym = b.y + b.dy / 2;
                let c2 = layout.flow?.active.has(b) ? flowColor.mul(op) : col4;
                drawA(state, new Vec3(busX, ym, z), new Vec3(b.x + b.dx + pad * 0.5, ym, z), w, normal, c2, true);
            }
        }
    }

    // the Euler strip: x(t=1) … x(t=0); board j → board j+1 (− v_j / K); the current board
    // feeds the x_t input of the DiT
    function drawEulerStrip() {
        let S = layout.strip;
        let eu = layout.euler;
        for (let j = 0; j < S.length; j++) {
            if (S[j].opacity <= 0) continue;
            if (j + 1 < S.length && S[j + 1].opacity > 0) {
                drawArrowBetween(S[j], BlockPos.Right, S[j + 1], BlockPos.Left, 3);
            }
            if (j < S.length - 1) {
                if (eu && eu.k === j) drawArrowBetween(S[j], BlockPos.Bot, fm.x1, BlockPos.Top, 3);
                else rec(S[j], fm.x1);
            }
        }
    }

    // v_k → board k+1: the one return line (down, out to the right of the scene, up, back in)
    function drawEulerReturn() {
        let eu = layout.euler;
        let S = layout.strip;
        if (!eu) return;
        for (let j = 1; j < S.length; j++) if (S[j].opacity > 0 && v.opacity > 0) rec(v, S[j]);
        let dst = S[eu.k + 1];
        if (!dst || eu.ret <= 0 || dst.opacity <= 0 || v.opacity <= 0) return;
        let w = 3;
        let normal = new Vec3(0, 0, 1);
        let col = flowColor.mul(eu.ret);
        let z = v.z + v.dz / 2;
        let xFar = Math.max(...layout.cubes.filter(c => c.opacity > 0).map(c => c.x + c.dx)) + 3 * layout.margin;
        let yLow = v.y + v.dy + layout.margin / 2;
        let yTop = dst.y - layout.margin;
        let sx = v.x + v.dx / 2, dx = dst.x + dst.dx / 2;
        drawA(state, new Vec3(sx, v.y + v.dy + pad, z), new Vec3(sx, yLow - w / 2, z), w, normal, col, false);
        drawA(state, new Vec3(sx - w / 2, yLow, z), new Vec3(xFar + w / 2, yLow, z), w, normal, col, false, CornerMode.Left);
        drawA(state, new Vec3(xFar, yLow + w / 2, z), new Vec3(xFar, yTop - w / 2, z), w, normal, col, false, CornerMode.Left);
        drawA(state, new Vec3(xFar + w / 2, yTop, z), new Vec3(dx - w / 2, yTop, z), w, normal, col, false, CornerMode.Left);
        drawA(state, new Vec3(dx, yTop - w / 2, z), new Vec3(dx, dst.y - pad, z), w, normal, col, true, CornerMode.Left);
        drawWrapLabel({ ...dst, y: yTop - 2, dy: 0 }, `v(t=${(1 - eu.k / (S.length - 1)).toFixed(1)}): x - v/${S.length - 1}`, -1);
    }

    // straight down from src's bottom into dest's top at src's x (dest is wider)
    function drawStraightDown(src: IBlkDef, dest: IBlkDef, width: number = 6, x?: number) {
        let opacity = Math.min(src.opacity, dest.opacity);
        if (opacity === 0) return;
        rec(src, dest);
        let start = blockPos(src, BlockPos.Bot);
        if (x !== undefined) start.x = x;
        let end = new Vec3(start.x, dest.y - pad, dest.z + dest.dz / 2);
        drawA(state, start, end, width, new Vec3(0, 0, 1), arrowColor(src, dest, opacity), true);
    }

    function blkColor(src: IBlkDef) {
        return src.t === 'w' ? weightColor : dataColor;
    }

    // during a data-pass animation the arrow into the block being computed is lit and arrows
    // into blocks the pass has not reached yet are faded
    function arrowColor(src: IBlkDef, dest: IBlkDef, opacity: number) {
        let sm = members.get(src) ?? [src], dm = (members.get(dest) ?? [dest]).filter(b => b.opacity > 0);
        let base = sm.every(b => ditSet.has(b)) && src.t !== 'w' ? expertColor : blkColor(src);
        let f = layout.flow;
        if (f && dm.length) {
            if (dm.some(b => f!.active.has(b))) return flowColor.mul(opacity);
            if (dm.every(b => f!.pending.has(b))) return base.mul(opacity * 0.18);
        }
        return base.mul(opacity);
    }

    // a residual add: the skip operand is named on the ⊕ block instead of a line running past
    // the sublayers (the edge is still recorded so the audit sees the dependency)
    function drawSkip(src: IBlkDef, dest: IBlkDef, text: string) {
        let opacity = Math.min(src.opacity, dest.opacity);
        if (opacity === 0) return;
        rec(src, dest);
        if (dry) return;
        let size = 4;
        let f = layout.flow;
        let a = f && f.pending.has(dest) ? 0.25 : 1;
        let mtx = Mat4f.fromTranslation(new Vec3(dest.x + dest.dx / 2, dest.y - size * 1.4, dest.z + dest.dz / 2));
        let col = f && f.active.has(dest) ? flowColor : new Vec4(0.2, 0.45, 0.2, 1);
        let opts = { color: col.mul(opacity * a), size, mtx };
        let w = measureText(state.modelFontBuf, text, opts);
        let prev = state.sharedRender.activePhase;
        state.sharedRender.activePhase = RenderPhase.Overlay;
        drawText(state.modelFontBuf, text, -w / 2, 0, opts);
        state.sharedRender.activePhase = prev;
    }

    function drawVertArrow(src: IBlkDef, dest: IBlkDef, width: number = 6) {
        drawArrowBetween(src, BlockPos.Bot, dest, BlockPos.Top, width);
    }

    function drawHorizArrow(src: IBlkDef, dest: IBlkDef, width: number = 6) {
        drawArrowBetween(src, BlockPos.Right, dest, BlockPos.Left, width);
    }

    // over-the-top route: down from src, across above dest (in the gap above it), down into dest
    function drawArrowOverTo(src: IBlkDef, dest: IBlkDef, width: number = 6) {
        let opacity = Math.min(src.opacity, dest.opacity);
        if (opacity === 0) {
            return;
        }
        rec(src, dest);
        let normal = new Vec3(0, 0, 1);
        let color = arrowColor(src, dest, opacity);
        let start = blockPos(src, BlockPos.Bot);
        let end = blockPos(dest, BlockPos.Top);
        let yMid = dest.y - layout.margin / 2;
        let z = dest.z + dest.dz / 2;
        drawA(state, start, new Vec3(start.x, yMid - width / 2, z), width, normal, color, false);
        drawA(state, new Vec3(start.x - width / 2, yMid, z), new Vec3(end.x + width / 2, yMid, z), width, normal, color, false, CornerMode.Left);
        drawA(state, new Vec3(end.x, yMid + width / 2, z), end, width, normal, color, true, CornerMode.Left);
    }

    function drawWrapLabel(blk: IBlkDef, text: string, dir: number) {
        if (dry) return;
        let size = 6;
        let x = blk.x + blk.dx / 2;
        let y = dir > 0 ? blk.y + blk.dy + 3 * size : blk.y - size;
        let mtx = Mat4f.fromTranslation(new Vec3(x, y, blk.z + blk.dz / 2));
        let opts = { color: new Vec4(0.3, 0.3, 0.35, 1), size, mtx };
        let w = measureText(state.modelFontBuf, text, opts);
        let prev = state.sharedRender.activePhase;
        state.sharedRender.activePhase = RenderPhase.Overlay;
        drawText(state.modelFontBuf, text, -w / 2, 0, opts);
        state.sharedRender.activePhase = prev;
    }

    function blockPos(block: IBlkDef, pos: BlockPos) {
        let z = block.z + block.dz / 2;
        switch (pos) {
            case BlockPos.Left: return new Vec3(block.x - pad, block.y + block.dy / 2, z);
            case BlockPos.Right: return new Vec3(block.x + block.dx + pad, block.y + block.dy / 2, z);
            case BlockPos.Top: return new Vec3(block.x + block.dx / 2, block.y - pad, z);
            case BlockPos.Bot: return new Vec3(block.x + block.dx / 2, block.y + block.dy + pad, z);
        }
    }

    function drawArrowBotToSide(src: IBlkDef, dest: IBlkDef, offset: number, width: number = 6, forceOffset: boolean = false) {
        let start = blockPos(src, BlockPos.Bot);
        let left = start.z > dest.z + dest.dz / 2;
        let end = new Vec3(dest.x + dest.dx / 2, dest.y + layout.cell * (offset + 0.5), left ? dest.z + dest.dz / 2 + pad : dest.z - pad);

        let opacity = Math.min(src.opacity, dest.opacity);
        if (opacity === 0) {
            return;
        }
        rec(src, dest);


        let normal = new Vec3(0, 0, 1);
        let color = arrowColor(src, dest, opacity);
        let endDir = new Vec3(0, 0, left ? -1 : 1);

        let areClose = Math.abs(start.z - (dest.z + dest.dz / 2)) < 1.0;
        if (areClose && !forceOffset) {
            endDir = undefined!;
            end = blockPos(dest, BlockPos.Top);
        }

        drawA(state, start, end, width, normal, color, true, CornerMode.None, endDir);
    }
    function drawArrowBetween(src: IBlkDef, srcPos: BlockPos, dest: IBlkDef, destPos: BlockPos, width: number = 6) {
        let start = blockPos(src, srcPos);
        let end = blockPos(dest, destPos);

        let opacity = Math.min(src.opacity, dest.opacity);
        if (opacity === 0) {
            return;
        }
        rec(src, dest);

        let normal = new Vec3(0, 0, 1);
        let color = arrowColor(src, dest, opacity);

        if (srcPos === BlockPos.Left && destPos === BlockPos.Right) {
            // horizontal: at dest's mid-height if that lies on src, else at src's mid-height
            if (end.y >= src.y && end.y <= src.y + src.dy) start.y = end.y;
            else end.y = start.y;
        }

        if (srcPos === BlockPos.Right && destPos === BlockPos.Top) {
            // dogleg right => down
            let mid0 = new Vec3(end.x - width / 2, start.y, start.z);
            let mid1 = new Vec3(end.x, start.y + width / 2, end.z);

            drawA(state, start, mid0, width, normal, color, false);
            drawA(state, mid1, end, width, normal, color, true, CornerMode.Left);

        } else if (srcPos === BlockPos.Bot && destPos === BlockPos.Right) {
            // dogleg down => right
            let mid0 = new Vec3(start.x, end.y - width / 2, end.z);
            let mid1 = new Vec3(start.x - width / 2, end.y, end.z);

            drawA(state, start, mid0, width, normal, color, false);
            drawA(state, mid1, end, width, normal, color, true, CornerMode.Left);

        } else if (srcPos === BlockPos.Bot && destPos === BlockPos.Left) {
            // dogleg down => left
            let mid0 = new Vec3(start.x, end.y - width / 2, end.z);
            let mid1 = new Vec3(start.x + width / 2, end.y, end.z);

            drawA(state, start, mid0, width, normal, color, false, CornerMode.None, new Vec3(0, 1, 0));
            drawA(state, mid1, end, width, normal, color, true, CornerMode.Right);

        } else {

            drawA(state, start, end, width, normal, color, true);
        }
    }
}

enum BlockPos {
    Left,
    Right,
    Top,
    Bot,
}

export enum CornerMode {
    None,
    Left,
    Right,
}

let _bezierLineBuf = new Float32Array(4 * 3);
export function drawArrow(state: IRenderState, start: Vec3, end: Vec3, width: number, normal: Vec3, color: Vec4, drawHead: boolean = true, drawCorner: CornerMode = CornerMode.None, endDir?: Vec3) {

    let dir = end.sub(start);
    dir.z = 0.0;
    dir = dir.normalize();

    let len = end.sub(start).len();
    let headExtra = 3.0;
    let headDepth = drawHead ? Math.min(len * 0.7, headExtra * 1.0) : 0;
    let mtx = new Mat4f();

    let side = Vec3.cross(dir, normal).mul(-1).normalize();
    normal = Vec3.cross(side, dir).normalize();

    mtx[0] = side.x;
    mtx[1] = side.y;
    mtx[2] = side.z;
    mtx[4] = dir.x;
    mtx[5] = dir.y;
    mtx[6] = dir.z;
    mtx[8] = normal.x;
    mtx[9] = normal.y;
    mtx[10] = normal.z;

    start = mtx.mulVec3Proj(start);
    end = mtx.mulVec3Proj(end);

    let borderColor = color.mul(0.8);
    let ribbonColor = color.mul(0.3);

    let opts: IArrowOpts = {
        width,
        borderColor,
        ribbonColor,
        headDepth,
        headExtra,
        lineThick: 1.2,
        mtx,
    };

    endDir = endDir ? mtx.mulVec3ProjVec(endDir) : undefined;

    // matrix is constructed such that the ribbons are always in the x-y plane, and going from -y to +y
    // The ribbon will be curved if it is not in the x-y plane, but the start and end remain tangent to the plane

    function drawBezierRibbon() {
        let dist = Math.max(headDepth, Math.abs(start.y - end.y - headDepth) / 2);
        let p0 = new Vec3(start.x, start.y               , start.z);
        let p1 = new Vec3(start.x, start.y + dist        , start.z);
        let p2 = new Vec3(end.x, end.y - headDepth - dist, end.z);
        let p3 = new Vec3(end.x, end.y - headDepth       , end.z);
        if (endDir) {
            p2 = end.mulAdd(endDir, -headDepth - dist);
            p3 = end.mulAdd(endDir, -headDepth);
        }
        // the flatness test is absolute (∝ L²), so long arrows (28 layers of K/V links) would get
        // thousands of segments; scale it to keep the per-arrow segment count roughly size-independent
        let chord = p3.sub(p0).len();
        let steps = bezierCurveBuild(p0, p1, p2, p3, 0.1 * Math.max(1, (chord / 10) ** 2));
        let nPts = steps.length / 3;

        let headNPts = drawHead ? 3 : 0;
        let nFloats = (nPts * 2 + headNPts) * 3;
        if (_bezierLineBuf.length < nFloats) {
            _bezierLineBuf = new Float32Array(nFloats);
        }
        let arrowLine = _bezierLineBuf.subarray(0, nFloats);

        for (let i = 0; i < nPts - 1; i++) {
            let p0 = new Vec3(steps[i*3+0], steps[i*3+1], steps[i*3+2]);
            let p1 = new Vec3(steps[i*3+3], steps[i*3+4], steps[i*3+5]);
            drawArrowSeg(state, p0, p1, opts);
        }

        let arrowStartIdx = nPts;
        let arrowEndIdx = arrowStartIdx + headNPts;
        for (let i = 0; i < nPts; i++) {
            let j = arrowEndIdx + i;
            arrowLine[j*3+0] = steps[i*3+0] + opts.width / 2;
            arrowLine[j*3+1] = steps[i*3+1];
            arrowLine[j*3+2] = steps[i*3+2];
            let k = nPts - i - 1;
            arrowLine[k*3+0] = steps[i*3+0] - opts.width / 2;
            arrowLine[k*3+1] = steps[i*3+1];
            arrowLine[k*3+2] = steps[i*3+2];
        }

        if (drawHead) {
            let headExtra = 3.0;
            endDir = endDir ?? new Vec3(0, 1, 0);
            let endA = end.mulAdd(endDir, -headDepth);
            let i = arrowStartIdx;
            arrowLine[i*3+0] = endA.x - opts.width / 2 - headExtra;
            arrowLine[i*3+1] = endA.y;
            arrowLine[i*3+2] = endA.z;
            i += 1;
            arrowLine[i*3+0] = end.x;
            arrowLine[i*3+1] = end.y;
            arrowLine[i*3+2] = end.z;
            i += 1;
            arrowLine[i*3+0] = endA.x + opts.width / 2 + headExtra;
            arrowLine[i*3+1] = endA.y;
            arrowLine[i*3+2] = endA.z;
        }

        let lineOpts = makeLineOpts({ thick: opts.lineThick, mtx: opts.mtx, color: opts.borderColor });
        drawLineSegs(state.lineRender, arrowLine, lineOpts);
    }

    if (Math.abs(start.z - end.z) > 0.01 || endDir) {
        drawBezierRibbon();

    } else {
        drawArrowSeg(state, start, end.sub(new Vec3(0, headDepth)), opts);
    }

    if (drawCorner !== CornerMode.None) {
        drawArrowCorner(state, start.sub(new Vec3(0, width/2)), drawCorner, opts);
    }
    if (drawHead) {
        endDir = endDir ?? new Vec3(0, 1, 0);
        drawArrowHead(state, end.mulAdd(endDir, -headDepth), end, opts);
    }
}

interface IArrowOpts {
    width: number;
    headExtra: number;
    headDepth: number;
    borderColor: Vec4;
    ribbonColor: Vec4;
    lineThick: number;
    mtx: Mat4f;
}

// Inner loop of bezier curve, so make this fast
let _segTl = new Vec3();
let _segBr = new Vec3();
function drawArrowSeg(state: IRenderState, start: Vec3, end: Vec3, opts: IArrowOpts) {
    _segTl.x = start.x - opts.width/2;
    _segTl.y = start.y;
    _segTl.z = start.z;

    _segBr.x = end.x + opts.width/2;
    _segBr.y = end.y;
    _segBr.z = end.z;

    addQuad(state.triRender, _segTl, _segBr, opts.ribbonColor, opts.mtx);



}

let _headTl = new Vec3();
let _headTr = new Vec3();
let _headBr = new Vec3();
let _headLeft = new Vec3();
let _headRight = new Vec3();
let _headTip = new Vec3();
let _headN = new Vec3(0, 0, 1);
function drawArrowHead(state: IRenderState, a: Vec3, b: Vec3, opts: IArrowOpts) {
    let headExtra = 3.0;

    _headTl.copy_(a);
    _headTl.x -= opts.width/2;
    _headTr.copy_(a);
    _headTr.x += opts.width/2;
    _headBr.copy_(b);
    _headBr.x += opts.width/2;
    _headLeft.copy_(a);
    _headLeft.x = _headTl.x - headExtra;
    _headRight.copy_(a);
    _headRight.x = _headBr.x + headExtra;
    _headTip.copy_(b);
    _headTip.x = _headTl.x + opts.width / 2;

    addVert(state.triRender, _headLeft, opts.ribbonColor, _headN, opts.mtx);
    addVert(state.triRender, _headTip, opts.ribbonColor, _headN, opts.mtx);
    addVert(state.triRender, _headRight, opts.ribbonColor, _headN, opts.mtx);
    addPrimitiveRestart(state.triRender);

}

let _cornerPivot = new Vec3();
let _cornerN = new Vec3(0, 0, 1);
let _cornerCurrP = new Vec3();
let _cornerPrevP = new Vec3();
function drawArrowCorner(state: IRenderState, center: Vec3, mode: CornerMode, opts: IArrowOpts) {

    // coming from the left (-x), and going downwards (+y)
    let mul = mode === CornerMode.Left ? 1 : -1;

    _cornerPivot.x = center.x + opts.width / 2 * mul;
    _cornerPivot.y = center.y + opts.width / 2;
    _cornerPivot.z = center.z;

    _cornerCurrP.z = center.z;
    _cornerPrevP.z = center.z;
    let count = 8;

    for (let i = 0; i < count; i++) {
        let theta = i / (count - 1) * Math.PI / 2;
        let c = opts.width * Math.cos(theta) * mul;
        let s = opts.width * Math.sin(theta);
        _cornerCurrP.x = _cornerPivot.x - c;
        _cornerCurrP.y = _cornerPivot.y - s;

        addVert(state.triRender, _cornerCurrP, opts.ribbonColor, _cornerN, opts.mtx);
        addVert(state.triRender, _cornerPivot, opts.ribbonColor, _cornerN, opts.mtx);

        let tmp = _cornerPrevP;
        _cornerPrevP = _cornerCurrP;
        _cornerCurrP = tmp;
    }

    addPrimitiveRestart(state.triRender);
}
