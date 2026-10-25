import { IModelFacts } from "../ModelFacts";
import { TextAlignHoriz, TextAlignVert } from "../Annotations";
import { IBlkDef, IBlkLabel, IVlaModelLayout } from "../VlaModelLayout";
import { measureTextWidth, writeTextToBuffer } from "../render/fontRender";
import { addLine } from "../render/lineRender";
import { IRenderState } from "../render/modelRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";

export type IFitsView = (mn: Vec3, mx: Vec3) => boolean;

export function drawBlockLabels(state: IRenderState, layout: IVlaModelLayout, facts: IModelFacts, focus?: Vec3, fits?: IFitsView) {
    // same-text labels: only the one nearest the camera focus is drawn; a label (block + its text)
    // that does not fit entirely on screen is dropped: up close it would only be a giant fragment
    let pend = new Map<string, { d: number, draw: () => void }>();
    let labelFits = (text: string, e: IExtent, fs: number, pad: number, z: number) => {
        if (!fits) return true;
        let tw = measureTextWidth(state.modelFontBuf, text, fs);
        return fits(new Vec3(e.minX - tw - 2 * pad, e.minY, z), new Vec3(e.maxX, e.maxY, z));
    };
    let put = (text: string, e: IExtent | null, draw: () => void, fs = 10, pad = 10, z?: number) => {
        if (!e) return;
        if (!labelFits(text, e, fs, pad, z ?? e.midZ)) return;
        let d = focus ? new Vec3((e.minX + e.maxX) / 2, (e.minY + e.maxY) / 2, e.midZ).sub(focus).len() : 0;
        let cur = pend.get(text);
        if (!cur || d < cur.d) pend.set(text, { d, draw });
    };

    // probably should limit this to specific blocks (defined by walkthrough)
    // can configure this on the blocks themselves, or the layout object
    // either way, this runs after the walkthrough, for positioning, and the walkthrough
    // needs to configure this is some way

    let baseColor = new Vec4(0.4, 0.4, 0.4, 1.0);

    {
        let color = baseColor.mul(layout.embedLabel.visible);
        let tl = new Vec3(layout.tokEmbedObj.x - layout.margin * 2, layout.tokEmbedObj.y, 0);
        let br = new Vec3(layout.tokEmbedObj.x - layout.margin * 2, layout.tokEmbedObj.y + layout.tokEmbedObj.dy, 0);
        let tw = measureTextWidth(state.modelFontBuf, "Input Embedding", 6);
        if (!fits || fits(new Vec3(tl.x - tw - 8, tl.y, 0), new Vec3(tl.x, br.y, 0)))
            drawSectionLabel(state, "Input Embedding", tl, br, { color, fontSize: 6, pad: 4 });
    }

    // every label is pinned to the blocks it names: a bracket along their left edge (or a header
    // line along their top edge) at their own position, never at a fixed offset in empty space
    // the two towers are one column of layer pairs: a title over each tower, and one header
    // per pair spanning trunk layer i and expert layer i (they share layer i's joint attention)
    let nPairs = Math.min(layout.blocks.length, layout.expertLayers.length);
    let towerTitles: [IExtent | null, string[]][] = [
        [extent(layout.blocks[0]?.transformerLabel.cubes ?? []), [facts.trunkLabel, `x${layout.blocks.length} layers`]],
        [extent(layout.expertLayers[0]?.transformerLabel.cubes ?? []), [facts.expertLabel, `x${layout.expertLayers.length} layers`]],
    ];
    let tVis = layout.towersLabel.visible;
    // titles sit in the empty space beside the column: trunk on the left, expert on the right
    towerTitles.forEach(([e, lines], side) => {
        if (!e || tVis <= 0) return;
        let fs = 40, y0 = e.minY + 40;
        let tw = Math.max(...lines.map(l => measureTextWidth(state.modelFontBuf, l, fs)));
        let x0 = side === 0 ? e.minX - 50 - tw : e.maxX + 50;
        if (fits && !fits(new Vec3(x0, y0, e.midZ), new Vec3(x0 + tw, y0 + 2 * (fs + 8), e.midZ))) return;
        let mtx = new Mat4f(); mtx[14] = e.midZ;
        lines.forEach((l, k) => {
            let w = measureTextWidth(state.modelFontBuf, l, fs);
            let x = side === 0 ? x0 + tw - w : x0;
            writeTextToBuffer(state.modelFontBuf, l, new Vec4(0.75, 0.75, 0.75, 1).mul(e.op * tVis), x, y0 + k * (fs + 8), fs, mtx);
        });
    });
    for (let i = 0; i < nPairs; i++) {
        let T = layout.blocks[i], E = layout.expertLayers[i];
        let eT = extent(T.transformerLabel.cubes), eE = extent(E.transformerLabel.cubes);
        let vis = Math.max(eT ? eT.op * T.transformerLabel.visible : 0, eE ? eE.op * E.transformerLabel.visible : 0);
        let e = eT && eE ? union(eT, eE) : (eT ?? eE);
        if (!e || vis <= 0) continue;
        let fs = 26;
        if (fits && !fits(new Vec3(e.minX, e.minY - 12 - fs, e.midZ), new Vec3(e.maxX, e.maxY, e.midZ))) continue;
        drawPairHeader(state, `Layer pair ${i}:  trunk L${i}  <->  expert E${i}`, e, [eT, eE], baseColor.mul(vis), fs);
    }
    let mlpLbl = `MLP (${facts.mlpName})`;
    let towers: ILayerLike[][] = [layout.blocks, layout.expertLayers];
    for (let layers of towers) {
        layers.forEach((L) => {
            let attnCubes = L.heads.flatMap(h => h.cubes);
            let eAttn = extent(attnCubes);
            if (eAttn) {
                let color = baseColor.mul(eAttn.op * L.selfAttendLabel.visible);
                if (color.w > 0) put('Joint Attention', eAttn, () => pinLeft(state, 'Joint Attention', eAttn!, color, 12, eAttn!.maxZ, 26), 12, 26, eAttn.maxZ);
            }
            let eProj = extent(L.projLabel.cubes.filter(c => c !== L.attnResidual));
            if (eProj && L.projLabel.visible > 0) put('Out Proj', eProj, () => pinLeft(state, 'Out Proj', eProj!, baseColor.mul(eProj!.op * L.projLabel.visible), 10, eProj!.midZ), 10, 10);
            let eMlp = extent(L.mlpLabel.cubes);
            if (eMlp && L.mlpLabel.visible > 0) put(mlpLbl, eMlp, () => pinLeft(state, mlpLbl, eMlp!, baseColor.mul(eMlp!.op * L.mlpLabel.visible), 12, eMlp!.midZ), 12, 10);

            L.heads.forEach((head, hi) => {
                let z = head.qBlock.z + head.qBlock.dz / 2;
                let eHead = extent(head.cubes.filter(c => Math.abs(c.z + c.dz / 2 - z) < 1e-3));
                let nH = L.heads.length;
                let hText = `Head ${hi}  (x${nH} heads, same shape)`;
                let hc = baseColor.mul(head.attnMtx.opacity * head.headLabel.visible);
                if (eHead && hc.w > 0) put(hText, eHead, () => pinLeft(state, hText, eHead!, hc, 10, z, 18), 10, 18, z);
                for (let [lbl, blk, txt] of [[head.qLabel, head.qBlock, 'Q'], [head.kLabel, head.kBlock, 'K'], [head.vLabel, head.vBlock, 'V']] as [IBlkLabel, IBlkDef, string][]) {
                    let e = extent(lbl.cubes.filter(c => Math.abs(c.z + c.dz / 2 - (blk.z + blk.dz / 2)) < 1e-3));
                    let lc = baseColor.mul(blk.opacity * lbl.visible);
                    if (e && lc.w > 0) put(txt, e, () => pinLeft(state, txt, e!, lc, 6, blk.z + blk.dz / 2, 4), 6, 4, blk.z + blk.dz / 2);
                }
            });
        });
    }
    for (let { draw } of pend.values()) draw();
}

interface ILayerLike {
    transformerLabel: IBlkLabel; selfAttendLabel: IBlkLabel; projLabel: IBlkLabel; mlpLabel: IBlkLabel;
    attnResidual: IBlkDef;
    heads: { cubes: IBlkDef[], qBlock: IBlkDef, kBlock: IBlkDef, vBlock: IBlkDef, attnMtx: IBlkDef,
        headLabel: IBlkLabel, qLabel: IBlkLabel, kLabel: IBlkLabel, vLabel: IBlkLabel }[];
}

interface IExtent { minX: number, maxX: number, minY: number, maxY: number, minZ: number, maxZ: number, midZ: number, op: number }

function extent(cubes: IBlkDef[]): IExtent | null {
    let e: IExtent = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity, midZ: 0, op: 0 };
    for (let c of cubes) {
        if (c.opacity <= 0) continue;
        e.minX = Math.min(e.minX, c.x); e.maxX = Math.max(e.maxX, c.x + c.dx);
        e.minY = Math.min(e.minY, c.y); e.maxY = Math.max(e.maxY, c.y + c.dy);
        e.minZ = Math.min(e.minZ, c.z); e.maxZ = Math.max(e.maxZ, c.z + c.dz);
        e.op = Math.max(e.op, c.opacity);
    }
    if (e.op <= 0) return null;
    e.midZ = (e.minZ + e.maxZ) / 2;
    return e;
}

// a bracket along the left edge of the blocks, text left of it
function pinLeft(state: IRenderState, text: string, e: IExtent, color: Vec4, fontSize: number, z: number, pad = 10) {
    if (color.w <= 0) return;
    drawSectionLabel(state, text, new Vec3(e.minX, e.minY, z), new Vec3(e.minX, e.maxY, z), { color, fontSize, pad });
}

function union(a: IExtent, b: IExtent): IExtent {
    let e = { minX: Math.min(a.minX, b.minX), maxX: Math.max(a.maxX, b.maxX), minY: Math.min(a.minY, b.minY),
        maxY: Math.max(a.maxY, b.maxY), minZ: Math.min(a.minZ, b.minZ), maxZ: Math.max(a.maxZ, b.maxZ), midZ: 0, op: Math.max(a.op, b.op) };
    e.midZ = (e.minZ + e.maxZ) / 2;
    return e;
}

// a header above a layer pair: text, then a bar along the pair's top edge with a tick down
// to each member (trunk layer, expert layer), so the bar visibly joins the two
function drawPairHeader(state: IRenderState, text: string, e: IExtent, parts: (IExtent | null)[], color: Vec4, fontSize: number) {
    let mtx = new Mat4f();
    mtx[14] = e.midZ;
    let y = e.minY - 10;
    writeTextToBuffer(state.modelFontBuf, text, color, e.minX, y - fontSize - 2, fontSize, mtx);
    let lc = color.mul(0.6);
    addLine(state.lineRender, 2.0, lc, new Vec3(e.minX, y, e.midZ), new Vec3(e.maxX, y, e.midZ), undefined);
    for (let p of parts) {
        if (!p) continue;
        for (let x of [p.minX, p.maxX]) addLine(state.lineRender, 2.0, lc, new Vec3(x, y, e.midZ), new Vec3(x, p.minY, e.midZ), undefined);
    }
}

interface ILabelOpts {
    color: Vec4;
    fontSize: number;
    textAlign?: TextAlignHoriz;
    textAlignV?: TextAlignVert;
    pad?: number;
}

function drawSectionLabel(state: IRenderState, text: string, tl: Vec3, br: Vec3, opts: ILabelOpts) {
    let mtx = new Mat4f();
    mtx[14] = (tl.z + br.z) / 2;

    let color = opts.color;
    let fontScale = opts.fontSize;
    let pad = opts.pad ?? 10;

    let textColor = color;
    let lineColor = color.mul(0.4);

    let tw = measureTextWidth(state.modelFontBuf, text, fontScale);

    writeTextToBuffer(state.modelFontBuf, text, textColor, tl.x - tw - 2 * pad, (tl.y + br.y) / 2 - fontScale / 2, fontScale, mtx);

    let p0 = new Vec3(tl.x, tl.y, (tl.z + br.z) / 2);
    let p1 = new Vec3(br.x, br.y, (tl.z + br.z) / 2);

    if (tl.z != br.z) {
        p0 = new Vec3(tl.x, (tl.y + br.y) / 2, tl.z);
        p1 = new Vec3(tl.x, (tl.y + br.y) / 2, br.z);
    }

    let inward = new Vec3(1, 0, 0);

    addLine(state.lineRender, 1.0, lineColor, p0.mulAdd(inward, -pad), p1.mulAdd(inward, -pad), undefined);

    addLine(state.lineRender, 1.0, lineColor, p0.mulAdd(inward, -pad), p0, undefined);
    addLine(state.lineRender, 1.0, lineColor, p1.mulAdd(inward, -pad), p1, undefined);

}
