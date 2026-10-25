import { cellPosition, IBlkDef, IModelLayout } from "./VlaModelLayout";
import { addLine } from "./render/lineRender";
import { measureTextWidth, writeTextToBuffer } from "./render/fontRender";
import { lerpSmoothstep } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { DimStyle, dimStyleColor, dimStyleText } from "./walkthrough/WalkthroughTools";
import { IProgramState } from "./Program";
import { camScaleToScreen } from "./Camera";
import { isNil } from "@/src/utils/data";

export function blockDimension(state: IProgramState, layout: IModelLayout, blk: IBlkDef, dim: Dim, style: DimStyle, t: number) {
    let render = state.render;

    // Render |----- T ------| along the appropriate dimension

    let { vecId } = dimConsts(dim);
    let { cx } = dimProps(blk, dim);

    let offVecId = vecId === 0 ? 1 : 0;

    let text = dimStyleText(style);
    if (style === DimStyle.None) {
        return;
    }

    let start = cellPosition(layout, blk, dim, 0);
    let end = cellPosition(layout, blk, dim, cx - 1) + layout.cell;
    let mid = (end + start) / 2;
    let midPos = dim === Dim.X ? new Vec3(mid, blk.y + blk.dz, blk.z + blk.dz) : new Vec3(blk.x, mid, blk.z + blk.dz);
    let mul = dim === Dim.X ? -1.0 : 1.0;

    let scale = camScaleToScreen(state, midPos);
    scale = Math.min(scale, 1);

    let fontSize = 2.5 * scale;
    let tw = measureTextWidth(render.modelFontBuf, text, fontSize);
    let th = fontSize;
    let textPad = dim === Dim.X ? tw / 2 + fontSize * 0.4 : dim === Dim.Y ? th / 2 + fontSize * 0.4 : 0;

    let botPad = fontSize * 0.3;
    let edgeH2 = fontSize / 2 * 0.5;

    let color = dimStyleColor(style).mul(t);

    let zOffset = 0.1;

    let textPos = new Vec3(blk.x, blk.y + blk.dy, blk.z + blk.dz + zOffset)
        .setAt(vecId, mid)
        .withAddAt(offVecId, -mul * (fontSize / 2 + botPad));

    let textYOff = 0;
    let tooSmall = tw > end - start - textPad * 2;
    if (tooSmall) {
        textYOff = mul * fontSize;
    }

    let mtx = Mat4f.fromTranslation(textPos);

    let textDx = dim === Dim.X ? -tw / 2 : dim === Dim.Y ? -tw / 2 : 0;
    let textDy = dim === Dim.X ? -textYOff - fontSize / 2 : dim === Dim.Y ? -fontSize / 2 : 0;

    writeTextToBuffer(render.modelFontBuf, text, color, textDx, textDy, fontSize, mtx);

    let lX = blk.x;
    let lY = blk.y + blk.dy;
    let lZ = blk.z + blk.dz + zOffset;
    let thickness = fontSize * 0.02;
    let n = new Vec3(0, 0, 1);

    if (tooSmall) {
        textPad = 0;

    }

    let base = new Vec3(lX, lY, lZ).withAddAt(offVecId, -mul * (fontSize / 2 + botPad));
    let vStart = base.withSetAt(vecId, start);
    let vEnd = base.withSetAt(vecId, end);
    let vMid1 = base.withSetAt(vecId, mid - textPad);
    let vMid2 = base.withSetAt(vecId, mid + textPad);

    addLine(render.lineRender, thickness, color, vStart, vMid1, n);
    addLine(render.lineRender, thickness, color, vEnd  , vMid2, n);
    addLine(render.lineRender, thickness, color, vStart.withAddAt(offVecId, edgeH2), vStart.withAddAt(offVecId, -edgeH2), n);
    addLine(render.lineRender, thickness, color, vEnd.withAddAt(offVecId, edgeH2), vEnd.withAddAt(offVecId, -edgeH2), n);
}

interface IDimConst {
    vecId: number;
    xName: string;
    dxName: string;
    cxName: string;
    offXName: string;
    sizeXName: string;
}

let dimConstX: IDimConst = { vecId: 0, xName: 'x', dxName: 'dx', cxName: 'cx', offXName: 'offX', sizeXName: 'sizeX' };
let dimConstY: IDimConst = { vecId: 1, xName: 'y', dxName: 'dy', cxName: 'cy', offXName: 'offY', sizeXName: 'sizeY' };
let dimConstZ: IDimConst = { vecId: 2, xName: 'z', dxName: 'dz', cxName: 'cz', offXName: 'offZ', sizeXName: 'sizeZ' };

function dimConsts(dim: Dim) {
    return dim === Dim.X ? dimConstX : dim === Dim.Y ? dimConstY : dimConstZ;
}

export function dimProps(blk: IBlkDef, dim: Dim) {
    switch (dim) {
        case Dim.X: return { x: blk.x, cx: blk.cx, dx: blk.dx, rangeOffsets: blk.rangeOffsetsX, offX: blk.offX ?? 0, sizeX: blk.sizeX ?? blk.cx };
        case Dim.Y: return { x: blk.y, cx: blk.cy, dx: blk.dy, rangeOffsets: blk.rangeOffsetsY, offX: blk.offY ?? 0, sizeX: blk.sizeY ?? blk.cy };
        case Dim.Z: return { x: blk.z, cx: blk.cz, dx: blk.dz, rangeOffsets: blk.rangeOffsetsZ, offX: blk.offZ ?? 0, sizeX: blk.sizeZ ?? blk.cz };
    }
}

export function duplicateGrid(layout: IModelLayout, blk: IBlkDef): IBlkDef {
    let newBlk = { ...blk, access: blk.access ? { ...blk.access } : undefined };
    newBlk.name = '';
    layout.cubes.push(newBlk);
    return newBlk;
}

export function splitGridForHighlight(layout: IModelLayout, blk: IBlkDef, dim: Dim, xSplit: number) {
    let { cx, rangeOffsets } = dimProps(blk, dim);

    if (cx <= 1) {
        return blk;
    }

    // if there exists some rangeOffsets in our dim, then we iter over the sub-blocks, and split them
    if (rangeOffsets && blk.subs) {
        for (let s of blk.subs) {
            let res = splitGrid(layout, s, dim, xSplit, 0);
            if (res) {
                return res;
            }
        }
    }

    return splitGrid(layout, blk, dim, xSplit, 0);
}

export function splitGrid(layout: IModelLayout, blk: IBlkDef, dim: Dim, xSplit: number, splitAmt: number) {
    // generate several new blocks (let's say up to 5) that are neighbouring the zSplit point

    // main-left, left, center, right, main-right

    // choose center as floor(zSplit), left is floor(zSplit) - 1, right is floor(zSplit) + 1
    // main-left and main-right are the remaining
    // only create those if there's space

    // The splitAmt governs the overall gap between blocks
    // Want a rotating-block-under-examination effect. When zSplit is right down the center (x + 0.5),
    // have max seperation, and effectively join left & right with their main
    // For non 0.5 zSplits, will show 2 gaps

    let { offX, sizeX } = dimProps(blk, dim);


    let blocks: IBlkDef[] = [];
    let rangeOffsets: [number, number][] = [];

    let colX = Math.floor(xSplit) - offX;
    if (colX < 0 || colX >= sizeX) {
        return null;
    }

    if (sizeX <= 1) {
        return blk;
    }

    function addSubBlockLocal(iStart: number, iEnd: number, xOffset: number) {
        let res = addSubBlock(layout, blk, dim, iStart, iEnd, xOffset);
        if (res) {
            blocks.push(res.subBlock);
            rangeOffsets.push(res.rangeOffset);
        }
        return res?.subBlock ?? null;
    }

    let midBlock: IBlkDef | null;
    if (splitAmt === 0) {
        addSubBlockLocal(0, colX, 0.0);
        midBlock = addSubBlockLocal(colX, colX + 1, 0.0);
        addSubBlockLocal(colX + 1, sizeX, 0.0);

    } else {
        let scale = 0.5;
        let fract = (xSplit - colX - 0.5) * scale + 0.5;

        let addMidBlockBefore = fract + scale < 1.0;
        let addMidBlockAfter = fract - scale > 0.0;
        let offset = lerpSmoothstep(-splitAmt, 0, (xSplit - 0.5) * scale + 0.5);

        addSubBlockLocal(0     , colX - (addMidBlockBefore ? 1 : 0), offset + 0.0);

        if (addMidBlockBefore) {
            addSubBlockLocal(colX - 1, colX    , offset + lerpSmoothstep(splitAmt, 0, fract + scale));
        }

        midBlock = addSubBlockLocal(colX    , colX + 1, offset + lerpSmoothstep(splitAmt, 0, fract));

        if (addMidBlockAfter) {
            addSubBlockLocal(colX + 1, colX + 2, offset + lerpSmoothstep(splitAmt, 0, fract - scale));
        }

        addSubBlockLocal(colX + (addMidBlockAfter ? 2 : 1), sizeX, offset + splitAmt);
    }

    if (blocks.length > 0) {
        if (dim === Dim.X) blk.rangeOffsetsX = rangeOffsets;
        if (dim === Dim.Y) blk.rangeOffsetsY = rangeOffsets;
        if (dim === Dim.Z) blk.rangeOffsetsZ = rangeOffsets;
        blk.subs = blocks;
        return midBlock;

    } else {
        return null;
    }
}

interface ISubBlockInfo {
    subBlock: IBlkDef;
    rangeOffset: [number, number];
}

function addSubBlock(layout: IModelLayout, blk: IBlkDef, dim: Dim, iStart: number, iEnd: number, xOffset: number): ISubBlockInfo | null {
    let { x, sizeX, offX } = dimProps(blk, dim);
    let { vecId, xName, dxName, offXName, sizeXName } = dimConsts(dim);

    if (iStart >= iEnd || iEnd <= 0 || iStart >= sizeX) {
        return null;
    }

    let scale = (iEnd - iStart) / sizeX;
    let translate = iStart / sizeX;

    let mtx = Mat4f.fromScaleTranslation(new Vec3(1,1,1).setAt(vecId, scale), new Vec3().setAt(vecId, translate));

    let subBlock: IBlkDef = { ...blk,
        [dxName]: (iEnd - iStart) * layout.cell,
        // [cxName]: iEnd - iStart,
        access: blk.access && { ...blk.access },
        localMtx: (blk.localMtx ?? new Mat4f()).mul(mtx),
        [xName]: x + (iStart * layout.cell + xOffset),
        [offXName]: iStart + offX,
        [sizeXName]: iEnd - iStart,
    };
    return { subBlock, rangeOffset: [iEnd, xOffset] };
}


export function splitGridAll(layout: IModelLayout, blk: IBlkDef, dim: Dim) {
    let { dx } = dimProps(blk, dim);

    let nCells = Math.ceil(dx / layout.cell);

    let blocks: IBlkDef[] = [];
    let rangeOffsets: [number, number][] = [];

    for (let i = 0; i < nCells; i += 1) {
        let res = addSubBlock(layout, blk, dim, i, i + 1, 0)!;
        blocks.push(res.subBlock);
        rangeOffsets.push(res.rangeOffset);
    }

    if (dim === Dim.X) blk.rangeOffsetsX = rangeOffsets;
    if (dim === Dim.Y) blk.rangeOffsetsY = rangeOffsets;
    if (dim === Dim.Z) blk.rangeOffsetsZ = rangeOffsets;
    blk.subs = blocks;

    return blocks;
}

export interface IColorMix {
    color1?: Vec4;
    color2: Vec4;
    mixes: number[];
}

/* Returns all subblocks along a given dimension that overlap the provided range

Used in combination with splitGrid. To find all blocks up to, but not including the target idx 3, use:

    findSubBlocks(blk, Dim.X, null, 3)

To find the exact block at idx 3, use:

    findSubBlocks(blk, Dim.X, 3, 3)

To find all blocks after idx 3, use:

    findSubBlocks(blk, Dim.X, 4, null)

*/
export function findSubBlocks(blk: IBlkDef, dim: Dim, idxLow: number | null, idxHi: number | null) {
    if (!blk.subs) {
        return [];
    }

    let offsets = dim === Dim.X ? blk.rangeOffsetsX : dim === Dim.Y ? blk.rangeOffsetsY : blk.rangeOffsetsZ;
    idxLow = idxLow === null ? null : Math.floor(idxLow);
    idxHi = idxHi === null ? null : Math.floor(idxHi);

    let subBlocks: IBlkDef[] = [];
    let startIdx = 0;
    for (let i = 0; i < blk.subs.length; i += 1) {
        let endIdx = offsets?.[i]?.[0];
        if (isNil(endIdx)) {
            break;
        }
        if ((idxLow === null || idxLow < endIdx) && (idxHi === null || idxHi >= startIdx)) {
            subBlocks.push(blk.subs[i]);
        }
        startIdx = endIdx;
    }
    return subBlocks;
}

export enum TextAlignVert {
    Top,
    Middle,
    Bottom,
}

export enum TextAlignHoriz {
    Left,
    Center,
    Right,
}

