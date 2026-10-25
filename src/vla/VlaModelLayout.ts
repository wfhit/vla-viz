import { IVlaModelLink, IModelShape, IRealTex, IRmsNormLink, IBlockLayerLink, IExpertLayerLink, modelVariant } from "./VlaModel";
import { isNil } from "@/src/utils/data";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3 } from "@/src/utils/vector";
import { IBufferTex } from "@/src/utils/bufferTex";
import { dimProps } from "./Annotations";
import { DimStyle } from "./walkthrough/WalkthroughTools";

export interface IBlkDef {
    idx: number; // index in the layout.cubes array
    noTag?: boolean; // no name tag above the block (the page labels it another way)
    absent?: boolean; // the model has no such tensor: never drawn (the page zeroes its opacity each frame)
    t: 'w' | 'i' | 'a', // weights; intermediate value; aggregate (just LN & softmax)
    x: number;
    y: number;
    z: number;
    dx: number; // units: model-space
    dy: number;
    dz: number;
    cx: number; // units: number of cells
    cy: number;
    cz: number;
    access?: IBlkAccess;
    deps?: IBlkDeps;
    dimX: DimStyle;
    dimY: DimStyle;
    name: string;
    small: boolean; // small enough to not be worth rendering in large models
    // implicit dimZ = DimStyle.Batch for t === 'i'

    // fields that are post-added by the walk-through for various rendering configurations

    localMtx?: Mat4f; // for creating blocks that are sub-parts of a block
    // what to do for different axes?
    rangeOffsetsX?: [number, number][]; // if this block has been split, map from [[s0, xOff], [s1, xOff], ...] to the original block
    rangeOffsetsY?: [number, number][];
    rangeOffsetsZ?: [number, number][];
    highlight: number; // 0 - 1 (0 = no highlight, 1 = full highlight)
    opacity: number; // 0 - 1 (0 = transparent, 1 = opaque)
    special: BlkSpecial;
    transpose?: boolean; // transpose the process direction
    subs?: IBlkDef[]; // substitutes for this block (i.e. render these instead)
    offX?: number; // offset from the original block
    offY?: number;
    offZ?: number;
    sizeX?: number; // size of the sub block
    sizeY?: number;
    sizeZ?: number;
}

export enum BlkSpecial {
    None,
    Attention,
}

// define how a cell is computed from other blocks
// matrix-mulplication: cell(x, y, b) = sum_i(A[i, y] * B[x, i, b]) + C[0, y]
export interface IBlkDeps {
    dot?: [IBlkCellDep, IBlkCellDep];
    dotLen?: number;
    add?: IBlkCellDep[];
    special: BlKDepSpecial;
    lowerTri?: boolean;
}

export interface IBlkCellDep {
    src: IBlkDef;
    srcIdxMtx: Mat4f; // inputs: [x, y, b, [i]], outputs: [x, y, b]
}

interface IBlkDepArgs {
    dot?: [[IBlkDef, string], [IBlkDef, string]];
    dotLen?: number;
    add?: [IBlkDef, string][];
    lowerTri?: boolean; // only use the lower triangle of the matrix (causal attention matrices)
    special?: BlKDepSpecial;
}

export enum BlKDepSpecial {
    None,
    Softmax,
    SwiGlu,
    LayerNorm,
    InputEmbed,
    LayerNormMu,
    LayerNormSigma,
    SoftmaxAggMax,
    SoftmaxAggExp,
    Attention,
    AdaLayerNorm,
}

let depIdxVars = '0xybi';
function parseDepIdxStr(str: string): Mat4f {
    let mtx = Mat4f.zeros();
    for (let destI = 0; destI < str.length; destI++) {
        let srcIdx = depIdxVars.indexOf(str[destI]);
        if (srcIdx > 0) {
            mtx.s(destI, srcIdx - 1, 1.0);
        }
    }
    return mtx;
}

function depArgsToDeps(args: IBlkDepArgs): IBlkDeps {
    let makeBlkDeps = (src: IBlkDef, depStr: string) => ({ src, srcIdxMtx: parseDepIdxStr(depStr) });
    return {
        dot: args.dot && args.dot.map(([src, depStr]) => makeBlkDeps(src, depStr)) as [IBlkCellDep, IBlkCellDep],
        dotLen: args.dotLen,
        add: args.add && args.add.map(([src, depStr]) => makeBlkDeps(src, depStr)),
        special: args.special ?? BlKDepSpecial.None,
        lowerTri: args.lowerTri,
    };
}

export function getBlkDimensions(blk: IBlkDef) {
    let { x, y, z, dx, dy, dz } = blk;
    return {
        tl: new Vec3(x, y, z),
        br: new Vec3(x + dx, y + dy, z + dz),
    };
}

export function setBlkPosition(blk: IBlkDef, pos: Vec3) {
    blk.x = pos.x;
    blk.y = pos.y;
    blk.z = pos.z;
}

export interface IBlkAccess {
    src: IBufferTex;
    channel: 'r' | 'g' | 'b';
    scale: number;
    // Packed as GPU row-major mat4x2: first 4 floats = texX row (coeffs of cell x,y,z,1),
    // next 4 floats = texY row. See getBlockValueAtIdx / blockRender u_accessMtx.
    mat: Mat4f;
    disable?: boolean;
}

export interface IBlkAccessDefArgs {
    src?: IBufferTex;
    channel?: 'r' | 'g' | 'b';
    scale?: number;
    // texX = x[0]*cellX + x[1]*cellY + x[2]*cellZ + x[3]
    // texY = y[0]*cellX + y[1]*cellY + y[2]*cellZ + y[3]
    // (missing trailing coeffs are treated as 0)
    x: number[];
    y: number[];
}

export interface IBlkDefArgs {
    t: 'w' | 'i' | 'a', // weights; intermediate value
    xL?: number; // pos of Left edge
    xR?: number; // Right
    xM?: number; // Middle
    zF?: number; // Front
    zB?: number; // Back
    zM?: number; // Middle
    name?: string;
    y: number;
    cx: number; // units: number of cells
    cz: number;
    cy: number;
    dimX: DimStyle;
    dimY: DimStyle;
    special?: BlkSpecial;
    access?: IBlkAccessDefArgs;
    deps?: IBlkDepArgs;
    small?: boolean;
    hidden?: boolean;
    transpose?: boolean;
}

export interface IBlkLabel {
    visible: number;
    cubes: IBlkDef[];
}

export interface IModelLayout {
    cell: number;
    height: number;
    margin: number;
    cubes: IBlkDef[];
}

// what the page-independent code (render, hover, labels) needs from any page's scene layout
export interface ISceneLayout extends IModelLayout {
    labels: IBlkLabel[];
}

export function cellPosition(layout: IModelLayout, blk: IBlkDef, dim: Dim, index: number) {
    let { x, rangeOffsets } = dimProps(blk, dim);
    let base = x + layout.cell * index;
    if (!rangeOffsets) {
        return base;
    }
    for (let [s, xOff] of rangeOffsets!) {
        if (index < s) {
            return base + xOff;
        }
    }
    return base;
}

export function makeBlk(args: IBlkDefArgs, cell: number, offset: Vec3 = new Vec3()): IBlkDef {
    let xDef = [args.xL, args.xR, args.xM].map(a => +!isNil(a)).reduce((a, b) => a + b, 0);
    let yDef = [args.zF, args.zB, args.zM].map(a => +!isNil(a)).reduce((a, b) => a + b, 0);
    if (xDef !== 1 || yDef !== 1) {
        throw new Error(`Must supply exactly 1 x arg & 1 y arg: ${JSON.stringify(args)}`);
    }
    let dx = args.cx * cell;
    let dy = args.cz * cell;
    let x = !isNil(args.xL) ? args.xL : !isNil(args.xR) ? args.xR - dx : args.xM! - dx / 2;
    let z = !isNil(args.zB) ? args.zB : !isNil(args.zF) ? args.zF - dy : args.zM! - dy / 2;

    function ensure4(a: number[]) {
        return a.length >= 4 ? a.slice(0, 4) : [...a, 0, 0, 0, 0].slice(0, 4);
    }

    return {
        dx: args.cx * cell,
        dy: args.cy * cell,
        dz: args.cz * cell,
        t: args.t,
        x: x + offset.x,
        y: args.y + offset.y,
        z: z + offset.z,
        cx: args.cx,
        cy: args.cy,
        cz: args.cz,
        dimX: args.dimX,
        dimY: args.dimY,
        name: args.name ?? "<unknown>",
        access: args.access?.src ? {
            channel: args.access.channel ?? 'r',
            src: args.access.src,
            scale: args.access.scale ?? 1.0,
            mat: Mat4f.fromColMajor([...ensure4(args.access.x), ...ensure4(args.access.y), 0, 0, 0, 0, 0, 0, 0, 0]),
        } : undefined,
        deps: args.deps ? depArgsToDeps(args.deps) : undefined,
        opacity: args.hidden ? 0.0 : 1.0,
        absent: !!(args.access?.src as IRealTex | undefined)?.meta?.absent || undefined,
        highlight: 0.0,
        small: args.small ?? false,
        special: args.special ?? BlkSpecial.None,
        transpose: args.transpose,
        idx: -1,
    };
}

export type IVlaModelLayout = ReturnType<typeof genVlaModelLayout>;

// The k005b trunk laid out in 3D. Geometry (cells) = the REAL exported windows
// (see IModelShape); every block samples a real tensor window through `access`.
// Structure follows the real Qwen3-VL-2B trunk of k005b: RMSNorm (γ only), GQA
// attention (16 query heads, 8 kv groups; each K/V group serves 2 query heads),
// per-head q_norm/k_norm, SwiGLU MLP (gate, up, silu(gate)·up, down), final norm,
// then the h* boundary state feeding the knack + done heads, and the action
// expert's suffix feeding action_out_proj → velocity → action chunk.
export function genVlaModelLayout(shape: IModelShape, vlaGpuModel: IVlaModelLink | null = null, offset: Vec3 = new Vec3(0, 0, 0)) {
    let { B, T, C, vocabSize, nHeads, nKvHeads, A, F, nBlocks, nKnack, nDone, actH, actD, nSuffix } = shape;
    let groupSize = nHeads / nKvHeads;

    let y = 0;

    let cell = 1.5;
    let margin = Math.max(12, C / 10);

    function acc(src: IRealTex | undefined, x: number[], yy: number[], scaleOverride?: number): IBlkAccessDefArgs | undefined {
        if (!src) return undefined;
        return { src, x, y: yy, scale: scaleOverride ?? src.meta.absScale };
    }

    function mk(args: IBlkDefArgs): IBlkDef {
        return makeBlk(args, cell, offset);
    }

    function mkLabel(cubes?: IBlkDef[]): IBlkLabel {
        return { visible: 0, cubes: cubes ?? [] };
    }

    let m = vlaGpuModel;
    let V = modelVariant(m?.config);
    let ditName = V.id === 'pi0' ? 'Expert' : 'DiT';
    let gluName = (k: 'swiglu' | 'geglu') => k === 'geglu' ? 'GELU(gate)*up' : 'SiLU(gate)·up';
    let cubes: IBlkDef[] = [];

    // [Wt] token ids
    let idxObj = mk({
        t: 'i', cx: T, cz: B, cy: 1, y: y,
        xM: 0, zM: 0,
        access: acc(m?.inputTokens, [0, 0, 0], [1, 0, 0], 1 / (m?.config.vocab_size ?? 1)),
        dimX: DimStyle.T, dimY: DimStyle.None,
        name: 'Prompt token ids',
    });

    let leftX = -T * cell / 2 - margin;
    let rightX = T * cell / 2 + margin;

    y += cell + margin;

    // embed table [vocab, C] (tied) — cells: x = vocab row, y = channel
    let tokEmbedObj = mk({
        t: 'w',
        xR: leftX, zM: 0, y: y,
        cx: vocabSize, cz: 1, cy: C,
        access: acc(m?.embedWeight, [0, 1, 0], [1, 0, 0]),
        dimX: DimStyle.n_vocab, dimY: DimStyle.C,
        name: 'Embed Table',
    });

    let residual0 = mk({
        t: 'i',
        xM: 0, zM: 0, y: y,
        cx: T, cz: B, cy: C,
        access: acc(m?.embedOut, [0, 1, 0], [1, 0, 0]),
        deps: { add: [[tokEmbedObj, 'iy'], [idxObj, 'x0']], special: BlKDepSpecial.InputEmbed },
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: 'Input Embed (prefix)',
    });
    cubes.push(idxObj, tokEmbedObj, residual0);

    let embedLabel = mkLabel([idxObj, tokEmbedObj, residual0]);

    y += C * cell + margin;

    function createLn(x: number, src: IBlkDef, target?: IRmsNormLink) {
        return createLnAt(leftX + x, T, src, target);
    }

    // RMSNorm group whose blocks end at x = lnLeftX (the residual stream sits right of it).
    // With `ada` it is the expert's AdaRMSNorm: γβ(t) (the timestep modulation) sits left of w.
    function createLnAt(lnLeftX: number, Tt: number, src: IBlkDef, target?: IRmsNormLink,
            ada?: { tex: IRealTex | undefined, row0: number, name: string, rows: number }) {
        let resLeftX = lnLeftX - Tt * cell - margin;

        let lnAgg1 = mk({
            t: 'a', cx: Tt, cz: B, cy: 1, y: y,
            xR: lnLeftX, zM: 0,
            access: acc(target?.agg, [0, 0, 0, 0], [1, 0, 0]),
            deps: { add: [[src, 'xi']], special: BlKDepSpecial.LayerNormMu },
            dimX: DimStyle.T, dimY: DimStyle.None, small: true,
            name: 'RMS Agg',
        });
        let lnAgg2 = mk({
            t: 'a', cx: Tt, cz: B, cy: 1, y: y + cell,
            xR: lnLeftX, zM: 0,
            access: acc(target?.agg, [0, 0, 0, 1], [1, 0, 0]),
            deps: { add: [[src, 'xi']], special: BlKDepSpecial.LayerNormSigma },
            dimX: DimStyle.T, dimY: DimStyle.None, small: true,
            name: '',
        });

        y += 2 * cell + margin;

        let lnSigma = mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: y,
            xR: resLeftX, zM: 0,
            access: acc(target?.weight, [0, 0, 0], [0, 1, 0]),
            dimX: DimStyle.None, dimY: DimStyle.C,
            name: ada ? 'w' : 'γ', small: true,
        });
        // RMSNorm has no β: placeholder only (never rendered)
        let lnMu = mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: y,
            xR: resLeftX - cell * 1 - margin, zM: 0,
            dimX: DimStyle.None, dimY: DimStyle.C,
            name: '', small: true, hidden: true,
        });
        // AdaRMSNorm: [γ(t) | β(t)] (pi0: [scale | shift | gate]) for this sublayer (columns =
        // modulation rows; rows = channels), right of the norm output, where the te(t) bus reaches it
        let adaBlk = ada ? mk({
            t: 'i', cx: ada.rows, cz: B, cy: C, y: y,
            xL: lnLeftX + margin, zM: 0,
            access: acc(ada.tex, [0, 1, 0], [1, 0, 0, ada.row0]),
            dimX: DimStyle.None, dimY: DimStyle.C,
            name: ada.name,
        }) : null;
        let adds: [IBlkDef, string][] = [[src, 'xy'], [lnAgg1, 'xi'], [lnAgg2, 'xi'], [lnSigma, '0y']];
        if (adaBlk) adds.push([adaBlk, '0y']);
        let lnResid = mk({
            t: 'i', cx: Tt, cz: B, cy: C, y: y,
            xR: lnLeftX, zM: 0,
            access: acc(target?.output, [0, 1, 0], [1, 0, 0]),
            deps: { add: adds, special: ada ? BlKDepSpecial.AdaLayerNorm : BlKDepSpecial.LayerNorm },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: ada ? 'AdaRMSNorm' : 'RMSNorm',
        });
        let lnCubes = adaBlk ? [lnAgg1, lnAgg2, lnSigma, adaBlk, lnResid] : [lnAgg1, lnAgg2, lnSigma, lnResid];
        return { lnAgg1, lnAgg2, lnResid, lnSigma, lnMu, ada: adaBlk, cubes: lnCubes };
    }

    let seam = cell;
    let lnLeftX = leftX - (T + 2) * cell - 2 * margin;
    let resMidX = 0;
    // pre-offset coordinates of a block (mk() adds `offset` again)
    let bx = (b: IBlkDef) => b.x - offset.x;
    let by = (b: IBlkDef) => b.y - offset.y;
    let bxR = (b: IBlkDef) => b.x - offset.x + b.dx;

    function createLayer(src: IBlkDef, target: IBlockLayerLink | undefined) {
        let ln1 = createLn(0, src, target?.ln_1);

        let interGroupMargin = 3 * margin + (C * cell) / 16;
        let qkvMargin = 1 * margin + (C * cell) / 16;
        let slot = B * cell + qkvMargin;           // z pitch inside a kv group
        let groupWidth = (groupSize + 2) * slot + interGroupMargin;
        let groupSlots = groupSize + 2;

        let attn1Y = y + A * cell + margin;
        let vOutY = attn1Y + T * cell + margin;

        let attnLeftX = lnLeftX;
        let qkvValLeftX = attnLeftX - T * cell - margin;
        let qkvBiasLeftX = qkvValLeftX - C * cell - margin;
        let attn2LeftX = attnLeftX - (T + 2) * cell - 2 * margin;

        // z order inside group g: q(2g) | K_g | V_g | q(2g+1)
        // one representative head per view (bbycroft style, "x16 heads"): every head / kv group
        // shares head 0's slots (q | K | V around z = 0); only head 0 / kv 0 are drawn
        let slotZ = (_g: number, s: number) => (s - 1) * slot;
        let qSlot = (_j: number) => 0;

        let groups = [];
        for (let g = 0; g < nKvHeads; g++) {
            let kv = target?.kv[g];
            let kMid = slotZ(g, 1);
            let vMid = slotZ(g, 2);
            let kWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y,
                xR: qkvValLeftX, zM: kMid,
                access: acc(kv?.kWeight, [1, 0, 0], [0, 1, 0]),
                dimX: DimStyle.C, dimY: DimStyle.A,
                name: `K Weights (kv ${g})`,
            });
            let vWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y,
                xR: qkvValLeftX, zM: vMid,
                access: acc(kv?.vWeight, [1, 0, 0], [0, 1, 0]),
                dimX: DimStyle.C, dimY: DimStyle.A,
                name: `V Weights (kv ${g})`,
            });
            let kBiasBlock = mk({
                t: 'w', cx: 1, cz: 1, cy: A, y: y,
                xR: qkvBiasLeftX, zM: kMid,
                access: acc(kv?.kNorm, [0, 0, 0], [0, 1, 0]),
                dimX: DimStyle.None, dimY: DimStyle.A, small: true,
                name: 'k_norm γ',
            });
            // V has no bias / norm in Qwen3: placeholder only
            let vBiasBlock = mk({
                t: 'w', cx: 1, cz: 1, cy: A, y: y,
                xR: qkvBiasLeftX, zM: vMid,
                dimX: DimStyle.None, dimY: DimStyle.A, small: true, hidden: true,
                name: '',
            });
            let kBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: y,
                xR: attnLeftX, zM: kMid,
                access: acc(kv?.k, [0, 1, 0], [1, 0, 0]),
                deps: { dot: [[kWeightBlock, 'iy'], [ln1.lnResid, 'xi']], add: [[kBiasBlock, '0y']], dotLen: C },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: `K vectors (kv ${g})`,
            });
            let vBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: y,
                xR: attnLeftX, zM: vMid,
                access: acc(kv?.v, [0, 1, 0], [1, 0, 0]),
                deps: { dot: [[vWeightBlock, 'iy'], [ln1.lnResid, 'xi']], dotLen: C },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: `V vectors (kv ${g})`,
            });
            groups.push({ kWeightBlock, vWeightBlock, kBiasBlock, vBiasBlock, kBlock, vBlock,
                cubes: [kWeightBlock, vWeightBlock, kBiasBlock, kBlock, vBlock] });
        }

        let heads = [];
        for (let i = 0; i < nHeads; i++) {
            let g = Math.floor(i / groupSize);
            let grp = groups[g];
            let ht = target?.heads[i];
            let headZMid = slotZ(g, qSlot(i % groupSize));

            let qWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y,
                xR: qkvValLeftX, zM: headZMid,
                access: acc(ht?.qWeight, [1, 0, 0], [0, 1, 0]),
                dimX: DimStyle.C, dimY: DimStyle.A,
                name: `Q Weights (head ${i})`,
            });
            let qBiasBlock = mk({
                t: 'w', cx: 1, cz: 1, cy: A, y: y,
                xR: qkvBiasLeftX, zM: headZMid,
                access: acc(ht?.qNorm, [0, 0, 0], [0, 1, 0]),
                dimX: DimStyle.None, dimY: DimStyle.A, small: true,
                name: 'q_norm γ',
            });
            let qBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: y,
                xR: attnLeftX, zM: headZMid,
                access: acc(ht?.q, [0, 1, 0], [1, 0, 0]),
                deps: { dot: [[qWeightBlock, 'iy'], [ln1.lnResid, 'xi']], add: [[qBiasBlock, '0y']], dotLen: C },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: `Q vectors (head ${i})`,
            });

            // prefix q · prefix k (causal): the prefix never sees the suffix
            let attnMtx = mk({
                t: 'i', cx: T, cz: B, cy: T, y: attn1Y,
                xR: attnLeftX, zM: headZMid,
                access: acc(ht?.attnScores, [1, 0, 0], [0, 1, 0]),
                deps: { dot: [[qBlock, 'yi'], [grp.kBlock, 'xi']], lowerTri: V.trunk_causal, dotLen: A, special: BlKDepSpecial.Attention },
                dimX: DimStyle.T, dimY: DimStyle.T,
                special: BlkSpecial.Attention,
                transpose: true,
                name: 'Joint Attention',
            });

            let attnMtxAgg1 = mk({
                t: 'a', cx: 1, cz: B, cy: T, y: attn1Y,
                xR: attnLeftX - T * cell - margin - cell, zM: headZMid,
                access: acc(ht?.attnAgg, [0, 0, 0, 1], [0, 1, 0]),
                deps: { add: [[attnMtx, 'iy']], special: BlKDepSpecial.SoftmaxAggExp },
                dimX: DimStyle.None, dimY: DimStyle.T, small: true,
                name: '',
            });

            let attnMtxAgg2 = mk({
                t: 'a', cx: 1, cz: B, cy: T, y: attn1Y,
                xR: attnLeftX - T * cell - margin, zM: headZMid,
                access: acc(ht?.attnAgg, [0, 0, 0, 0], [0, 1, 0]),
                deps: { add: [[attnMtx, 'iy']], special: BlKDepSpecial.SoftmaxAggMax },
                dimX: DimStyle.None, dimY: DimStyle.T, small: true,
                name: '',
            });

            let attnMtxSm = mk({
                t: 'i', cx: T, cz: B, cy: T, y: attn1Y,
                xR: attn2LeftX, zM: headZMid,
                access: acc(ht?.attnProbs, [1, 0, 0], [0, 1, 0]),
                deps: { add: [[attnMtx, 'xy'], [attnMtxAgg1, 'iy'], [attnMtxAgg2, 'iy']], lowerTri: V.trunk_causal, special: BlKDepSpecial.Softmax },
                dimX: DimStyle.T, dimY: DimStyle.T,
                special: BlkSpecial.Attention,
                transpose: true,
                name: 'Prefix Softmax',
            });

            let vOutBlock = mk({
                t: 'i', cx: T, cz: B, cy: A, y: vOutY,
                xR: attnLeftX, zM: headZMid,
                access: acc(ht?.attnOut, [0, 1, 0], [1, 0, 0]),
                deps: { dot: [[grp.vBlock, 'iy'], [attnMtxSm, 'ix']], dotLen: A },
                dimX: DimStyle.T, dimY: DimStyle.A,
                name: 'V Output',
            });

            let ownCubes = [qWeightBlock, qBiasBlock, qBlock, attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSm, vOutBlock];
            // the head's K/V group is shared with its sibling head; it is pushed once (by the group)
            let headCubes = [...ownCubes, ...grp.cubes];

            let headLabel = mkLabel(headCubes);
            let qLabel = mkLabel([qWeightBlock, qBiasBlock, qBlock]);
            let kLabel = mkLabel([grp.kWeightBlock, grp.kBiasBlock, grp.kBlock]);
            let vLabel = mkLabel([grp.vWeightBlock, grp.vBlock]);
            let biasLabel = mkLabel([qBiasBlock, grp.kBiasBlock]);
            let mtxLabel = mkLabel([attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSm]);
            let vectorLabel = mkLabel([vOutBlock]);

            heads.push({
                kvGroup: g,
                qWeightBlock, kWeightBlock: grp.kWeightBlock, vWeightBlock: grp.vWeightBlock,
                qBiasBlock, kBiasBlock: grp.kBiasBlock, vBiasBlock: grp.vBiasBlock,
                qBlock, kBlock: grp.kBlock, vBlock: grp.vBlock,
                attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSm, vOutBlock,
                qLabel, kLabel, vLabel, biasLabel, mtxLabel, vectorLabel, headLabel,
                cubes: headCubes,
                ownCubes,
                labels: [qLabel, kLabel, vLabel, biasLabel, mtxLabel, vectorLabel, headLabel],
            });
        }

        let attnCubes = [...groups.flatMap(g => g.cubes), ...heads.flatMap(h => h.ownCubes)];

        let vOutCombined = mk({
            t: 'i', cx: T, cz: B, cy: C, y: vOutY,
            xR: attnLeftX, zF: -groupWidth * nKvHeads / 2,
            dimX: DimStyle.T, dimY: DimStyle.C,
            hidden: true,
            name: 'V Output Combined',
        });

        let vFinalZ = Math.max(
            vOutY + A * cell + 2 * margin,
            y + C * cell + margin,
        );

        let projWeight = mk({
            t: 'w', cx: C, cz: 1, cy: C, y: vFinalZ,
            xR: qkvValLeftX, zM: 0,
            access: acc(target?.oWeight, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.C, dimY: DimStyle.C,
            name: 'o_proj Weights',
        });

        // o_proj has no bias: placeholder only
        let projBias = mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: vFinalZ,
            xR: qkvValLeftX - C * cell - margin, zM: 0,
            dimX: DimStyle.None, dimY: DimStyle.C, small: true, hidden: true,
            name: '',
        });

        let attnOut = mk({
            t: 'i', cx: T, cz: B, cy: C, y: vFinalZ,
            xR: attnLeftX, zM: 0,
            access: acc(target?.oOut, [0, 1, 0], [1, 0, 0]),
            deps: {
                dot: [[projWeight, 'iy'], [vOutCombined, 'xi']], dotLen: C,
                add: heads.map(h => [h.vOutBlock, 'xi'] as [IBlkDef, string]),
            },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'Attention Output',
        });

        let attnResidual = mk({
            t: 'i', cx: T, cz: B, cy: C, y: vFinalZ,
            xM: resMidX, zM: 0,
            access: acc(target?.attnResid, [0, 1, 0], [1, 0, 0]),
            deps: { add: [[attnOut, 'xy'], [src, 'xy']] },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'Attention Residual',
        });

        // room for the S-shaped entry arrow ⊕ → next norm
        y = vFinalZ + C * cell + 3 * margin;

        let ln2 = createLn(0, attnResidual, target?.ln_2);

        // SwiGLU: gate & up weights [F, C], drawn side by side
        let mlpFcWeight = mk({
            t: 'w', cx: F, cz: 1, cy: C, y: y,
            xR: attnLeftX, zM: 0,
            access: acc(target?.gateWeight, [0, 1, 0], [1, 0, 0]),
            dimX: DimStyle.C4, dimY: DimStyle.C,
            name: 'gate_proj Weights',
        });
        let mlpUpWeight = mk({
            t: 'w', cx: F, cz: 1, cy: C, y: y,
            xR: attnLeftX - F * cell - margin, zM: 0,
            access: acc(target?.upWeight, [0, 1, 0], [1, 0, 0]),
            dimX: DimStyle.C4, dimY: DimStyle.C,
            name: 'up_proj Weights',
        });
        // no MLP biases in Qwen3: placeholder only
        let mlpFcBias = mk({
            t: 'w', cx: F, cz: 1, cy: 1, y: y - 1 * cell - margin,
            xR: attnLeftX, zM: 0,
            dimX: DimStyle.C4, dimY: DimStyle.None, small: true, hidden: true,
            name: '',
        });

        y += C * cell + margin;

        let mlpFc = mk({
            t: 'i', cx: F, cz: B, cy: T, y: y,
            xR: attnLeftX, zM: 0,
            access: acc(target?.gate, [1, 0, 0], [0, 1, 0]),
            deps: { dot: [[mlpFcWeight, 'xi'], [ln2.lnResid, 'yi']], dotLen: C },
            dimX: DimStyle.C4, dimY: DimStyle.T,
            name: 'gate',
            transpose: true,
        });
        let mlpUp = mk({
            t: 'i', cx: F, cz: B, cy: T, y: y,
            xR: attnLeftX - F * cell - margin, zM: 0,
            access: acc(target?.up, [1, 0, 0], [0, 1, 0]),
            deps: { dot: [[mlpUpWeight, 'xi'], [ln2.lnResid, 'yi']], dotLen: C },
            dimX: DimStyle.C4, dimY: DimStyle.T,
            name: 'up',
            transpose: true,
        });
        y += T * cell + margin;

        let mlpAct = mk({
            t: 'i', cx: F, cz: B, cy: T, y: y,
            xR: attnLeftX, zM: 0,
            access: acc(target?.act, [1, 0, 0], [0, 1, 0]),
            deps: { add: [[mlpFc, 'xy'], [mlpUp, 'xy']], special: BlKDepSpecial.SwiGlu },
            dimX: DimStyle.C4, dimY: DimStyle.T,
            name: gluName(V.trunk_mlp),
            transpose: true,
        });
        y += T * cell + margin;

        let mlpProjWeight = mk({
            t: 'w', cx: F, cz: 1, cy: C, y: y,
            xR: attnLeftX, zM: 0,
            access: acc(target?.downWeight, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.C4, dimY: DimStyle.C,
            name: 'down_proj Weights',
        });

        let mlpProjBias = mk({
            t: 'w', cx: 1, cz: 1, cy: C, y: y,
            xR: attnLeftX - F * cell - margin, zM: 0,
            dimX: DimStyle.None, dimY: DimStyle.C, small: true, hidden: true,
            name: '',
        });

        let mlpResult = mk({
            t: 'i', cx: T, cz: B, cy: C, y: y,
            xL: attnLeftX + margin, zM: 0,
            access: acc(target?.mlpOut, [0, 1, 0], [1, 0, 0]),
            deps: { dot: [[mlpProjWeight, 'iy'], [mlpAct, 'ix']], dotLen: F },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'MLP Result',
        });

        let mlpResidual = mk({
            t: 'i', cx: T, cz: B, cy: C, y: y,
            xM: resMidX, zM: 0,
            access: acc(target?.mlpResid, [0, 1, 0], [1, 0, 0]),
            deps: { add: [[mlpResult, 'xy'], [attnResidual, 'xy']] },
            dimX: DimStyle.T, dimY: DimStyle.C,
            name: 'MLP Residual',
        });

        y += C * cell - margin;

        let mlpCubes = [...ln2.cubes, mlpFcWeight, mlpUpWeight, mlpFc, mlpUp, mlpAct, mlpProjWeight, mlpResult, mlpResidual];
        let projCubes = [projWeight, attnOut, attnResidual];
        let blockCubes = [...ln1.cubes, ...attnCubes, ...projCubes, ...mlpCubes];
        let headCubes = [...ln1.cubes, ...attnCubes];

        let transformerLabel = mkLabel(blockCubes);
        let selfAttendLabel = mkLabel([...headCubes, ...projCubes]);
        let projLabel = mkLabel(projCubes);
        let mlpLabel = mkLabel(mlpCubes);

        cubes.push(...blockCubes);

        return {
            ln1,
            heads,
            kvGroups: groups,
            labels: [transformerLabel, projLabel, selfAttendLabel, mlpLabel, ...heads.flatMap(h => h.labels)],
            cubes: blockCubes,
            transformerLabel,
            projLabel,
            selfAttendLabel,
            mlpLabel,
            projWeight,
            projBias,
            attnOut,
            attnResidual,
            mlpFc,
            mlpFcWeight,
            mlpFcBias,
            mlpUp,
            mlpUpWeight,
            mlpAct,
            mlpProjWeight,
            mlpProjBias,
            mlpResult,
            mlpResidual,
            ln2,
        };
    }

    // ---- one layer of the action expert (DiT), drawn in exactly the trunk layer's visual language:
    // same y anchors (spacing by the trunk's T), same block roles and field names. Differences are
    // the real ones: 17 suffix tokens, AdaRMSNorm (γβ(t) beside w), q/k/v biases (no q/k norm),
    // 16 query heads sharing 2 kv heads (8 per group), and a joint score row
    // [prefix keys (the trunk's cached K of the same layer) | suffix keys].
    let nHeadsE = m?.config.expert_heads ?? 16;
    let nKvE = m?.config.expert_kv_heads ?? 2;
    let groupSizeE = nHeadsE / nKvE;
    // the AdaNorm slot + the te(t) bus between the norm and the residual stream
    let adaRoom = V.ada_rows * cell + margin;
    let adaName = (i: number) => V.ada_rows === 3 ? `scale${i} shift${i} gate${i} (t)` : `gamma${i} beta${i} (t)`;

    function createExpertLayer(resMid: number, src: IBlkDef, el: IExpertLayerLink | undefined,
            adaTex: IRealTex | undefined, li: number, trunk: ReturnType<typeof createLayer> | null) {
        let Tt = nSuffix;
        let eLeftX = resMid - Tt * cell / 2 - margin;
        let ln1 = createLnAt(eLeftX - adaRoom, Tt, src, el?.ln_1, { tex: adaTex, row0: 0, name: adaName(1), rows: V.ada_rows });

        let interGroupMargin = 3 * margin + (C * cell) / 16;
        let qkvMargin = 1 * margin + (C * cell) / 16;
        let slot = B * cell + qkvMargin;
        let groupSlots = groupSizeE + 2;
        let groupWidth = groupSlots * slot + interGroupMargin;

        let attn1Y = y + A * cell + margin;
        let vOutY = attn1Y + T * cell + margin;

        let attnLeftX = eLeftX - (Tt + 2) * cell - 2 * margin;
        let qkvValLeftX = attnLeftX - Tt * cell - margin;
        let qkvBiasLeftX = qkvValLeftX - C * cell - margin;
        // one joint score row: [prefix keys | suffix keys], a 1-cell seam between them
        let scPR = attnLeftX - Tt * cell - seam;
        let aggR = scPR - T * cell - margin;
        let smR = aggR - cell - margin;
        let smPR = smR - Tt * cell - seam;

        // one representative head per view (bbycroft style, "x16 heads"): every head / kv group
        // shares head 0's slots (q | K | V around z = 0); only head 0 / kv 0 are drawn
        let slotZ = (_g: number, s: number) => (s - 1) * slot;
        let qSlot = (_j: number) => 0;

        let groups = [];
        for (let g = 0; g < nKvE; g++) {
            let kv = el?.kv[g];
            let kMid = slotZ(g, 1);
            let vMid = slotZ(g, 2);
            let kWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y, xR: qkvValLeftX, zM: kMid,
                access: acc(kv?.kWeight, [1, 0, 0], [0, 1, 0]),
                dimX: DimStyle.C, dimY: DimStyle.A, name: `K Weights (kv ${g})`,
            });
            let vWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y, xR: qkvValLeftX, zM: vMid,
                access: acc(kv?.vWeight, [1, 0, 0], [0, 1, 0]),
                dimX: DimStyle.C, dimY: DimStyle.A, name: `V Weights (kv ${g})`,
            });
            let kBiasBlock = mk({
                t: 'w', cx: 1, cz: 1, cy: A, y: y, xR: qkvBiasLeftX, zM: kMid,
                access: acc(kv?.kBias, [0, 0, 0], [0, 1, 0]),
                dimX: DimStyle.None, dimY: DimStyle.A, small: true, name: 'k bias',
            });
            let vBiasBlock = mk({
                t: 'w', cx: 1, cz: 1, cy: A, y: y, xR: qkvBiasLeftX, zM: vMid,
                access: acc(kv?.vBias, [0, 0, 0], [0, 1, 0]),
                dimX: DimStyle.None, dimY: DimStyle.A, small: true, name: 'v bias',
            });
            let kBlock = mk({
                t: 'i', cx: Tt, cz: B, cy: A, y: y, xR: attnLeftX, zM: kMid,
                access: acc(kv?.k, [0, 1, 0], [1, 0, 0]),
                deps: { dot: [[kWeightBlock, 'iy'], [ln1.lnResid, 'xi']], add: [[kBiasBlock, '0y']], dotLen: C },
                dimX: DimStyle.T, dimY: DimStyle.A, name: `K vectors (kv ${g})`,
            });
            let vBlock = mk({
                t: 'i', cx: Tt, cz: B, cy: A, y: y, xR: attnLeftX, zM: vMid,
                access: acc(kv?.v, [0, 1, 0], [1, 0, 0]),
                deps: { dot: [[vWeightBlock, 'iy'], [ln1.lnResid, 'xi']], add: [[vBiasBlock, '0y']], dotLen: C },
                dimX: DimStyle.T, dimY: DimStyle.A, name: `V vectors (kv ${g})`,
            });
            groups.push({ kWeightBlock, vWeightBlock, kBiasBlock, vBiasBlock, kBlock, vBlock,
                cubes: [kWeightBlock, vWeightBlock, kBiasBlock, vBiasBlock, kBlock, vBlock] });
        }

        let heads = [];
        for (let i = 0; i < nHeadsE; i++) {
            let g = Math.floor(i / groupSizeE);
            let grp = groups[g];
            let ht = el?.heads[i];
            let headZMid = slotZ(g, qSlot(i % groupSizeE));
            // the prefix keys / values this head reads: the trunk's kv group of the same layer
            // (expert q head i ↔ trunk kv group ⌊i / 2⌋, the shared attention's GQA tiling)
            let tg = trunk ? trunk.kvGroups[Math.floor(i / (nHeads / nKvHeads))] : null;

            let qWeightBlock = mk({
                t: 'w', cx: C, cz: 1, cy: A, y: y, xR: qkvValLeftX, zM: headZMid,
                access: acc(ht?.qWeight, [1, 0, 0], [0, 1, 0]),
                dimX: DimStyle.C, dimY: DimStyle.A, name: `Q Weights (head ${i})`,
            });
            let qBiasBlock = mk({
                t: 'w', cx: 1, cz: 1, cy: A, y: y, xR: qkvBiasLeftX, zM: headZMid,
                access: acc(ht?.qBias, [0, 0, 0], [0, 1, 0]),
                dimX: DimStyle.None, dimY: DimStyle.A, small: true, name: 'q bias',
            });
            let qBlock = mk({
                t: 'i', cx: Tt, cz: B, cy: A, y: y, xR: attnLeftX, zM: headZMid,
                access: acc(ht?.q, [0, 1, 0], [1, 0, 0]),
                deps: { dot: [[qWeightBlock, 'iy'], [ln1.lnResid, 'xi']], add: [[qBiasBlock, '0y']], dotLen: C },
                dimX: DimStyle.T, dimY: DimStyle.A, name: `Q vectors (head ${i})`,
            });
            let attnMtxP = mk({
                t: 'i', cx: T, cz: B, cy: Tt, y: attn1Y, xR: scPR, zM: headZMid,
                access: acc(ht?.scoresP, [1, 0, 0], [0, 1, 0]),
                deps: tg ? { dot: [[qBlock, 'yi'], [tg.kBlock, 'xi']], dotLen: A, special: BlKDepSpecial.Attention } : undefined,
                dimX: DimStyle.T, dimY: DimStyle.T, special: BlkSpecial.Attention, transpose: true,
                name: 'Joint Attention',
            });
            let attnMtx = mk({
                t: 'i', cx: Tt, cz: B, cy: Tt, y: attn1Y, xR: attnLeftX, zM: headZMid,
                access: acc(ht?.scoresS, [1, 0, 0], [0, 1, 0]),
                deps: { dot: [[qBlock, 'yi'], [grp.kBlock, 'xi']], dotLen: A, special: BlKDepSpecial.Attention },
                dimX: DimStyle.T, dimY: DimStyle.T, special: BlkSpecial.Attention, transpose: true,
                name: '',
            });
            let attnMtxAgg1 = mk({
                t: 'a', cx: 1, cz: B, cy: Tt, y: attn1Y, xR: aggR - cell, zM: headZMid,
                access: acc(ht?.attnAgg, [0, 0, 0, 1], [0, 1, 0]),
                deps: { add: [[attnMtxP, 'iy'], [attnMtx, 'iy']], special: BlKDepSpecial.SoftmaxAggExp },
                dimX: DimStyle.None, dimY: DimStyle.T, small: true, name: '',
            });
            let attnMtxAgg2 = mk({
                t: 'a', cx: 1, cz: B, cy: Tt, y: attn1Y, xR: aggR, zM: headZMid,
                access: acc(ht?.attnAgg, [0, 0, 0, 0], [0, 1, 0]),
                deps: { add: [[attnMtxP, 'iy'], [attnMtx, 'iy']], special: BlKDepSpecial.SoftmaxAggMax },
                dimX: DimStyle.None, dimY: DimStyle.T, small: true, name: '',
            });
            let attnMtxSmP = mk({
                t: 'i', cx: T, cz: B, cy: Tt, y: attn1Y, xR: smPR, zM: headZMid,
                access: acc(ht?.probsP, [1, 0, 0], [0, 1, 0]),
                deps: { add: [[attnMtxP, 'xy'], [attnMtxAgg1, 'iy'], [attnMtxAgg2, 'iy']], special: BlKDepSpecial.Softmax },
                dimX: DimStyle.T, dimY: DimStyle.T, special: BlkSpecial.Attention, transpose: true,
                name: 'Joint Softmax',
            });
            let attnMtxSm = mk({
                t: 'i', cx: Tt, cz: B, cy: Tt, y: attn1Y, xR: smR, zM: headZMid,
                access: acc(ht?.probsS, [1, 0, 0], [0, 1, 0]),
                deps: { add: [[attnMtx, 'xy'], [attnMtxAgg1, 'iy'], [attnMtxAgg2, 'iy']], special: BlKDepSpecial.Softmax },
                dimX: DimStyle.T, dimY: DimStyle.T, special: BlkSpecial.Attention, transpose: true,
                name: '',
            });
            let vOutBlock = mk({
                t: 'i', cx: Tt, cz: B, cy: A, y: vOutY, xR: attnLeftX, zM: headZMid,
                access: acc(ht?.attnOut, [0, 1, 0], [1, 0, 0]),
                deps: { dot: [[grp.vBlock, 'iy'], [attnMtxSm, 'ix']], dotLen: Tt },
                dimX: DimStyle.T, dimY: DimStyle.A, name: 'V Output',
            });

            let ownCubes = [qWeightBlock, qBiasBlock, qBlock, attnMtxP, attnMtx, attnMtxAgg1, attnMtxAgg2,
                attnMtxSmP, attnMtxSm, vOutBlock];
            let headCubes = [...ownCubes, ...grp.cubes];
            let headLabel = mkLabel(headCubes);
            let qLabel = mkLabel([qWeightBlock, qBiasBlock, qBlock]);
            let kLabel = mkLabel([grp.kWeightBlock, grp.kBiasBlock, grp.kBlock]);
            let vLabel = mkLabel([grp.vWeightBlock, grp.vBiasBlock, grp.vBlock]);
            let biasLabel = mkLabel([qBiasBlock, grp.kBiasBlock, grp.vBiasBlock]);
            let mtxLabel = mkLabel([attnMtxP, attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSmP, attnMtxSm]);
            let vectorLabel = mkLabel([vOutBlock]);

            heads.push({
                kvGroup: g,
                qWeightBlock, kWeightBlock: grp.kWeightBlock, vWeightBlock: grp.vWeightBlock,
                qBiasBlock, kBiasBlock: grp.kBiasBlock, vBiasBlock: grp.vBiasBlock,
                qBlock, kBlock: grp.kBlock, vBlock: grp.vBlock,
                attnMtxP, attnMtx, attnMtxAgg1, attnMtxAgg2, attnMtxSmP, attnMtxSm, vOutBlock,
                trunkK: tg?.kBlock ?? null, trunkV: tg?.vBlock ?? null,
                qLabel, kLabel, vLabel, biasLabel, mtxLabel, vectorLabel, headLabel,
                cubes: headCubes,
                ownCubes,
                labels: [qLabel, kLabel, vLabel, biasLabel, mtxLabel, vectorLabel, headLabel],
            });
        }
        // the softmax'd prefix part also feeds V Output (with the trunk's cached V): visual dep only
        let attnCubes = [...groups.flatMap(g => g.cubes), ...heads.flatMap(h => h.ownCubes)];

        let vOutCombined = mk({
            t: 'i', cx: Tt, cz: B, cy: C, y: vOutY,
            xR: attnLeftX, zF: -groupWidth * nKvE / 2,
            dimX: DimStyle.T, dimY: DimStyle.C, hidden: true, name: 'V Output Combined',
        });
        let vFinalZ = Math.max(vOutY + A * cell + 2 * margin, y + C * cell + margin);

        let projWeight = mk({
            t: 'w', cx: C, cz: 1, cy: C, y: vFinalZ, xR: qkvValLeftX, zM: 0,
            access: acc(el?.oWeight, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.C, dimY: DimStyle.C, name: 'o_proj Weights',
        });
        let attnOut = mk({
            t: 'i', cx: Tt, cz: B, cy: C, y: vFinalZ, xR: attnLeftX, zM: 0,
            access: acc(el?.oOut, [0, 1, 0], [1, 0, 0]),
            deps: {
                dot: [[projWeight, 'iy'], [vOutCombined, 'xi']], dotLen: C,
                add: heads.map(h => [h.vOutBlock, 'xi'] as [IBlkDef, string]),
            },
            dimX: DimStyle.T, dimY: DimStyle.C, name: 'Attention Output',
        });
        let attnResidual = mk({
            t: 'i', cx: Tt, cz: B, cy: C, y: vFinalZ, xM: resMid, zM: 0,
            access: acc(el?.attnResid, [0, 1, 0], [1, 0, 0]),
            deps: { add: [[attnOut, 'xy'], [src, 'xy']] },
            dimX: DimStyle.T, dimY: DimStyle.C, name: 'Attention Residual',
        });

        y = vFinalZ + C * cell + 3 * margin;

        let ln2 = createLnAt(eLeftX - adaRoom, Tt, attnResidual, el?.ln_2, { tex: adaTex, row0: V.ada_rows, name: adaName(2), rows: V.ada_rows });

        let mlpFcWeight = mk({
            t: 'w', cx: F, cz: 1, cy: C, y: y, xR: attnLeftX, zM: 0,
            access: acc(el?.gateWeight, [0, 1, 0], [1, 0, 0]),
            dimX: DimStyle.C4, dimY: DimStyle.C, name: 'gate_proj Weights',
        });
        let mlpUpWeight = mk({
            t: 'w', cx: F, cz: 1, cy: C, y: y, xR: attnLeftX - F * cell - margin, zM: 0,
            access: acc(el?.upWeight, [0, 1, 0], [1, 0, 0]),
            dimX: DimStyle.C4, dimY: DimStyle.C, name: 'up_proj Weights',
        });

        y += C * cell + margin;

        let mlpFc = mk({
            t: 'i', cx: F, cz: B, cy: Tt, y: y, xR: attnLeftX, zM: 0,
            access: acc(el?.gate, [1, 0, 0], [0, 1, 0]),
            deps: { dot: [[mlpFcWeight, 'xi'], [ln2.lnResid, 'yi']], dotLen: C },
            dimX: DimStyle.C4, dimY: DimStyle.T, name: 'gate', transpose: true,
        });
        let mlpUp = mk({
            t: 'i', cx: F, cz: B, cy: Tt, y: y, xR: attnLeftX - F * cell - margin, zM: 0,
            access: acc(el?.up, [1, 0, 0], [0, 1, 0]),
            deps: { dot: [[mlpUpWeight, 'xi'], [ln2.lnResid, 'yi']], dotLen: C },
            dimX: DimStyle.C4, dimY: DimStyle.T, name: 'up', transpose: true,
        });

        y += T * cell + margin;

        let mlpAct = mk({
            t: 'i', cx: F, cz: B, cy: Tt, y: y, xR: attnLeftX, zM: 0,
            access: acc(el?.act, [1, 0, 0], [0, 1, 0]),
            deps: { add: [[mlpFc, 'xy'], [mlpUp, 'xy']], special: BlKDepSpecial.SwiGlu },
            dimX: DimStyle.C4, dimY: DimStyle.T, name: gluName(V.expert_mlp), transpose: true,
        });

        y += T * cell + margin;

        let mlpProjWeight = mk({
            t: 'w', cx: F, cz: 1, cy: C, y: y, xR: attnLeftX, zM: 0,
            access: acc(el?.downWeight, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.C4, dimY: DimStyle.C, name: 'down_proj Weights',
        });
        let mlpResult = mk({
            t: 'i', cx: Tt, cz: B, cy: C, y: y, xL: attnLeftX + margin, zM: 0,
            access: acc(el?.mlpOut, [0, 1, 0], [1, 0, 0]),
            deps: { dot: [[mlpProjWeight, 'iy'], [mlpAct, 'ix']], dotLen: F },
            dimX: DimStyle.T, dimY: DimStyle.C, name: 'MLP Result',
        });
        let mlpResidual = mk({
            t: 'i', cx: Tt, cz: B, cy: C, y: y, xM: resMid, zM: 0,
            access: acc(el?.mlpResid, [0, 1, 0], [1, 0, 0]),
            deps: { add: [[mlpResult, 'xy'], [attnResidual, 'xy']] },
            dimX: DimStyle.T, dimY: DimStyle.C, name: `${ditName} ${li} Out`,
        });

        y += C * cell - margin;

        let mlpCubes = [...ln2.cubes, mlpFcWeight, mlpUpWeight, mlpFc, mlpUp, mlpAct, mlpProjWeight, mlpResult, mlpResidual];
        let projCubes = [projWeight, attnOut, attnResidual];
        let blockCubes = [...ln1.cubes, ...attnCubes, ...projCubes, ...mlpCubes];

        let transformerLabel = mkLabel(blockCubes);
        let selfAttendLabel = mkLabel([...ln1.cubes, ...attnCubes, ...projCubes]);
        let projLabel = mkLabel(projCubes);
        let mlpLabel = mkLabel(mlpCubes);

        return {
            layer: li,
            ln1, heads, kvGroups: groups,
            ada: ln1.ada!, ada2: ln2.ada!,
            projWeight, attnOut, attnResidual, ln2,
            mlpFcWeight, mlpUpWeight, mlpFc, mlpUp, mlpAct, mlpProjWeight, mlpResult, mlpResidual,
            transformerLabel, selfAttendLabel, projLabel, mlpLabel,
            labels: [transformerLabel, projLabel, selfAttendLabel, mlpLabel, ...heads.flatMap(h => h.labels)],
            cubes: blockCubes,
        };
    }

    let blockHalfMargin = 2 * margin;

    y += blockHalfMargin;

    // one column of 28 layer pairs: trunk layer i and expert layer i side by side, so the
    // overview reads as two towers paired level by level (close-ups come from the camera)
    let blocks: ReturnType<typeof createLayer>[] = [];
    let blockSrc = residual0;

    // the expert tower stands right of the trunk tower, its layer i level with trunk layer i
    // (they meet in layer i's joint attention). Its x offset: a dry-run layer measures its left extent.
    let yProbe = y;
    let probe = createExpertLayer(0, residual0, undefined, undefined, -1, null);
    y = yProbe;
    let eDX = T * cell / 2 + 4 * margin - Math.min(...probe.cubes.map(c => bx(c)));

    // the DiT's suffix input (17 tokens: state token + 16 noisy action tokens), level with the
    // prefix's input embedding, at the head of the expert's residual stream
    let expertIn = mk({
        t: 'i', cx: nSuffix, cy: C, cz: B, y: by(residual0),
        xM: eDX, zM: 0,
        access: acc(m?.expertIn, [0, 1, 0], [1, 0, 0]),
        dimX: DimStyle.T, dimY: DimStyle.C,
        name: V.state_token ? 'DiT Suffix In' : `${ditName} Suffix In`,
    });

    // ---- flow-matching suffix inputs (model._suffix_embeds), all real tensors ----
    // token 0: state_proj(state75); tokens 1..16: time_mlp_out(silu(time_mlp_in(cat[action_in_proj(x_t), te(t)])))
    // The x_t chain runs right → left into expertIn's right face; the state chain sits above it.
    let fmM = m?.fm;
    let nState = fmM?.state75.meta.fullShape[0] ?? 75;
    let fmY = by(expertIn);
    let fmX = bxR(expertIn) + margin;
    let chainX = fmX;
    let nextX = (cx: number) => { let x = chainX; chainX += cx * cell + margin; return x; };
    // a block of the other chain variant: never drawn
    let absentBlk = (name: string) => { let b = mk({ t: 'i', cx: 1, cy: 1, cz: B, y: fmY, xL: fmX, zM: 0,
        dimX: DimStyle.None, dimY: DimStyle.None, hidden: true, name }); b.absent = true; return b; };

    // k005b: suffix = [state_proj(state75) | time_mlp_out(silu(time_mlp_in([action_in_proj(x_t), te])))]
    function fmChainTokens() {
        let fmActionTime = mk({
            t: 'i', cx: actH, cy: C, cz: B, y: fmY, xL: nextX(actH), zM: 0,
            access: acc(fmM?.actionTime, [0, 1, 0], [1, 0, 0]),
            dimX: DimStyle.T, dimY: DimStyle.C, name: 'time_mlp_out -> tokens 1..16',
        });
        let fmTimeOutW = mk({
            t: 'w', cx: C, cy: C, cz: 1, y: fmY, xL: nextX(C), zM: 0,
            access: acc(fmM?.timeOutW, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.C, dimY: DimStyle.C, name: 'action_time_mlp_out W',
        });
        let fmTimeHidden = mk({
            t: 'i', cx: actH, cy: C, cz: B, y: fmY, xL: nextX(actH), zM: 0,
            access: acc(fmM?.timeHidden, [0, 1, 0], [1, 0, 0]),
            dimX: DimStyle.T, dimY: DimStyle.C, name: 'SiLU(time_mlp_in)',
        });
        let fmTimeInW = mk({
            t: 'w', cx: C, cy: C, cz: 1, y: fmY, xL: nextX(C), zM: 0,
            access: acc(fmM?.timeInW, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.C, dimY: DimStyle.C, name: 'action_time_mlp_in W',
        });
        let fmActionEmb = mk({
            t: 'i', cx: actH, cy: C, cz: B, y: fmY, xL: nextX(actH), zM: 0,
            access: acc(fmM?.actionEmb, [0, 1, 0], [1, 0, 0]),
            dimX: DimStyle.T, dimY: DimStyle.C, name: 'action_in_proj(x_t)',
        });
        let fmActionInW = mk({
            t: 'w', cx: actD, cy: C, cz: 1, y: fmY, xL: nextX(actD), zM: 0,
            access: acc(fmM?.actionInW, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.None, dimY: DimStyle.C, name: 'action_in_proj W',
        });
        let fmX1 = mk({
            t: 'i', cx: actH, cy: actD, cz: B, y: fmY, xL: nextX(actH), zM: 0,
            access: acc(m?.actionX1, [0, 1, 0], [1, 0, 0]),
            dimX: DimStyle.T, dimY: DimStyle.None, name: 'x_t (t=1: noise x1)',
        });
        let fmTimeEmb = mk({
            t: 'i', cx: C, cy: 1, cz: B, y: fmY - margin - cell, xL: bx(fmTimeInW), zM: 0,
            access: acc(fmM?.timeEmb, [0, 0, 0], [1, 0, 0]),
            dimX: DimStyle.C, dimY: DimStyle.None, name: 'te = sincos(t)',
        });
        let fmStateEmb = mk({
            t: 'i', cx: 1, cy: C, cz: B, y: fmY - margin - C * cell, xL: bx(expertIn), zM: 0,
            access: acc(fmM?.stateEmb, [0, 0, 0], [0, 1, 0]),
            dimX: DimStyle.None, dimY: DimStyle.C, name: 'state_proj -> token 0',
        });
        let fmStateProjW = mk({
            t: 'w', cx: nState, cy: C, cz: 1, y: by(fmStateEmb), xL: bx(expertIn) + cell + margin, zM: 0,
            access: acc(fmM?.stateProjW, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.None, dimY: DimStyle.C, name: 'state_proj W',
        });
        let fmState75 = mk({
            t: 'i', cx: nState, cy: 1, cz: B, y: by(fmStateEmb) - margin - cell, xL: bx(fmStateProjW), zM: 0,
            access: acc(fmM?.state75, [0, 0, 0], [1, 0, 0]),
            dimX: DimStyle.None, dimY: DimStyle.None, name: 'state75 (normalized)',
        });
        let fmCond = absentBlk('cond');
        return { fmState75, fmStateProjW, fmStateEmb, fmX1, fmActionInW, fmActionEmb, fmTimeEmb,
            fmTimeInW, fmTimeHidden, fmTimeOutW, fmActionTime, fmCond };
    }

    // pi0: the action tokens are action_in_proj(x_t) alone; t enters only through the AdaNorms.
    // x_t chain on the expertIn row (x_t → action_in_proj → expertIn); the time MLP on the row
    // above (te → time_mlp_in → swish → time_mlp_out → swish = cond, the te(t) bus source)
    function fmChainCond() {
        let fmActionInW = mk({
            t: 'w', cx: actD, cy: C, cz: 1, y: fmY, xL: nextX(actD), zM: 0,
            access: acc(m?.fm.actionInW, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.None, dimY: DimStyle.C, name: 'action_in_proj W',
        });
        let fmX1 = mk({
            t: 'i', cx: actH, cy: actD, cz: B, y: fmY, xL: nextX(actH), zM: 0,
            access: acc(m?.actionX1, [0, 1, 0], [1, 0, 0]),
            dimX: DimStyle.T, dimY: DimStyle.None, name: 'x_t (t=1: noise x1)',
        });
        let tY = fmY - 2 * margin - C * cell;
        chainX = fmX;
        let fmCond = mk({
            t: 'i', cx: 1, cy: C, cz: B, y: tY, xL: nextX(1), zM: 0,
            access: acc(m?.fm.cond, [0, 1, 0], [0, 0, 0]),
            dimX: DimStyle.None, dimY: DimStyle.C, name: 'cond = swish(time_mlp_out)',
        });
        let fmTimeOutW = mk({
            t: 'w', cx: C, cy: C, cz: 1, y: tY, xL: nextX(C), zM: 0,
            access: acc(m?.fm.timeOutW, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.C, dimY: DimStyle.C, name: 'time_mlp_out W',
        });
        let fmTimeHidden = mk({
            t: 'i', cx: 1, cy: C, cz: B, y: tY, xL: nextX(1), zM: 0,
            access: acc(m?.fm.timeHidden, [0, 1, 0], [0, 0, 0]),
            dimX: DimStyle.None, dimY: DimStyle.C, name: 'swish(time_mlp_in)',
        });
        let fmTimeInW = mk({
            t: 'w', cx: C, cy: C, cz: 1, y: tY, xL: nextX(C), zM: 0,
            access: acc(m?.fm.timeInW, [1, 0, 0], [0, 1, 0]),
            dimX: DimStyle.C, dimY: DimStyle.C, name: 'time_mlp_in W',
        });
        let fmTimeEmb = mk({
            t: 'i', cx: C, cy: 1, cz: B, y: tY - margin - cell, xL: bx(fmTimeInW), zM: 0,
            access: acc(m?.fm.timeEmb, [0, 0, 0], [1, 0, 0]),
            dimX: DimStyle.C, dimY: DimStyle.None, name: 'te = sincos(t)',
        });
        return { fmState75: absentBlk('state'), fmStateProjW: absentBlk('state_proj'), fmStateEmb: absentBlk('state token'),
            fmX1, fmActionInW, fmActionEmb: absentBlk('action emb'), fmTimeEmb, fmTimeInW, fmTimeHidden, fmTimeOutW,
            fmActionTime: absentBlk('action time'), fmCond };
    }

    let { fmState75, fmStateProjW, fmStateEmb, fmX1, fmActionInW, fmActionEmb, fmTimeEmb, fmTimeInW, fmTimeHidden,
        fmTimeOutW, fmActionTime, fmCond } = V.time_into_tokens ? fmChainTokens() : fmChainCond();
    let fmCubes = [fmState75, fmStateProjW, fmStateEmb, fmX1, fmActionInW, fmActionEmb, fmTimeEmb,
        fmTimeInW, fmTimeHidden, fmTimeOutW, fmActionTime, fmCond];
    // top of the input cluster: the te(t) bus runs across just above it
    let fmTopY = Math.min(...fmCubes.filter(c => !c.absent).map(c => by(c)));

    // ---- the Euler trajectory strip (decode only): x(t=1.0) … x(t=0.1), x(t=0), above the
    // x_t input. Board k feeds pass k; pass k's v writes board k+1 = board k − v/K.
    let fmS = m?.fmSteps;
    let nEuler = fmS?.K ?? 10;
    let stripY = by(fmTimeEmb) - margin - actD * cell;
    let strip: IBlkDef[] = [];
    for (let k = 0; k <= nEuler; k++) {
        let tex = k === 0 ? fmS?.steps[0]?.tex['x'] : fmS?.steps[k - 1]?.tex['x_next'];
        let tk = 1 - k / nEuler;
        strip.push(mk({
            t: 'i', cx: actH, cy: actD, cz: B, y: stripY, xL: bx(fmX1) + k * (actH * cell + margin), zM: 0,
            access: acc(tex, [0, 1, 0], [1, 0, 0]),
            dimX: DimStyle.T, dimY: DimStyle.None, hidden: true,
            name: k === nEuler ? `x(t=0) = action (K=${nEuler})` : `x(t=${tk.toFixed(1)})`,
        }));
    }

    let expertLayers: ReturnType<typeof createExpertLayer>[] = [];
    let expertSrc = expertIn;
    for (let i = 0; i < nBlocks; i++) {
        y += blockHalfMargin;
        let y0 = y;
        let block = createLayer(blockSrc, m?.blocks[i]);
        let y1 = y;
        y = y0;
        let ex = createExpertLayer(resMidX + eDX, expertSrc, m?.expertLayers[i], fmM?.ada[i], i, block);
        y = Math.max(y, y1);
        blocks.push(block);
        expertLayers.push(ex);
        blockSrc = block.mlpResidual;
        expertSrc = ex.mlpResidual;
        y += blockHalfMargin;
    }

    y += blockHalfMargin;
    let yF = y;
    let ln_f = createLn(0, blockSrc, m?.ln_f);
    let yF1 = y;
    y = yF;
    // the DiT's final RMSNorm (plain, not t-modulated), level with the trunk's ln_f
    let expertNormX = resMidX + eDX - nSuffix * cell / 2 - margin;
    let expertNorm = V.expert_final_ada
        ? createLnAt(expertNormX - adaRoom, nSuffix, expertSrc, m?.expertNorm,
            { tex: m?.expertFinalAda, row0: 0, name: 'scale shift (t)', rows: 2 })
        : createLnAt(expertNormX, nSuffix, expertSrc, m?.expertNorm);
    expertNorm.lnResid.name = `${ditName} final ${V.expert_final_ada ? 'AdaRMSNorm' : 'RMSNorm'}`;
    let expertOut = expertNorm.lnResid;
    y = Math.max(y, yF1);

    cubes.push(...ln_f.cubes);

    y += C * cell + margin;

    // ---- the heads: h* = ln_f out at the boundary token (the last token of the window) ----
    let hR = bxR(ln_f.lnResid);
    let hStar = mk({
        t: 'i', cx: 1, cy: C, cz: B, y: y,
        xR: hR, zM: 0,
        access: acc(m?.hStar, [0, 0, 0], [0, 1, 0]),
        deps: { add: [[ln_f.lnResid, '0y']] },
        dimX: DimStyle.None, dimY: DimStyle.C,
        name: 'h* (boundary state)',
    });

    let knackY = y + C * cell + margin;

    let lmHeadWeight = mk({
        t: 'w', cx: C, cy: nKnack, cz: 1, y: knackY,
        xR: hR - cell - margin, zM: 0,
        access: acc(m?.knackWeight, [1, 0, 0], [0, 1, 0]),
        dimX: DimStyle.C, dimY: DimStyle.n_vocab,
        name: 'knack_head Weights',
    });

    let logits = mk({
        t: 'i', cx: 1, cy: nKnack, cz: B, y: knackY,
        xR: hR, zM: 0,
        access: acc(m?.knackLogits, [0, 0, 0], [0, 1, 0]),
        deps: { dot: [[lmHeadWeight, 'iy'], [hStar, '0i']], dotLen: C },
        dimX: DimStyle.None, dimY: DimStyle.n_vocab,
        name: 'Knack Logits',
    });

    let logitsAgg2 = mk({
        t: 'a', cx: 1, cy: 1, cz: B, y: knackY + nKnack * cell + margin,
        xR: hR, zM: 0,
        access: acc(m?.knackAgg, [0, 0, 0], [0, 0, 0, 0]),
        deps: { add: [[logits, 'xi']], special: BlKDepSpecial.SoftmaxAggMax },
        dimX: DimStyle.None, dimY: DimStyle.None,
        name: 'Softmax Agg',
    });

    let logitsAgg1 = mk({
        t: 'a', cx: 1, cy: 1, cz: B, y: knackY + nKnack * cell + margin + cell,
        xR: hR, zM: 0,
        access: acc(m?.knackAgg, [0, 0, 0], [0, 0, 0, 1]),
        deps: { add: [[logits, 'xi'], [logitsAgg2, 'x0']], special: BlKDepSpecial.SoftmaxAggExp },
        dimX: DimStyle.None, dimY: DimStyle.None,
        name: '',
    });

    let probsY = knackY + nKnack * cell + 2 * cell + 2 * margin;

    let logitsSoftmax = mk({
        t: 'i', cx: 1, cy: nKnack, cz: B, y: probsY,
        xR: hR, zM: 0,
        access: acc(m?.knackProbs, [0, 0, 0], [0, 1, 0], 1.0),
        deps: { add: [[logits, 'xy'], [logitsAgg1, 'x0'], [logitsAgg2, 'x0']], special: BlKDepSpecial.Softmax },
        dimX: DimStyle.None, dimY: DimStyle.n_vocab,
        name: 'Knack Softmax',
    });

    // done head on h*'s row, LEFT of h* (the right side belongs to the DiT's action group)
    let doneR = hR - cell - margin;
    let doneWeight = mk({
        t: 'w', cx: C, cy: nDone, cz: 1, y: y,
        xR: doneR - cell - margin, zM: 0,
        access: acc(m?.doneWeight, [1, 0, 0], [0, 1, 0]),
        dimX: DimStyle.C, dimY: DimStyle.None,
        name: 'done_head Weights',
    });
    let doneLogits = mk({
        t: 'i', cx: 1, cy: nDone, cz: B, y: y,
        xR: doneR, zM: 0,
        access: acc(m?.doneLogits, [0, 0, 0], [0, 1, 0]),
        deps: { dot: [[doneWeight, 'iy'], [hStar, '0i']], dotLen: C },
        dimX: DimStyle.None, dimY: DimStyle.None,
        name: 'Done Logits',
    });
    let doneAgg2 = mk({
        t: 'a', cx: 1, cy: 1, cz: B, y: y + nDone * cell + margin,
        xR: doneR, zM: 0,
        access: acc(m?.doneAgg, [0, 0, 0], [0, 0, 0, 0]),
        deps: { add: [[doneLogits, 'xi']], special: BlKDepSpecial.SoftmaxAggMax },
        dimX: DimStyle.None, dimY: DimStyle.None,
        name: '',
    });
    let doneAgg1 = mk({
        t: 'a', cx: 1, cy: 1, cz: B, y: y + nDone * cell + margin + cell,
        xR: doneR, zM: 0,
        access: acc(m?.doneAgg, [0, 0, 0], [0, 0, 0, 1]),
        deps: { add: [[doneLogits, 'xi'], [doneAgg2, 'x0']], special: BlKDepSpecial.SoftmaxAggExp },
        dimX: DimStyle.None, dimY: DimStyle.None,
        name: '',
    });
    let doneProbs = mk({
        t: 'i', cx: 1, cy: nDone, cz: B, y: y + nDone * cell + 2 * cell + 2 * margin,
        xR: doneR, zM: 0,
        access: acc(m?.doneProbs, [0, 0, 0], [0, 1, 0], 1.0),
        deps: { add: [[doneLogits, 'xy'], [doneAgg1, 'x0'], [doneAgg2, 'x0']], special: BlKDepSpecial.Softmax },
        dimX: DimStyle.None, dimY: DimStyle.None,
        name: 'Done Softmax',
    });

    // DiT exit, straight under its final norm: drop token 0, action_out_proj → v (16 × 23);
    // the one-step decode x1 − v below it. In the K-step decode v instead feeds the strip.
    let ax = bx(expertOut) + cell;
    let actionVelocity = mk({
        t: 'i', cx: actH, cy: actD, cz: B, y: y,
        xL: ax, zM: 0,
        access: acc(m?.actionVelocity, [0, 1, 0], [1, 0, 0]),
        dimX: DimStyle.T, dimY: DimStyle.None,
        name: 'Velocity v',
    });
    let actionWeight = mk({
        t: 'w', cx: C, cy: actD, cz: 1, y: y,
        xL: ax + actH * cell + margin, zM: 0,
        access: acc(m?.actionWeight, [1, 0, 0], [0, 1, 0]),
        dimX: DimStyle.C, dimY: DimStyle.None,
        name: 'action_out_proj Weights',
    });
    (actionVelocity.deps as any) = depArgsToDeps({ dot: [[actionWeight, 'iy'], [expertOut, 'xi']], dotLen: C });
    let chunkY = y + actD * cell + margin;
    let actionChunk = mk({
        t: 'i', cx: actH, cy: actD, cz: B, y: chunkY,
        xL: ax, zM: 0,
        access: acc(m?.actionChunk, [0, 1, 0], [1, 0, 0]),
        dimX: DimStyle.T, dimY: DimStyle.None,
        name: 'Action Chunk = x1 - v (K=1)',
    });
    let actionX1 = mk({
        t: 'i', cx: actH, cy: actD, cz: B, y: chunkY,
        xL: ax + actH * cell + margin, zM: 0,
        access: acc(m?.actionX1, [0, 1, 0], [1, 0, 0]),
        dimX: DimStyle.T, dimY: DimStyle.None,
        name: 'x1 (skip: same input x1)',
    });
    actionChunk.deps = depArgsToDeps({ add: [[actionX1, 'xy'], [actionVelocity, 'xy']] });

    y = Math.max(chunkY + actD * cell + margin, probsY + nKnack * cell + margin);

    // 3D bar field of the action chunk (drawn by components/ActionViz.ts): dims along x,
    // the 16 steps receding along z, bar height = real value
    let barPitch = 4;
    let actionField = {
        x: bxR(actionWeight) + 4 * margin,
        y: actionChunk.y + actionChunk.dy,
        z: -actH * barPitch / 2,
        pitch: barPitch,
        heightScale: 12,
        nSteps: actH,
        nDims: actD,
    };

    let full = shape.full;
    let weightCount = full ? full.n_params : 0;

    let headCubes = [hStar, lmHeadWeight, logits, logitsAgg1, logitsAgg2, logitsSoftmax,
        doneWeight, doneLogits, doneAgg1, doneAgg2, doneProbs];
    let actionCubes = [actionVelocity, actionWeight, actionChunk, actionX1];
    // cube order = compute order: the prefix (trunk + heads) is complete before the suffix
    // starts, so the trunk chapters' sweeps never walk into the expert
    let expertLayerCubes = expertLayers.flatMap(e => e.cubes);
    let expertCubes = [...fmCubes, expertIn, ...expertLayerCubes, ...expertNorm.cubes];
    cubes.push(...headCubes);
    cubes.push(...expertCubes, ...actionCubes, ...strip);
    // every block that belongs to the DiT (its inputs, layers, final norm, outputs, the strip)
    let ditCubes = [...expertCubes, ...actionCubes, ...strip];

    for (let i = 0; i < cubes.length; i++) {
        cubes[i].idx = i;
    }

    let headsLabel = mkLabel(headCubes);
    let actionLabel = mkLabel(actionCubes);
    let expertLabel = mkLabel(expertCubes);

    return {
        cubes,
        cell,
        margin,
        idxObj,
        tokEmbedObj,
        residual0,
        ln_f,
        hStar,
        lmHeadWeight,
        logits,
        logitsAgg1,
        logitsAgg2,
        logitsSoftmax,
        doneWeight,
        doneLogits,
        doneAgg1,
        doneAgg2,
        doneProbs,
        expertOut,
        actionWeight,
        actionVelocity,
        actionX1,
        actionChunk,
        actionField,
        expertIn,
        expertLayers,
        expertNorm,
        ditCubes,
        strip,
        // set by the decode chapter: the current Euler step k (the return arrow v_k → board k+1)
        euler: null as null | { k: number, ret: number },
        fm: { state75: fmState75, stateProjW: fmStateProjW, stateEmb: fmStateEmb, x1: fmX1,
            actionInW: fmActionInW, actionEmb: fmActionEmb, timeEmb: fmTimeEmb, timeInW: fmTimeInW,
            timeHidden: fmTimeHidden, timeOutW: fmTimeOutW, actionTime: fmActionTime, cond: fmCond,
            // where the te(t) bus starts (te itself, or pi0's cond) and the height it crosses at
            busSrc: V.time_into_tokens ? fmTimeEmb : fmCond, topY: fmTopY + offset.y, cubes: fmCubes },
        variant: V,
        expertLabel,
        embedLabel,
        // tower titles over the column of layer pairs (overview steps only)
        towersLabel: mkLabel([]),
        headsLabel,
        actionLabel,
        blocks,
        height: y,
        logitsTransposed: false,
        // set by a walkthrough's data-pass animation: arrows into `active` blocks are lit,
        // arrows into `pending` (not yet computed) blocks are faded
        flow: null as null | { active: Set<IBlkDef>, pending: Set<IBlkDef> },
        model: vlaGpuModel,
        labels: [embedLabel, ...blocks.flatMap(b => b.labels), ...expertLayers.flatMap(e => e.labels), headsLabel, actionLabel, expertLabel],
        weightCount,
        shape,
        extraSources: {
            idx: vlaGpuModel?.inputTokens,
            tokEmbedOut: vlaGpuModel?.embedOut,
            posEmbedOut: undefined as IBufferTex | undefined,
        },
    };
}
