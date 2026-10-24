import { IBlkDef, IGptModelLayout } from "../GptModelLayout";

// The data pass through the layer stack: one forward = for each layer pair, the trunk's
// sublayers in order (the expert's sublayers run in lock-step, sharing the attention step).
// `pos` counts stages: [-1, 0) is the entry (embeddings → layer 0), then S stages per layer.

export interface IFlowStage {
    name: string;
    trunk: IBlkDef[];
    expert: IBlkDef[];
}

export const flowStageNames = ['RMSNorm', 'Q K V', 'attention', 'softmax · V', 'o_proj', '+ residual',
    'RMSNorm', 'gate / up', 'SiLU · up', 'down_proj', '+ residual'];
export const nFlowStages = flowStageNames.length;

export function layerStages(L: IGptModelLayout, i: number): IFlowStage[] {
    let b = L.blocks[i];
    let e = L.expertLayers[i];
    let st = (k: number, trunk: IBlkDef[], expert: IBlkDef[]) => ({ name: flowStageNames[k], trunk, expert });
    return [
        st(0, b.ln1.cubes, e ? [e.ada, e.ln1] : []),
        st(1, [...b.kvGroups.flatMap(g => [g.kWeightBlock, g.vWeightBlock, g.kBlock, g.vBlock]),
            ...b.heads.flatMap(h => [h.qWeightBlock, h.qBlock])], []),
        st(2, b.heads.flatMap(h => [h.attnMtx, h.attnMtxAgg1, h.attnMtxAgg2]), e ? [e.attn] : []),
        st(3, b.heads.flatMap(h => [h.attnMtxSm, h.vOutBlock]), []),
        st(4, [b.projWeight, b.attnOut], e ? [e.o] : []),
        st(5, [b.attnResidual], e ? [e.res1] : []),
        st(6, b.ln2.cubes, e ? [e.ada2, e.ln2] : []),
        st(7, [b.mlpFcWeight, b.mlpUpWeight, b.mlpFc, b.mlpUp], []),
        st(8, [b.mlpAct], []),
        st(9, [b.mlpProjWeight, b.mlpResult], e ? [e.mlp] : []),
        st(10, [b.mlpResidual], e ? [e.out] : []),
    ];
}

// everything computed after the stack: final norms, heads, expert output, action decode
function downstream(L: IGptModelLayout): IBlkDef[] {
    return [...L.ln_f.cubes, L.hStar, L.logits, L.logitsAgg1, L.logitsAgg2, L.logitsSoftmax,
        L.doneLogits, L.doneAgg1, L.doneAgg2, L.doneProbs,
        L.expertOut, L.actionVelocity, L.actionChunk];
}

function disable(b: IBlkDef) {
    if (b.t !== 'i') return;
    if (b.access) b.access.disable = true;
    b.subs?.forEach(disable);
}

// Apply the pass at `pos` to the (per-frame) layout: blocks not reached yet lose their data,
// the current stage's blocks are highlighted, and `layout.flow` tells the arrow renderer which
// arrows to light / fade. Returns the current layer and stage (stage -1 = the entry).
export function applyFlowPass(L: IGptModelLayout, pos: number, opts: { expert?: boolean } = {}) {
    let nL = L.blocks.length;
    let total = nL * nFlowStages;
    let g = Math.floor(pos);
    let active = new Set<IBlkDef>(), pending = new Set<IBlkDef>();
    let withExpert = opts.expert ?? true;
    for (let i = 0; i < nL; i++) {
        let stages = layerStages(L, i);
        for (let s = 0; s < nFlowStages; s++) {
            let k = i * nFlowStages + s;
            let blks = withExpert ? [...stages[s].trunk, ...stages[s].expert] : stages[s].trunk;
            if (k === g) blks.forEach(b => active.add(b));
            else if (k > g) blks.forEach(b => pending.add(b));
        }
    }
    if (g < 0) {
        active.add(L.residual0);
        if (withExpert) active.add(L.expertIn);
    }
    if (g < total) downstream(L).forEach(b => pending.add(b));
    for (let b of pending) disable(b);
    for (let b of active) if (b.t === 'i') b.highlight = Math.max(b.highlight, 0.45);
    L.flow = g >= total ? null : { active, pending };
    let layer = Math.max(0, Math.min(nL - 1, Math.floor(g / nFlowStages)));
    let stage = g < 0 ? -1 : g >= total ? nFlowStages : g % nFlowStages;
    return { layer, stage };
}
