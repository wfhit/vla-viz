import { IBlkDef, IVlaModelLayout } from "../VlaModelLayout";
import { IVlaModelLink } from "../VlaModel";

// The data pass through the layer stack: one forward = for each layer pair, the trunk's
// sublayers in order (the expert's sublayers run in lock-step, sharing the attention step).
// `pos` counts stages: [-1, 0) is the entry (embeddings → layer 0), then S stages per layer.

interface IFlowStage {
    name: string;
    trunk: IBlkDef[];
    expert: IBlkDef[];
}

const flowStageNames = ['RMSNorm', 'Q K V', 'attention', 'softmax · V', 'o_proj', '+ residual',
    'RMSNorm', 'gate / up', 'SiLU · up', 'down_proj', '+ residual'];
export const nFlowStages = flowStageNames.length;

function layerStages(L: IVlaModelLayout, i: number): IFlowStage[] {
    let b = L.blocks[i];
    let e = L.expertLayers[i];
    let st = (k: number, trunk: IBlkDef[], expert: IBlkDef[]) => ({ name: k === 8 && L.variant.trunk_mlp === 'geglu'
        ? 'GELU · up' : flowStageNames[k], trunk, expert });
    let eh = e ? e.heads : [];
    return [
        st(0, b.ln1.cubes, e ? e.ln1.cubes : []),
        st(1, [...b.kvGroups.flatMap(g => [g.kWeightBlock, g.vWeightBlock, g.kBlock, g.vBlock]),
            ...b.heads.flatMap(h => [h.qWeightBlock, h.qBlock])],
            e ? [...e.kvGroups.flatMap(g => [g.kWeightBlock, g.vWeightBlock, g.kBlock, g.vBlock]),
                ...eh.flatMap(h => [h.qWeightBlock, h.qBlock])] : []),
        st(2, b.heads.flatMap(h => [h.attnMtx, h.attnMtxAgg1, h.attnMtxAgg2]),
            eh.flatMap(h => [h.attnMtxP, h.attnMtx, h.attnMtxAgg1, h.attnMtxAgg2])),
        st(3, b.heads.flatMap(h => [h.attnMtxSm, h.vOutBlock]), eh.flatMap(h => [h.attnMtxSmP, h.attnMtxSm, h.vOutBlock])),
        st(4, [b.projWeight, b.attnOut], e ? [e.projWeight, e.attnOut] : []),
        st(5, [b.attnResidual], e ? [e.attnResidual] : []),
        st(6, b.ln2.cubes, e ? e.ln2.cubes : []),
        st(7, [b.mlpFcWeight, b.mlpUpWeight, b.mlpFc, b.mlpUp], e ? [e.mlpFcWeight, e.mlpUpWeight, e.mlpFc, e.mlpUp] : []),
        st(8, [b.mlpAct], e ? [e.mlpAct] : []),
        st(9, [b.mlpProjWeight, b.mlpResult], e ? [e.mlpProjWeight, e.mlpResult] : []),
        st(10, [b.mlpResidual], e ? [e.mlpResidual] : []),
    ];
}

// everything computed after the stack: final norms, heads, expert output, action decode
function downstream(L: IVlaModelLayout): IBlkDef[] {
    return [...L.ln_f.cubes, L.hStar, L.logits, L.logitsAgg1, L.logitsAgg2, L.logitsSoftmax,
        L.doneLogits, L.doneAgg1, L.doneAgg2, L.doneProbs,
        ...L.expertNorm.cubes, L.actionVelocity, L.actionChunk];
}

function disable(b: IBlkDef) {
    if (b.t !== 'i') return;
    if (b.access) b.access.disable = true;
    b.subs?.forEach(disable);
}

// Apply the pass at `pos` to the (per-frame) layout: blocks not reached yet lose their data,
// the current stage's blocks are highlighted, and `layout.flow` tells the arrow renderer which
// arrows to light / fade. Returns the current layer and stage (stage -1 = the entry).
export function applyFlowPass(L: IVlaModelLayout, pos: number, opts: { expert?: boolean, suffixOnly?: boolean } = {}) {
    let nL = L.blocks.length;
    let total = nL * nFlowStages;
    let g = Math.floor(pos);
    let active = new Set<IBlkDef>(), pending = new Set<IBlkDef>();
    let withExpert = opts.expert ?? true;
    for (let i = 0; i < nL; i++) {
        let stages = layerStages(L, i);
        for (let s = 0; s < nFlowStages; s++) {
            let k = i * nFlowStages + s;
            // suffix-only (a decode pass): the prefix is the cached encode — only the DiT runs
            let blks = opts.suffixOnly ? stages[s].expert
                : withExpert ? [...stages[s].trunk, ...stages[s].expert] : stages[s].trunk;
            if (k === g) blks.forEach(b => active.add(b));
            else if (k > g) blks.forEach(b => pending.add(b));
        }
    }
    if (g < 0) {
        if (!opts.suffixOnly) active.add(L.residual0);
        if (withExpert) active.add(L.expertIn);
    }
    if (g < total) (opts.suffixOnly ? [...L.expertNorm.cubes, L.actionVelocity] : downstream(L)).forEach(b => pending.add(b));
    for (let b of pending) disable(b);
    for (let b of active) if (b.t === 'i') b.highlight = Math.max(b.highlight, 0.45);
    L.flow = g >= total ? null : { active, pending };
    let layer = Math.max(0, Math.min(nL - 1, Math.floor(g / nFlowStages)));
    let stage = g < 0 ? -1 : g >= total ? nFlowStages : g % nFlowStages;
    return { layer, stage };
}

// Decode pass k (0-based) of the K-step Euler loop: every DiT block that has a per-step capture
// shows step k's real tensor; the per-head blocks, norm aggregates and MLP lanes were only
// exported for pass 0, so on later passes they are greyed out instead of showing pass-0 values.
export function applyFmStep(L: IVlaModelLayout, m: IVlaModelLink | null, k: number) {
    let fs = m?.fmSteps;
    if (!fs) return false;
    let st = fs.steps[Math.max(0, Math.min(fs.steps.length - 1, k))];
    let set = (b: IBlkDef, key: string) => {
        let t = st.tex[key];
        if (t && b.access) { b.access.src = t; b.access.disable = false; }
    };
    for (let i = 0; i < L.expertLayers.length; i++) {
        let e = L.expertLayers[i];
        let per: [IBlkDef, string][] = [[e.ln1.lnResid, 'ln1'], [e.attnOut, 'o'], [e.attnResidual, 'res1'],
            [e.ln2.lnResid, 'ln2'], [e.mlpResult, 'mlp'], [e.mlpResidual, 'out']];
        for (let [b, n] of per) set(b, `E${i}.${n}`);
        set(e.ada, `E${i}.ada`);
        set(e.ada2, `E${i}.ada`);
        if (k > 0) {
            e.heads.forEach(h => [h.qBlock, h.attnMtxP, h.attnMtx, h.attnMtxAgg1, h.attnMtxAgg2,
                h.attnMtxSmP, h.attnMtxSm, h.vOutBlock].forEach(disable));
            e.kvGroups.forEach(g => { disable(g.kBlock); disable(g.vBlock); });
            [e.ln1.lnAgg1, e.ln1.lnAgg2, e.ln2.lnAgg1, e.ln2.lnAgg2, e.mlpFc, e.mlpUp, e.mlpAct].forEach(disable);
        }
    }
    set(L.expertIn, 'expert_in');
    set(L.expertOut, 'expert_out');
    if (L.expertNorm.ada) set(L.expertNorm.ada, 'expert_final_ada');
    set(L.fm.cond, 'fm_cond');
    if (k > 0) [L.expertNorm.lnAgg1, L.expertNorm.lnAgg2].forEach(disable);
    set(L.fm.x1, 'x');
    set(L.fm.actionEmb, 'fm_action_emb');
    set(L.fm.timeHidden, 'fm_time_mlp_hidden');
    set(L.fm.actionTime, 'fm_action_time');
    set(L.fm.timeEmb, 'fm_time_emb');
    set(L.actionVelocity, 'v');
    L.fm.x1.name = `x_t (t=${st.t.toFixed(1)})`;
    L.fm.timeEmb.name = `te = sincos(t=${st.t.toFixed(1)})`;
    return true;
}
