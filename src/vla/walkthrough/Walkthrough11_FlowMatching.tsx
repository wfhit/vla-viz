import { Phase } from "./Walkthrough";
import { commentary, frameCubes, setFocus, frameTo, initialFrame, IWalkthroughArgs } from "./WalkthroughTools";
import { clamp } from "@/src/utils/data";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { IBlkDef, IVlaModelLayout } from "../VlaModelLayout";
import { describeActionRow, fmtSigned, normalizeAction, unnormalizeAction } from "../actionDims";
import { layerMass } from "../components/ActionViz";
import { IProgramState } from "../Program";
import { applyFlowPass, applyFmStep, nFlowStages } from "./FlowPass";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

// The DiT action-expert chapter group D0 … D7. The expert tower is drawn in the same visual
// language as the trunk chapters (03 RMSNorm … 08 stack): each expert layer is a full layer view
// (AdaRMSNorm → Q/K/V → joint attention → o_proj → ⊕ → AdaRMSNorm → SwiGLU → ⊕), 28 of them
// stacked. The time bus and the Euler strip are annotations on top of that language.
// Every tensor is a real k005b capture; the K-step decode (D7) runs on the step_130000 weights,
// trained at t = 1 only, so it is marked as a pre-stamp preview.

type IFrameBox = { x: number, y: number, z: number, dx: number, dy: number, dz: number };

function fieldBox(L: IVlaModelLayout): IFrameBox {
    let f = L.actionField;
    let h = f.heightScale * 2.6;
    return { x: f.x - 8, y: f.y - h - 4, z: f.z, dx: f.nDims * f.pitch + 30, dy: 2 * h + 24, dz: f.nSteps * f.pitch + 4 };
}

// fade everything except `keep` (arrows between dimmed blocks dim too)
function dimExcept(L: IVlaModelLayout, keep: IBlkDef[], level: number) {
    let k = new Set(keep);
    for (let c of L.cubes) {
        if (!k.has(c) && c.opacity > 0) c.opacity = Math.min(c.opacity, level);
    }
}

// expert blocks from `blk` on (cube order = compute order) have not been computed yet
function blankFrom(L: IVlaModelLayout, blk: IBlkDef) {
    let dit = new Set(L.ditCubes);
    let i0 = L.cubes.indexOf(blk);
    for (let i = i0; i < L.cubes.length; i++) {
        let c = L.cubes[i];
        if (c.t !== 'w' && dit.has(c) && c.access) c.access.disable = true;
    }
}

// expert layer i's sublayer chain (the head fan extends in z and is left out, as in chapter 08)
function eMain(L: IVlaModelLayout, i: number) {
    let e = L.expertLayers[i], h = e.heads[0];
    return [...e.ln1.cubes, h.qBlock, h.attnMtxP, h.attnMtx, h.attnMtxSm, e.attnOut, e.attnResidual, ...e.ln2.cubes,
        e.mlpFc, e.mlpUp, e.mlpAct, e.mlpResult, e.mlpResidual];
}

// the camera rides along a pass: frames expert layer `pos` (continuous between two layers)
function followLayer(state: IProgramState, L: IVlaModelLayout, pos: number, angleX: number, angleY: number, fill: number) {
    let nL = L.expertLayers.length;
    let i0 = clamp(Math.floor(pos), 0, nL - 1), i1 = Math.min(nL - 1, i0 + 1);
    let u = clamp(pos - i0, 0, 1);
    u = u * u * (3 - 2 * u);
    let a = frameCubes(eMain(L, i0), angleX, angleY, { fill });
    let b = frameCubes(eMain(L, i1), angleX, angleY, { fill });
    setFocus(state, eMain(L, u < 0.5 ? i0 : i1));
    state.camera.center = a.center.lerp(b.center, u);
    state.camera.angle = a.angle.lerp(b.angle, u);
    state.walkthrough.markDirty();
}

// pass progress u ∈ [0, 1] → FlowPass stage position: layer 0 gets 30% of the time
function passPos(u: number, nL: number) {
    if (u < 0.3) return u / 0.3 * nFlowStages;
    return nFlowStages + (u - 0.3) / 0.7 * (nL - 1) * nFlowStages;
}

function rms(a: Float32Array | undefined) {
    if (!a) return NaN;
    let s = 0;
    for (let v of a) s += v * v;
    return Math.sqrt(s / a.length);
}

function mse(a: Float32Array | undefined, b: Float32Array | undefined) {
    if (!a || !b) return NaN;
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return s / a.length;
}

export function walkthrough11_FlowMatching(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout: L, tools: { afterTime, breakAfter, c_blockRef, cleanup } } = args;
    let cfg = L.shape.full;
    let m = state.vlaModel;
    let viz = state.display.actionViz!;
    let fm = L.fm;
    let fmc = cfg?.fm;
    let fs = m?.fmSteps;
    let H = cfg?.action_horizon ?? 16, D = cfg?.action_dim ?? 23;
    let nPairs = cfg?.n_layer_pairs ?? 28;
    let x1 = m?.actionX1.localBuffer, vel = m?.actionVelocity.localBuffer, chunk = m?.actionChunk.localBuffer;
    let tgt = m?.fm.target.localBuffer;
    let f3 = (v: number | undefined) => v === undefined || isNaN(v) ? '…' : fmtSigned(v);
    let pct = (v: number) => (v * 100).toFixed(1) + '%';
    let el = L.expertLayers;
    let nL = el.length;
    let K = fs?.K ?? 10;
    let ckpt = fs ? `step_${fs.checkpointStep}` : 'step_130000';
    let actionBlocks = [L.expertOut, L.actionWeight, L.actionVelocity, L.actionX1, L.actionChunk];
    let fBox = fieldBox(L);
    let kvOf = (i: number) => L.blocks[i].kvGroups.flatMap(g => [g.kBlock, g.vBlock]);
    let e0 = el[0], b0 = L.blocks[0];
    let eh = e0.heads[0];
    let S = L.strip;
    let dit = L.ditCubes;
    // the field is hidden unless a phase asks for it; the Euler strip belongs to D7 only
    viz.field = 0;
    if (wt.phase !== Phase.D7_Euler) for (let b of S) b.opacity = 0;

    // real values: suffix token 1 (the first action token), window channel c
    let F = state.page.facts!;
    let pi0 = F.id === 'pi0';
    let fmVis = fm.cubes.filter(c => !c.absent);
    let el0 = m?.expertLayers[0];
    let ada0 = m?.fm.ada[0].localBuffer;
    // ada rows per norm: [γ, β] (k005b) or [scale, shift, gate] (pi0)
    let R = L.variant.ada_rows;
    let wc = ada0 ? ada0.length / (2 * R) : 64;
    let adaW = pi0 ? 'scale/shift/gate' : 'γβ';
    let at = (t: { localBuffer: Float32Array } | undefined, r: number, c: number) => t?.localBuffer[r * wc + c];
    let st = (i: number) => {
        let s = m?.fm.ada[i].meta.stats;
        return s ? `[${fmtSigned(s.min, 2)}, ${fmtSigned(s.max, 2)}]` : '…';
    };
    let xin = m?.expertIn.localBuffer[wc], o10 = at(el0?.oOut, 1, 0), r10 = at(el0?.attnResid, 1, 0);
    let mlp10 = at(el0?.mlpOut, 1, 0), out10 = at(el0?.mlpResid, 1, 0);
    let gA = ada0?.[2 * wc], gF = ada0?.[5 * wc];
    let preStamp = pi0 ? '' : `（数据说明：本组激活来自 ${ckpt} 旧权重、t = 1 的一次前向；t = 1 正是它训练时见过的分布，但仍属 pre-stamp 预览，step_160000 印章检查点落盘后会换数据。）`;
    // the trunk recedes in the one-layer chapters (it is the prefix the expert reads)
    let dimTrunk = (keep: IBlkDef[], level: number) => dimExcept(L, [...dit, ...keep], level);

    switch (wt.phase) {

    case Phase.D0_Towers: {
        let pair0 = [L.residual0, L.expertIn, ...b0.ln1.cubes, b0.attnResidual, b0.mlpResidual, ...eMain(L, 0)];
        initialFrame(state, pair0, 270, 6, { fill: 1.2 });
        if (pi0) commentary(wt, null, 0)`

这一章组讲动作分支：动作专家与流匹配（flow matching）。

π0.5 里有两个 Transformer。左边是前面几章一层层拆开过的主干 ${F.trunkName}；右边是第二座塔：${F.expertName}，一个规格更小的 Gemma（Gemma-300M 规格），用时间 t 做条件。

两座塔用同一套画法。专家塔的每一层也是：Norm → Q/K/V → 注意力 → softmax·V → o_proj → ⊕ → Norm → GeGLU MLP → ⊕。区别只有两处：Norm 换成带时间条件的${F.adaName}，两个 ⊕ 在相加前先乘一个随 t 变化的门控 gate。
`;
        else commentary(wt, null, 0)`

这一章组讲动作分支：DiT 动作专家与流匹配（flow matching）。

k005b 里有两个 Transformer。左边是前面几章一层层拆开过的主干 Qwen3-VL；右边是第二座塔：动作专家，一个用时间 t 做条件的 Transformer（DiT）。

两座塔用同一套画法。专家塔的每一层也是：RMSNorm → Q/K/V → 注意力 → softmax·V → o_proj → ⊕ → RMSNorm → SwiGLU MLP → ⊕，只是规格更小，Norm 换成带时间条件的 AdaRMSNorm。
`;
        breakAfter();
        let tPull = afterTime(null, 2.0);
        // the top layer pairs (headers join trunk Li and expert Ei); the column continues below
        let towers = [L.residual0, L.expertIn, ...L.blocks.slice(0, 3).flatMap(b => b.cubes),
            ...el.slice(0, 3).flatMap(e => e.cubes)];
        frameTo(state, tPull, towers, 280, 6, { fill: 1.0 });
        if (tPull.t > 0) {
            for (let X of [...L.blocks, ...el]) X.transformerLabel.visible = tPull.t;
            L.towersLabel.visible = tPull.t;
        }
        if (pi0) commentary(wt)`

两座塔各 ${nPairs} 层，第 i 层与第 i 层同高并排：它们在这一层共用一次联合注意力（D3），这就是耦合。

主干 ↔ 专家（真实配置）：
隐藏维 ${F.dModel} ↔ ${F.expertD}
token：前缀 ${cfg ? cfg.L_joint - cfg.n_suffix : 885} 个（图中 ${L.shape.T} 列窗口）↔ 后缀 ${F.nSuffix} 个
注意力：${F.nHead} 个 Q 头 / ${F.nKv} 组 KV ↔ ${F.expertHeads} 个 Q 头 / ${F.expertKv} 组 KV（两边都是 MQA），head_dim 都是 ${F.headDim}（联合注意力要求两边一致）
MLP：${F.ffn} ↔ ${F.expertFfn}（都是 GeGLU）
Norm：RMSNorm ↔ 门控 adaRMSNorm（scale/shift/gate 随 t 变化）
`;
        else commentary(wt)`

两座塔各 ${nPairs} 层，第 i 层与第 i 层同高并排：它们在这一层共用一次联合注意力（D3），这就是耦合。

主干 ↔ 专家（真实配置）：
隐藏维 ${cfg?.n_embd ?? 2048} ↔ ${cfg?.expert_hidden ?? 768}
token：前缀 ${cfg ? cfg.L_joint - cfg.n_suffix : 1240} 个（图中 ${L.shape.T} 列窗口）↔ 后缀 ${cfg?.n_suffix ?? 17} 个
注意力：${cfg?.n_head ?? 16} 个 Q 头 / ${cfg?.n_kv_head ?? 8} 组 KV ↔ ${cfg?.expert_heads ?? 16} 个 Q 头 / ${cfg?.expert_kv_heads ?? 2} 组 KV，head_dim 都是 ${cfg?.head_dim ?? 128}（联合注意力要求两边一致）
MLP：${cfg?.intermediate ?? 6144} ↔ ${cfg?.expert_intermediate ?? 2752}
Norm：RMSNorm ↔ AdaRMSNorm（γβ 随 t 变化）
`;
        breakAfter();
        let tBus = afterTime(null, 1.5);
        let busView = [L.expertIn, ...fmVis, ...el.slice(0, 2).flatMap(e => [e.ada, e.ada2, e.ln1.lnResid, e.attnResidual, e.mlpResidual])];
        frameTo(state, tBus, busView, 270, 6, { fill: 1.1 });
        if (pi0) commentary(wt)`

专家塔多出来的只有两样注解：
① 顶部的输入：${c_blockRef('x_t', fm.x1)} 经 ${c_blockRef('action_in_proj', fm.actionInW)} 直接成为 ${F.nSuffix} 个动作 token，进入 ${c_blockRef('Expert Suffix In', L.expertIn)}。π0.5 没有状态 token：本体状态已离散化成整数写进 prompt 文本，由主干读（见输入嵌入章）。D1 讲。
② 青色的时间总线：${c_blockRef('te = sincos(t)', fm.timeEmb)} 经 time MLP 得到 ${c_blockRef('cond', fm.cond)}，沿总线送到每层两个 Norm 旁边的 scale/shift/gate 小块（${c_blockRef('第一组', e0.ada)}、${c_blockRef('第二组', e0.ada2)}），D2 讲。

D2–D5 按主干 03–07 的节奏，把专家第 0 层逐个子层展开；D6 把 ${nPairs} 层走一遍；D7 是 ${K} 步 Euler 解码。
`;
        else commentary(wt)`

专家塔多出来的只有两样注解：
① 顶部的输入簇（${c_blockRef('state75', fm.state75)}、${c_blockRef('x_t', fm.x1)}、${c_blockRef('te = sincos(t)', fm.timeEmb)}）拼成 ${c_blockRef('DiT Suffix In', L.expertIn)}，D1 讲；
② 青色的时间总线：te 沿它送到每层两个 Norm 旁边的 γβ 小块（${c_blockRef('γ₁β₁', e0.ada)}、${c_blockRef('γ₂β₂', e0.ada2)}），D2 讲。

D2–D5 按主干 03–07 的节奏，把专家第 0 层逐个子层展开；D6 把 ${nPairs} 层走一遍；D7 是 ${K} 步 Euler 解码。
`;
        viz.activeLayer = 0;
        viz.links = 0.6;
        for (let b of [...L.blocks, ...el]) b.transformerLabel.visible = 1;
        if (tBus.active) {
            fm.timeEmb.highlight = 0.45;
            for (let e of el) { e.ada.highlight = 0.3; e.ada2.highlight = 0.3; }
        }
    } break;

    case Phase.D1_SuffixInput: {
        initialFrame(state, [...fmVis, L.expertIn], 272, 10, { fill: 1.1 });
        let s75 = m?.fm.state75.localBuffer;
        let nValid = fmc?.state_valid ?? 0;
        let te = m?.fm.timeEmb.localBuffer;
        blankFrom(L, L.expertIn);
        if (pi0) commentary(wt, null, 0)`

专家塔的输入是 ${F.nSuffix} 个后缀 token，对应主干的${c_blockRef('输入嵌入', L.residual0)}（前缀 ${cfg ? cfg.L_joint - cfg.n_suffix : 885} 个 token）。两段在联合注意力里拼成一条长 ${cfg?.L_joint ?? 917} 的序列（本样本 episode ${cfg?.sample.episode ?? '…'}，frame ${cfg?.sample.frame ?? '…'}）。

π0.5 的后缀只有动作 token。它没有状态 token：本体状态已离散化成 0–255 的整数写进 prompt 文本，由主干读。时间 t 也不拼进 token，而是走另一条路（下一段）。

第一条路：${c_blockRef('x_t', fm.x1)} 是 ${H} × ${D} 的当前动作板（t = 1 时就是纯噪声，x[0][0] = ${f3(x1?.[0])}）。前 ${F.actDReal} 维是 R1 Pro 的动作，后 ${D - F.actDReal} 维是训练时补零的占位维，输出时丢弃。每一行经 ${c_blockRef('action_in_proj', fm.actionInW)}（${D} → ${F.expertD}，带偏置）直接成为一个动作 token，进入 ${c_blockRef('Expert Suffix In', L.expertIn)}。教科书 DiT 把图像切 patch 当 token；这里每个动作步一行就是一个 token。
`;
        else commentary(wt, null, 0)`

专家塔的输入是 ${cfg?.n_suffix ?? 17} 个后缀 token，对应主干的${c_blockRef('输入嵌入', L.residual0)}（前缀 ${cfg ? cfg.L_joint - cfg.n_suffix : 1240} 个 token）。两段在联合注意力里拼成一条长 ${cfg?.L_joint ?? 1257} 的序列。后缀由三个输入拼成 ${c_blockRef('DiT Suffix In', L.expertIn)}（本样本 episode ${cfg?.sample.episode ?? '…'}，frame ${cfg?.sample.frame ?? '…'}）。

token 0：${c_blockRef('state75', fm.state75)}（bounds99 归一化的本体状态，${nValid} 个非零槽位，前几个值 ${s75 ? Array.from(s75.slice(0, 4)).map(v => fmtSigned(v)).join(' / ') : '…'}）经 ${c_blockRef('state_proj', fm.stateProjW)}（75 → 768）得到 ${c_blockRef('状态 token', fm.stateEmb)}。主干那 16 个状态 token 在联合注意力里对专家是屏蔽的，所以这是专家看到本体状态的唯一入口。
`;
        breakAfter();
        let tA = afterTime(null, 2.5);
        breakAfter();
        if (pi0) commentary(wt)`

第二条路是时间：t 先变成正弦嵌入 ${c_blockRef('te = sincos(t)', fm.timeEmb)}（te[0..3] = ${te ? Array.from(te.slice(0, 4)).map(v => fmtSigned(v)).join(' / ') : '…'}），经 ${c_blockRef('time_mlp_in', fm.timeInW)} + swish 得到 ${c_blockRef('隐层', fm.timeHidden)}，再经 ${c_blockRef('time_mlp_out', fm.timeOutW)} + swish 得到 ${c_blockRef('cond', fm.cond)}（${F.expertD} 维）。

cond 不进 token，而是沿青色时间总线送到每一层的 adaRMSNorm 去调制归一化（D2）。教科书 DiT 也是这样用条件向量调制 Norm；k005b 则把 t 拼进动作 token。
`;
        else commentary(wt)`

token 1..16：${c_blockRef('x_t', fm.x1)} 是 ${H} × ${D} 的当前动作板（t = 1 时就是纯噪声，x[0][0] = ${f3(x1?.[0])}）。每一行先经 ${c_blockRef('action_in_proj', fm.actionInW)}（23 → 768）成为 ${c_blockRef('动作嵌入', fm.actionEmb)}，再与 ${c_blockRef('te', fm.timeEmb)}（t 的正弦嵌入，te[0..3] = ${te ? Array.from(te.slice(0, 4)).map(v => fmtSigned(v)).join(' / ') : '…'}）拼成 1536 维，经 ${c_blockRef('time_mlp_in', fm.timeInW)} + SiLU 得到 ${c_blockRef('隐层', fm.timeHidden)}，再经 ${c_blockRef('time_mlp_out', fm.timeOutW)} 成为 ${c_blockRef('16 个动作 token', fm.actionTime)}，从右侧进入 DiT Suffix In。

教科书 DiT 把图像切 patch 当 token；这里每个动作步一行就是一个 token。
`;
        breakAfter();
        let tB = afterTime(null, 4.0);
        breakAfter();
        let tIn = afterTime(null, 1.0);
        commentary(wt)`

${c_blockRef('DiT Suffix In', L.expertIn)}（${F.nSuffix} × ${F.expertD}，图中 64 通道窗口）就是专家塔第 0 层的输入。下一章开始，按主干 03–07 章同样的画法把这一层逐个子层展开。
${preStamp}
`;
        let chainA = pi0 ? [fm.x1, fm.actionInW, L.expertIn] : [fm.state75, fm.stateProjW, fm.stateEmb];
        let chainB = pi0 ? [fm.timeEmb, fm.timeInW, fm.timeHidden, fm.timeOutW, fm.cond]
            : [fm.x1, fm.actionInW, fm.actionEmb, fm.timeEmb, fm.timeInW, fm.timeHidden, fm.timeOutW, fm.actionTime];
        let light = (chain: IBlkDef[], t: number) => {
            let k = Math.min(chain.length - 1, Math.floor(t * chain.length));
            chain[k].highlight = 0.5;
        };
        if (tA.active && tA.t < 1) light(chainA, tA.t);
        if (tB.active && tB.t < 1) light(chainB, tB.t);
        let info = startProcessBefore(state, L.expertIn);
        processUpTo(state, tIn, L.expertIn, info);
    } break;

    case Phase.D2_AdaNorm: {
        let ln = e0.ln1;
        let ada = ln.ada!;
        initialFrame(state, [L.expertIn, ...ln.cubes], 275, 8, { fill: 1.1 });
        wt.dimHighlightBlocks = [L.expertIn, ...ln.cubes];
        dimTrunk([], 0.15);
        blankFrom(L, ln.lnAgg1);
        e0.transformerLabel.visible = 1;
        // the channel where the plain RMSNorm value is largest, so the modulation is visible
        let cPick = 0;
        if (ada0 && el0) {
            let best = -1;
            for (let c = 0; c < wc; c++) {
                let pre = (el0.ln_1.output.localBuffer[wc + c] - ada0[wc + c]) / (1 + ada0[c]);
                if (Math.abs(pre) > best) { best = Math.abs(pre); cPick = c; }
            }
        }
        let g1 = ada0?.[cPick], b1 = ada0?.[wc + cPick], post1 = at(el0?.ln_1.output, 1, cPick);
        let pre1 = g1 !== undefined && b1 !== undefined && post1 !== undefined ? (post1 - b1) / (1 + g1) : undefined;
        let te = m?.fm.timeEmb.localBuffer;

        if (pi0) commentary(wt, null, 0)`

专家第 0 层的第一步和主干 03 章一样是归一化：对 ${c_blockRef('Expert Suffix In', L.expertIn)} 的每一列（每个 token）做 RMSNorm。区别：归一化之后，要用随时间 t 变化的 scale、shift 再调制一次，同时再给出一个 gate，留到残差处用。这就是_门控 adaRMSNorm_。
`;
        else commentary(wt, null, 0)`

专家第 0 层的第一步和主干 03 章一样是归一化：对 ${c_blockRef('DiT Suffix In', L.expertIn)} 的每一列（每个 token）做 RMSNorm。区别只有一处：归一化之后，还要用随时间 t 变化的一对 γβ 再调制一次。这就是 _AdaRMSNorm_。
${preStamp}
`;
        breakAfter();
        let tCam = afterTime(null, 1.0);
        frameTo(state, tCam, ln.cubes, 272, 6, { fill: 1.2 });
        breakAfter();
        commentary(wt)`

先和主干完全相同：两个聚合格是真实的 ${c_blockRef('mean(x²)', ln.lnAgg1)} 与 ${c_blockRef('rsqrt(mean(x²) + ε)', ln.lnAgg2)}，对完整的 ${cfg?.expert_hidden ?? 768} 个通道计算（图中显示 64 个）。
`;
        breakAfter();
        let tAgg = afterTime(null, 2.0);
        breakAfter();
        if (pi0) commentary(wt)`

时间条件。D1 的 ${c_blockRef('cond', fm.cond)} 沿青色时间总线送到每一层。在这一层，cond 经一个线性层（${F.expertD} → 3 × ${F.expertD}）得到 ${c_blockRef('[scale₁ | shift₁ | gate₁]', ada)}：三列，每个通道一组值（真值，与 pi0-fm.json 的 E0.ada 前三行一致）。t 是整条样本共用的，所以这三列对 ${F.nSuffix} 个 token 都一样。

教科书 DiT 的条件是 t 嵌入加类别 c 嵌入；π0.5 只有 t，任务和场景信息不走条件向量，而是由注意力读主干（D3）。
`;
        else commentary(wt)`

时间条件。t 先变成正弦嵌入 ${c_blockRef('te = sincos(t)', fm.timeEmb)}（这里 t = 1，te 前两维 = ${f3(te?.[0])}、${f3(te?.[1])}），沿青色时间总线送到每一层。在这一层，te 经线性层得到 ${c_blockRef('[γ₁ | β₁]', ada)}：两列，每个通道一对值（真值，与 k005b-fm.json 的 E0.ada 前两行一致）。

教科书 DiT 的条件是 t 嵌入加类别 c 嵌入；k005b 只有 t，任务和场景信息不走条件向量，而是由注意力读主干（D3）。
`;
        breakAfter();
        let tAda = afterTime(null, 1.5);
        breakAfter();
        if (pi0) commentary(wt)`

然后逐列算出${c_blockRef('adaRMSNorm 输出', ln.lnResid)}：
out = x / sqrt(mean(x²) + ε) · (1 + scale(t)) + shift(t)
没有单独的可学习 w：可学的部分全在那个线性层里。gate₁ 这一步不用，留到 D4 的残差。

真值核对（token 1，通道 ${cPick}）：scale₁ = ${f3(g1)}，shift₁ = ${f3(b1)}，调制前 ${f3(pre1)} → 调制后 ${f3(post1)}。
`;
        else commentary(wt)`

然后逐列算出${c_blockRef('AdaRMSNorm 输出', ln.lnResid)}：
out = (1 + γ(t)) · [x / sqrt(mean(x²) + ε) · ${c_blockRef('w', ln.lnSigma)}] + β(t)
γ、β 零初始化，所以训练初期它就是普通 RMSNorm。

真值核对（token 1，通道 ${cPick}）：γ₁ = ${f3(g1)}，β₁ = ${f3(b1)}，调制前 ${f3(pre1)} → 调制后 ${f3(post1)}。
`;
        breakAfter();
        let tNorm = afterTime(null, 4.0);
        breakAfter();
        let tCmp = afterTime(null, 1.2);
        let chartBox = { x: ada.x + ada.dx, y: ada.y - 14, z: ada.z, dx: 56, dy: ada.dy + 14, dz: 2 };
        frameTo(state, tCmp, [ln.lnResid, ada, chartBox], 270, 2, { fill: 1.15 });
        if (pi0) commentary(wt)`

${adaW} 旁的条形图钉在通道行上：token 1 的 64 个通道，灰条是调制前的 RMSNorm(x)，橙条是调制后的真实输出。

第 0 层 6 × ${F.expertD} 个 scale/shift/gate 值域 ${st(0)}，第 ${nPairs - 1} 层 ${st(nPairs - 1)}。和教科书 adaLN-Zero 对照：用 RMSNorm 而非 LayerNorm；每层 2 处 [scale, shift, gate]，gate 就是 adaLN-Zero 的 α 门控（k005b 没有这个门控）。
`;
        else commentary(wt)`

γβ 旁的条形图钉在通道行上：token 1 的 64 个通道，灰条是调制前的 RMSNorm(x)·w，橙条是调制后的真实输出。灰条普遍很短、橙条很长：t = 1 时 γ 把多数通道放大十倍以上，β 再整体平移。

第 0 层 4 × 768 组 γβ 值域 ${st(0)}，第 ${nPairs - 1} 层 ${st(nPairs - 1)}。和教科书 adaLN-Zero 的区别：用 RMSNorm 而非 LayerNorm；每层 2 处 γβ，没有 α 门控，残差直接相加。
`;
        let info = startProcessBefore(state, ln.lnAgg1);
        try {
            processUpTo(state, tAgg, ln.lnAgg2, info);
            processUpTo(state, tAda, ada, info);
            processUpTo(state, tNorm, ln.lnResid, info);
        } catch (e) { console.log(e); }
        if (tAda.active && tAda.t < 1) (pi0 ? fm.cond : fm.timeEmb).highlight = 0.45;
        if (tCmp.active) {
            viz.normCmp = tCmp.t;
            viz.normLayer = 0;
            viz.normWhich = 1;
            viz.normTok = 1;
        }
    } break;

    case Phase.D3_JointAttn: {
        let g0 = e0.kvGroups[0];
        let tK = eh.trunkK!, tV = eh.trunkV!;
        let mass = (i: number) => m ? layerMass(m, i) : [0, 0, 0, 0];
        let line = (i: number) => {
            let q = mass(i);
            return pi0 ? `第 ${i} 层：状态文本 ${pct(q[0])}，图像 ${pct(q[1])}，其余文本 ${pct(q[2])}，后缀自身 ${pct(q[3])}`
                : `第 ${i} 层：视觉 ${pct(q[1])}，文本 ${pct(q[2])}，后缀自身 ${pct(q[3])}`;
        };
        let mid = Math.floor(nPairs / 2);
        let qkv = [e0.ln1.lnResid, eh.qWeightBlock, eh.qBlock, eh.kBlock, eh.vBlock, eh.kWeightBlock, eh.vWeightBlock];
        initialFrame(state, [e0.ln1.lnResid, ...eh.cubes], 285, 12, { fill: 1.1 });
        dimTrunk([...kvOf(0)], 0.15);
        blankFrom(L, g0.kBlock);
        e0.transformerLabel.visible = 1;
        e0.selfAttendLabel.visible = 1;
        b0.transformerLabel.visible = 1;
        if (pi0) commentary(wt, null, 0)`

注意力和主干 04 章同构：归一化后的每一列（token）分别乘 Q、K、V 权重，得到每个 token 的 ${c_blockRef('Q', eh.qBlock)}、${c_blockRef('K', eh.kBlock)}、${c_blockRef('V', eh.vBlock)} 向量。

专家这边是 ${F.expertHeads} 个 Q 头，只有 ${F.expertKv} 组 K/V（MQA，与主干一样），Q/K/V 都没有偏置。head_dim = ${F.headDim}，与主干相同。图中只画 head 0：其余 ${F.expertHeads - 1} 个 head 结构相同、只是权重不同，标注 ×${F.expertHeads}，不重复铺开。
`;
        else commentary(wt, null, 0)`

注意力和主干 04 章同构：归一化后的每一列（token）分别乘 Q、K、V 权重，得到每个 token 的 ${c_blockRef('Q', eh.qBlock)}、${c_blockRef('K', eh.kBlock)}、${c_blockRef('V', eh.vBlock)} 向量。

专家这边是 ${cfg?.expert_heads ?? 16} 个 Q 头，只有 ${cfg?.expert_kv_heads ?? 2} 组 K/V（每 8 个 Q 头共用一组，GQA），Q/K/V 都带偏置（图中细条）。head_dim = ${cfg?.head_dim ?? 128}，与主干相同。图中只画 head 0 与 K/V 组 0：其余 15 个 head 结构相同、只是权重不同，标注 ×16，不重复铺开。
`;
        breakAfter();
        let tQkv = afterTime(null, 4.0);
        breakAfter();
        let tWide = afterTime(null, 1.5);
        frameTo(state, tWide, [tK, tV, eh.qBlock, eh.attnMtxP, eh.attnMtx, eh.vOutBlock], 285, 12, { fill: 1.1 });
        if (pi0) commentary(wt)`

接着打分：后缀每个 query 与所有 key 做点积。这里和主干的区别是“联合”：一行 key 分成两段。

右段 ${c_blockRef('后缀 × 后缀', eh.attnMtx)}：${F.nSuffix} 个动作 token 彼此注意（双向，没有因果掩码）。
左段 ${c_blockRef('后缀 × 前缀', eh.attnMtxP)}：专家的 Q 去读主干同一层的 ${c_blockRef('K', tK)}（图中 ${L.shape.T} 列窗口）。琥珀色线就是这条耦合：专家塔从这里拿到图像、任务和状态信息。

掩码（make_attn_mask）：后缀能看到全部 ${cfg?.suffix_visible_prefix ?? '…'} 个前缀 token（${cfg?.suffix_visible_visual ?? '…'} 个图像 token + ${cfg ? cfg.suffix_visible_prefix - cfg.suffix_visible_visual : '…'} 个文本 token，文本里含离散化的状态）。反过来，主干的 Q 看不到后缀，所以主干不受专家影响。
`;
        else commentary(wt)`

接着打分：后缀每个 query 与所有 key 做点积。这里和主干的区别是“联合”：一行 key 分成两段。

右段 ${c_blockRef('后缀 × 后缀', eh.attnMtx)}：17 个后缀 token 彼此注意（双向，没有因果掩码）。
左段 ${c_blockRef('后缀 × 前缀', eh.attnMtxP)}：专家的 Q 去读主干同一层的 ${c_blockRef('K', tK)}（图中 32 列窗口）。琥珀色线就是这条耦合：专家塔从这里拿到图像和任务信息。

掩码（build_fused_mask）决定后缀能看哪些前缀列：有效且不属于答案的 prompt 列，本样本 ${cfg?.suffix_visible_prefix ?? '…'} 列（${cfg?.suffix_visible_visual ?? '…'} 视觉 + ${cfg ? cfg.suffix_visible_prefix - cfg.suffix_visible_visual : '…'} 文本）；16 个状态列和答案列被屏蔽。反过来，主干的 Q 看不到后缀，所以主干不受专家影响。
`;
        breakAfter();
        let tClose = afterTime(null, 1.0);
        frameTo(state, tClose, [eh.qBlock, eh.kBlock, eh.attnMtxP, eh.attnMtx, eh.attnMtxSm, eh.attnMtxSmP], 275, 8, { fill: 1.15 });
        let tScore = afterTime(null, 3.0);
        breakAfter();
        commentary(wt)`

softmax 对整行 [前缀 | 后缀] 一起做：两个聚合格是整行的最大值与 exp 之和，所以 ${c_blockRef('前缀段', eh.attnMtxSmP)} 和 ${c_blockRef('后缀段', eh.attnMtxSm)} 的概率加起来才是 1。
`;
        breakAfter();
        let tSm = afterTime(null, 2.5);
        breakAfter();
        commentary(wt)`

最后乘 V：${c_blockRef('V Output', eh.vOutBlock)} = 前缀段概率 · 主干 ${c_blockRef('V', tV)}（品红色线）+ 后缀段概率 · 专家自己的 V。其余 ${e0.heads.length - 1} 个 head（×${e0.heads.length}）同样各得一个 V Output。
`;
        breakAfter();
        let tV1 = afterTime(null, 2.0);
        breakAfter();
        commentary(wt)`

耦合强度逐层不同。下面沿专家塔扫过 ${nPairs} 层，质量条钉在每层 head 的打分块旁（真值）：${line(0)}；${line(mid)}；${line(nPairs - 1)}。${pi0 ? '' : '浅层几乎只读文本，中层开始读图像，深层更多看后缀自己。'}
`;
        let tSweep = afterTime(null, 9.0);

        let info = startProcessBefore(state, g0.kBlock);
        try {
            processUpTo(state, tQkv, eh.qBlock, info);
            processUpTo(state, tScore, eh.attnMtx, info);
            processUpTo(state, tSm, eh.attnMtxSm, info);
            processUpTo(state, tV1, eh.vOutBlock, info);
        } catch (e) { console.log(e); }
        if (!tWide.active) wt.dimHighlightBlocks = [...qkv];
        viz.activeLayer = 0;
        viz.links = tWide.active ? 1 : 0.3;
        viz.quadLabel = tWide.active && !tSweep.active ? 1 : 0;
        if (tSweep.active) {
            let pos = tSweep.t * (nL - 1);
            if (tSweep.t < 1) followLayer(state, L, pos, 285, 10, 1.3);
            viz.activeLayer = tSweep.t >= 1 ? nL - 1 : Math.round(pos);
            viz.massBars = 1;
            dimTrunk(L.blocks.flatMap((_, i) => kvOf(i)), 0.15);
        }
    } break;

    case Phase.D4_Projection: {
        // one representative head (x16): the others share its slot and are not drawn
        let v0 = eh.vOutBlock;
        let outBlocks = [v0];
        let n = e0.heads.length;
        initialFrame(state, [...outBlocks, e0.projWeight, e0.attnOut], 284, 6, { fill: 1.15 });
        wt.dimHighlightBlocks = [...outBlocks, e0.projWeight, e0.attnOut];
        dimTrunk([], 0.15);
        blankFrom(L, e0.attnOut);
        e0.transformerLabel.visible = 1;
        e0.projLabel.visible = 1;
        eh.headLabel.visible = 1;
        commentary(wt, null, 0)`

和主干 06 章一样：注意力算完后，每个 head 都有自己的${c_blockRef('输出向量', outBlocks)}（图中只画 head 0，×${n}）。把 ${n} 个首尾相接（concat）：每个后缀 token 上，${n} 个长度 ${cfg?.head_dim ?? 128} 的向量拼成一个长度 ${n} × ${cfg?.head_dim ?? 128} = ${n * (cfg?.head_dim ?? 128)} 的向量。
`;
        breakAfter();
        let tFade = afterTime(null, 1.0, 0.5);
        let tStack = afterTime(null, 1.0);
        frameTo(state, tStack, [{ ...v0, z: e0.attnOut.z }, e0.projWeight, e0.attnOut], 278, 6, { fill: 1.15 });
        breakAfter();
        commentary(wt)`

与主干不同，这里拼出来的 ${n * (cfg?.head_dim ?? 128)} 维比专家的隐藏维 ${cfg?.expert_hidden ?? 768} 大：${c_blockRef('o_proj', e0.projWeight)} 把它投影回 ${cfg?.expert_hidden ?? 768} 维（图中是真实权重的 64 × 64 窗口），逐列做矩阵-向量乘法，得到${c_blockRef('注意力输出', e0.attnOut)}。
`;
        breakAfter();
        let tProc = afterTime(null, 3.0);
        breakAfter();
        if (pi0) commentary(wt)`

然后是门控残差：注意力输出先乘 D2 那组的第三列 gate₁，再逐元素加回这一层的输入 ${c_blockRef('Expert Suffix In', L.expertIn)}，得到 ${c_blockRef('残差流', e0.attnResidual)}：x + gate₁ · attn。

真值核对（token 1，通道 0）：输入 ${f3(xin)} + gate₁ ${f3(gA)} × o_proj ${f3(o10)} = 残差 ${f3(r10)}。教科书 DiT（adaLN-Zero）在这里乘的门控 α 就是它；k005b 没有 α，直接相加。
`;
        else commentary(wt)`

然后是残差：注意力输出逐元素加回这一层的输入 ${c_blockRef('DiT Suffix In', L.expertIn)}，得到 ${c_blockRef('残差流', e0.attnResidual)}。

真值核对（token 1，通道 0）：输入 ${f3(xin)} + o_proj ${f3(o10)} = 残差 ${f3(r10)}。教科书 DiT（adaLN-Zero）在这里还会乘一个门控 α；k005b 没有 α，直接相加。
`;
        breakAfter();
        let tZoom = afterTime(null, 1.0, 0.5);
        cleanup(tZoom, [tFade, tStack]);
        frameTo(state, tZoom, [e0.projWeight, e0.attnOut, e0.attnResidual, L.expertIn], 272, 6, { fill: 1.1 });
        let tRes = afterTime(null, 3.0);
        breakAfter();
        commentary(wt)`

注意力部分到此结束，接下来是这一层的后半部分：MLP。
`;
        if (tFade.active) {
            for (let h of e0.heads) for (let b of h.ownCubes) if (b !== h.vOutBlock) b.opacity = lerpSmoothstep(1, 0.15, tFade.t);
            for (let g of e0.kvGroups) for (let b of g.cubes) b.opacity = lerpSmoothstep(1, 0.15, tFade.t);
        }
        if (tStack.active) {
            v0.z = lerp(v0.z, e0.attnOut.z, tStack.t);
        }
        let info = startProcessBefore(state, e0.attnOut);
        processUpTo(state, tProc, e0.attnOut, info);
        processUpTo(state, tRes, e0.attnResidual, info);
    } break;

    case Phase.D5_Mlp: {
        let mlpSet = [...e0.ln2.cubes, e0.mlpFcWeight, e0.mlpUpWeight, e0.mlpFc, e0.mlpUp, e0.mlpAct, e0.mlpProjWeight, e0.mlpResult, e0.mlpResidual];
        initialFrame(state, [e0.attnResidual, ...mlpSet], 280, 6, { fill: 1.05 });
        wt.dimHighlightBlocks = [e0.attnResidual, ...mlpSet];
        dimTrunk([], 0.15);
        blankFrom(L, e0.ln2.lnAgg1);
        e0.transformerLabel.visible = 1;
        e0.mlpLabel.visible = 1;
        if (pi0) commentary(wt, null, 0)`

后半层和主干 07 章同一套结构：先归一化，再过 GeGLU MLP，最后门控加回残差。

第一步是第二个 ${c_blockRef('adaRMSNorm', e0.ln2.lnResid)}：同一个 cond，另一个线性层给出 ${c_blockRef('[scale₂ | shift₂ | gate₂]', e0.ada2)}（pi0-fm.json E0.ada 的后三行）。
`;
        else commentary(wt, null, 0)`

后半层和主干 07 章同一套结构：先归一化，再过 SwiGLU MLP，最后加回残差。

第一步是第二个 ${c_blockRef('AdaRMSNorm', e0.ln2.lnResid)}：同一个 te，另一组线性层给出 ${c_blockRef('[γ₂ | β₂]', e0.ada2)}（k005b-fm.json E0.ada 的后两行）。
`;
        breakAfter();
        let tLn = afterTime(null, 3.0);
        breakAfter();
        commentary(wt)`

每一列分别乘 ${c_blockRef('gate_proj', e0.mlpFcWeight)} 和 ${c_blockRef('up_proj', e0.mlpUpWeight)}：${cfg?.expert_hidden ?? 768} → ${cfg?.expert_intermediate ?? 2752}，得到 ${c_blockRef('gate', e0.mlpFc)} 和 ${c_blockRef('up', e0.mlpUp)}（图中 64 维窗口）。
`;
        breakAfter();
        let tGU = afterTime(null, 3.0);
        breakAfter();
        if (pi0) commentary(wt)`

逐元素算 ${c_blockRef('GELU(gate) · up', e0.mlpAct)}（tanh 近似的 GELU）。教科书 DiT 用 GELU 的点式 MLP，π0.5 与主干一样用 GeGLU。
`;
        else commentary(wt)`

逐元素算 ${c_blockRef('SiLU(gate) · up', e0.mlpAct)}。教科书 DiT 用 GELU 的点式 MLP，k005b 与主干一样用 SwiGLU。
`;
        breakAfter();
        let tAct = afterTime(null, 2.0);
        breakAfter();
        commentary(wt)`

${c_blockRef('down_proj', e0.mlpProjWeight)} 把 ${cfg?.expert_intermediate ?? 2752} 维投影回 ${cfg?.expert_hidden ?? 768} 维，得到 ${c_blockRef('MLP 输出', e0.mlpResult)}。
`;
        breakAfter();
        let tDown = afterTime(null, 2.5);
        breakAfter();
        if (pi0) commentary(wt)`

再门控加回残差：x + gate₂ · MLP，得到 ${c_blockRef('Expert 0 Out', e0.mlpResidual)}，它就是专家第 1 层的输入。

真值核对（token 1，通道 0）：残差 ${f3(r10)} + gate₂ ${f3(gF)} × MLP ${f3(mlp10)} = 输出 ${f3(out10)}。
`;
        else commentary(wt)`

再加回残差，得到 ${c_blockRef('DiT 0 Out', e0.mlpResidual)}，它就是专家第 1 层的输入。

真值核对（token 1，通道 0）：残差 ${f3(r10)} + MLP ${f3(mlp10)} = 输出 ${f3(out10)}。
`;
        breakAfter();
        let tRes = afterTime(null, 2.0);
        let info = startProcessBefore(state, e0.ln2.lnAgg1);
        try {
            processUpTo(state, tLn, e0.ln2.lnResid, info);
            processUpTo(state, tGU, e0.mlpUp, info);
            processUpTo(state, tAct, e0.mlpAct, info);
            processUpTo(state, tDown, e0.mlpResult, info);
            processUpTo(state, tRes, e0.mlpResidual, info);
        } catch (e) { console.log(e); }
        if (tLn.active && tLn.t < 1) (pi0 ? fm.cond : fm.timeEmb).highlight = 0.45;
    } break;

    case Phase.D6_Stack: {
        let kv = L.blocks.flatMap((_, i) => kvOf(i));
        initialFrame(state, [L.expertIn, ...eMain(L, 0)], 270, 3, { fill: 1.6 });
        dimExcept(L, [...dit, ...kv], 0.2);
        for (let e of el) e.transformerLabel.visible = 1;
        if (pi0) commentary(wt, null, 0)`

和主干 08 章一样，把真实数据推过整个 ${nL} 层的专家塔。每层同构，顺序相同：
adaRMSNorm → Q/K/V → 联合注意力 → softmax·V → o_proj → 门控⊕ → adaRMSNorm → gate/up → GELU·up → down_proj → 门控⊕。

主干在这里只剩各层的 K/V 块：掩码让前缀看不到后缀，所以主干的 K/V 与 x_t、t 无关。部署时（sample_actions）先跑一次前缀前向，把 ${nPairs} 层 K/V 缓存下来；专家每层的联合注意力读的就是同一层的缓存（琥珀/品红线）。导出器核对过：读缓存与两段联合前向得到的 v 最大差 ${fmc ? fmc.asserts['cache_vs_joint_v_maxabs(step0)']?.toExponential(1) : '…'}。
`;
        else commentary(wt, null, 0)`

和主干 08 章一样，把真实数据推过整个 ${nL} 层的专家塔。每层同构，顺序相同：
AdaRMSNorm → Q/K/V → 联合注意力 → softmax·V → o_proj → ⊕ → AdaRMSNorm → gate/up → SiLU·up → down_proj → ⊕。

主干在这里只剩各层的 K/V 块：掩码让前缀看不到后缀，所以主干的 K/V 与 x_t、t 无关，部署时先跑一次 encode_prefix 把 ${nPairs} 层 K/V 缓存下来。专家每层的联合注意力读的就是同一层的缓存（琥珀/品红线）。
`;
        breakAfter();
        let tEnter = afterTime(null, 1.5);
        let tL0 = afterTime(null, nFlowStages * 0.9);
        breakAfter();
        commentary(wt)`

第 0 层走完。接下来一层层穿过剩下的 ${nL - 1} 层，每层有自己的 ${adaW}（同一个${pi0 ? ' cond' : ' te'}，不同的线性层），值域从第 0 层的 ${st(0)} 到第 ${nPairs - 1} 层的 ${st(nPairs - 1)}。
`;
        let tRest = afterTime(null, (nL - 1) * 0.7);
        breakAfter();
        let tExit = afterTime(null, 1.5);
        frameTo(state, tExit, [el[nL - 1].mlpResidual, ...L.expertNorm.cubes, ...actionBlocks], 276, 8, { fill: 1.1 });
        if (pi0) commentary(wt)`

${nL} 层之后：${c_blockRef('final adaRMSNorm', L.expertOut)}（同样受 t 调制，只用 scale/shift）→ ${c_blockRef('action_out_proj', L.actionWeight)}（${F.expertD} → ${D}，带偏置）→ ${c_blockRef('速度场 v', L.actionVelocity)}，${H} × ${D}，例如 v[0][14] = ${f3(vel?.[14])}。

这是 t = 1 这一遍的速度。${c_blockRef('x1 − v', L.actionChunk)} 相当于把 ${K} 步并成一大步（x[0][14]：${f3(x1?.[14])} → ${f3(chunk?.[14])}），只用来示意方向；部署输出是 D7 的 ${K} 步积分。
`;
        else commentary(wt)`

${nL} 层之后：${c_blockRef('DiT final RMSNorm', L.expertOut)}（普通 RMSNorm，不受 t 调制）→ 丢掉状态 token → ${c_blockRef('action_out_proj', L.actionWeight)}（768 → 23）→ ${c_blockRef('速度场 v', L.actionVelocity)}，16 × 23，例如 v[0][14] = ${f3(vel?.[14])}。

一步解码：${c_blockRef('动作块', L.actionChunk)} = ${c_blockRef('x1', L.actionX1)} − v，例如 x1[0][14] = ${f3(x1?.[14])}，动作 = ${f3(chunk?.[14])}。多步解码见 D7。
`;
        breakAfter();
        if (pi0) commentary(wt)`

教科书 DiT ↔ π0.5 动作专家：
① patchify → 每个动作步一行一个 token，共 ${F.nSuffix} 个，没有状态 token（D1）
② 自注意力 → 联合注意力，K/V 含主干前缀（D3）
③ 条件 t + 类别 c → 只有 t：cond = time MLP(sincos t)（D1、D2）
④ LayerNorm + adaLN-Zero(γβα) → 门控 adaRMSNorm（scale/shift/gate，gate 即 α）
⑤ GELU MLP → GeGLU ${F.expertD}→${F.expertFfn}→${F.expertD}
⑥ N 块 → ${nL} 层，与主干逐层配对
⑦ final adaLN + Linear + unpatchify → final adaRMSNorm → action_out_proj → ${H}×${D}（前 ${F.actDReal} 维有效）
⑧ K 步采样 → Euler ${K} 步，x ← x − v/${K}（D7）
`;
        else commentary(wt)`

导出器核对过前缀缓存：用 ${fmc?.n_particles ?? 8} 张不同噪声各做一次前向，${nPairs} 层前缀 K/V 的最大差异 = ${fmc ? fmc.asserts.prefix_kv_max_abs_diff_across_seeds.toFixed(1) : '…'}。所以 knack / done 在多步解码中逐位不变，变的只有专家塔。

教科书 DiT ↔ k005b：
① patchify → 每个动作步一行一个 token，加 1 个 state token，共 17 个（D1）
② 自注意力 → 联合注意力，K/V 含主干前缀（D3）
③ 条件 t + 类别 c → 只有 t（D2）
④ LayerNorm + adaLN-Zero(γβα) → AdaRMSNorm（γβ 零初始化，无 α）
⑤ GELU MLP → SwiGLU ${cfg?.expert_hidden ?? 768}→${cfg?.expert_intermediate ?? 2752}→${cfg?.expert_hidden ?? 768}
⑥ N 块 → ${nL} 层，与主干逐层配对
⑦ final adaLN + Linear + unpatchify → RMSNorm → action_out_proj → 16×23
⑧ K 步采样 → Euler ${K} 步，x ← x − v/${K}（D7）
`;
        let pos: number;
        if (!tEnter.active) pos = -1;
        else if (!tL0.active) pos = -1 + tEnter.t;
        else if (!tRest.active) pos = tL0.t * nFlowStages;
        else pos = nFlowStages + tRest.t * (nL - 1) * nFlowStages;
        if (!(tRest.active && tRest.t >= 1)) {
            applyFlowPass(L, pos, { suffixOnly: true });
            if (tEnter.active) followLayer(state, L, Math.max(0, pos / nFlowStages - 0.5), 270, 3, 1.6);
            viz.activeLayer = clamp(Math.floor(Math.max(0, pos) / nFlowStages), 0, nL - 1);
            viz.links = 1;
        }
        if (tExit.active) for (let b of [L.expertOut, L.actionWeight, L.actionVelocity]) b.highlight = 0.3;
    } break;

    case Phase.D7_Euler: {
        let kv = L.blocks.flatMap((_, i) => kvOf(i));
        // the strip + the DiT input chain: where every pass starts and every return lands
        let stripView = [...S.map(b => ({ x: b.x, y: b.y, z: b.z, dx: b.dx, dy: b.dy, dz: b.dz })), fm.x1];
        initialFrame(state, stripView, 272, 10, { fill: 1.2 });
        dimExcept(L, [...dit, ...kv], 0.2);
        let tgtArr = tgt;
        let stepInfo = (k: number) => {
            let st = fs?.steps[k];
            if (!st) return '…';
            let v = st.tex['v']?.localBuffer, xn = st.tex['x_next']?.localBuffer;
            let base = `t = ${st.t.toFixed(1)}：|v| = ${rms(v).toFixed(2)}，x[0][14] ${f3(st.tex['x']?.localBuffer[14])} → ${f3(xn?.[14])}`;
            return pi0 ? base : `${base}，MSE(x, a) = ${mse(xn, tgtArr).toFixed(3)}`;
        };
        if (pi0) commentary(wt, null, 0)`

部署时的 flow matching 解码（openpi 的 sample_actions）：从噪声出发，专家塔被调用 ${K} 次，每次是一个 Euler 步。每次穿行就是 D6 那一遍，画法不变；Euler 轨迹带和回程箭头只是加在上面的注解。

轨迹带：${K + 1} 块 ${H} × ${D} 的板，${c_blockRef('x(t=1.0)', S[0])} 是纯噪声，${c_blockRef('x(t=0)', S[K])} 是最终动作块。每次穿行：
① 当前板 x_t 进入专家的输入链；
② 时间总线换成新的 t，${nPairs} 层的 scale/shift/gate 全部换成这个 t 的真值；
③ 专家塔从第 0 层穿到第 ${nPairs - 1} 层，联合注意力读缓存的主干 K/V；
④ 底部得到 v，黄色回程箭头把 x − v/${K} 写进下一块板。

这组权重（${cfg?.train_config ?? '…'}）正是按这条 ${K} 步路径训练的：t 从 1.0 到 0.1 都在训练分布内。
`;
        else commentary(wt, null, 0)`

部署时的 flow matching 解码（sample_action_chunk）：从噪声出发，专家塔被调用 ${K} 次，每次是一个 Euler 步。每次穿行就是 D6 那一遍，画法不变；Euler 轨迹带和回程箭头只是加在上面的注解。

轨迹带：${K + 1} 块 16 × 23 的板，${c_blockRef('x(t=1.0)', S[0])} 是纯噪声，${c_blockRef('x(t=0)', S[K])} 是最终动作块。每次穿行：
① 当前板 x_t 进入专家的输入链；
② 时间总线换成新的 t，${nPairs} 层的 γβ 全部换成这个 t 的真值；
③ 专家塔从第 0 层穿到第 ${nPairs - 1} 层，联合注意力读缓存的主干 K/V；
④ 底部得到 v，黄色回程箭头把 x − v/${K} 写进下一块板。

注意：这是 ${ckpt} 的旧权重，只在 t = 1 上训练过，t < 1 是分布外，这里是 pre-stamp 预览（轨迹带标题已标注）。
`;
        breakAfter();
        // pass k: 0..0.85 = the DiT traverse, 0.85..1 = the return v → board k+1
        let durs = Array.from({ length: K }, (_, k) => k === 0 ? 20 : k < 3 ? 7 : 2.5);
        let passes = durs.map((d, k) => {
            let t = afterTime(null, d);
            if (k === 0 || k === 2) breakAfter();
            if (k === 0) commentary(wt)`

第 1 次穿行（${stepInfo(0)}）。以后每次穿行用同一组权重，只换两样东西：输入板 x_t，和总线上的 t。接下来两次中速。
`;
            if (k === 2) commentary(wt)`

第 2、3 次：${stepInfo(1)}；${stepInfo(2)}。剩下 ${K - 3} 次快放，轨迹带上的步进格每完成一次走一格。
`;
            return t;
        });
        breakAfter();
        if (pi0) commentary(wt)`

${K} 次穿行完成，板 ${K} 就是解码出的动作块。逐步真值：
${Array.from({ length: K }, (_, k) => stepInfo(k)).filter((_, k) => k % 3 === 0 || k === K - 1).join('\n\n')}

导出器核对：每一步 x_next = x + dt · v（dt = ${fmc ? fmc.dt : '…'}）的恒等误差 ${fmc ? fmc.asserts.euler_identity_maxabs?.toExponential(1) : '…'}。
`;
        else commentary(wt)`

${K} 次穿行完成，板 ${K} 就是解码出的动作块。逐步真值：
${Array.from({ length: K }, (_, k) => stepInfo(k)).filter((_, k) => k % 3 === 0 || k === K - 1).join('\n')}

如实说明：在这组 pre-stamp 权重上，${K} 步后 MSE = ${fs ? mse(fs.steps[K - 1].tex['x_next']?.localBuffer, tgtArr).toFixed(3) : '…'}，比一步解码（x1 − v）的 ${fmc ? fmc.mse.toFixed(3) : '…'} 更差。原因正是 t < 1 分布外；带 action_decode 印章的 step_160000 才是为这条 ${K} 步路径训练的。
`;
        // which pass is running
        let k = passes.findIndex(p => p.active && p.t < 1);
        let finished = passes[K - 1].t >= 1;
        if (k < 0) k = finished ? K : 0;
        let u = k < K ? passes[k].t : 1;
        let started = passes[0].active;
        let ret = clamp((u - 0.85) / 0.15, 0, 1);
        // boards 0..k are written; board k+1 fades in with the return arrow
        for (let j = 0; j < S.length; j++) {
            S[j].opacity = j <= k ? 1 : j === k + 1 ? ret : 0;
        }
        viz.eulerPass = started ? k : -1;
        if (k < K) {
            applyFmStep(L, m, k);
            viz.massStep = k;
            if (started) {
                L.euler = { k, ret };
                let pp = passPos(clamp(u / 0.85, 0, 1), nL);
                applyFlowPass(L, pp, { suffixOnly: true });
                let li = clamp(Math.floor(pp / nFlowStages), 0, nL - 1);
                viz.activeLayer = li;
                viz.links = 1;
                if (k === 0 && u < 0.85) followLayer(state, L, Math.max(0, li - 0.5), 280, 10, 1.4);
                else if (k === 0) {
                    // glide from the last layer back out to the strip for the return arrow
                    followLayer(state, L, nL - 1, 280, 10, 1.4);
                    let w = frameCubes(stripView, 272, 10, { fill: 1.2 });
                    let s = ret * ret * (3 - 2 * ret);
                    state.camera.center = state.camera.center.lerp(w.center, s);
                    state.camera.angle = state.camera.angle.lerp(w.angle, s);
                }
            }
        } else {
            applyFmStep(L, m, K - 1);
            viz.massStep = K - 1;
            S[K].highlight = 0.4;
        }
        let kFin = fs?.steps[K - 1]?.tex['x_next']?.localBuffer;
        breakAfter();
        let tChunk = afterTime(null, 1.2);
        frameTo(state, tChunk, pi0 ? [S[K - 1], S[K]] : [L.strip[K], ...actionBlocks], 284, 12, { fill: 1.1 });
        if (pi0) {
            let rowK = cfg && kFin ? describeActionRow(cfg, kFin, 0) : [];
            commentary(wt)`

最后看动作块本身：${H} 步 × ${D} 维，所有值都在 bounds99 归一化空间：norm = (x − q01) / (q99 − q01) · 2 − 1。只有前 ${F.actDReal} 维是 R1 Pro 的动作（策略只返回前 ${F.actDReal} 维），后 ${D - F.actDReal} 维是训练时补零的占位维，直接丢弃。

${F.actDReal} 维依次是：0–2 底盘速度 base_vx / base_vy / base_yaw；3–6 躯干 trunk_1–4；7–13 左臂 7 个关节；14 左夹爪；15–21 右臂 7 个关节；22 右夹爪（openpi b1k 配置）。
`;
            breakAfter();
            commentary(wt)`

轨迹带上的板 ${K}（${c_blockRef(`x(t=0), K=${K}`, L.strip[K])}）就是部署输出。按同一组 q01 / q99 反归一化，第 0 步（括号内为机器人单位）：

${rowK.join('\n\n')}
`;
            break;
        }
        commentary(wt)`

最后看动作块本身：${H} 步 × ${D} 维，所有值都在 bounds99 归一化空间。

轨迹带上的板 ${K}（${c_blockRef(`x(t=0), K=${K}`, L.strip[K])}）是 ${K} 步解码的结果，例如第 0 步左夹爪 = ${f3(kFin?.[14])}；在本检查点（${ckpt}，只训练过 t = 1）上，与它分布内一致的是一步解码 ${c_blockRef('x1 − v', L.actionChunk)}，同一分量 = ${f3(chunk?.[14])}。下面的柱场展示一步解码（这组权重的真实部署输出）；step_160000 到位后换成 ${K} 步结果。
`;
        breakAfter();
        let tMove = afterTime(null, 1.5);
        frameTo(state, tMove, [fBox], 300, 22, { fill: 1.05 });
        let tDec = afterTime(null, 4.0);
        breakAfter();
        commentary(wt)`

柱场先摆出 x1，白色竖线是每个分量的 −v，随后所有柱子从 x1 移到 x1 − v（中间帧是直线插值，标题标了 interp only）。例如第 0 步左夹爪：x1 = ${f3(x1?.[14])}，v = ${f3(vel?.[14])}，x1 − v = ${f3(chunk?.[14])}。
`;
        breakAfter();
        let err = m?.fm.errDim;
        let order = err ? Array.from(err).map((v, d) => [v, d] as [number, number]).sort((a, b) => a[0] - b[0]) : [];
        let nm = (d: number) => cfg?.action_dim_names[d] ?? `${d}`;
        commentary(wt)`

训练怎么教会它？训练目标是速度 u = x1 − a（a 是标签动作），损失是 v 与 u 在 16 × 23 上的 MSE。

由于 x0 − a = (x1 − v) − (x1 − u) = u − v，“速度误差”和“动作误差”完全相等。场景里的线框就是标签 a，红色竖线是每个分量的误差。
`;
        breakAfter();
        let tShow = afterTime(null, 1.5);
        breakAfter();
        commentary(wt)`

如实给出本样本的数值：MSE(x1 − v, a) = ${fmc ? fmc.mse.toFixed(3) : '…'}，而直接预测全 0 的 MSE 是 ${fmc ? fmc.mse_zero.toFixed(3) : '…'}。也就是说在这一帧上，单次解码并不比“输出 0”更好，这是单个噪声样本的局限，下一节的多噪声实验会说明原因。

前排的红色短柱是逐维 MSE：最好的维是 ${order.slice(0, 2).map(([v, d]) => `${d} ${nm(d)}（${v.toFixed(2)}）`).join('、')}；最差的是 ${order.slice(-2).map(([v, d]) => `${d} ${nm(d)}（${v.toFixed(2)}）`).join('、')}。夹爪：标签 a[0][14] = ${f3(tgt?.[14])}、预测 ${f3(chunk?.[14])}；a[0][22] = ${f3(tgt?.[22])}、预测 ${f3(chunk?.[22])}。
`;
        let row0 = cfg && chunk ? describeActionRow(cfg, chunk, 0) : [];
        let zeroVel = cfg ? [0, 1, 2].map(d => `${cfg!.action_dim_names[d]} ${fmtSigned(normalizeAction(cfg!, d, 0))}`).join('，') : '…';
        breakAfter();
        let tRobot = afterTime(null, 0.01);
        commentary(wt)`

最后，动作块要落到机器人上。${D} 维依次是：0–2 底盘速度 base_vx / base_vy / base_yaw（速度量）；3–6 躯干 trunk_1–4；7–13 左臂 7 个关节；14 左夹爪；15–21 右臂 7 个关节；22 右夹爪。3–22 是位置量。夹爪 +1 张开、−1 闭合（harness 以负值判定闭合）。

模型输出在 bounds99 归一化空间：norm = (x − q01) / (q99 − q01) · 2 − 1。服务端按同一组 q01 / q99 反归一化，柱场下方的标签现在附上了第 0 步的机器人单位值。trunk_4（dim 6）在数据里恒为 0（q01 = q99 = 0），反归一化后总是 0。
`;
        breakAfter();
        commentary(wt)`

本样本第 0 步（括号内为机器人单位）：

${row0.join('\n\n')}
`;
        breakAfter();
        commentary(wt)`

settle 语义：一旦 done 为真，harness 播完当前块后改发“停住保持”块：dim 0–2 的速度在机器人单位下清零（归一化后为 ${zeroVel}，并不是 0），dim 3–22 的躯干、手臂和夹爪位置保持上一行的值。训练标签中每个 knack 结束帧之后的行也按同一规则合成。

本样本是 frame ${cfg?.sample.frame ?? '…'}，预测 done = ${String(cfg?.pred_done ?? '…')}，不处于 settle 状态，所以这是一段正常的运动块。第 0 步 base_vx 反归一化为 ${chunk && cfg ? fmtSigned(unnormalizeAction(cfg, 0, chunk[0])) : '…'}。
`;
        let P = m?.fm.particlesMass;
        // how much the suffix's visual attention varies across the K noises (layer mid)
        let spread = '…';
        if (P && cfg && fmc) {
            let K = fmc.n_particles, NLp = nPairs, NS = cfg.n_suffix, li = Math.floor(NLp / 2);
            let vis: number[] = [];
            for (let k = 0; k < K; k++) {
                let s = 0;
                for (let q = 0; q < NS; q++) s += P[((k * NLp + li) * NS + q) * 4 + 1] / NS;
                vis.push(s);
            }
            spread = `${pct(Math.min(...vis))} – ${pct(Math.max(...vis))}`;
        }
        breakAfter();
        let tPart = afterTime(null, 1.0);
        commentary(wt)`

附录：多噪声（粒子）。

同一个速度场可以搬运不同的起点。导出器用 ${fmc?.n_particles ?? 8} 张噪声（本样本自带的 x1，加上种子 ${fmc ? fmc.seeds.slice(1, 3).join('、') : '…'}…${fmc ? fmc.seeds[fmc.seeds.length - 1] : '…'}）各做了一次真实的 t = 1 一步解码。

场景切到“粒子轴”：23 根竖轴各对应一个动作维（第 0 步），每根轴上 ${fmc?.n_particles ?? 8} 个彩色粒子，从各自的 x1_k 出发，被各自的 v_k 推到 x1_k − v_k。
`;
        breakAfter();
        let tPMove = afterTime(null, 4.0);
        breakAfter();
        commentary(wt)`

起点几乎是随机散开的，终点却聚拢了：${fmc?.n_particles ?? 8} 个动作块之间的平均标准差是 ${fmc ? fmc.particle_chunk_std.toFixed(2) : '…'}，单块 MSE 在 ${fmc ? `${Math.min(...fmc.mse_per_particle).toFixed(2)} – ${Math.max(...fmc.mse_per_particle).toFixed(2)}` : '…'} 之间。黑色横标是 ${fmc?.n_particles ?? 8} 块的均值，它的 MSE 降到 ${fmc ? fmc.mse_mean_chunk.toFixed(3) : '…'}，红色横标是标签。

这说明 t = 1 的单步解码本质上是在回归“给定观测时动作的条件均值”，再叠加一部分来自噪声的随机性。单个样本的误差大，主要来自这部分随机性。
`;
        breakAfter();
        commentary(wt)`

换噪声时，主干完全不受影响：前缀 K/V 在 ${fmc?.n_particles ?? 8} 次前向中逐位相同（最大差异 ${fmc ? fmc.asserts.prefix_kv_max_abs_diff_across_seeds.toFixed(1) : '…'}）。变的只有后缀：第 ${Math.floor(nPairs / 2)} 层后缀落在视觉 token 上的注意力质量，随噪声在 ${spread} 之间变化。
`;
        if (!tMove.active) break;
        viz.field = 1;
        viz.fieldMode = 'decode';
        viz.decode = tDec.active ? tDec.t : 0;
        viz.showVelocity = tDec.t >= 1 ? 0.35 : 1;
        viz.target = tShow.active && !tRobot.active ? tShow.t : 0;
        viz.errBars = tShow.active && !tRobot.active ? tShow.t : 0;
        viz.robotUnits = tRobot.active && !tPart.active ? 1 : 0;
        if (tPart.active) {
            viz.field = 0;
            viz.particles = 1;
            viz.particleTau = tPMove.active ? tPMove.t : 0;
            viz.particleStep = 0;
        }
    } break;
    }
}
