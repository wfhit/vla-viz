import { Phase } from "./Walkthrough";
import { commentary, frameCubes, frameTo, initialFrame, IWalkthroughArgs } from "./WalkthroughTools";
import { clamp } from "@/src/utils/data";
import { IBlkDef, IGptModelLayout } from "../GptModelLayout";
import { describeActionRow, fmtSigned, normalizeAction, unnormalizeAction } from "../actionDims";
import { layerMass } from "../components/ActionViz";
import { IProgramState } from "../Program";
import { applyFlowPass, nFlowStages } from "./FlowPass";

// Flow matching chapter (docs/design/flow-matching-viz.md §5, FM-0 … FM-9). Everything in the
// scene is a real k005b tensor; the only non-model frames are the τ interpolation of FM-6/FM-8,
// which the text and the in-scene title mark as "interp only".

type IFrameBox = { x: number, y: number, z: number, dx: number, dy: number, dz: number };

function fieldBox(L: IGptModelLayout): IFrameBox {
    let f = L.actionField;
    let h = f.heightScale * 2.6;
    return { x: f.x - 8, y: f.y - h - 4, z: f.z, dx: f.nDims * f.pitch + 30, dy: 2 * h + 24, dz: f.nSteps * f.pitch + 4 };
}

// fade everything except `keep` (blocks stay visible: arrows between dimmed blocks dim too)
function dimExcept(L: IGptModelLayout, keep: IBlkDef[], level: number) {
    let k = new Set(keep);
    for (let c of L.cubes) {
        if (!k.has(c) && c.opacity > 0) c.opacity = Math.min(c.opacity, level);
    }
}

// follow a moving layer index with the camera (continuous between two layers)
function followLayer(state: IProgramState, L: IGptModelLayout, pos: number, angleX: number, angleY: number, extra: (i: number) => IBlkDef[] = () => []) {
    let nL = L.expertLayers.length;
    let i0 = clamp(Math.floor(pos), 0, nL - 1), i1 = Math.min(nL - 1, i0 + 1);
    let u = clamp(pos - i0, 0, 1);
    let cubesOf = (i: number) => [L.expertLayers[i].attn, L.expertLayers[i].out, L.expertLayers[i].ada, L.blocks[i].ln1.lnResid, ...extra(i)];
    let a = frameCubes(cubesOf(i0), angleX, angleY, { fill: 1.1 });
    let b = frameCubes(cubesOf(i1), angleX, angleY, { fill: 1.1 });
    state.camera.center = a.center.lerp(b.center, u);
    state.camera.angle = a.angle.lerp(b.angle, u);
    state.walkthrough.markDirty();
}

export function walkthrough11_FlowMatching(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout: L, tools: { afterTime, breakAfter, c_blockRef } } = args;
    let cfg = L.shape.full;
    let m = state.jsGptModel;
    let viz = state.display.actionViz!;
    let fm = L.fm;
    let fmc = cfg?.fm;
    let H = cfg?.action_horizon ?? 16, D = cfg?.action_dim ?? 23;
    let nPairs = cfg?.n_layer_pairs ?? 28;
    let x1 = m?.actionX1.localBuffer, vel = m?.actionVelocity.localBuffer, chunk = m?.actionChunk.localBuffer;
    let tgt = m?.fm.target.localBuffer;
    let f3 = (v: number | undefined) => v === undefined ? '…' : fmtSigned(v);
    let pct = (v: number) => (v * 100).toFixed(1) + '%';
    let el = L.expertLayers;
    let nL = el.length;
    let actionBlocks = [L.expertOut, L.actionWeight, L.actionVelocity, L.actionX1, L.actionChunk];
    let fBox = fieldBox(L);
    // the field is hidden unless a phase asks for it
    viz.field = 0;

    switch (wt.phase) {

    case Phase.FM_Overview: {
        let expertCol = [L.expertIn, ...fm.cubes, ...el.flatMap(e => e.cubes), L.expertOut];
        let trunkCol = [L.residual0, ...L.blocks.flatMap(b => [b.attnResidual, b.mlpResidual])];
        let top = [L.idxObj, L.residual0, L.expertIn, ...fm.cubes, ...el.slice(0, 3).flatMap(e => [e.attn, e.out]), ...L.blocks.slice(0, 3).map(b => b.mlpResidual)];
        initialFrame(state, top, 284, 14, { fill: 1.0 });
        dimExcept(L, [...expertCol, ...trunkCol, ...actionBlocks], 0.3);

        commentary(wt, null, 0)`

这一章讲动作分支：动作专家与流匹配（flow matching）。

k005b 的一次前向里有两条并行的流。左边是主干 Qwen3-VL（隐藏维 ${cfg?.n_embd ?? 2048}），负责处理图像、文本和状态 token，并给出 knack / done。右边橙色的一列是动作专家（隐藏维 ${cfg?.expert_hidden ?? 768}），负责产出动作块。两者逐层配对，一共 ${nPairs} 个层对，每一对共用一次联合注意力。

场景顶部新出现的一簇块是专家的输入：${c_blockRef('state75', fm.state75)}、${c_blockRef('噪声 x1', fm.x1)} 和 ${c_blockRef('时间嵌入 te', fm.timeEmb)}。从这里开始，每根箭头都能追溯到一个起点块和一个终点块。
`;
        breakAfter();
        commentary(wt)`

“流匹配”原本是一个多步过程：从纯噪声 x1（t = 1）出发，沿模型预测的速度场 v 积分，逐步走到动作 x0（t = 0）。π0 推理时走 10 步 Euler。

k005b 不一样：训练和部署都固定在 t = 1，只走 1 步，即 action = x1 − v。所以本章不会伪造“第 2…10 步”的中间张量。观众看到的“迭代”来自两种真实数据：

1. 深度方向：${nPairs} 个层对逐层改写 17 个后缀 token（FM-3 到 FM-5）。

2. 分布方向：用 ${fmc?.n_particles ?? 8} 个不同噪声各做一次真实前向，看同一个速度场如何搬运不同的起点（FM-8）。
`;
    } break;

    case Phase.FM_Inputs: {
        initialFrame(state, [...fm.cubes, L.expertIn], 270, 8, { fill: 1.05 });
        dimExcept(L, [...fm.cubes, L.expertIn], 0.25);
        let s75 = m?.fm.state75.localBuffer;
        let nValid = fmc?.state_valid ?? 0;
        let te = m?.fm.timeEmb.localBuffer;

        commentary(wt, null, 0)`

动作专家有三个输入，全部来自本样本（episode ${cfg?.sample.episode ?? '…'}，frame ${cfg?.sample.frame ?? '…'}）。

① ${c_blockRef('state75', fm.state75)}：机器人本体状态，经 bounds99 归一化后放进 75 个槽位，未使用的槽位补 0。本样本有 ${nValid} 个非零槽位，前几个值是 ${s75 ? Array.from(s75.slice(0, 4)).map(v => fmtSigned(v)).join(' / ') : '…'}。它是专家看到本体状态的唯一入口：主干那 16 个状态 token 在联合注意力里对专家是被屏蔽的。
`;
        breakAfter();
        let t1 = afterTime(null, 1.0);
        commentary(wt)`

② ${c_blockRef('x1', fm.x1)}：一张 ${H} × ${D} 的噪声（16 个时间步 × 23 个动作维），也就是流匹配的起点。本样本 x1[0][0] = ${f3(x1?.[0])}，x1[0][14] = ${f3(x1?.[14])}。训练时噪声按 episode / 帧取种子，所以同一帧每次训练都用同一张噪声；部署时固定 seed 0。这里显示的就是训练行里存好的那张 action_noise。
`;
        breakAfter();
        let t2 = afterTime(null, 1.0);
        commentary(wt)`

③ ${c_blockRef('te', fm.timeEmb)}：时间 t 的正弦嵌入，共 768 维（前 384 维是 sin，后 384 维是 cos，周期从 0.004 到 4）。k005b 的 t 恒为 1，所以 te 是一组常数：te[0..3] = ${te ? Array.from(te.slice(0, 4)).map(v => fmtSigned(v)).join(' / ') : '…'}。图中只画了前 64 维。
`;
        if (!t1.active) fm.state75.highlight = 0.4;
        else if (!t2.active) fm.x1.highlight = 0.4;
        else fm.timeEmb.highlight = 0.4;
    } break;

    case Phase.FM_Suffix: {
        initialFrame(state, [...fm.cubes, L.expertIn], 272, 10, { fill: 1.0 });
        dimExcept(L, [...fm.cubes, L.expertIn, el[0].ada, el[0].ada2, el[0].ln1, el[0].ln2, el[0].out], 0.25);

        commentary(wt, null, 0)`

三个输入要拼成 ${cfg?.n_suffix ?? 17} 个后缀 token，每个 token 768 维，作为 ${c_blockRef('Expert Suffix In', L.expertIn)}。

token 0：${c_blockRef('state75', fm.state75)} 经过 ${c_blockRef('state_proj', fm.stateProjW)}（75 → 768）得到 ${c_blockRef('状态 token', fm.stateEmb)}，再沿箭头落到 Expert Suffix In 的第 0 列。
`;
        breakAfter();
        let tA = afterTime(null, 2.5);
        breakAfter();
        commentary(wt)`

token 1..16：x1 的每一行（一个时间步的 23 维）先经 ${c_blockRef('action_in_proj', fm.actionInW)}（23 → 768）得到 ${c_blockRef('动作嵌入', fm.actionEmb)}；再与 te 拼成 1536 维，经 ${c_blockRef('time_mlp_in', fm.timeInW)}（1536 → 768）和 SiLU 得到 ${c_blockRef('隐层', fm.timeHidden)}；最后经 ${c_blockRef('time_mlp_out', fm.timeOutW)}（768 → 768）得到 ${c_blockRef('16 个动作 token', fm.actionTime)}，从右侧进入 Expert Suffix In。权重块只画了前 64 × 64 的真实窗口。
`;
        breakAfter();
        let tB = afterTime(null, 4.0);
        breakAfter();
        commentary(wt)`

te 还有第二条去路：它是专家每一层 AdaRMSNorm 的条件。每一层旁边有两根窄块：${c_blockRef('γ₁β₁(te)', el[0].ada)} 调制输入 AdaRMSNorm（箭头指向 ${c_blockRef('AdaRMSNorm1', el[0].ln1)}），${c_blockRef('γ₂β₂(te)', el[0].ada2)} 调制注意力后的 AdaRMSNorm（指向 ${c_blockRef('AdaRMSNorm2', el[0].ln2)}）。FM-4 会展开讲。
`;
        // light the chain in data-flow order
        let chainA = [fm.state75, fm.stateProjW, fm.stateEmb, L.expertIn];
        let chainB = [fm.x1, fm.actionInW, fm.actionEmb, fm.timeEmb, fm.timeInW, fm.timeHidden, fm.timeOutW, fm.actionTime, L.expertIn];
        let light = (chain: IBlkDef[], t: number) => {
            let k = Math.min(chain.length - 1, Math.floor(t * chain.length));
            chain[k].highlight = 0.5;
        };
        if (tA.active && tA.t < 1) light(chainA, tA.t);
        if (tB.active && tB.t < 1) light(chainB, tB.t);
        if (tB.t >= 1) { el[0].ada.highlight = 0.5; el[0].ada2.highlight = 0.5; }
    } break;

    case Phase.FM_JointAttn: {
        let layer0 = [L.blocks[0].ln1.lnResid, ...L.blocks[0].kvGroups.flatMap(g => [g.kBlock, g.vBlock]), el[0].attn, el[0].out, L.expertIn];
        initialFrame(state, layer0, 300, 16, { fill: 1.05 });
        let mass = (i: number) => m ? layerMass(m, i) : [0, 0, 0, 0];
        let line = (i: number) => { let q = mass(i); return `第 ${i} 层：视觉 ${pct(q[1])}，文本 ${pct(q[2])}，后缀自身 ${pct(q[3])}`; };
        let mid = Math.floor(nPairs / 2);

        commentary(wt, null, 0)`

现在看一个层对内部。在第 i 个层对里，主干第 i 层用自己的权重算出 1240 个前缀 token 的 K/V（8 个 KV 组，图中左侧一排排的 K、V 块）；专家第 i 层用自己的权重算出 17 个后缀 token 的 Q/K/V（专家的 2 个 KV 头复制成 8 个以对齐）。两边沿 token 轴拼成长 ${cfg?.L_joint ?? 1257} 的序列，统一做 RoPE，跑一次共享注意力，再拆回两路。

${c_blockRef('Joint Attn (expert i)', el[0].attn)} 就是后缀 query 那一半的真实注意力（16 个头取平均）：前 ${cfg?.window.Wt ?? 32} 列是前缀窗口里的 key，后 17 列是后缀 key。橙色箭头从主干 K 组、紫色箭头从 V 组指向它，线宽正比于该层后缀 query 落在前缀上的注意力质量。Q 来自上一层专家输出（竖直箭头）。
`;
        breakAfter();
        commentary(wt)`

掩码（build_fused_mask）决定后缀能看到什么。它只能看两类 key：后缀自身，以及有效且不属于答案的 prompt 列，本样本共 ${cfg?.suffix_visible_prefix ?? '…'} 列（${cfg?.suffix_visible_visual ?? '…'} 个视觉 + ${cfg ? cfg.suffix_visible_prefix - cfg.suffix_visible_visual : '…'} 个文本）。16 个状态 token 列和 ${cfg ? cfg.T_text - cfg.suffix_visible_prefix : '…'} 个答案列被屏蔽，下方质量条里画叉的两行就是它们，质量恰为 0。

反方向上，前缀的 Q 看不到后缀。
`;
        breakAfter();
        let tSweep = afterTime(null, 9.0);
        breakAfter();
        commentary(wt)`

逐层扫过 ${nPairs} 个层对（质量条随层变化，均为真值）：${line(0)}；${line(mid)}；${line(nPairs - 1)}。浅层几乎只读文本，中层开始读图像，深层更多看后缀自己。
`;
        breakAfter();
        commentary(wt)`

由于前缀看不到后缀，主干每层的 K/V 与 x1、t 完全无关。导出器用 ${fmc?.n_particles ?? 8} 张不同噪声各做一次前向核对过：${nPairs} 层前缀 K/V 的最大差异 = ${fmc ? fmc.asserts.prefix_kv_max_abs_diff_across_seeds.toFixed(1) : '…'}。这就是 π0 所说的“前缀 KV cache 可复用”：换噪声、多步积分时，主干只需算一次。
`;
        let pos = tSweep.active ? tSweep.t * (nL - 1) : 0;
        if (tSweep.active && tSweep.t < 1) {
            followLayer(state, L, pos, 300, 16);
        }
        let active = tSweep.t >= 1 ? nL - 1 : Math.round(pos);
        viz.activeLayer = active;
        viz.links = 1;
        viz.massBars = 1;
        el[active].attn.highlight = 0.35;
    } break;

    case Phase.FM_TimeMod: {
        let around = (i: number) => [el[i].out, el[i].ada, el[i].ada2, el[i].attn];
        initialFrame(state, around(0), 250, 10, { fill: 1.4 });
        let st = (i: number) => {
            let s = m?.fm.ada[i].meta.stats;
            return s ? `[${fmtSigned(s.min, 2)}, ${fmtSigned(s.max, 2)}]` : '…';
        };
        commentary(wt, null, 0)`

t 对专家的第二个作用是逐层调制。专家每层的两个 RMSNorm 都是 AdaRMSNorm：

out = (1 + γ(te)) · RMSNorm(x) · w + β(te)

γ、β 都是以 te 为输入的线性层。${c_blockRef('γ₁β₁', el[0].ada)} 与 ${c_blockRef('γ₂β₂', el[0].ada2)} 两根窄块合起来就是第 0 层真实的 4 × 768 组 γ/β（每根 2 行，图中前 64 通道），整组值域 ${st(0)}。
`;
        breakAfter();
        let t2 = afterTime(null, 2.0);
        commentary(wt)`

因为 t 恒为 1，te 是常数，所以对 k005b 而言每层的 AdaRMSNorm 实际上就是一组固定的逐通道缩放 + 平移，各层不同：第 ${nPairs - 1} 层值域 ${st(nPairs - 1)}。专家最后的 final norm 是普通 RMSNorm，不受 t 调制。
`;
        if (t2.active) frameTo(state, t2, around(nPairs - 1), 250, 10, { fill: 1.4 });
        let i = t2.t >= 1 ? nPairs - 1 : 0;
        el[i].ada.highlight = 0.5;
        el[i].ada2.highlight = 0.5;
        viz.activeLayer = -1;
    } break;

    case Phase.FM_Velocity: {
        let col1 = el.slice(0, 14).flatMap(e => [e.attn, e.out]);
        initialFrame(state, col1, 292, 12, { fill: 0.9 });
        commentary(wt, null, 0)`

${nPairs} 个层对逐层改写 17 个后缀 token，这是深度方向的“迭代”：每一层都在上一层的残差上加一点修正。下面后缀流从 ${c_blockRef('Expert 0', el[0].ln1)} 一层层穿到 ${c_blockRef(`Expert ${nPairs - 1} Out`, el[nPairs - 1].out)}：每层依次点亮 AdaRMSNorm → 联合注意力（与主干同层 K/V 的耦合线同时加亮）→ o_proj → ⊕ → AdaRMSNorm → MLP → ⊕。
`;
        breakAfter();
        let tDepth = afterTime(null, 14.0);
        breakAfter();
        let tMove = afterTime(null, 1.5);
        frameTo(state, tMove, [...actionBlocks, fBox], 300, 22, { fill: 1.05 });
        breakAfter();
        commentary(wt)`

最后一层输出经 ${c_blockRef('final RMSNorm', L.expertOut)}，丢掉第 0 个（状态）token，剩下 16 个 token 各自经 ${c_blockRef('action_out_proj', L.actionWeight)}（768 → 23）得到 ${c_blockRef('速度场 v', L.actionVelocity)}，形状 ${H} × ${D}。

右侧柱场现在显示的就是 v 本身（蓝为正、红为负，柱高为值）：每根柱子对应一个（时间步，动作维）。例如 v[0][15] = ${f3(vel?.[15])}，v[0][14] = ${f3(vel?.[14])}。v 的含义是“从 x1 指向动作 x0 的方向”的反向，即 v ≈ x1 − x0。
`;
        if (tDepth.active && tDepth.t < 1) {
            applyFlowPass(L, tDepth.t * nL * nFlowStages);
            followLayer(state, L, Math.max(0, tDepth.t * nL - 0.5), 292, 12);
        }
        viz.field = tMove.active ? 1 : 0;
        viz.fieldMode = 'velocity';
    } break;

    case Phase.FM_Euler: {
        initialFrame(state, [fBox], 300, 22, { fill: 1.05 });
        let cf = fmc?.cf_euler ?? [];
        commentary(wt, null, 0)`

解码就是 Euler 积分：x(t + Δt) = x(t) + Δt · v。k005b 只走一步，Δt = −1，所以 ${c_blockRef('action', L.actionChunk)} = ${c_blockRef('x1', L.actionX1)} − ${c_blockRef('v', L.actionVelocity)}。

换个角度看，这一步就是一次残差连接：x1 是“残差”，−v 是网络算出的修正量。底部的 x1 块与顶部输入簇里的 x1 是同一个张量（skip）。
`;
        breakAfter();
        let tDec = afterTime(null, 5.0);
        breakAfter();
        commentary(wt)`

柱场先摆出 x1，白色竖线是每个分量的 −v。随后所有柱子沿 x(t) = x1 − (1 − t)·v 从 t = 1 走到 t = 0，停下时就是动作块。例如第 0 步左夹爪：x1 = ${f3(x1?.[14])}，v = ${f3(vel?.[14])}，x1 − v = ${f3(chunk?.[14])}。

注意：中间帧只是直线插值（标题标了 interp only），不是模型输出。模型只在 t = 1 被调用一次。
`;
        breakAfter();
        commentary(wt)`

如果硬把 t 改成 1.0, 0.9, …, 0.1 做 10 步 Euler（导出器真实跑过的反事实实验），结果是：${cf.length ? cf.filter((_, j) => j % 3 === 0 || j === cf.length - 1).map(c => `t=${c.t.toFixed(1)} 时 |v| = ${c.vnorm.toFixed(2)}、MSE = ${c.mse.toFixed(3)}`).join('；') : '…'}。10 步后 MSE 为 ${cf.length ? cf[cf.length - 1].mse.toFixed(3) : '…'}，远差于 1 步的 ${fmc ? fmc.mse.toFixed(3) : '…'}。原因是模型从没在 t < 1 上训练过，所以这里不做多步动画。
`;
        viz.field = 1;
        viz.fieldMode = 'decode';
        viz.decode = tDec.active ? tDec.t : 0;
        viz.showVelocity = tDec.t >= 1 ? 0.35 : 1;
    } break;

    case Phase.FM_Loss: {
        initialFrame(state, [fBox], 300, 18, { fill: 1.0 });
        let err = m?.fm.errDim;
        let order = err ? Array.from(err).map((v, d) => [v, d] as [number, number]).sort((a, b) => a[0] - b[0]) : [];
        let nm = (d: number) => cfg?.action_dim_names[d] ?? `${d}`;
        commentary(wt, null, 0)`

训练怎么教会它？训练目标是速度 u = x1 − a（a 是标签动作），损失是 v 与 u 在 16 × 23 上的 MSE。

由于 x0 − a = (x1 − v) − (x1 − u) = u − v，“速度误差”和“动作误差”完全相等。场景里的线框就是标签 a，红色竖线是每个分量的误差。
`;
        breakAfter();
        let tShow = afterTime(null, 1.5);
        commentary(wt)`

如实给出本样本的数值：MSE(x1 − v, a) = ${fmc ? fmc.mse.toFixed(3) : '…'}，而直接预测全 0 的 MSE 是 ${fmc ? fmc.mse_zero.toFixed(3) : '…'}。也就是说在这一帧上，单次解码并不比“输出 0”更好，这是单个噪声样本的局限，下一节的多噪声实验会说明原因。

前排的红色短柱是逐维 MSE：最好的维是 ${order.slice(0, 2).map(([v, d]) => `${d} ${nm(d)}（${v.toFixed(2)}）`).join('、')}；最差的是 ${order.slice(-2).map(([v, d]) => `${d} ${nm(d)}（${v.toFixed(2)}）`).join('、')}。夹爪：标签 a[0][14] = ${f3(tgt?.[14])}、预测 ${f3(chunk?.[14])}；a[0][22] = ${f3(tgt?.[22])}、预测 ${f3(chunk?.[22])}。
`;
        viz.field = 1;
        viz.target = tShow.active ? tShow.t : 0;
        viz.errBars = tShow.active ? tShow.t : 0;
    } break;

    case Phase.FM_Particles: {
        initialFrame(state, [fBox], 270, 4, { fill: 1.0 });
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
        commentary(wt, null, 0)`

同一个速度场可以搬运不同的起点。导出器用 ${fmc?.n_particles ?? 8} 张噪声（本样本自带的 x1，加上种子 ${fmc ? fmc.seeds.slice(1, 3).join('、') : '…'}…${fmc ? fmc.seeds[fmc.seeds.length - 1] : '…'}）各做了一次真实的 t = 1 前向。

场景切到“粒子轴”：23 根竖轴各对应一个动作维（第 0 步），每根轴上 ${fmc?.n_particles ?? 8} 个彩色粒子，从各自的 x1_k 出发，被各自的 v_k 推到 x1_k − v_k。
`;
        breakAfter();
        let tMove = afterTime(null, 4.0);
        breakAfter();
        commentary(wt)`

起点几乎是随机散开的，终点却聚拢了：${fmc?.n_particles ?? 8} 个动作块之间的平均标准差是 ${fmc ? fmc.particle_chunk_std.toFixed(2) : '…'}，单块 MSE 在 ${fmc ? `${Math.min(...fmc.mse_per_particle).toFixed(2)} – ${Math.max(...fmc.mse_per_particle).toFixed(2)}` : '…'} 之间。黑色横标是 ${fmc?.n_particles ?? 8} 块的均值，它的 MSE 降到 ${fmc ? fmc.mse_mean_chunk.toFixed(3) : '…'}，红色横标是标签。

这说明 t = 1 的单步解码本质上是在回归“给定观测时动作的条件均值”，再叠加一部分来自噪声的随机性。单个样本的误差大，主要来自这部分随机性。
`;
        breakAfter();
        commentary(wt)`

换噪声时，主干完全不受影响：前缀 K/V 在 ${fmc?.n_particles ?? 8} 次前向中逐位相同（最大差异 ${fmc ? fmc.asserts.prefix_kv_max_abs_diff_across_seeds.toFixed(1) : '…'}）。变的只有后缀：第 ${Math.floor(nPairs / 2)} 层后缀落在视觉 token 上的注意力质量，随噪声在 ${spread} 之间变化。
`;
        viz.field = 0;
        viz.particles = 1;
        viz.particleTau = tMove.active ? tMove.t : 0;
        viz.particleStep = 0;
    } break;

    case Phase.FM_Robot: {
        initialFrame(state, [fBox], 300, 22, { fill: 1.05 });
        let row0 = cfg && chunk ? describeActionRow(cfg, chunk, 0) : [];
        let zeroVel = cfg ? [0, 1, 2].map(d => `${cfg!.action_dim_names[d]} ${fmtSigned(normalizeAction(cfg!, d, 0))}`).join('，') : '…';
        commentary(wt, null, 0)`

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
        viz.field = 1;
        viz.robotUnits = 1;
    } break;
    }
}
