import { Phase } from "./Walkthrough";
import { commentary, frameCubes, frameTo, initialFrame, IWalkthroughArgs } from "./WalkthroughTools";
import { clamp } from "@/src/utils/data";
import { IBlkDef, IVlaModelLayout } from "../VlaModelLayout";
import { describeActionRow, fmtSigned, normalizeAction, unnormalizeAction } from "../actionDims";
import { layerMass } from "../components/ActionViz";
import { IProgramState } from "../Program";
import { applyFlowPass, applyFmStep, nFlowStages } from "./FlowPass";

// The DiT action-expert chapter group D0 … D7 (docs/design/dit-section-design.md §2, §3, §5).
// Every tensor in the scene is a real k005b capture; the K-step decode (D6) runs on the
// step_130000 weights, which were trained at t = 1 only — the scene and the text mark it as a
// pre-stamp preview until the action_decode-stamped checkpoint is exported.

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

// the joint block of layer pair i: the trunk lane's chain and the DiT lane beside it
function pairPath(L: IVlaModelLayout, i: number, withTrunk = true) {
    let b = L.blocks[i], e = L.expertLayers[i];
    let h = b.heads.length - 1;
    let dit = [e.ln1, e.ada, e.o, e.res1, e.ln2, e.mlp, e.out, e.heads[h].scS, e.heads[h].vo];
    if (!withTrunk) return [...dit, b.attnResidual, b.mlpResidual];
    return [...b.ln1.cubes, b.heads[h].attnMtx, b.attnOut, b.attnResidual, ...b.ln2.cubes,
        b.mlpFc, b.mlpAct, b.mlpResult, b.mlpResidual, ...dit];
}

// the camera rides along a pass: frames layer pair `pos` (continuous between two layers)
function followPair(state: IProgramState, L: IVlaModelLayout, pos: number, angleX: number, angleY: number, fill: number, withTrunk = true) {
    let nL = L.blocks.length;
    let i0 = clamp(Math.floor(pos), 0, nL - 1), i1 = Math.min(nL - 1, i0 + 1);
    let u = clamp(pos - i0, 0, 1);
    u = u * u * (3 - 2 * u);
    let a = frameCubes(pairPath(L, i0, withTrunk), angleX, angleY, { fill });
    let b = frameCubes(pairPath(L, i1, withTrunk), angleX, angleY, { fill });
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
    let { walkthrough: wt, state, layout: L, tools: { afterTime, breakAfter, c_blockRef } } = args;
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
    let hF = b0.heads.length - 1;
    // the field is hidden unless a phase asks for it
    viz.field = 0;

    switch (wt.phase) {

    case Phase.D0_TwoTransformers: {
        let lane0 = [b0.ln1.lnResid, b0.attnOut, b0.attnResidual, b0.mlpResult, b0.mlpResidual,
            e0.ln1, e0.ada, e0.o, e0.res1, e0.ln2, e0.mlp, e0.out];
        let top = [L.residual0, L.expertIn, ...fm.cubes, ...lane0];
        initialFrame(state, top, 280, 8, { fill: 1.0 });
        commentary(wt, null, 0)`

这一章组讲动作分支：DiT 动作专家与流匹配（flow matching）。

k005b 里其实有两个 Transformer。主干是 Qwen3-VL（隐藏维 ${cfg?.n_embd ?? 2048}），处理图像、文本和状态 token，给出 knack / done。动作专家是一个 DiT：一个用时间 t 做条件的 Transformer（隐藏维 ${cfg?.expert_hidden ?? 768}），负责产出 ${H} × ${D} 的动作块。

两者逐层配对，一共 ${nPairs} 个层对。场景里每一层就是一个“传送门”：左边是主干泳道，右边紧贴着的橙色窄条是 DiT 泳道（17 个后缀 token，图中 64 通道窗口）。两条泳道在这一层共用一次联合注意力，这就是它们的耦合。
`;
        breakAfter();
        let tPull = afterTime(null, 2.0);
        commentary(wt)`

所以不存在“两边各一条与中间无关的流”：主干流和 DiT 流都从顶部进入第 0 层，一格一格穿过同一组 ${nPairs} 个层对。

另外两样东西只属于 DiT：
① 顶部右侧的输入簇（${c_blockRef('state75', fm.state75)}、${c_blockRef('x_t', fm.x1)}、${c_blockRef('te = sincos(t)', fm.timeEmb)}），拼成 ${c_blockRef('DiT Suffix In', L.expertIn)}；
② 一条青色的时间总线：te 沿着它送到每一层 DiT 泳道的 γβ 窄块。这是 DiT 的“时间条件化 adaLN”。
`;
        frameTo(state, tPull, [...top, ...pairPath(L, 1, false), ...pairPath(L, 2, false), ...pairPath(L, 3, false)], 280, 8, { fill: 1.0 });
        viz.activeLayer = 0;
        viz.links = 0.6;
    } break;

    case Phase.D1_DitBlock: {
        // D1a–D1h (docs/design/dit-framework-design.md §1): one DiT block, part by part
        let eh = e0.heads[hF];
        let laneTop = [e0.ln1, e0.ada, e0.o, e0.res1, b0.ln1.lnResid, b0.attnResidual, b0.attnOut];
        let laneBot = [e0.res1, e0.ln2, e0.ada2, e0.gate, e0.act, e0.mlp, e0.out, b0.mlpResidual];
        let lane = [...laneTop, ...laneBot];
        let st = (i: number) => {
            let s = m?.fm.ada[i].meta.stats;
            return s ? `[${fmtSigned(s.min, 2)}, ${fmtSigned(s.max, 2)}]` : '…';
        };
        // real values of suffix token 1 (the first action token), window channel c
        let el0 = m?.expertLayers[0];
        let ada0 = m?.fm.ada[0].localBuffer;
        let wc = ada0 ? ada0.length / 4 : 64;
        let at = (t: { localBuffer: Float32Array } | undefined, r: number, c: number) => t?.localBuffer[r * wc + c];
        // the channel where the plain RMSNorm value is largest, so the modulation is visible
        let cPick = 0;
        if (ada0 && el0) {
            let best = -1;
            for (let c = 0; c < wc; c++) {
                let pre = (el0.ln1.localBuffer[wc + c] - ada0[wc + c]) / (1 + ada0[c]);
                if (Math.abs(pre) > best) { best = Math.abs(pre); cPick = c; }
            }
        }
        let g1 = ada0?.[cPick], b1 = ada0?.[wc + cPick], post1 = at(el0?.ln1, 1, cPick);
        let pre1 = g1 !== undefined && b1 !== undefined && post1 !== undefined ? (post1 - b1) / (1 + g1) : undefined;
        let xin = m?.expertIn.localBuffer[wc], o10 = at(el0?.o, 1, 0), r10 = at(el0?.res1, 1, 0);
        let mlp10 = at(el0?.mlp, 1, 0), out10 = at(el0?.out, 1, 0);
        let te = m?.fm.timeEmb.localBuffer;
        let preStamp = `（数据说明：这里的激活来自 ${ckpt} 旧权重、t = 1 的一次前向；t = 1 正是它训练时见过的分布，但仍属 pre-stamp 预览，step_160000 印章检查点落盘后会换数据。）`;

        initialFrame(state, lane, 270, 4, { fill: 1.05 });
        commentary(wt, null, 0)`

D1a 全貌。动作专家就是一个 DiT：用时间 t 做条件的 Transformer。这一章把它的一个块从头到尾拆开讲。

教科书 DiT 的一块是：adaLN 条件化的 Norm → 注意力 → 残差 → adaLN Norm → MLP → 残差。堆 N 块，最后接 final norm 和线性输出。

场景里右侧橙色窄条就是第 0 层的 DiT 块。下面先让真实数据把这一层完整走一遍，黄色是正在计算的部件。
${preStamp}
`;
        breakAfter();
        let tA = afterTime(null, nFlowStages * 0.6);
        breakAfter();

        // B and C share the close frontal view of gamma/beta (wider frames put the layer-0 heads in front of the lane)
        let chartBox = { x: e0.ada.x + e0.ada.dx, y: e0.ada.y - 14, z: e0.ada.z, dx: 56, dy: e0.ada.dy + 14, dz: 2 };
        let adaView = [b0.ln1.lnResid, e0.ln1, e0.ada, chartBox];
        let tB = afterTime(null, 1.2);
        frameTo(state, tB, adaView, 270, 2, { fill: 1.15 });
        commentary(wt)`

D1b 时间条件。t 先变成正弦嵌入 ${c_blockRef('te = sincos(t)', fm.timeEmb)}（min_period 4e-3、max_period 4.0，代码 :378），这里 t = 1，te 前两维 = ${f3(te?.[0])}、${f3(te?.[1])}。

te 沿时间总线（γβ 右侧的竖线，D0 里看过全貌）从上方送到每一层的两个 γβ 小块：${c_blockRef('γ₁β₁', e0.ada)} 和 ${c_blockRef('γ₂β₂', e0.ada2)}。

教科书 DiT 的条件是 t 嵌入加上类别 c 嵌入。k005b 只有 t：任务和场景信息不走条件向量，而是由注意力去读主干（D1d / D3）。
`;
        breakAfter();

        let tC = afterTime(null, 1.2);
        frameTo(state, tC, adaView, 270, 2, { fill: 1.15 });
        commentary(wt)`

D1c adaLN 调制。${c_blockRef('AdaRMSNorm1', e0.ln1)} 的公式：
out = (1 + γ(te)) · RMSNorm(x) · w + β(te)
γ、β 由 te 经线性层得到，零初始化。

右侧条形图是后缀 token 1（第一个动作 token）的 64 个通道：灰条是调制前的 RMSNorm(x)·w，橙条是调制后的真实输出。例如通道 ${cPick}：γ₁ = ${f3(g1)}，β₁ = ${f3(b1)}，调制前 ${f3(pre1)} → 调制后 ${f3(post1)}。灰条普遍很短，橙条很长：t = 1 时 γ 把多数通道放大了十倍以上，β 再整体平移。

第 0 层 4 × 768 组 γβ 值域 ${st(0)}，第 ${nPairs - 1} 层 ${st(nPairs - 1)}。和教科书 adaLN-Zero 的区别：用 RMSNorm 而非 LayerNorm；每层只有 2 处 γβ，没有 α 门控，残差直接相加。
`;
        breakAfter();

        let tD = afterTime(null, 1.2);
        frameTo(state, tD, [eh.q, eh.scP, eh.scS, eh.pP, eh.pS, eh.vo, b0.heads[hF].attnMtx], 278, 8, { fill: 1.5 });
        commentary(wt)`

D1d 注意力。17 个后缀 token 生成 ${c_blockRef('Q', eh.q)}、K、V，接着打分（${c_blockRef('scP', eh.scP)} 对主干前缀，${c_blockRef('scS', eh.scS)} 对后缀自身），softmax 后乘 V 得到 ${c_blockRef('vo', eh.vo)}。

教科书是自注意力，只看自己的 token。k005b 是联合注意力：K/V 同时包含主干前缀和后缀，这是 DiT 拿到场景与任务信息的唯一入口（D3 细讲）。
`;
        breakAfter();
        let tDrun = afterTime(null, 3 * 0.9);
        breakAfter();

        let tE = afterTime(null, 1.2);
        frameTo(state, tE, [e0.o, b0.attnResidual, e0.res1], 270, 4, { fill: 1.3 });
        commentary(wt)`

D1e 残差 1。${c_blockRef('o_proj', e0.o)} 的输出是增量，在 ⊕ 处加回主线，得到 ${c_blockRef('res1', e0.res1)}。

真值核对（token 1，通道 0）：输入 ${f3(xin)} + o ${f3(o10)} = res1 ${f3(r10)}。每个块只学增量。
`;
        breakAfter();
        let tErun = afterTime(null, 2 * 0.9);
        breakAfter();

        let tF = afterTime(null, 1.2);
        frameTo(state, tF, [e0.ln2, e0.ada2, e0.gate, e0.up, e0.act, e0.mlp, e0.out], 270, 4, { fill: 1.15 });
        commentary(wt)`

D1f MLP。${c_blockRef('AdaRMSNorm2', e0.ln2)}（由 ${c_blockRef('γ₂β₂', e0.ada2)} 调制）→ ${c_blockRef('gate', e0.gate)} / ${c_blockRef('up', e0.up)} → ${c_blockRef('SiLU(gate)·up', e0.act)} → ${c_blockRef('down_proj', e0.mlp)} → ⊕ → ${c_blockRef('out', e0.out)}。

教科书用 GELU 的点式 MLP；k005b 用 SwiGLU，768 → 2752 → 768，逐 token 独立。
真值核对（token 1，通道 0）：res1 ${f3(r10)} + mlp ${f3(mlp10)} = out ${f3(out10)}。
`;
        breakAfter();
        let tFrun = afterTime(null, 5 * 0.9);
        breakAfter();

        let cG = commentary(wt)`

D1g 堆叠。第 0 层的 out 就是第 1 层的输入，同样的块一层层叠下去。教科书 DiT-XL 有 28 块；k005b 也是 ${nL} 层，每层与主干同名层配对。

每层有自己的 γβ（同一个 te，不同的线性层），值域从第 0 层的 ${st(0)} 到第 ${nPairs - 1} 层的 ${st(nPairs - 1)}。
`;
        breakAfter();
        let tG = afterTime(null, 4.0);
        breakAfter();
        let tGpull = afterTime(null, 1.5);
        frameTo(state, tGpull, el.flatMap(e => [e.ln1, e.out]), 280, 10, { fill: 1.05 });
        breakAfter();

        let tH = afterTime(null, 1.5);
        frameTo(state, tH, [el[nL - 1].out, L.expertOut, L.actionWeight, L.actionVelocity], 270, 6, { fill: 1.45 });
        commentary(wt)`

D1h 输出。${nL} 层之后：${c_blockRef('final RMSNorm', L.expertOut)}（普通 RMSNorm，final_norm_adanorm = False，不受 t 调制）→ 丢掉 state token → ${c_blockRef('action_out_proj', L.actionWeight)}（768 → 23）→ ${c_blockRef('速度场 v', L.actionVelocity)}，16 × 23，例如 v[0][14] = ${f3(vel?.[14])}。

教科书 DiT 在这里是 final adaLN + Linear + unpatchify。

教科书 DiT ↔ k005b 对照：

① patchify → token：每个动作步一行就是一个 token，再加 1 个 state token，共 17 个（:356-386，D2）

② 自注意力 → 联合注意力，K/V 含主干前缀（:86-140，D3）

③ 条件 t + 类别 c → 只有 t（:378）

④ LayerNorm + adaLN-Zero(γβα) → AdaRMSNorm（γβ 零初始化，无 α）

⑤ GELU MLP → SwiGLU 768→2752→768

⑥ N 块 → ${nL} 层，与主干配对

⑦ final adaLN + Linear + unpatchify → RMSNorm → action_out_proj → 16×23（:214, :441）

⑧ K 步采样 → Euler ${K} 步，x ← x − v/${K}（:526-549，D6）

${preStamp}
`;

        // which sub-step is on screen (a … h)
        let sub = tH.active ? 7 : cG.active ? 6 : tF.active ? 5 : tE.active ? 4 : tD.active ? 3 : tC.active ? 2 : tB.active ? 1 : 0;
        let S = nFlowStages;
        if (sub === 0) {
            if (tA.active && tA.t < 1) applyFlowPass(L, tA.t * S, { suffixOnly: true });
        } else if (sub === 1) {
            applyFlowPass(L, 0, { suffixOnly: true });
            fm.timeEmb.highlight = 0.45;
            e0.ada2.highlight = 0.45;
            dimExcept(L, [...fm.cubes, L.expertIn, e0.ln1, e0.ln2, ...el.flatMap(e => [e.ada, e.ada2])], 1 - 0.85 * tB.t);
        } else if (sub === 2) {
            applyFlowPass(L, 0, { suffixOnly: true });
            viz.normCmp = tC.t;
            viz.normLayer = 0;
            viz.normWhich = 1;
            viz.normTok = 1;
        } else if (sub === 3) {
            applyFlowPass(L, 1 + (tDrun.active ? Math.min(2.99, tDrun.t * 3) : 0), { suffixOnly: true });
            viz.activeLayer = 0;
            viz.links = 0.6;
        } else if (sub === 4) {
            applyFlowPass(L, 4 + (tErun.active ? Math.min(1.99, tErun.t * 2) : 0), { suffixOnly: true });
        } else if (sub === 5) {
            applyFlowPass(L, 6 + (tFrun.active ? Math.min(4.99, tFrun.t * 5) : 0), { suffixOnly: true });
        } else if (sub === 6) {
            // the pass descends layers 0 → 2 with the camera, then the stack pulls back
            if (!tGpull.active) {
                let pos = S + tG.t * 2 * S;
                applyFlowPass(L, Math.min(pos, 3 * S - 0.01), { suffixOnly: true });
                followPair(state, L, Math.max(0, pos / S - 0.5), 270, 4, 1.6, false);
            } else {
                let li = Math.min(nL - 1, Math.floor(tGpull.t * nL));
                for (let i = 0; i <= li; i++) {
                    el[i].ada.highlight = Math.max(el[i].ada.highlight, i === li ? 0.45 : 0.2);
                    el[i].ada2.highlight = Math.max(el[i].ada2.highlight, i === li ? 0.45 : 0.2);
                }
            }
        } else {
            for (let b of [L.expertOut, L.actionWeight, L.actionVelocity]) b.highlight = 0.45;
        }
        e0.ada.highlight = Math.max(e0.ada.highlight, sub <= 2 ? 0.45 : 0.25);
    } break;

    case Phase.D2_Concat: {
        initialFrame(state, [...fm.cubes, L.expertIn, L.residual0, L.idxObj], 272, 10, { fill: 1.0 });
        let s75 = m?.fm.state75.localBuffer;
        let nValid = fmc?.state_valid ?? 0;
        let te = m?.fm.timeEmb.localBuffer;
        commentary(wt, null, 0)`

联合注意力要求两路 token 拼成一条序列：前缀（主干）${cfg ? cfg.L_joint - cfg.n_suffix : 1240} 个 token ‖ 后缀（DiT）${cfg?.n_suffix ?? 17} 个 token，总长 ${cfg?.L_joint ?? 1257}。前缀就是 ${c_blockRef('主干嵌入', L.residual0)}。后缀的 17 个 token 由三个输入拼成 ${c_blockRef('DiT Suffix In', L.expertIn)}（本样本 episode ${cfg?.sample.episode ?? '…'}，frame ${cfg?.sample.frame ?? '…'}）。

token 0：${c_blockRef('state75', fm.state75)}（bounds99 归一化的本体状态，${nValid} 个非零槽位，前几个值 ${s75 ? Array.from(s75.slice(0, 4)).map(v => fmtSigned(v)).join(' / ') : '…'}）经 ${c_blockRef('state_proj', fm.stateProjW)}（75 → 768）得到 ${c_blockRef('状态 token', fm.stateEmb)}。主干那 16 个状态 token 在联合注意力里对 DiT 是屏蔽的，所以这是 DiT 看到本体状态的唯一入口。
`;
        breakAfter();
        let tA = afterTime(null, 2.5);
        breakAfter();
        commentary(wt)`

token 1..16：${c_blockRef('x_t', fm.x1)} 是 ${H} × ${D} 的当前动作板（t = 1 时就是纯噪声，x[0][0] = ${f3(x1?.[0])}）。每一行先经 ${c_blockRef('action_in_proj', fm.actionInW)}（23 → 768）成为 ${c_blockRef('动作嵌入', fm.actionEmb)}，再与 ${c_blockRef('te', fm.timeEmb)}（t 的正弦嵌入，te[0..3] = ${te ? Array.from(te.slice(0, 4)).map(v => fmtSigned(v)).join(' / ') : '…'}）拼成 1536 维，经 ${c_blockRef('time_mlp_in', fm.timeInW)} + SiLU 得到 ${c_blockRef('隐层', fm.timeHidden)}，再经 ${c_blockRef('time_mlp_out', fm.timeOutW)} 成为 ${c_blockRef('16 个动作 token', fm.actionTime)}，从右侧进入 DiT Suffix In。
`;
        breakAfter();
        let tB = afterTime(null, 4.0);
        let chainA = [fm.state75, fm.stateProjW, fm.stateEmb, L.expertIn];
        let chainB = [fm.x1, fm.actionInW, fm.actionEmb, fm.timeEmb, fm.timeInW, fm.timeHidden, fm.timeOutW, fm.actionTime, L.expertIn];
        let light = (chain: IBlkDef[], t: number) => {
            let k = Math.min(chain.length - 1, Math.floor(t * chain.length));
            chain[k].highlight = 0.5;
        };
        if (tA.active && tA.t < 1) light(chainA, tA.t);
        if (tB.active && tB.t < 1) light(chainB, tB.t);
        if (!tA.active) { L.residual0.highlight = 0.35; L.expertIn.highlight = 0.35; }
    } break;

    case Phase.D3_JointAttn: {
        let bh = b0.heads[hF], eh = e0.heads[hF];
        let g0 = b0.kvGroups[bh.kvGroup];
        let quad = [bh.attnMtx, eh.scP, eh.scS, bh.attnMtxSm, eh.pP, eh.pS, bh.qBlock, eh.q, eh.vo, bh.vOutBlock, g0.kBlock];
        initialFrame(state, quad, 278, 8, { fill: 1.5 });
        let mass = (i: number) => m ? layerMass(m, i) : [0, 0, 0, 0];
        let line = (i: number) => { let q = mass(i); return `第 ${i} 层：视觉 ${pct(q[1])}，文本 ${pct(q[2])}，后缀自身 ${pct(q[3])}`; };
        let mid = Math.floor(nPairs / 2);
        commentary(wt, null, 0)`

联合注意力就是两流耦合的地方。把一个头的打分矩阵画成四个象限（行是 query，列是 key）：

左上 ${c_blockRef('前缀 × 前缀', bh.attnMtx)}：主干自己的注意力（窗口）。
右上：空的。掩码规定前缀看不到后缀，场景里这一格标着 “prefix Q cannot see suffix K”。
左下 ${c_blockRef('scP 后缀 × 前缀', eh.scP)}：DiT 的 Q 去读主干的 K。这一象限就是耦合本身。
右下 ${c_blockRef('scS 后缀 × 后缀', eh.scS)}：DiT 的 17 个 token 彼此注意。

琥珀色细线从主干 K 组指向它所服务的两个头的 scP；品红色细线从主干 V 组指向这两个头的 ${c_blockRef('softmax·V 输出', eh.vo)}：DiT 读到的内容来自主干的 V。专家只有 2 个 KV 头，复制到 8 组以对齐主干。
`;
        breakAfter();
        commentary(wt)`

掩码（build_fused_mask）还决定后缀能看哪些前缀列：有效且不属于答案的 prompt 列，本样本共 ${cfg?.suffix_visible_prefix ?? '…'} 列（${cfg?.suffix_visible_visual ?? '…'} 个视觉 + ${cfg ? cfg.suffix_visible_prefix - cfg.suffix_visible_visual : '…'} 个文本）。16 个状态列和 ${cfg ? cfg.T_text - cfg.suffix_visible_prefix : '…'} 个答案列被屏蔽，质量条里画叉的两行就是它们，质量恰为 0。
`;
        breakAfter();
        let tSweep = afterTime(null, 9.0);
        breakAfter();
        commentary(wt)`

逐层扫过 ${nPairs} 个层对（耦合线粗细与质量条都是真值）：${line(0)}；${line(mid)}；${line(nPairs - 1)}。浅层几乎只读文本，中层开始读图像，深层更多看后缀自己。
`;
        let pos = tSweep.active ? tSweep.t * (nL - 1) : 0;
        if (tSweep.active && tSweep.t < 1) followPair(state, L, pos, 284, 10, 1.2, false);
        let active = tSweep.t >= 1 ? nL - 1 : Math.round(pos);
        viz.activeLayer = active;
        viz.links = 1;
        viz.massBars = 1;
        viz.quadLabel = 1;
        el[active].heads.forEach(h => { h.scP.highlight = 0.3; });
    } break;

    case Phase.D4_Forward: {
        initialFrame(state, [L.residual0, L.expertIn, ...pairPath(L, 0)], 280, 6, { fill: 1.6 });
        commentary(wt, null, 0)`

现在把真实数据推过整个层堆，两条流同时走。主干流（${c_blockRef('嵌入', L.residual0)}）和 DiT 流（${c_blockRef('DiT Suffix In', L.expertIn)}）从顶部同一个入口进入第 0 层：箭头落在每个“主干 | DiT”成对块的外框上，所以每根箭头都对应两路同时计算。

黄色是正在计算的子层，未到达的块留空、箭头变淡。每到联合注意力，这一层的耦合线就加亮。这次前向对应训练时的 forward（t = 1）。
`;
        breakAfter();
        let tL0 = afterTime(null, nFlowStages * 0.9);
        breakAfter();
        commentary(wt)`

第 0 层走完。⊕ 块上的小字是残差的另一个加数（skip），它在 ⊕ 处相加，没有绕过层堆的直线。接下来穿过剩下的 ${nL - 1} 层。
`;
        let tRest = afterTime(null, (nL - 1) * 0.7);
        breakAfter();
        let tExit = afterTime(null, 1.5);
        frameTo(state, tExit, [L.ln_f.lnResid, L.expertOut, L.hStar, L.logits, L.doneLogits, ...actionBlocks], 282, 10, { fill: 1.1 });
        commentary(wt)`

${nL} 层穿完，两条流在这里分开出去：
主干：${c_blockRef('ln_f', L.ln_f.lnResid)} → ${c_blockRef('h*', L.hStar)} → knack / done 两个头。
DiT：${c_blockRef('final RMSNorm', L.expertOut)} 丢掉状态 token，16 个 token 经 ${c_blockRef('action_out_proj', L.actionWeight)}（768 → 23）得到 ${c_blockRef('速度场 v', L.actionVelocity)}，例如 v[0][14] = ${f3(vel?.[14])}。
一步解码：${c_blockRef('动作块', L.actionChunk)} = ${c_blockRef('x1', L.actionX1)} − v，例如 x1[0][14] = ${f3(x1?.[14])}，动作 = ${f3(chunk?.[14])}。多步解码见 D6。
`;
        let pos: number;
        if (!tL0.active) pos = -0.5;
        else if (!tRest.active) pos = tL0.t * nFlowStages;
        else pos = nFlowStages + tRest.t * (nL - 1) * nFlowStages;
        if (!(tRest.active && tRest.t >= 1)) {
            applyFlowPass(L, pos);
            if (tL0.active) followPair(state, L, Math.max(0, pos / nFlowStages - 0.5), 280, 6, 1.6);
            let li = clamp(Math.floor(Math.max(0, pos) / nFlowStages), 0, nL - 1);
            viz.activeLayer = li;
        }
    } break;

    case Phase.D5_PrefixCache: {
        let kv = L.blocks.flatMap((_, i) => kvOf(i));
        initialFrame(state, [...pairPath(L, 0), ...pairPath(L, 3)], 284, 10, { fill: 1.2 });
        commentary(wt, null, 0)`

在解码时，前缀只算一次。

掩码规定前缀看不到后缀，所以主干每一层的 K/V 与 x_t、t 完全无关。部署代码因此先跑一次 encode_prefix：主干完整前向一次，把 ${nPairs} 层的 K/V 存成缓存。

场景里主干其余的块淡出，只留下每层的 ${c_blockRef('K', b0.kvGroups[0].kBlock)} / ${c_blockRef('V', b0.kvGroups[0].vBlock)} 块：这就是缓存。之后的每一次 DiT 穿行，耦合线都从这些缓存的 K/V 读数。
`;
        let tCache = afterTime(null, 1.5);
        let keep = [...L.ditCubes, ...kv];
        let lvl = 1 - 0.8 * (tCache.active ? tCache.t : 0);
        dimExcept(L, keep, lvl);
        breakAfter();
        commentary(wt)`

导出器核对过：用 ${fmc?.n_particles ?? 8} 张不同噪声各做一次前向，${nPairs} 层前缀 K/V 的最大差异 = ${fmc ? fmc.asserts.prefix_kv_max_abs_diff_across_seeds.toFixed(1) : '…'}。所以 knack / done 在 10 次穿行中逐位不变，变的只有 DiT 一侧。
`;
        viz.activeLayer = 0;
        viz.links = 0.8;
    } break;

    case Phase.D6_Decode: {
        let kv = L.blocks.flatMap((_, i) => kvOf(i));
        let S = L.strip;
        // the strip + the DiT input chain: where every pass starts and every return lands
        let stripView = [...S.map(b => ({ x: b.x, y: b.y, z: b.z, dx: b.dx, dy: b.dy, dz: b.dz })), fm.x1];
        initialFrame(state, stripView, 272, 10, { fill: 1.2 });
        dimExcept(L, [...L.ditCubes, ...kv], 0.2);
        let tgtArr = tgt;
        let stepInfo = (k: number) => {
            let st = fs?.steps[k];
            if (!st) return '…';
            let v = st.tex['v']?.localBuffer, xn = st.tex['x_next']?.localBuffer;
            return `t = ${st.t.toFixed(1)}：|v| = ${rms(v).toFixed(2)}，x[0][14] ${f3(st.tex['x']?.localBuffer[14])} → ${f3(xn?.[14])}，MSE(x, a) = ${mse(xn, tgtArr).toFixed(3)}`;
        };
        commentary(wt, null, 0)`

现在是部署时的 flow matching 解码（sample_action_chunk）：从噪声出发，DiT 被调用 ${K} 次，每次是一个 Euler 步。

场景上方是 Euler 轨迹带：${K + 1} 块 16 × 23 的板，${c_blockRef('x(t=1.0)', S[0])} 是纯噪声，${c_blockRef(`x(t=0)`, S[K])} 是最终动作块。每次穿行：
① 当前板 x_t 进入 DiT 的输入链（箭头从板落到 x_t）；
② 时间总线换成新的 t，${nPairs} 层的 γβ 全部换成这个 t 的真值；
③ 只有 DiT 泳道从第 0 层穿到第 ${nPairs - 1} 层，联合注意力读缓存的主干 K/V；
④ 底部得到 v，黄色回程箭头把 x − v/${K} 写进下一块板。

注意：这是 ${ckpt} 的旧权重，它只在 t = 1 上训练过，t < 1 是分布外，这里是 pre-stamp 预览（场景标题已标注）。step_160000 落盘后会替换成真值。
`;
        breakAfter();
        // pass k: 0..0.85 = the DiT traverse, 0.85..1 = the return v → board k+1
        let durs = Array.from({ length: K }, (_, k) => k === 0 ? 20 : k < 3 ? 7 : 2.5);
        let passes = durs.map((d, k) => {
            let t = afterTime(null, d);
            if (k === 0 || k === 2) breakAfter();
            if (k === 0) commentary(wt)`

第 1 次穿行（${stepInfo(0)}）。以后每次穿行用的是同一组权重，只有两样东西换了：输入板 x_t，和总线上的 t。接下来两次中速。
`;
            if (k === 2) commentary(wt)`

第 2、3 次：${stepInfo(1)}；${stepInfo(2)}。剩下 ${K - 3} 次快放，左上角的步进格每完成一次走一格。
`;
            return t;
        });
        breakAfter();
        commentary(wt)`

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
                applyFlowPass(L, passPos(clamp(u / 0.85, 0, 1), nL), { suffixOnly: true });
                let li = clamp(Math.floor(passPos(clamp(u / 0.85, 0, 1), nL) / nFlowStages), 0, nL - 1);
                viz.activeLayer = li;
                viz.links = 1;
                if (k === 0 && u < 0.85) followPair(state, L, Math.max(0, li - 0.5), 280, 10, 1.4, false);
                else if (k === 0) {
                    // glide from the last layer back out to the strip for the return arrow
                    followPair(state, L, nL - 1, 280, 10, 1.4, false);
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
    } break;

    case Phase.D7_ActionChunk: {
        let kFin = fs?.steps[K - 1]?.tex['x_next']?.localBuffer;
        initialFrame(state, [L.strip[K], ...actionBlocks], 284, 12, { fill: 1.1 });
        L.strip[K].opacity = 1;
        commentary(wt, null, 0)`

最后看动作块本身：${H} 步 × ${D} 维，所有值都在 bounds99 归一化空间。

D6 的板 ${K}（${c_blockRef(`x(t=0), K=${K}`, L.strip[K])}）是 ${K} 步解码的结果，例如第 0 步左夹爪 = ${f3(kFin?.[14])}；在本检查点（${ckpt}，只训练过 t = 1）上，与它分布内一致的是一步解码 ${c_blockRef('x1 − v', L.actionChunk)}，同一分量 = ${f3(chunk?.[14])}。下面的柱场展示一步解码（这组权重的真实部署输出）；step_160000 到位后换成 ${K} 步结果。
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
