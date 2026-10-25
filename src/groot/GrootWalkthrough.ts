import { IProgramState } from "@/src/vla/Program";
import { IRenderView } from "@/src/vla/render/modelRender";
import { IBlkDef } from "@/src/vla/VlaModelLayout";
import { endWalkthroughFrame, stepWalkthrough } from "@/src/vla/walkthrough/Walkthrough";
import { commentary, frameTo, initialFrame, IPhaseGroup, phaseTools } from "@/src/vla/walkthrough/WalkthroughTools";
import { fmtSigned } from "@/src/vla/actionDims";
import { IGrootModel, texNorm } from "./GrootModel";
import { applyGrootStep, IGrootLayout } from "./GrootLayout";

export enum GPhase {
    None,
    G0_Overview,
    G1_Inputs,
    G2_Backbone,
    G3_Adapters,
    G4_ActionTokens,
    G5_CrossBlock,
    G6_SelfBlock,
    G7_Stack,
    G8_Euler,
    G9_Compare,
}

enum GGroup { Intro, System2, System1, FlowMatching }

export const grootPhaseList: IPhaseGroup[] = [{
    groupId: GGroup.Intro,
    title: '概览',
    phases: [
        { id: GPhase.G0_Overview, title: 'G0 GR00T N1.7 全貌' },
    ],
}, {
    groupId: GGroup.System2,
    title: 'System 2：视觉语言主干',
    phases: [
        { id: GPhase.G1_Inputs, title: 'G1 三路相机 + 指令 = 216 个 token' },
        { id: GPhase.G2_Backbone, title: 'G2 16 层 LM 与 VL 头' },
    ],
}, {
    groupId: GGroup.System1,
    title: 'System 1：DiT 动作头',
    phases: [
        { id: GPhase.G3_Adapters, title: 'G3 本体适配器（slot 23）' },
        { id: GPhase.G4_ActionTokens, title: 'G4 动作 token 与时间嵌入' },
        { id: GPhase.G5_CrossBlock, title: 'G5 D0 解剖：交叉注意力' },
        { id: GPhase.G6_SelfBlock, title: 'G6 D1 解剖：自注意力' },
        { id: GPhase.G7_Stack, title: 'G7 32 块交替堆叠' },
    ],
}, {
    groupId: GGroup.FlowMatching,
    title: '流匹配',
    phases: [
        { id: GPhase.G8_Euler, title: 'G8 4 步 Euler 去噪' },
        { id: GPhase.G9_Compare, title: 'G9 三个模型对照' },
    ],
}];

const f3 = (v: number | undefined, d = 3) => v === undefined || isNaN(v) ? '...' : fmtSigned(v, d);
const fB = (n: number | undefined) => n === undefined ? '...' : n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : (n / 1e6).toFixed(0) + 'M';

function dim(cubes: IBlkDef[], keep: IBlkDef[], level: number) {
    let k = new Set(keep);
    for (let c of cubes) if (!k.has(c) && c.opacity > 0) c.opacity = Math.min(c.opacity, level);
}

// share of a block's attention mass (mean over heads and queries) on each camera's image tokens
function cameraShares(m: IGrootModel | null, k: number, b: number) {
    let mass = m?.steps[k]?.tex[`D${b}.attn_mass`]?.localBuffer;
    if (!m || !mass) return null;
    return m.cfg.image_runs.map(([s, e]) => {
        let t = 0;
        for (let i = s; i < e; i++) t += mass![i];
        return t;
    });
}

export function runGrootWalkthrough(state: IProgramState, view: IRenderView) {
    let wt = state.walkthrough;
    stepWalkthrough(state, view);
    let L = state.layout as IGrootLayout;
    let m = state.pageData as IGrootModel | null;
    let { afterTime, breakAfter, c_blockRef } = phaseTools(state);
    let cfg = m?.cfg;
    let S = cfg?.S ?? 216, H = cfg?.action_horizon ?? 40, AD = cfg?.action_dim ?? 132;
    let nImg = cfg ? cfg.image_runs.reduce((a, r) => a + r[1] - r[0], 0) : 192;
    let np = cfg?.n_params;
    let a = m?.acts ?? {};
    let all = L.cubes;
    let anat = L.anat.flatMap(p => p.cubes);
    let toyNote = 'slot 23 在这个检查点里没有训练过（逐槽标准差 = 初始化的 0.02）：适配器权重是真权重，但数值等于初始化，属于 toy；解码出的动作没有物理意义。';

    switch (wt.phase) {

    case GPhase.G0_Overview: {
        initialFrame(state, all, 278, 6, { fill: 1.0 });
        commentary(wt, null, 0)`

这一页讲 NVIDIA GR00T N1.7（本地 ${cfg?.model ?? 'GR00T-N1.7-3B'}，${cfg?.global_step ?? 'global_step196000'}）。场景里每一块都是真实权重或真实激活：输入是 BEHAVIOR 2026-challenge-demos 第 0 条示范的第 0 帧，任务是 “${cfg?.sample.task ?? '...'}”。

GR00T 是“双系统”结构。左半边是 System 2：一个视觉语言主干（Cosmos-Reason2-2B，Qwen3-VL 架构，${fB(np?.backbone)} 参数），慢，负责“看懂并理解指令”。右半边是 System 1：一个 32 块的 DiT 动作头（${fB(np?.action_head)} 参数），快，负责把理解变成 ${H} 步 × ${AD} 维的动作块。

两个系统之间只有一条接口：主干最后输出的 ${S} 个 VL token。DiT 通过交叉注意力读取它们（中间那两根竖条），而不是像 k005b 那样逐层联合注意力。

所有张量都画成竖立的“墙”：横向是 token，竖向是通道，数据从上往下流。通道只画一个窗口：重复堆叠的层画前 16 个通道，单独的块画 32 个，D0/D1 解剖面板画 64 个。
`;
        breakAfter();
        let t1 = afterTime(null, 1.5);
        frameTo(state, t1, [...L.ditStack, ...L.frontEnd, ...L.decode], 278, 6, { fill: 1.05 });
        commentary(wt)`

右边的 System 1 从上到下：三条输入链（状态、带噪动作、时间 t）汇成一条 ${H + 1} 个 token 的 ${c_blockRef('sa 序列', L.saIn)}，穿过 32 个 DiT 块，经 final AdaLN 和动作解码器得到速度 ${c_blockRef('v', L.v)}；左下的轨迹带把 v 积分 4 次，得到最终的 ${c_blockRef('动作块', L.track[4])}。

DiT 右侧的青色竖线是时间总线：时间嵌入 temb 送到每一块的 AdaLN（scale、shift 两根小棒）。
`;
    } break;

    case GPhase.G1_Inputs: {
        let view = [L.ids, L.vitPatch, L.vitB0, L.vitB23, L.visTok, L.lm[0]];
        dim(all, view, 0.15);
        initialFrame(state, view, 278, 6, { fill: 1.1 });
        let tn = a['lm.token_norm']?.localBuffer;
        let maxTok = 0;
        if (tn) for (let i = 0; i < S; i++) if (tn[i] > tn[maxTok]) maxTok = i;
        commentary(wt, null, 0)`

输入有两类。一是三路相机：头部（720²）、左腕和右腕（480²）各一帧，统一缩放到 256²。二是语言指令，套进 Qwen3-VL 的对话模板。

视觉编码器是 ${cfg?.vit.n_block ?? 24} 块的 ViT（宽 ${cfg?.vit.width ?? 1024}）。每帧切成 16×16 的 patch，得到 256 个 ${c_blockRef('patch 嵌入', L.vitPatch)}；穿过 24 块后，2×2 合并成 64 个 ${c_blockRef('视觉 token', L.visTok)}，三路一共 ${nImg} 个。中间灰色线框是没有导出的第 1–22 块。第 ${(cfg?.vit.deepstack ?? [5, 11, 17]).join('、')} 块的特征还会作为 deepstack 直接加进 LM 的前几层。

${c_blockRef('token id', L.ids)} 一共 ${S} 个：3 段各 64 个 <|image_pad|>（被视觉 token 替换），其余 ${S - nImg} 个是模板和指令文字。两路汇合成 ${c_blockRef('LM 第 0 层输入', L.lm[0])}。
`;
        breakAfter();
        commentary(wt)`

一个真值：进入 LM 前，第 ${maxTok} 个 token（“${cfg?.tokens[maxTok] ?? '?'}”）的残差范数是 ${tn ? tn[maxTok].toFixed(1) : '...'}，其余 token 大多在个位到几十。少数 token 带着极大的范数，这是 Qwen 系 LM 里常见的 massive activation 现象。
`;
    } break;

    case GPhase.G2_Backbone: {
        let view = [...L.lm, L.attn0, L.attn15, L.lmOut, L.vlln, ...L.vlsa];
        dim(all, view, 0.15);
        initialFrame(state, view, 278, 6, { fill: 1.05 });
        let tn = a['lm.token_norm']?.localBuffer;
        let vn = a['vl.token_norm'];
        commentary(wt, null, 0)`

LM 原本有 ${cfg?.lm.n_layer_full ?? 28} 层（宽 ${cfg?.lm.width ?? 2048}，${cfg?.lm.heads ?? 16} 个 Q 头、${cfg?.lm.kv_heads ?? 8} 个 KV 头）。GR00T 只用前 ${cfg?.lm.n_layer ?? 16} 层：配置 select_layer = 16，后 12 层在加载时就被删掉了，推理时根本不跑（红色线框）。

每一层都是因果自注意力。右边两块是真实的注意力图（16 个头平均）：${c_blockRef('第 0 层', L.attn0)} 和 ${c_blockRef('第 15 层', L.attn15)}。
`;
        breakAfter();
        let t1 = afterTime(null, 1.2);
        frameTo(state, t1, [L.lm[15], L.lmOut, L.vlln, ...L.vlsa], 278, 6, { fill: 1.2 });
        commentary(wt)`

主干输出 ${c_blockRef('lm.out', L.lmOut)} 之后还有一个 VL 头，它属于动作头一侧、和 DiT 一起训练：先 ${c_blockRef('vlln', L.vlln)}（LayerNorm），再 4 块 ${c_blockRef('vl_self_attention', L.vlsa[3])}（32 头 × 64）。它的输出就是 DiT 读取的 K/V 来源。

真值：主干输出的 token 范数最大 ${tn ? Math.max(...Array.from(tn.slice(16 * S, 17 * S))).toFixed(0) : '...'}，经过 VL 头后被压到 ${vn ? `${vn.meta.stats.min.toFixed(1)}–${vn.meta.stats.max.toFixed(1)}，平均 ${vn.meta.stats.mean.toFixed(1)}` : '...'}。导出器用官方模块核对过这一段，最大误差 ${cfg ? cfg.asserts['vlsa_vs_official_maxabs'].toExponential(1) : '...'}。
`;
    } break;

    case GPhase.G3_Adapters: {
        let view = [L.stateRaw, L.stateNorm, L.stateL1, L.stateH, L.stateL2, L.stateFeat, L.xt, L.aW1, L.aEmb, L.decL1, L.decH, L.decL2, L.v];
        dim(all, view, 0.15);
        for (let b of L.slotted) b.highlight = 0.35;
        initialFrame(state, [L.stateRaw, L.stateNorm, L.stateL1, L.stateL2, L.stateFeat, L.xt, L.aW1, L.aW3, L.posEmb], 278, 6, { fill: 1.15 });
        let sr = a['state.raw'];
        commentary(wt, null, 0)`

GR00T 用一个模型服务很多种机器人。机器人相关的部分只有三个“本体适配器”：状态编码器、动作编码器、动作解码器。每个都有 32 个槽位，按 embodiment id 选一个。这个检查点里 BEHAVIOR R1 Pro 模拟身体是 sim_behavior_r1_pro，id = ${cfg?.embodiment.slot ?? 23}。场景里每个带 [slot 23] 的权重后面的灰色线框，代表另外 31 个槽。

状态：BEHAVIOR 给的是 ${cfg?.behavior_state_dim ?? 61} 维本体感觉（${c_blockRef('state', L.stateRaw)}，值域 ${sr ? `${f3(sr.meta.stats.min, 2)}..${f3(sr.meta.stats.max, 2)}` : '...'}）。按 q01/q99 归一化到 [-1, 1]，填进 ${cfg?.state_dim ?? 132} 维槽的前 61 维（其余为 0），然后两层 MLP 变成 1 个 ${c_blockRef('状态 token', L.stateFeat)}。

动作同理：61 维状态和 ${cfg?.behavior_action_dim ?? 23} 维动作各自投影到隐空间，互不相干。动作块的 ${AD} 维里只有前 ${cfg?.behavior_action_dim ?? 23} 维对应 R1 Pro 的关节。

注意：${toyNote}
`;
    } break;

    case GPhase.G4_ActionTokens: {
        let view = [L.xt, L.aW1, L.aEmb, L.tau, L.aW2, L.w2, L.aW3, L.aFeat, L.posEmb, L.aFeatPos, L.tProj, L.timeLin1, L.temb, L.saIn, L.stateFeat];
        dim(all, view, 0.15);
        initialFrame(state, view, 278, 6, { fill: 1.1 });
        let st0 = m?.steps[0];
        commentary(wt, null, 0)`

DiT 的输入序列有 ${H + 1} 个 token：1 个状态 token + ${H} 个动作 token。

动作 token 来自带噪动作 ${c_blockRef('x_t', L.xt)}（${H} × ${AD}，第 0 步就是纯高斯噪声，seed 0）。动作编码器做三件事：W1 把每个时刻投影到 1536 维；把它和时间的正弦编码 ${c_blockRef('tau', L.tau)} 拼起来，经 W2 + swish 混合；W3 再投影一次，最后加上可学习的位置编码 ${c_blockRef('pos_emb', L.posEmb)}。所以时间 t 在这里已经进入了每个动作 token。

时间还有第二条路：t 离散成 0–999 的整数（第 0 步 t_disc = ${st0?.t_disc ?? 0}），经 ${c_blockRef('Timesteps(256)', L.tProj)} 正弦编码和两层 MLP 变成 ${c_blockRef('temb', L.temb)}（1536 维，真值范数 ${f3(texNorm(st0?.tex['temb']), 2)}，这是窗口内 64 维的范数）。temb 不进入序列，而是走右边的时间总线去调制每一块的 AdaLN。

拼好的 ${c_blockRef('sa 序列', L.saIn)} 就是 DiT 第 0 块的输入。
`;
    } break;

    case GPhase.G5_CrossBlock: {
        let p = L.anat[0];
        dim(all, [...p.cubes, L.temb], 0.12);
        initialFrame(state, p.cubes, 276, 6, { fill: 1.2 });
        let sc = a['s0.D0.scale']?.localBuffer, sh = a['s0.D0.shift']?.localBuffer;
        let pr = a['s0.D0.probs'];
        commentary(wt, null, 0)`

现在把第 0 块拆开。它是一个“交叉注意力：文本”块。这是第 0 步的真实激活。

第一步是 AdaLN：${c_blockRef('norm1.linear', p.ada)} 把 SiLU(temb) 线性映射成 ${c_blockRef('scale', p.scale)} 和 ${c_blockRef('shift', p.shift)}，然后
AdaLN(x) = LayerNorm(x) · (1 + scale) + shift。
真值：scale 前 4 个通道 = ${sc ? Array.from(sc.slice(0, 4)).map(v => f3(v)).join('、') : '...'}；shift 前 4 个 = ${sh ? Array.from(sh.slice(0, 4)).map(v => f3(v)).join('、') : '...'}。这些数随 t 变化，是 DiT 知道“现在去噪到哪一步”的唯一途径。
`;
        breakAfter();
        let t1 = afterTime(null, 1.2);
        frameTo(state, t1, [p.n1, p.q, p.ctx!, p.k, p.v, p.probs, p.attn], 278, 6, { fill: 1.1 });
        commentary(wt)`

Q 来自动作序列：${c_blockRef('Q', p.q)}（${H + 1} 行）。K 和 V 不来自自己，而来自 ${c_blockRef('VL token', p.ctx!)}（全部 ${S} 个，${cfg?.dit.heads ?? 32} 头 × ${cfg?.dit.head_dim ?? 48} 维）。这个块是“文本”块，所以图像 token 被 mask 掉：${c_blockRef('softmax 概率', p.probs)} 里 ${nImg} 列图像 key 全是 0（值域 ${pr ? `${pr.meta.stats.min.toFixed(0)}..${pr.meta.stats.max.toFixed(2)}` : '...'}），动作 token 只看那 ${S - nImg} 个文字和模板 token。

然后是残差 ${c_blockRef('x + attn', p.r1)}，再过一个不带仿射参数的 ${c_blockRef('LayerNorm', p.n3)}，接 GELU 前馈（1536 → 6144 → 1536），再残差。注意第二个 norm 不受 temb 调制：GR00T 每块只有 1 处 AdaLN，k005b 每层有 2 处。

导出器用官方 DiT 模块逐块核对，最大误差 ${cfg ? cfg.asserts['dit_vs_official_maxabs(step2)'].toExponential(1) : '...'}。
`;
    } break;

    case GPhase.G6_SelfBlock: {
        let p = L.anat[1];
        dim(all, [...p.cubes, L.temb], 0.12);
        initialFrame(state, p.cubes, 276, 6, { fill: 1.2 });
        let sc = a['s0.D1.scale'];
        commentary(wt, null, 0)`

第 1 块是自注意力块：结构和第 0 块完全相同，唯一的区别是 K、V 来自序列自己（${c_blockRef('AdaLN(x)', p.n1)} 同时给出 Q、K、V），所以 ${c_blockRef('注意力矩阵', p.probs)} 是 ${H + 1} × ${H + 1} 的方阵：状态 token 和 40 个动作 token 互相看，不带因果 mask。

这一块的 scale 值域是 ${sc ? `${f3(sc.meta.stats.min, 2)}..${f3(sc.meta.stats.max, 2)}，均值 ${f3(sc.meta.stats.mean, 2)}` : '...'}。均值为负，意味着 t = 0 时它整体把归一化后的激活压小。

交叉块从主干“读信息”，自注意力块在动作 token 之间“传信息”。这两种块交替出现。
`;
    } break;

    case GPhase.G7_Stack: {
        let view = [L.saIn, ...L.ditStack, L.finNorm];
        dim(all, [...view, L.temb, L.vlsa[3]], 0.12);
        initialFrame(state, view, 278, 6, { fill: 1.1 });
        let sh2 = cameraShares(m, 0, 2), sh30 = cameraShares(m, 0, 30);
        let pct = (v: number) => (v * 100).toFixed(0) + '%';
        let bn = m?.steps[0]?.tex['block_norm']?.localBuffer;
        let meanRow = (r: number) => { if (!bn) return NaN; let s = 0; for (let i = 0; i < H + 1; i++) s += bn[r * (H + 1) + i]; return s / (H + 1); };
        commentary(wt, null, 0)`

32 个块按 “文本交叉 → 自注意力 → 图像交叉 → 自注意力” 的节奏重复 8 次（interleave_self_attention = true）。右侧标签标出每一块的类型。粉色箭头从 ${c_blockRef('文本 token 竖条', L.vlText)} 进入文本块，紫色箭头从 ${c_blockRef('图像 token 竖条', L.vlImage)} 进入图像块。两根竖条是同一份 VL token 按 image_mask 分成的两组。

真值：图像块的注意力在三路相机之间怎么分？第 2 块：头部 ${sh2 ? pct(sh2[0]) : '...'}、左腕 ${sh2 ? pct(sh2[1]) : '...'}、右腕 ${sh2 ? pct(sh2[2]) : '...'}；第 30 块：头部 ${sh30 ? pct(sh30[0]) : '...'}、左腕 ${sh30 ? pct(sh30[1]) : '...'}、右腕 ${sh30 ? pct(sh30[2]) : '...'}。

序列的平均 token 范数从输入的 ${f3(meanRow(0), 1)} 涨到第 32 块输出的 ${f3(meanRow(32), 1)}。最后的 ${c_blockRef('final AdaLN', L.finNorm)} 又用 temb 调制一次，然后 proj_out_2 投到 1024 维、动作解码器投到 ${AD} 维。
`;
    } break;

    case GPhase.G8_Euler: {
        // fixed boxes: the track cells fade in step by step, the frame must not follow them
        let view = [L.finNorm, L.ditOut, L.projOut2, L.v, L.decL2, L.track[0], L.track[4]].map(b => ({ x: b.x, y: b.y, z: b.z, dx: b.dx, dy: b.dy, dz: b.dz }));
        initialFrame(state, view, 276, 4, { fill: 1.3, shiftX: 0.3 });
        commentary(wt, null, 0)`

流匹配把“生成动作”变成“沿一个速度场走”。GR00T 的约定：t = 0 是纯噪声，t = 1 是动作（和 k005b、π0 的方向相反，它们从 t = 1 走到 0）。

每一步：用当前的 x_t 和 t 跑一次完整的 DiT（32 块），得到速度 v，然后 x ← x + v · 1/4。一共 ${cfg?.K ?? 4} 步，t = ${(cfg?.t_cont ?? [0, .25, .5, .75]).join('、')}。主干的 VL token 只算一次，4 步都复用。

下面依次播放 4 步。每一步 DiT 里所有块的激活、AdaLN 的 scale/shift、temb 都换成这一步的真值，左下轨迹带上多出一格。
`;
        breakAfter();
        let tRun = afterTime(null, 10);
        let u = tRun.t;
        let k = Math.min(3, Math.floor(u * 4));
        let frac = u * 4 - k;
        applyGrootStep(L, m, k);
        for (let j = 0; j < L.track.length; j++) L.track[j].opacity = j <= k + (frac > 0.85 ? 1 : 0) ? 1 : 0.15;
        if (tRun.active && u < 1) {
            let bi = Math.min(31, Math.floor(frac / 0.8 * 32));
            if (frac < 0.8) L.dit[bi].out.highlight = 0.5;
            L.temb.highlight = 0.3;
        }
        let st = m?.steps[k];
        commentary(wt, tRun)`

当前：第 ${k + 1} 步，t = ${st?.t ?? '...'}（t_disc = ${st?.t_disc ?? '...'}），速度范数 |v| = ${f3(st?.v_norm, 2)}，其中 R1 Pro 的 23 维部分 ${f3(st?.v23_norm, 2)}。4 步的 |v|：${m ? m.steps.map(s => s.v_norm.toFixed(2)).join(' → ') : '...'}。

Euler 恒等式（x_next = x_t + v/4）导出时核对过，最大误差 ${cfg ? cfg.asserts['euler_identity_maxabs'].toExponential(1) : '...'}。最后一格就是输出的 ${c_blockRef('动作块', L.track[4])}（${H} × ${AD}，范数 ${f3(texNorm(m?.steps[3]?.tex['x_next']), 2)}，窗口内）。再提醒一次：slot 23 未训练，所以这个动作块本身没有物理意义，但每一步的计算都是真实的。
`;
    } break;

    case GPhase.G9_Compare: {
        initialFrame(state, all, 278, 6, { fill: 1.0 });
        commentary(wt, null, 0)`

三个模型并排对照（k005b 见 /vla，π0.5 见 /pi0）：

主干：k005b 是 Qwen3-VL，28 层全用；GR00T 是 Cosmos-Reason2-2B（同为 Qwen3-VL 架构），只用前 16 层；π0.5 是 PaliGemma（Gemma 2B）。

主干和动作专家怎么耦合：k005b 和 π0.5 是逐层联合注意力，动作 token 在每一层都和主干 token 拼在一起做注意力；GR00T 是交叉注意力，DiT 只读主干最后输出的 VL token，并且文本块和图像块交替读。

时间条件：k005b 用 AdaRMSNorm，每层 2 处；GR00T 用 AdaLN（scale、shift），每块 1 处，外加 final AdaLN；π0.5 用 adaRMSNorm。

去噪：k005b 10 步，16 × 23 动作块；GR00T 4 步，${H} × ${AD}（R1 Pro 用前 23 维）；π0.5 10 步。时间方向：GR00T 从 0 走到 1，另外两个从 1 走到 0。

本体：k005b 只服务 R1 Pro；GR00T 用 32 槽的本体适配器，这里选 slot 23（未训练）。
`;
    } break;
    }

    // TEMP camera-tuning hook
    endWalkthroughFrame(wt);
}
