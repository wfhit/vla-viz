import { IProgramState } from "@/src/vla/Program";
import { IRenderView } from "@/src/vla/render/modelRender";
import { IBlkDef } from "@/src/vla/VlaModelLayout";
import { endWalkthroughFrame, stepWalkthrough } from "@/src/vla/walkthrough/Walkthrough";
import { commentary, frameTo, initialFrame, IPhaseGroup, phaseTools } from "@/src/vla/walkthrough/WalkthroughTools";
import { fmtSigned } from "@/src/vla/actionDims";
import { texNorm } from "@/src/groot/GrootModel";
import { IPi0Model } from "./Pi0Model";
import { applyPi0Step, IPi0Layout } from "./Pi0Layout";

export enum PPhase {
    None,
    P0_Overview,
    P1_Vision,
    P2_Prompt,
    P3_Prefix,
    P4_ActionTime,
    P5_Layer0,
    P6_Towers,
    P7_Euler,
    P8_Compare,
}

enum PGroup { Intro, Prefix, Expert, FlowMatching }

export const pi0PhaseList: IPhaseGroup[] = [{
    groupId: PGroup.Intro,
    title: '概览',
    phases: [
        { id: PPhase.P0_Overview, title: 'P0 π0.5 全貌' },
    ],
}, {
    groupId: PGroup.Prefix,
    title: '前缀：PaliGemma',
    phases: [
        { id: PPhase.P1_Vision, title: 'P1 三路相机 → SigLIP → 768 个图像 token' },
        { id: PPhase.P2_Prompt, title: 'P2 指令 + 离散化状态 = 117 个文本 token' },
        { id: PPhase.P3_Prefix, title: 'P3 18 层前缀塔' },
    ],
}, {
    groupId: PGroup.Expert,
    title: '动作专家',
    phases: [
        { id: PPhase.P4_ActionTime, title: 'P4 动作 token 与时间条件' },
        { id: PPhase.P5_Layer0, title: 'P5 第 0 层解剖：两流联合注意力' },
        { id: PPhase.P6_Towers, title: 'P6 两座塔逐层耦合' },
    ],
}, {
    groupId: PGroup.FlowMatching,
    title: '流匹配',
    phases: [
        { id: PPhase.P7_Euler, title: 'P7 10 步 Euler 去噪' },
        { id: PPhase.P8_Compare, title: 'P8 三个模型对照' },
    ],
}];

const f3 = (v: number | undefined, d = 3) => v === undefined || isNaN(v) ? '...' : fmtSigned(v, d);
const fB = (n: number | undefined) => n === undefined ? '...' : n >= 1e9 ? (n / 1e9).toFixed(2) + 'B' : (n / 1e6).toFixed(0) + 'M';
const fE = (v: number | undefined) => v === undefined ? '...' : v === 0 ? '0' : v.toExponential(1);

function dim(cubes: IBlkDef[], keep: IBlkDef[], level: number) {
    let k = new Set(keep);
    for (let c of cubes) if (!k.has(c) && c.opacity > 0) c.opacity = Math.min(c.opacity, level);
}

// mean over the 32 action queries of the share of attention on the prefix keys, per layer
function prefixShare(m: IPi0Model | null, k: number, l: number) {
    let s = m?.steps[k]?.tex['prefix_share']?.localBuffer;
    let H = m?.cfg.action_horizon ?? 32;
    if (!s) return NaN;
    let t = 0;
    for (let i = 0; i < H; i++) t += s[l * H + i];
    return t / H;
}

export function runPi0Walkthrough(state: IProgramState, view: IRenderView) {
    let wt = state.walkthrough;
    stepWalkthrough(state, view);
    let L = state.layout as IPi0Layout;
    let m = state.pageData as IPi0Model | null;
    let { afterTime, breakAfter, c_blockRef } = phaseTools(state);
    let cfg = m?.cfg;
    let T = cfg?.T ?? 117, Np = cfg?.Np ?? 885, nImg = cfg?.n_img ?? 768;
    let H = cfg?.action_horizon ?? 32, AD = cfg?.action_dim ?? 32, D23 = cfg?.behavior_action_dim ?? 23;
    let np = cfg?.n_params;
    let as = cfg?.asserts;
    let a = m?.acts ?? {};
    let all = L.cubes;
    let A = L.anat;
    let pct = (v: number) => isNaN(v) ? '...' : (v * 100).toFixed(0) + '%';

    switch (wt.phase) {

    case PPhase.P0_Overview: {
        initialFrame(state, [...L.vision, ...L.text, ...L.prefixStack, ...L.frontEnd, ...L.expStack, ...L.decode], 278, 6, { fill: 1.0 });
        commentary(wt, null, 0)`

这一页讲 π0.5（Physical Intelligence 的 openpi）。本地检查点是 openpi-comet 的 pi05-b1kpt12-cs32：π0.5 在 BEHAVIOR-1K 上的微调版（12 个任务，含 turning_on_radio）。场景里每一块都是这个检查点的真实权重或真实激活；输入是 BEHAVIOR 2026-challenge-demos 第 0 条示范的第 0 帧，任务 “${cfg?.sample.task ?? '...'}”。

π0.5 由两座并排的 Transformer 塔组成。左边是前缀：PaliGemma（SigLIP 视觉编码器 ${fB(np?.siglip)} + Gemma 2B 语言模型 ${fB(np?.prefix_llm)}），读图像和文字。右边是动作专家：一个更窄的 Gemma（${fB(np?.expert)}），读带噪动作，输出速度场。

两座塔层数相同（${cfg?.prefix.n_layer ?? 18} 层），每一层做一次“联合注意力”：动作 token 的 Q 同时看前缀的 K/V 和自己的 K/V。中间那排紫红色斜线就是这条耦合。

张量都画成竖立的“墙”：横向是 token，竖向是通道，数据从上往下流。通道只画窗口：塔里每层画 16 个通道，解剖面板画 64 个。
`;
        breakAfter();
        let t1 = afterTime(null, 1.5);
        frameTo(state, t1, [...L.prefixStack, ...L.frontEnd, ...L.expStack, ...L.decode], 278, 6, { fill: 1.05 });
        commentary(wt)`

右边的动作专家从上到下：纯噪声 ${c_blockRef('x_t', L.xt)}（${H} 步 × ${AD} 维）经 action_in_proj 变成 ${c_blockRef('动作 token', L.aTok)}，穿过 18 层，经 final adaRMS 和 action_out_proj 得到速度 ${c_blockRef('v', L.v)}。下方的轨迹带把 v 积分 ${cfg?.K ?? 10} 次（t 从 1 走到 0），得到 ${c_blockRef('动作块', L.track[L.track.length - 1])}，再反归一化成 R1 Pro 的 ${D23} 维 ${c_blockRef('动作', L.act23)}。

时间 t 不进入序列，而是经 ${c_blockRef('cond', L.cond)} 走青色时间总线，调制每层的 adaRMSNorm（scale、shift、gate 三根小棒）。
`;
    } break;

    case PPhase.P1_Vision: {
        let view = [...L.vision];
        dim(all, view, 0.15);
        initialFrame(state, view, 278, 6, { fill: 1.1 });
        let bn = a['vit.block_norm']?.localBuffer;
        let meanRow = (r: number) => { if (!bn) return NaN; let s = 0; for (let i = 0; i < nImg; i++) s += bn[r * nImg + i]; return s / nImg; };
        commentary(wt, null, 0)`

三路相机（头部 720²、左腕和右腕 480²）各取一帧，保持比例缩放并补黑边到 224²，像素映射到 [-1, 1]。

视觉编码器是 SigLIP So400m/14：${cfg?.vit.n_block ?? 27} 块，宽 ${cfg?.vit.width ?? 1152}。每帧切成 16 × 16 = 256 个 14 像素的 patch，三路共用同一套权重，得到 ${nImg} 个 token，不做合并。左列画的是头部相机的内部：${c_blockRef('patch 嵌入', L.stem)} → ${c_blockRef('第 0 块', L.vitB0)} → （第 1–25 块，灰色线框）→ ${c_blockRef('第 26 块', L.vitB26)} → ${c_blockRef('encoder_norm', L.vitEnc)}。两个腕部相机只画了 patch 嵌入。

最后一个线性层 ${c_blockRef('img head', L.vitHead)}（1152 → 2048）把它们投到 Gemma 的宽度，得到 ${c_blockRef('768 个图像 token', L.imgTok)}。
`;
        breakAfter();
        commentary(wt)`

真值（三路平均的 patch 残差范数）：第 0 块后 ${f3(meanRow(0), 1)}，第 13 块后 ${f3(meanRow(13), 1)}，第 26 块后 ${f3(meanRow(26), 1)}。
`;
    } break;

    case PPhase.P2_Prompt: {
        let view = [...L.text, L.pre[0]];
        dim(all, view, 0.15);
        initialFrame(state, view, 278, 6, { fill: 1.2 });
        let names = cfg?.action_dim_names ?? [];
        let ex = [3, 7, 21].map(i => cfg ? `${names[i]}：${f3(cfg.state23[i], 3)} → ${f3(cfg.state_norm[i], 3)} → ${cfg.state_tokens[i]}` : '...');
        commentary(wt, null, 0)`

π0 把机器人状态当作连续向量送进动作专家；π0.5 改成把状态离散化后写进文字提示，交给前缀处理。

R1 Pro 的本体感觉有 61 维，从中取出 ${D23} 维 ${c_blockRef('状态', L.state)}（底盘速度 3、躯干 4、左右臂各 7、两个夹爪宽度），按训练集的 q01/q99 归一化到 [-1, 1]，再均分成 256 档，每维变成一个 0–255 的整数。几个真值（原值 → 归一化 → 档位）：
${ex.join('\n')}

提示词是 “Task: …, State: 131 125 …;\\nAction: ”，经 PaliGemma 分词器得到 ${T} 个 ${c_blockRef('token id', L.ids)}（粉框里是状态段）。查 ${c_blockRef('词嵌入表', L.txtW)}（${(cfg?.prefix.vocab ?? 257152).toLocaleString()} 行）并乘 √2048，得到 ${c_blockRef('文本嵌入', L.txtEmb)}。

图像 token 在前、文本 token 在后，拼成 ${Np} 个 token 的前缀，进入 ${c_blockRef('前缀第 0 层', L.pre[0])}。
`;
    } break;

    case PPhase.P3_Prefix: {
        let view = [...L.prefixStack, L.imgTok, L.txtEmb];
        dim(all, view, 0.15);
        initialFrame(state, [...L.prefixStack, L.txtEmb], 278, 6, { fill: 1.1 });
        let tn = a['pre.token_norm'];
        let km = a['pre.key_mass']?.localBuffer;
        let top = (l: number) => {
            if (!km) return '...';
            let best = 0;
            for (let i = 0; i < Np; i++) if (km[l * Np + i] > km[l * Np + best]) best = i;
            let nm = best < nImg ? `图像 token ${best}` : `“${cfg?.tokens[best - nImg] ?? '?'}”`;
            return `${nm}（${pct(km[l * Np + best])}）`;
        };
        commentary(wt, null, 0)`

前缀是 Gemma 2B：${cfg?.prefix.n_layer ?? 18} 层，宽 ${cfg?.prefix.width ?? 2048}，${cfg?.prefix.heads ?? 8} 个 Q 头共享 ${cfg?.prefix.kv_heads ?? 1} 个 KV 头（多查询注意力），每头 ${cfg?.prefix.head_dim ?? 256} 维；MLP 是 GeGLU（2048 → ${(cfg?.prefix.mlp ?? 16384).toLocaleString()} → 2048）。

前缀内部的注意力是双向的：${Np} 个图像和文本 token 互相都能看到（PaliGemma 的 prefix-LM 约定）。

每层的板只画了一部分行：每路相机前 16 个 patch，加上全部 ${T} 个文本 token，一共 ${cfg?.prefix_rows.length ?? 165} 行。
`;
        breakAfter();
        commentary(wt)`

真值。每层注意力最集中的 key（各头、各 query 平均）：第 0 层是 ${top(0)}，第 17 层是 ${top(17)}。前缀残差的 token 范数从 ${tn ? f3(tn.meta.stats.min, 0) : '...'} 到 ${tn ? f3(tn.meta.stats.max, 0) : '...'}，少数 token 带着极大的范数（Gemma 系的 massive activation）。

注意：最后一层的 ${c_blockRef('前缀输出', L.preOut)} 不会被用到。动作专家只读每一层的 K/V，所以前缀塔的作用就是为 18 层各准备一份 K/V。
`;
    } break;

    case PPhase.P4_ActionTime: {
        let view = [...L.frontEnd, L.exp[0].out, L.exp[0].scale, L.exp[0].shift, L.exp[0].gate, L.exp[1].out];
        dim(all, view, 0.15);
        initialFrame(state, view, 278, 6, { fill: 1.2 });
        let sA = a['s0.L0.scale_a']?.localBuffer, gA = a['s0.L0.gate_a']?.localBuffer;
        commentary(wt, null, 0)`

动作专家的输入只有带噪动作。第 0 步的 ${c_blockRef('x_t', L.xt)} 是纯高斯噪声（${H} 步 × ${AD} 维，seed 0）。${AD} 维里前 ${D23} 维对应 R1 Pro 的关节，后 ${AD - D23} 维是补零的填充。${c_blockRef('action_in_proj', L.inProj)}（32 → 1024）把每个时刻投成一个 ${c_blockRef('动作 token', L.aTok)}。

和 π0 不同，π0.5 的时间 t 不拼进动作 token。t 先做正弦编码 ${c_blockRef('sincos(t)', L.pe)}，再经 ${c_blockRef('time_mlp_in', L.tIn)}、${c_blockRef('time_mlp_out', L.tOut)}（各接 swish），得到 ${c_blockRef('cond', L.cond)}。
`;
        breakAfter();
        commentary(wt)`

cond 沿时间总线送到每一层。每层有两个 adaRMSNorm（注意力前、MLP 前），各用一个 Dense 把 cond 映成三组 1024 维的向量：
adaRMS(x) = RMSNorm(x) · (1 + scale) + shift，
残差 x ← x + gate · 子层输出。
场景里每层右侧的三根小棒就是注意力那一组的 scale、shift、gate。

真值（第 0 步，第 0 层，第 7 个通道）：scale_a = ${f3(sA?.[7])}，gate_a = ${f3(gA?.[7])}。
`;
    } break;

    case PPhase.P5_Layer0: {
        dim(all, [...A.cubes, L.cond], 0.12);
        initialFrame(state, A.pre.cubes, 276, 6, { fill: 1.1 });
        let pr = a['pre.L0.probs'];
        commentary(wt, null, 0)`

把第 0 层拆开。左边是前缀那一路（Gemma 2B 的权重），右边是动作专家那一路（300M 的权重）。两路各有一整套自己的权重：norm、W_q/W_k/W_v、W_o、GeGLU 的 W_gate/W_up/W_down。所谓“联合”，只发生在注意力这一步。

前缀这一路就是一层普通的 Gemma：${c_blockRef('RMSNorm', A.pre.norm)} → ${c_blockRef('Q', A.pre.q)}、${c_blockRef('K', A.pre.k)}、${c_blockRef('V', A.pre.v)} → ${c_blockRef('注意力', A.pre.probs)}（只看前缀的 ${Np} 个 key，值域 ${pr ? `${f3(pr.meta.stats.min, 2)}..${f3(pr.meta.stats.max, 2)}` : '...'}）→ 残差 → GeGLU → 残差。
`;
        breakAfter();
        let t1 = afterTime(null, 1.3);
        frameTo(state, t1, [...A.exp.cubes, A.mask], 276, 6, { fill: 1.1 });
        let sh0 = prefixShare(m, 0, 0);
        commentary(wt)`

动作专家这一路：${c_blockRef('adaRMS', A.exp.norm)} 用 cond 调制；自己的 ${c_blockRef('Q', A.exp.q)} 去看 ${Np} + ${H} 个 key：前缀的 K（紫红色箭头，来自左边那一路）拼上自己的 K。${c_blockRef('注意力矩阵', A.exp.probs)} 的右端框出了 ${H} 个动作 key。

${c_blockRef('注意力掩码', A.mask)}（每 8 行取 1 行）说明了方向：动作 token 能看前缀，前缀看不到动作 token。导出器核对过：改变动作 token，前缀的激活变化为 ${fE(as?.['prefix_blind_to_suffix'])}。

真值：第 0 层、第 0 步，动作 query 把 ${pct(sh0)} 的注意力放在前缀上。

注意力输出乘 ${c_blockRef('gate_a', A.exp.gateA)} 后加回残差，MLP 同理（${c_blockRef('gate_f', A.exp.gateF)}）。门控残差恒等式 x_out = r1 + gate_f · mlp 核对误差 ${fE(as?.['gated_residual_identity_maxabs(L0,step0)'])}。
`;
    } break;

    case PPhase.P6_Towers: {
        let view = [...L.prefixStack, ...L.expStack, L.aTok, L.final];
        dim(all, [...view, L.cond], 0.12);
        initialFrame(state, view, 278, 6, { fill: 1.05 });
        let ln = m?.steps[0]?.tex['layer_norm']?.localBuffer;
        let meanRow = (r: number) => { if (!ln) return NaN; let s = 0; for (let i = 0; i < H; i++) s += ln[r * H + i]; return s / H; };
        commentary(wt, null, 0)`

两座塔逐层对齐：专家第 l 层的板和前缀第 l 层的输入板平齐，紫红色斜线表示“第 l 层的动作 query 读第 l 层的前缀 K/V”。右侧标签是这一层动作 query 放在前缀上的注意力份额（第 0 步，32 个 query 平均）：第 0 层 ${pct(prefixShare(m, 0, 0))}，第 8 层 ${pct(prefixShare(m, 0, 8))}，第 17 层 ${pct(prefixShare(m, 0, 17))}。

因为前缀看不到动作 token，前缀的 18 层 K/V 和去噪步无关：推理时只算一次存进 KV cache，10 个 Euler 步都复用，每一步只跑专家那座塔。导出器分别用“cache + 只跑专家”和“两塔联合前向”算第 0 步的 v，最大差 ${fE(as?.['cache_vs_joint_v_maxabs(step0)'])}。

动作 token 的平均范数从输入的 ${f3(meanRow(0), 1)} 涨到第 18 层输出的 ${f3(meanRow(18), 1)}，然后 ${c_blockRef('final adaRMS', L.final)} 把它拉回来，交给 action_out_proj。
`;
    } break;

    case PPhase.P7_Euler: {
        let view = [L.final, L.v, L.outProj, L.track[0], L.track[L.track.length - 1], L.act23].map(b => ({ x: b.x, y: b.y, z: b.z, dx: b.dx, dy: b.dy, dz: b.dz }));
        initialFrame(state, view, 276, 4, { fill: 1.2, shiftX: 0.2 });
        let K = cfg?.K ?? 10;
        commentary(wt, null, 0)`

流匹配把“生成动作”变成“沿速度场走”。openpi 的约定：t = 1 是纯噪声，t = 0 是动作。每一步用当前 x_t 和 t 跑一次专家塔（前缀 K/V 来自 cache），得到 v，然后 x ← x + v · dt，dt = ${cfg?.dt ?? -0.1}。一共 ${K} 步，t = 1.0, 0.9, …, 0.1。

下面依次播放 ${K} 步。每一步专家塔里所有层的激活、adaRMS 小棒、cond 都换成这一步的真值，轨迹带上多出一格。
`;
        breakAfter();
        let tRun = afterTime(null, 12);
        let u = tRun.t;
        let k = Math.min(K - 1, Math.floor(u * K));
        let frac = u * K - k;
        applyPi0Step(L, m, k);
        for (let j = 0; j < L.track.length; j++) L.track[j].opacity = j <= k + (frac > 0.85 ? 1 : 0) ? 1 : 0.15;
        L.act23.opacity = k === K - 1 && frac > 0.85 ? 1 : 0.15;
        if (tRun.active && u < 1) {
            let li = Math.min(L.exp.length - 1, Math.floor(frac / 0.8 * L.exp.length));
            if (frac < 0.8) L.exp[li].out.highlight = 0.5;
            L.cond.highlight = 0.3;
        }
        let st = m?.steps[k];
        commentary(wt, tRun)`

当前：第 ${k + 1} 步，t = ${st ? st.t.toFixed(1) : '...'}，速度范数 |v| = ${f3(st?.v_norm, 2)}，其中 R1 Pro 的 23 维部分 ${f3(st?.v23_norm, 2)}。${K} 步的 |v| 几乎不变（${m ? m.steps.map(s => s.v_norm.toFixed(1)).join(' → ') : '...'}）：这正是直线流的特点，速度场在整条轨迹上接近常数。
`;
        breakAfter();
        let an = cfg?.action_norm, a23 = cfg?.action23, names = cfg?.action_dim_names ?? [];
        commentary(wt)`

最后一格是 ${c_blockRef('动作块', L.track[L.track.length - 1])}（归一化空间，${H} × ${AD}）。取前 ${D23} 维、按 q01/q99 反归一化，得到 ${c_blockRef('R1 Pro 动作', L.act23)}。第 0 个时刻的几个值：${a23 ? [3, 7, 14, 21].map(i => `${names[i]} = ${f3(a23![0][i], 3)}`).join('，') : '...'}。

核对：同一组输入交给官方 openpi JAX 代码（Pi0Config(pi05=True).sample_actions），归一化动作的最大差 ${fE(as?.['vs_official_jax_action_maxabs'])}（第 0 个时刻前两维：本页 ${an ? `${f3(an[0][0], 4)}, ${f3(an[0][1], 4)}` : '...'}）。Euler 恒等式误差 ${fE(as?.['euler_identity_maxabs'])}。计算用 fp32（官方运行时是 bf16）。
`;
    } break;

    case PPhase.P8_Compare: {
        initialFrame(state, [...L.vision, ...L.text, ...L.prefixStack, ...L.frontEnd, ...L.expStack, ...L.decode], 278, 6, { fill: 1.0 });
        commentary(wt, null, 0)`

三个模型并排对照（k005b 见 /vla，GR00T 见 /groot）：

主干：k005b 是 Qwen3-VL，28 层；GR00T 是 Cosmos-Reason2-2B，只用前 16 层；π0.5 是 PaliGemma（SigLIP + Gemma 2B，18 层）。

主干和动作专家怎么耦合：k005b 和 π0.5 是逐层联合注意力，动作 token 每一层都读主干同一层的 K/V；GR00T 是交叉注意力，DiT 只读主干最后输出的 VL token。

状态怎么进来：k005b 用 state_proj（61 → 768）投成连续 token；GR00T 用本体适配器（slot 23）；π0.5 把状态离散成 256 档写进文字提示。

时间条件：k005b 用 AdaRMSNorm；GR00T 用 AdaLN（scale、shift）；π0.5 用 adaRMSNorm（scale、shift、gate），时间 t 不进入 token。

去噪：k005b 10 步，16 × 23；GR00T 4 步，40 × 132，t 从 0 走到 1；π0.5 10 步，${H} × ${AD}（前 ${D23} 维是 R1 Pro），t 从 1 走到 0。

三个模型都在 R1 Pro 上：k005b 只服务 R1 Pro；GR00T 和 π0.5 是多机器人模型，这里用的都是它们在 BEHAVIOR 上训练过的版本。本页的 π0.5 检查点训练过这个任务，所以它的动作块有物理意义。
`;
    } break;
    }

    endWalkthroughFrame(wt);
}
