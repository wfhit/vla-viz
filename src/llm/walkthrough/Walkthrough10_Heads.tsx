import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { cameraKeyframes, commentary, frameCubes, IWalkthroughArgs, phaseClock, setInitialCamera } from "./WalkthroughTools";
import { clamp } from "@/src/utils/data";
import { describeActionRow, fmtSigned, normalizeAction } from "../actionDims";

// the head-specific visuals are drawn by components/HeadsOverlay.tsx, driven by wt.time over [0, 12]
export const headsAnimLength = 12;

export function walkthrough10_Heads(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { afterTime } } = args;
    let cfg = layout.shape.full;
    let m = state.jsGptModel;

    let L = layout;
    let headBlocks = [L.ln_f.lnResid, L.hStar, L.lmHeadWeight, L.logits, L.logitsSoftmax, L.doneWeight, L.doneLogits, L.doneProbs];
    let fieldBox = (() => {
        let f = L.actionField;
        let h = f.heightScale * 2.6;
        return { x: f.x - 8, y: f.y - h, z: f.z, dx: f.nDims * f.pitch + 30, dy: 2 * h + 20, dz: f.nSteps * f.pitch + 4 };
    })();
    let actionBlocks = [L.expertOut, L.actionWeight, L.actionVelocity, L.actionX1, L.actionChunk, fieldBox];
    // the 2D panel sits on the right during the action chapters: keep the 3D content left of it
    let side = { shiftX: 0.9, aspect: 0.8 };

    if (wt.phase === Phase.Head_Knack || wt.phase === Phase.Head_Done || wt.phase === Phase.Harness_Law) {
        let cam = frameCubes(headBlocks, 282, 10, { fill: 1.3 });
        setInitialCamera(state, cam.center, cam.angle);
    }

    switch (wt.phase) {
    case Phase.Head_Knack: {
        let c = commentary(wt, null, 0)`

从 h* 出发，第一路是 knack 分类头：一个 17 × 2048 的线性层（knack_head），对 17 个 knack（${cfg?.knack_names.join(' / ') ?? '…'}）各输出一个 logit，再做 softmax。

右侧的条形图是本样本的真实 logits / 概率：argmax 为 ${cfg?.pred_knack ?? '…'}，标注目标为 ${cfg?.sample.target_knack ?? '…'}。

因为是一次分类，没有逐 token 的自回归解码，也不需要采样温度。

`;
        afterTime(c, headsAnimLength);
    } break;
    case Phase.Head_Done: {
        let c = commentary(wt, null, 0)`

第二路是 done 头：一个 2 × 2048 的线性层（done_head），输出 [NOT_DONE, DONE] 两个 logit。

done 只回答“当前 subtask 是否已经完成”，它与 knack 头共享 h*，但各自独立做 softmax。本样本预测 done = ${String(cfg?.pred_done ?? '…')}，目标 ${String(cfg?.sample.target_done ?? '…')}。

`;
        afterTime(c, headsAnimLength);
    } break;
    case Phase.Action_Expert: {
        let nPairs = cfg?.n_layer_pairs ?? 28;
        let pct = (v: number) => (v * 100).toFixed(1) + '%';
        let massLine = (layers: number[]) => !m || !cfg ? '…' : layers.map(i => {
            let ms = m!.expertLayers[i].mass, n = cfg!.n_suffix, acc = [0, 0, 0, 0];
            for (let q = 0; q < n; q++) for (let k = 0; k < 4; k++) acc[k] += ms[q * 4 + k] / n;
            return `第 ${i} 层：视觉 ${pct(acc[1])}、文本 ${pct(acc[2])}、后缀自身 ${pct(acc[3])}`;
        }).join('；');
        let c = commentary(wt, null, 0)`

动作专家是与主干并行、权重独立的另一个 transformer：隐藏维 ${cfg?.expert_hidden ?? 768}，${cfg?.expert_heads ?? 16} 个 query head / ${cfg?.expert_kv_heads ?? 2} 个 KV head，SwiGLU FF ${cfg?.expert_intermediate ?? 2752}，参数 ${cfg ? (cfg.n_params_expert / 1e6).toFixed(1) : '…'}M。它的每层 RMSNorm 是 AdaRMSNorm，以流匹配时间步 t = 1 的正弦嵌入为条件。

它的输入是 ${cfg?.n_suffix ?? 17} 个后缀 token：第 1 个是动作状态 token（75 维归一化本体状态经 state_proj 投到 ${cfg?.expert_hidden ?? 768} 维），后 ${cfg?.action_horizon ?? 16} 个是噪声动作 token（x1 的每一步 ${cfg?.action_dim ?? 23} 维经 action_in_proj，再与时间嵌入拼接过一个 SiLU MLP）。

逐层联合注意力：checkpoint 里专家有 ${cfg?.expert_n_layer ?? 36} 层，但前向只跑 ${nPairs} 个“层对”（层数取自主干，专家第 ${nPairs}–${(cfg?.expert_n_layer ?? 36) - 1} 层不参与计算）。在第 i 个层对里，主干第 i 层和专家第 i 层各自算 Q/K/V，沿 token 轴拼成长 ${cfg?.L_joint ?? 1257} 的序列（${cfg?.T_prefix ?? 1240} 前缀 + ${cfg?.n_suffix ?? 17} 后缀），统一做 RoPE 后跑一次共享注意力，再拆回两路，各自过 O 投影和 MLP。

掩码（build_fused_mask，§3.2）决定了信息流向。后缀的 Q 只能看两类 key：后缀自身，以及有效且不属于答案的 prompt 列。本样本这类前缀列共 ${cfg?.suffix_visible_prefix ?? '…'} 个（${cfg?.suffix_visible_visual ?? '…'} 个视觉 token + ${cfg ? cfg.suffix_visible_prefix - cfg.suffix_visible_visual : '…'} 个文本 token）。${cfg?.n_state_tokens ?? 16} 个行为状态 token 列和 ${cfg ? cfg.T_text - (cfg.suffix_visible_prefix) : '…'} 个教师强制的答案 token 列被屏蔽：本体状态只能经动作状态 token 进入专家，改写答案文本也不会改变动作输出。反方向上，前缀的 Q 看不到后缀，所以动作分支无法反过来改变 knack / done 的读数。

真实注意力质量（${cfg?.n_suffix ?? 17} 个后缀 query、${cfg?.n_head ?? 16} 个头取平均）：${massLine([0, Math.floor(nPairs / 2), nPairs - 1])}。落在状态 token 上的质量恰为 0。

图中是专家最终 RMSNorm 输出的真实值（${cfg?.n_suffix ?? 17} 个后缀 token × 前 64 维）。其中最后 ${cfg?.action_horizon ?? 16} 个 token 会送进 action_out_proj。

3D 场景：主干每层右侧的橙色列就是专家。每个层对有两块，上面是“Joint Attn (expert i)”，即该层 ${cfg?.n_suffix ?? 17} 个后缀 query 的真实注意力（对 ${cfg?.n_head ?? 16} 个头取平均）：前 ${cfg?.window.Wt ?? 32} 列是前缀窗口里的 key，后 ${cfg?.n_suffix ?? 17} 列是后缀 key。下面是“Expert i Out”，即该层输出的后缀残差流。橙色连线从主干第 i 层的 K/V 组连到专家第 i 层，亮度和粗细正比于该层后缀 query 落在前缀上的真实注意力质量。

`;
        afterTime(c, headsAnimLength);
        if (L.expertLayers.length > 0) {
            let t = phaseClock(state);
            let nL = L.expertLayers.length;
            let sweep = clamp((t - 1.5) / 5.5, 0, 1);
            let viz = state.display.actionViz!;
            viz.links = 1;
            viz.linkSweep = t < 1.5 ? 1 / nL : sweep;
            viz.activeLayer = t < 1.5 ? 0 : t < 7 ? Math.min(nL - 1, Math.floor(sweep * nL)) : -1;
            let b0 = L.blocks[0];
            let e0 = L.expertLayers[0];
            let layer0 = [b0.ln1.lnResid, ...b0.kvGroups.flatMap(g => [g.kBlock, g.vBlock]), L.expertIn, e0.attn, e0.out];
            let allExpert = [...L.blocks.flatMap(b => [b.attnResidual, b.mlpResidual, b.ln1.lnResid, ...b.kvGroups.map(g => g.kBlock)]), ...L.expertLayers.flatMap(e => [e.attn, e.out])];
            let tail = [...L.expertLayers.slice(nL - 3).flatMap(e => [e.attn, e.out]), L.expertOut, L.hStar];
            cameraKeyframes(state, t, [
                { t: 1.5, cam: frameCubes(layer0, 318, 24, { fill: 1.05 }) },
                { t: 7, cam: frameCubes(allExpert, 300, 14, { fill: 0.95 }) },
                { t: 9.5, cam: frameCubes(tail, 296, 12, { fill: 1.15, ...side }) },
            ]);
            if (t < 10) wt.markDirty();
        }
    } break;
    case Phase.Action_Denoise: {
        let H = cfg?.action_horizon ?? 16, D = cfg?.action_dim ?? 23;
        let chunk = m?.actionChunk.localBuffer, x1 = m?.actionX1.localBuffer, vel = m?.actionVelocity.localBuffer;
        let row0 = cfg && chunk ? describeActionRow(cfg, chunk, 0) : [];
        let rowL = cfg && chunk ? describeActionRow(cfg, chunk, H - 1) : [];
        let ex = (d: number) => cfg && chunk && x1 && vel
            ? `${cfg.action_dim_names[d]}：x1 = ${fmtSigned(x1[d])}，v = ${fmtSigned(vel[d])}，x1 − v = ${fmtSigned(chunk[d])}` : '…';
        let zeroVel = cfg ? ((c) => [0, 1, 2].map(d => `${c.action_dim_names[d]} ${fmtSigned(normalizeAction(c, d, 0))}`).join('，'))(cfg) : '…';
        let stats = m?.actionChunk.meta.stats;
        let c = commentary(wt, null, 0)`

动作头是流匹配（flow matching）：专家最后 ${H} 个后缀 token 经 action_out_proj（${cfg?.expert_hidden ?? 768} → ${D}）得到速度场 v，形状 ${H} × ${D}。它描述从噪声 x1（t = 1）流向真实动作 x0（t = 0）的方向。

解码就是沿速度场做 Euler 积分：x(t + Δt) = x(t) + Δt · v。k005b 训练和部署都固定在 t = 1，只积一步（Δt = −1），所以 action = x1 − v。训练目标为 u = x1 − 真实动作，对 16 × 23 逐元素做 MSE（done 行也参与训练）。本样本的 x1 是训练行里存好的 action_noise，不是渲染时现生成的随机数。

逐分量验证（第 0 步）：${ex(0)}；${ex(14)}。

动作块 = ${H} 步 × ${D} 维，${D} 维依次为：

0–2 底盘速度 base_vx / base_vy / base_yaw（这三维是速度）；3–6 躯干关节 trunk_1–4；7–13 左臂 7 个关节；14 左夹爪；15–21 右臂 7 个关节；22 右夹爪。3–22 都是位置量。

模型输出在 bounds99 归一化空间：norm = (x − q01) / (q99 − q01) · 2 − 1，即每维 1% 分位映到 −1、99% 分位映到 +1（可略超出 ±1）。服务端按同一组 q01 / q99 反归一化成机器人单位。夹爪的 q01 / q99 约为 ${cfg ? fmtSigned(cfg.action_q01[14], 2) : '…'} / ${cfg ? fmtSigned(cfg.action_q99[14], 2) : '…'}，归一化值与机器人单位几乎相同：+1 张开、−1 闭合（harness 以负值判定闭合）。trunk_4（dim 6）在数据中恒为 0（q01 = q99 = 0），无论输出多少，反归一化后都是 0。

本样本真实动作块（括号内为反归一化后的机器人单位），第 0 步：

${row0.join('\n\n')}

第 ${H - 1} 步：

${rowL.join('\n\n')}

整块统计：min ${stats ? fmtSigned(stats.min) : '…'}，max ${stats ? fmtSigned(stats.max) : '…'}，mean ${stats ? fmtSigned(stats.mean) : '…'}。

settle 语义：本样本 done = ${String(cfg?.pred_done ?? '…')}、knack = ${cfg?.pred_knack ?? '…'}，所以这是一段正常的运动块。一旦 done 为真，harness 播完当前块后改发“停住保持”块：dim 0–2 的速度在机器人单位下清零（归一化后为 ${zeroVel}，并不是 0），dim 3–22 的躯干、手臂和夹爪位置保持上一行的值（夹爪也保持，因为写 0 会变成闭合指令、写正值会变成张开）。训练标签里每个 knack 结束帧之后的行也按同一规则合成（先在机器人单位清零，再做 bounds99）。

3D 场景：右下是这 ${H} × ${D} 个真实分量的柱状场。横向是 ${D} 维（颜色按分组：底盘、躯干、左臂、左夹爪、右臂、右夹爪），纵深是 ${H} 个时间步（第 0 步在最前），柱高就是归一化值，两条参考线是 ±1（bounds99 的 1% / 99% 分位）。动画先摆出 x1（t = 1 的噪声），白色竖线是每个分量的 Euler 步长 −v，随后所有柱子沿 x(t) = x1 − (1 − t)·v 从 t = 1 走到 t = 0，停下时就是解码出的动作块 x1 − v。

`;
        afterTime(c, headsAnimLength);
        {
            let t = phaseClock(state);
            let viz = state.display.actionViz!;
            viz.links = 0.12;
            viz.decode = clamp((t - 2.5) / 5, 0, 1);
            viz.showVelocity = t < 1 ? 0 : t < 2 ? t - 1 : t < 9 ? 1 : Math.max(0.25, 1 - (t - 9));
            cameraKeyframes(state, t, [
                { t: 1.5, cam: frameCubes(actionBlocks, 300, 22, { fill: 1.1, ...side }) },
            ]);
            if (t < 11) wt.markDirty();
        }
    } break;
    case Phase.Harness_Law: {
        let c = commentary(wt, null, 0)`

最后是 Harness 法则，它规定了模型与系统之间的分工：

done = true → 播完当前 chunk → 服务端刹车并定格 → 推进到下一个 subtask。

模型只负责报告 done，绝不自己切换 subtask。本样本中模型的真实输出为 knack = ${cfg?.pred_knack ?? '…'}、done = ${String(cfg?.pred_done ?? '…')}。

`;
        afterTime(c, headsAnimLength);
    } break;
    }
}
