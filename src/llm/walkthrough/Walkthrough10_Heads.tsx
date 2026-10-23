import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

// the head-specific visuals are drawn by components/HeadsOverlay.tsx, driven by wt.time over [0, 12]
export const headsAnimLength = 12;

export function walkthrough10_Heads(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { afterTime } } = args;
    let cfg = layout.shape.full;

    setInitialCamera(state, new Vec3(-20.203, 0.000, -1642.819), new Vec3(281.600, -7.900, 2.298));

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
        let c = commentary(wt, null, 0)`

动作专家是与主干并行的另一个 transformer（隐藏维 ${cfg?.expert_hidden ?? 768}，${cfg?.expert_heads ?? 16} 个 query head / ${cfg?.expert_kv_heads ?? 2} 个 KV head，FF ${cfg?.expert_intermediate ?? 2752}）。

它处理 ${cfg?.n_suffix ?? 17} 个后缀 token，并与主干逐层做联合注意力：后缀的 Q 可以看到前缀的 K/V，因此动作生成始终以图像和语言上下文为条件。

图中是动作专家最终 RMSNorm 输出的真实值（${cfg?.n_suffix ?? 17} 个 token × 前 64 维）。

`;
        afterTime(c, headsAnimLength);
    } break;
    case Phase.Action_Denoise: {
        let c = commentary(wt, null, 0)`

动作专家的输出经 action_out_proj 得到速度 v（${cfg?.action_horizon ?? 16} 步 × ${cfg?.action_dim ?? 23} 维）。

从噪声 x1 出发，本导出使用一步解码：action = x1 − v。

图中依次是真实的 x1、v 与解码出的动作块。

`;
        afterTime(c, headsAnimLength);
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
