import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

// the head-specific visuals are drawn by components/HeadsOverlay.tsx, driven by wt.time over [0, 12]
export const headsAnimLength = 12;

export function walkthrough10_Heads(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, tools: { afterTime } } = args;

    setInitialCamera(state, new Vec3(-20.203, 0.000, -1642.819), new Vec3(281.600, -7.900, 2.298));

    switch (wt.phase) {
    case Phase.Head_Knack: {
        let c = commentary(wt, null, 0)`

从 h* 出发，网络分成两路。第一路是 knack 并行分类头（System One / Jev 式）。

h* 同时送入多个彼此独立的线性头：knack 词表、object（260 类）、target（261 类）、facing、kind、direction。每个字段有自己的 softmax，概率互不干扰。

动画里每个面板的概率条依次长出，每个字段各自只胜出一个值（白框标出）。六个胜出值合起来，就是这一步的 knack 答案。

因为各字段并行、独立，所以没有逐 token 的自回归解码，也不需要采样温度：取各字段 argmax 即可。

`;
        afterTime(c, headsAnimLength);
    } break;
    case Phase.Head_Done: {
        let c = commentary(wt, null, 0)`

第二路是 done 独立头，输出形状为 [B, 2]：done = false / done = true。

done 是 subtask 级别的信号，只回答“当前 subtask 是否已经完成”。它作为叠加输出附在 knack 答案旁边，绝不会覆盖或改写 knack 的答案——上方已锁定的 knack 字段在动画中始终保持不变。

当 done 的概率升高到 true，模型只是在“报告”；真正怎样处理，由后面的 harness 决定。

`;
        afterTime(c, headsAnimLength);
    } break;
    case Phase.Action_Expert: {
        let c = commentary(wt, null, 0)`

动作专家是与主干并行的另一条后缀流。它的输入有三部分：

1. 状态 token：本体状态槽（61+14 = 75 维）先线性投影到 768 维；
2. 16 个噪声动作 token x_t，初始来自高斯噪声；
3. 时间嵌入 t，通过 adaLN 为每一层产生 scale / shift 调制。

这些 token 组成联合去噪器，并与主干逐层做联合注意力：后缀（动作）流的 Q 可以看到前缀（VLM）流的 K/V，因此动作生成始终以图像和语言上下文为条件。

`;
        afterTime(c, headsAnimLength);
    } break;
    case Phase.Action_Denoise: {
        let c = commentary(wt, null, 0)`

去噪器输出的是速度场 v_t。从 t = 0 的纯噪声开始，用 Euler 积分一步步前进：

  x_{t+dt} = x_t + dt · v_t

积分到 t = 1，就得到干净的动作块：16 步 × 23 维。

块的末尾是 settle 行：这些行的速度被清零、位置保持不动，让机器人平稳停住。动作块永远不会跨越 knack 边界——一个动作块只属于一个 knack。

`;
        afterTime(c, headsAnimLength);
    } break;
    case Phase.Harness_Law: {
        let c = commentary(wt, null, 0)`

最后是 Harness 法则，它规定了模型与系统之间的分工：

done = true → 播完当前 chunk → 服务端刹车并定格 → 推进到下一个 subtask。

模型只负责报告 done，绝不自己切换 subtask。切换时机、刹车和定格都由 harness 在服务端完成，这样即使模型某一帧给出了不稳定的 done，也不会让动作在中途被打断。

`;
        afterTime(c, headsAnimLength);
    } break;
    }
}
