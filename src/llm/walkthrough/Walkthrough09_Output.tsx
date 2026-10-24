import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera, initialFrame } from "./WalkthroughTools";

export function walkthrough09_Output(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout } = args;

    if (wt.phase !== Phase.Input_Detail_Output) {
        return;
    }

    initialFrame(state, [args.layout.ln_f.lnResid, args.layout.hStar, args.layout.lmHeadWeight, args.layout.logits, args.layout.logitsSoftmax, args.layout.doneWeight, args.layout.doneProbs], 281.6, -7.9, { fill: 1.2 });

    let c0 = commentary(wt, null, 0)`

最后来到主干的末端。k005b 不接语言模型那样的 151936 类词表输出，而是只读一个位置。

最后一层输出之后，经过最终的 RMSNorm（ln_f），我们只在 prefix 上读取“答案边界” token 位置（p* = ${layout.shape.full?.p_star ?? '…'}）的隐藏状态，记作 h*（长度 2048）。
它汇聚了视觉 token、提示词与状态 token 的全部信息。

这个 h* 送入两个线性头：knack 头（17 类）与 done 头（2 类），见下一章。
每个头各自做 softmax 并取 argmax，因此不需要采样或 temperature。本样本的预测：knack = ${layout.shape.full?.pred_knack ?? '…'}，done = ${String(layout.shape.full?.pred_done ?? '…')}。

`;

}
