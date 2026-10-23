import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough09_Output(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Output) {
        return;
    }

    setInitialCamera(state, new Vec3(-20.203, 0.000, -1642.819), new Vec3(281.600, -7.900, 2.298));

    let c0 = commentary(wt, null, 0)`

最后来到主干的末端。k005b 不像语言模型那样接一个 final layernorm + 线性层去产生词表 logits。

最后一层输出之后，经过最终的 RMSNorm，我们只在 prefix 上读取“答案边界” token 位置的隐藏状态，记作 h*。
它汇聚了三路相机、语言（Goal / Plan / Last knack）与状态槽的全部信息。

这个 h* 是整个网络的分叉点：同一个隐藏状态会分别送入两组头——knack 分类头与 done 头（见下一章）。
knack 的每个字段各自做独立的 softmax，取概率最大的值（argmax），因此不需要采样或 temperature。

`;

}
