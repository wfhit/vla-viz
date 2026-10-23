import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, setInitialCamera } from "./WalkthroughTools";

export function walkthrough08_Transformer(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Transformer) {
        return;
    }

    setInitialCamera(state, new Vec3(-135.531, 0.000, -353.905), new Vec3(291.100, 13.600, 5.706));

    let c0 = commentary(wt, null, 0)`

到这里，一层完整的 transformer 就讲完了！

Qwen3-VL 主干把这样的一层重复堆叠 28 次，3D 场景中 28 层分两列排列，每层都是该层真实的权重与激活：
上一层的输出沿着残差通路直接成为下一层的输入。

每一层里，suffix（action expert）流都会通过 joint 注意力与 VLM prefix 流相遇——不是只在最后才汇合一次，而是逐层交互。

与深度学习中常见的情形一样，很难精确说出每一层在做什么，但有一个大致的规律：
靠前的层主要处理局部、视觉层面的特征（如图像 patch 中的边缘、物体轮廓），
靠后的层则更多承载任务层面的语义（目标是什么、下一步该做什么）。

`;

}
