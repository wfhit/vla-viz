import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, initialFrame } from "./WalkthroughTools";

export function walkthrough05_Softmax(args: IWalkthroughArgs) {
    let { walkthrough: wt, state } = args;

    if (wt.phase !== Phase.Input_Detail_Softmax) {
        return;
    }

    // one representative head (x16; the others share its slot and are not drawn)
    let h = args.layout.blocks[0].heads[0];
    initialFrame(state, [h.qBlock, h.kBlock, h.attnMtx, h.attnMtxAgg1, h.attnMtxAgg2, h.attnMtxSm, h.vOutBlock], 290, 8, { fill: 1.1 });

    let f = state.page.facts!;
    commentary(wt, null, 0)`

softmax 是注意力的核心运算：它把一个向量变成总和为 1.0 的权重。

做法是先对每个输入取指数 exp(x)，保证全部为正，再除以所有指数之和。这样每个结果都在 0.0 到 1.0 之间，可以当作对各个位置的注意力权重。

但如果输入值很大，指数会溢出，大数除以大数会带来浮点精度问题。softmax 有个性质：所有输入加上同一个常数，结果不变。因此先找出最大值并从每个输入中减去，使最大值变为 0.0，运算就数值稳定了。

${f.id === 'groot'
    ? 'GR00T 主干的语言模型层里，softmax 作用于注意力矩阵的每一行，范围是该行允许看到的列：因果掩码下只有自己和之前的 token，被掩码的列在 softmax 之后恰好是 0。主干里没有动作 token；DiT 的 softmax 在 D 章组里讲。'
    : f.id === 'pi0'
    ? 'π0.5 的联合注意力中，softmax 作用于注意力矩阵的每一行。前缀查询的范围是全部前缀列（双向，没有三角掩码）；动作专家的后缀查询范围是全部前缀列加全部后缀列；前缀不受后缀影响。'
    : '在 k005b 的联合注意力中，softmax 作用于注意力矩阵的每一行，范围是前缀（prefix）列：即 VLM 前缀流的 K。动作专家的后缀（suffix）查询对前缀列（以及自身）计算注意力；前缀不受后缀影响。'}

和 RMSNorm 一样，这里先有一个聚合步骤，为每一行保存最大值（max）以及“减去最大值后取指数”的求和（sum）。然后每个输出就只需：减去 max、取指数、除以 sum。

“softmax” 的名字来源于 argmax：argmax 只把最大值设为 1.0、其余为 0.0，是“硬”选择；softmax 则是“软”版本，既突出最大值，又保留其他位置的相对权重。
`;

}
