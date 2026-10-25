import React from 'react';
import { Phase } from "./Walkthrough";
import { commentary, IWalkthroughArgs, initialFrame } from "./WalkthroughTools";

export function walkthrough01_Prelim(args: IWalkthroughArgs) {
    let { state, walkthrough: wt } = args;

    if (wt.phase !== Phase.Intro_Prelim) {
        return;
    }

    initialFrame(state, args.layout.cubes, 296, 16, { fill: 1.0 });

    let f = state.page.facts!;
    if (f.id === 'pi0') commentary(wt, null, 0)`
在深入细节之前，先退一步看几个基本概念。

本讲解只关注 _推理_（inference），不涉及训练。3D 场景中的每个数值都来自真实的 π0.5 checkpoint：权重直接读自本地 openpi-comet checkpoint，激活是对一帧真实 BEHAVIOR 演示数据做前向时抓下来的（由 tools/export_pi0.py 导出，并与 openpi 的参考前向逐项核对）。因为真实张量太大，每个张量只画一个真实窗口，悬停可看到真实下标与整张量统计。

_Token_：模型处理的最小单位。相机画面经 ${f.visionName} 变成视觉 token；提示词按 PaliGemma 词表切成语言 token。π0.5 没有状态 token：机器人状态被离散成 0–255 的整数，直接写进提示词文本。动作专家另有 ${f.nSuffix} 个动作 token。

_嵌入_（embedding）：每个 token 先被映射成一个向量（主干中是 ${f.dModel} 维，动作专家中是 ${f.expertD} 维），之后所有计算都在这些向量上进行。

_向量与矩阵乘法_：一个向量乘以一个权重矩阵，得到另一个向量——每个输出元素都是一次点积。Q/K/V、输出投影、MLP 都只是不同的矩阵乘法。

好，开始吧！
`;
    else commentary(wt, null, 0)`
在深入细节之前，先退一步看几个基本概念。

本讲解只关注 _推理_（inference），不涉及训练。3D 场景中的每个数值都来自真实的 k005b checkpoint：权重直接读自 checkpoint，激活是对一个真实样本做前向时用 hook 抓下来的（由 tools/export_k005b.py 导出）。因为真实张量太大，每个张量只画一个真实窗口，悬停可看到真实下标与整张量统计。

_Token_：模型处理的最小单位。相机画面经 Qwen3-VL 视觉塔变成视觉 token；提示词按 Qwen 词表切成语言 token；机器人状态被投影成 16 个状态 token。

_嵌入_（embedding）：每个 token 先被映射成一个向量（k005b 主干中是 2048 维，动作专家中是 768 维），之后所有计算都在这些向量上进行。

_向量与矩阵乘法_：一个向量乘以一个权重矩阵，得到另一个向量——每个输出元素都是一次点积。Q/K/V、输出投影、MLP 都只是不同的矩阵乘法。

好，开始吧！
`;

}

export function embedInline(a: React.ReactNode) {
    return { insertInline: a };
}


// Another similar model is BERT (bidirectional encoder representations from transformers), a "context-aware text encoder" commonly
// used for tasks like document classification and search.  Newer models like Facebook's LLaMA (large language model architecture), continue to use
// a similar transformer architecture, albeit with some minor differences.
