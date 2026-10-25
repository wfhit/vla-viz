import { Phase } from "./Walkthrough";
import { commentary, frameCubes, IWalkthroughArgs, initialFrame } from "./WalkthroughTools";
import { applyFlowPass, nFlowStages } from "./FlowPass";
import { clamp } from "@/src/utils/data";
import { IVlaModelLayout } from "../VlaModelLayout";
import { IProgramState } from "../Program";

// the layer pair's sublayer chain (the head fans extend toward the camera and are left out
// of the framing so the whole chain stays in view)
function mainPath(L: IVlaModelLayout, i: number) {
    let b = L.blocks[i];
    return [...b.ln1.cubes, b.heads[0].qBlock, b.heads[0].attnMtx, b.attnOut, b.attnResidual, ...b.ln2.cubes,
        b.mlpFc, b.mlpUp, b.mlpAct, b.mlpResult, b.mlpResidual];
}

// the camera rides along the pass: frames layer pair `pos` (continuous between two layers)
function followPass(state: IProgramState, L: IVlaModelLayout, pos: number) {
    let nL = L.blocks.length;
    let i0 = clamp(Math.floor(pos), 0, nL - 1), i1 = Math.min(nL - 1, i0 + 1);
    let u = clamp(pos - i0, 0, 1);
    u = u * u * (3 - 2 * u);
    let cubesOf = (i: number) => mainPath(L, i);
    let a = frameCubes(cubesOf(i0), 270, 3, { fill: 1.8 });
    let b = frameCubes(cubesOf(i1), 270, 3, { fill: 1.8 });
    state.camera.center = a.center.lerp(b.center, u);
    state.camera.angle = a.angle.lerp(b.angle, u);
    state.walkthrough.markDirty();
}

export function walkthrough08_Transformer(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout: L, tools: { afterTime, breakAfter, c_blockRef } } = args;

    if (wt.phase !== Phase.Input_Detail_Transformer) {
        return;
    }
    let nL = L.blocks.length;
    initialFrame(state, [L.residual0, ...mainPath(L, 0)], 270, 3, { fill: 1.8 });

    commentary(wt, null, 0)`

前面几章把一层拆开讲完了。现在把真实数据整体推过整个 ${nL} 层的主干堆叠，看一次前向到底怎么走。

拼接好的主干 token 流（${c_blockRef('嵌入', L.residual0)}）从顶部进入第 0 层。黄色是正在计算的子层，后面还没算到的块保持空白、箭头变淡。每一层都按同一顺序走：
RMSNorm → Q/K/V → 注意力 → softmax·V → o_proj → ⊕残差 → RMSNorm → gate/up → SiLU·up → down_proj → ⊕残差。

本章只看主干。每一层右侧其实还紧贴着一条 DiT 动作专家泳道，它与主干同层共用一次联合注意力；两条流一起穿过 28 层的完整过程放在 D4 章。
`;
    breakAfter();
    let tEnter = afterTime(null, 1.5);
    let tL0 = afterTime(null, nFlowStages * 0.9);
    breakAfter();
    commentary(wt)`

第 0 层走完了。⊕ 块上的小字标出残差的另一个加数（skip）：它不是一根绕过子层的直线，而是在 ⊕ 处与子层输出相加。

接下来一层一层穿过剩下的 ${nL - 1} 层。
`;
    let tRest = afterTime(null, (nL - 1) * 0.8);
    breakAfter();
    commentary(wt)`

${nL} 层全部穿过，主干最后一层的输出将去往 h* 和 knack / done 头。

与深度学习中常见的情形一样，很难精确说出每一层在做什么，但有一个大致的规律：
靠前的层主要处理局部、视觉层面的特征（如图像 patch 中的边缘、物体轮廓），
靠后的层则更多承载任务层面的语义（目标是什么、下一步该做什么）。
`;

    // stage position of the pass
    let pos: number;
    if (!tEnter.active) pos = -1;
    else if (!tL0.active) pos = -1 + tEnter.t;
    else if (!tRest.active) pos = tL0.t * nFlowStages;
    else pos = nFlowStages + tRest.t * (nL - 1) * nFlowStages;
    let done = tRest.active && tRest.t >= 1;
    if (!done) {
        applyFlowPass(L, pos, { expert: false });
        if (tEnter.active) followPass(state, L, Math.max(0, pos / nFlowStages - 0.5));
    }
}
