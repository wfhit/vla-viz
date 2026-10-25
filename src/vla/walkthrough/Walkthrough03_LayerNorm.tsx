import { findSubBlocks, splitGrid } from "../Annotations";
import { drawDataFlow } from "../components/DataFlow";
import { drawDependences } from "../Interaction";
import { clamp } from "@/src/utils/data";
import { lerp } from "@/src/utils/math";
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";
import { commentary, DimStyle, IWalkthroughArgs, frameTo, initialFrame } from "./WalkthroughTools";

export function walkthrough03_LayerNorm(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter, cleanup } } = args;
    let { C } = layout.shape;

    if (wt.phase !== Phase.Input_Detail_LayerNorm) {
        return;
    }

    let ln = layout.blocks[0].ln1;

    initialFrame(state, [args.layout.residual0, ...args.layout.blocks[0].ln1.cubes], 281, 9, { fill: 1.0 });
    wt.dimHighlightBlocks = [layout.residual0, ...ln.cubes];

    commentary(wt, null, 0)`

上一节得到的${c_blockRef('_输入嵌入_', layout.residual0)}矩阵是 Qwen3-VL 第一个 transformer 层（L0）的输入（共 28 层，这里展开其中一层）。

层内第一步是 _RMSNorm_：对矩阵的每一列分别做归一化。图中数值均为 L0 input_layernorm 的真实值。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t_hideExtra = afterTime(null, 1.0, 1.0);
    let t_moveInputEmbed = afterTime(null, 1.0);
    let t_moveCameraClose = afterTime(null, 0.5);

    breakAfter();
    commentary(wt)`
归一化能让深层网络的训练更稳定。

各列互相独立，先看其中一列。`;

    breakAfter();
    let t_focusColumn = afterTime(null, 0.5);

    // mu ascii: \u03bc
    // sigma ascii: \u03c3
    breakAfter();
    commentary(wt)`
RMSNorm 用该列的均方根（root-mean-square）去缩放整列，使其尺度为 1；与 LayerNorm 不同，它不减去均值，也没有偏置 β。
图中两个聚合格分别是真实的 ${c_blockRef('mean(x²)', ln.lnAgg1)} 与 ${c_blockRef('rsqrt(mean(x²) + ε)', ln.lnAgg2)}，均对完整的 2048 个通道计算（不仅是图中显示的 64 个）。`;

    breakAfter();

    let t_calcMuAgg = afterTime(null, 0.5);
    let t_calcVarAgg = afterTime(null, 0.5);

    // 1e-5 as 1x10^-5, but with superscript: 1x10<sup>-5</sup>

    breakAfter();
    commentary(wt)`
RMSNorm 的公式为 x / sqrt(mean(x²) + ε)，其中均值取自长度为 ${c_dimRef('C', DimStyle.C)} 的整列，ε（${<>1&times;10<sup>-6</sup></>}）用来避免除以零。

这些聚合值只需对整列算一次，因此存放在聚合层中。

最后把每个元素乘以可学习的${c_blockRef('增益 (\u03b3)', ln.lnSigma)}，得到${c_blockRef('归一化结果', ln.lnResid)}。RMSNorm 没有偏置 β。`;

    breakAfter();

    let t_clean_aggs = afterTime(null, 0.2);
    cleanup(t_clean_aggs, [t_calcMuAgg, t_calcVarAgg]);
    let t_colSequence = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
对${c_blockRef('输入嵌入矩阵', layout.residual0)}的每一列都做同样的归一化，得到${c_blockRef('归一化后的输入嵌入', ln.lnResid)}，接下来送入联合注意力层。
`;

    breakAfter();
    let t_cleanupSplits = afterTime(null, 0.5);
    cleanup(t_cleanupSplits, [t_focusColumn]);
    if (t_cleanupSplits.t > 0) {
        t_colSequence.t = 0;
    }
    let t_runAggFull = afterTime(null, 2.0);
    let t_runNormFull = afterTime(null, 6.0);

    frameTo(state, t_moveCamera, args.layout.blocks[0].ln1.cubes, 281.5, 11, { fill: 1.1 });

    let exampleIdx = 3;
    let ln1 = layout.blocks[0].ln1;
    let inputBlock = layout.residual0;

    inputBlock.highlight = lerp(0, 0.3, t_hideExtra.t);

    let relevantBlocks = new Set([
        layout.residual0,
        ...ln1.cubes,
    ]);

    for (let blk of layout.cubes) {
            if (!relevantBlocks.has(blk)) {
            blk.opacity = lerp(1.0, 0.0, t_hideExtra.t);
        }
    }

    for (let blk of relevantBlocks) {
        if (blk != layout.residual0 && blk.t !== 'w') {
            blk.access!.disable = true;
        }
    }

    let startResidualY = layout.residual0.y;
    let endResidulY = ln1.lnResid.y;
    layout.residual0.y = lerp(startResidualY, endResidulY, t_moveInputEmbed.t);

    if (t_moveInputEmbed.t >= 0.0) {
        inputBlock.highlight = lerp(0.3, 0.0, t_moveInputEmbed.t);
    }

    frameTo(state, t_moveCameraClose, [args.layout.blocks[0].ln1.lnAgg1, args.layout.blocks[0].ln1.lnAgg2, args.layout.blocks[0].ln1.lnResid], 270, 4, { fill: 1.2 });

    let splitAmt = lerp(0.0, 2.0, t_focusColumn.t);
    let splitPos = exampleIdx + 0.5;

    let otherColOpacity = lerp(1.0, 0.3, t_focusColumn.t);
    ln1.lnAgg1.opacity = otherColOpacity;
    ln1.lnAgg2.opacity = otherColOpacity;
    ln1.lnResid.opacity = otherColOpacity;
    inputBlock.opacity = otherColOpacity;

    if (t_focusColumn.t > 0) {
        let aggMuCol = splitGrid(layout, ln1.lnAgg1, Dim.X, splitPos, splitAmt)!;
        let aggVarCol = splitGrid(layout, ln1.lnAgg2, Dim.X, splitPos, splitAmt)!;
        let residCol = splitGrid(layout, ln1.lnResid, Dim.X, splitPos, splitAmt)!;
        let inputCol = splitGrid(layout, inputBlock, Dim.X, splitPos, splitAmt)!;
        aggMuCol.opacity = 1.0;
        aggVarCol.opacity = 1.0;
        residCol.opacity = 1.0;
        inputCol.opacity = 1.0;

        let aggDestIdx = new Vec3(exampleIdx, 0, 0);
        if (t_calcMuAgg.t > 0.0) {
            let pinIdx = new Vec3(0, 10, 0);
            drawDependences(state, ln1.lnAgg1, aggDestIdx);
            drawDataFlow(state, ln1.lnAgg1, aggDestIdx, pinIdx);
            aggMuCol.access!.disable = false;
            inputCol.highlight = 0.3;
        }

        if (t_calcVarAgg.t > 0.0) {
            let pinIdx = new Vec3(9, 9, 0);
            drawDependences(state, ln1.lnAgg2, aggDestIdx);
            drawDataFlow(state, ln1.lnAgg2, aggDestIdx, pinIdx);
            aggVarCol.access!.disable = false;
        }

        if (t_colSequence.t > 0.0) {
            aggMuCol.access!.disable = false;
            aggVarCol.access!.disable = false;

            let pinIdx = new Vec3(-10, 0, 0);

            let cPos = t_colSequence.t * C;
            let cIdx = clamp(Math.floor(cPos), 0, C - 1);
            let destIdx = new Vec3(exampleIdx, cIdx, 0);
            drawDependences(state, ln1.lnResid, destIdx);
            drawDataFlow(state, ln1.lnResid, destIdx, pinIdx);

            let targetCell = splitGrid(layout, residCol, Dim.Y, cIdx + 0.5, 0.0)!;
            targetCell.highlight = 0.3;

            findSubBlocks(residCol, Dim.Y, 0, cIdx).forEach((blk) => {
                blk.access!.disable = false;
            });
        }
    }

    if (t_runAggFull.t > 0.0) {
        try {
            let processInfo = startProcessBefore(state, ln1.lnAgg1);
            processUpTo(state, t_runAggFull, ln1.lnAgg2, processInfo);
            processUpTo(state, t_runNormFull, ln1.lnResid, processInfo);
        } catch (e) {
            console.log(e);
        }
    }
}
