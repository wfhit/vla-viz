import { findSubBlocks, splitGrid } from "../Annotations";
import { drawDataFlow } from "../components/DataFlow";
import { drawDependences } from "../Interaction";
import { clamp } from "@/src/utils/data";
import { lerp } from "@/src/utils/math";
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";
import { embedInline } from "./Walkthrough01_Prelim";
import { fmtA, N, TermsTable } from "../components/TraceLedger";
import { commentary, DimStyle, IWalkthroughArgs, frameTo, initialFrame, drawCellTag, TraceColors } from "./WalkthroughTools";

export function walkthrough03_LayerNorm(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter, cleanup } } = args;

    if (wt.phase !== Phase.Input_Detail_LayerNorm) {
        return;
    }

    let ln = layout.blocks[0].ln1;
    let f = state.page.facts!;

    initialFrame(state, [args.layout.residual0, ...args.layout.blocks[0].ln1.cubes], 281, 9, { fill: 1.0 });
    wt.dimHighlightBlocks = [layout.residual0, ...ln.cubes];

    let tr = state.vlaModel?.trace;
    let cfg = layout.shape.full;
    let exampleIdx = layout.shape.T - 1;
    let tokStr = cfg ? JSON.stringify(cfg.window_tokens[exampleIdx]) : '…';
    let c = tr?.rms.c ?? 0;
    let pi0 = f.id === 'pi0';
    let lnName = pi0 ? 'pre_attention_norm' : 'input_layernorm';

    commentary(wt, null, 0)`
本章跟踪一个真实对象：窗口最后一列的 token ${tokStr}（id ${tr?.embed.id ?? '…'}，序列位置 ${tr?.p ?? '…'}），它的第 ${c} 个通道（下面在 3D 中把这一格单独高亮）。

它的数值来自上一章的查表：${embedInline(tr && <span>x[{c}] = 嵌入表第 {tr.embed.id} 行第 {c} 列 <N v={tr.embed.row_c} />{tr.embed.mult !== 1 ? <> × {tr.embed.mult_name} (<N v={tr.embed.mult} />)</> : null} = <N v={tr.embed.x_c} out /></span>)}。这个数是本章的起点，它要穿过 L0 的 ${embedInline(<code>{lnName}</code>)}（RMSNorm），变成下一章注意力的输入。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t_hideExtra = afterTime(null, 1.0, 1.0);
    let t_moveInputEmbed = afterTime(null, 1.0);
    let t_moveCameraClose = afterTime(null, 0.5);

    breakAfter();
    commentary(wt)`
RMSNorm 对每一列独立处理：这一列只用它自己的 ${f.dModel} 个通道（图中画通道 0–${layout.shape.C - 1}），不看别的 token。`;

    breakAfter();
    let t_focusColumn = afterTime(null, 0.5);

    breakAfter();
    commentary(wt)`
第一步，求这一列的均方：先把 ${f.dModel} 个通道各自平方再相加。我们的 x[${c}] 是其中一项：
${embedInline(tr && <TermsTable t={tr.rms.sumsq} wName='' xName={i => `x[${i}]`} carryFirst square />)}
mean(x²) = ${embedInline(tr && <span><N v={tr.rms.sumsq.sum} /> / {tr.rms.C} = <N v={tr.rms.mean} out /></span>)}，rsqrt(mean(x²) + ε) = ${embedInline(tr && <span>1 / √(<N v={tr.rms.mean} carry /> + 10<sup>−6</sup>) = <N v={tr.rms.rsqrt} out /></span>)}。

这两个数就是图中两个聚合格 ${c_blockRef('mean(x²)', ln.lnAgg1)}、${c_blockRef('rsqrt', ln.lnAgg2)} 在这一列的真值。与 LayerNorm 不同，RMSNorm 不减均值。`;

    breakAfter();

    let t_calcMuAgg = afterTime(null, 0.5);
    let t_calcVarAgg = afterTime(null, 0.5);

    breakAfter();
    commentary(wt)`
第二步，把 x[${c}] 乘以 rsqrt，再乘可学习的${c_blockRef(pi0 ? '增益 (1 + γ)' : '增益 γ', ln.lnSigma)}的第 ${c} 个分量：
${embedInline(tr && <div className='my-2 px-2 py-1 border-l-4 border-green-500 bg-slate-800/60 text-sm'>y[{c}] = <N v={tr.rms.x_c} carry /> × <N v={tr.rms.rsqrt} carry /> × {pi0 ? '(1 + γ)' : 'γ'}[{c}] (<N v={tr.rms.g_c} />) = <N v={tr.rms.y_c} out /></div>)}
这就是${c_blockRef('归一化结果', ln.lnResid)}里高亮格的数。下一章算 Q 时，${embedInline(tr && <N v={tr.rms.y_c} carry />)} 是点积中 i = ${c} 那一项的输入。

${pi0 ? 'Gemma 的 checkpoint 存的是 γ，前向时乘 1 + γ；没有偏置 β。' : f.id === 'groot' ? '没有偏置 β。（主干最后 4 块 VL 自注意力用的是带 β 的 LayerNorm。）' : '没有偏置 β。'}`;

    breakAfter();

    let t_clean_aggs = afterTime(null, 0.2);
    cleanup(t_clean_aggs, [t_calcMuAgg, t_calcVarAgg]);
    let t_colSequence = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
对${c_blockRef('输入嵌入矩阵', layout.residual0)}的每一列都做同样的归一化，得到${c_blockRef('归一化后的输入嵌入', ln.lnResid)}，接下来送入${f.id === 'groot' ? '自注意力层' : '联合注意力层'}。
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

    let ln1 = layout.blocks[0].ln1;
    let inputBlock = layout.residual0;

    inputBlock.highlight = lerp(0, 0.3, t_hideExtra.t);

    let relevantBlocks = new Set([
        layout.residual0,
        ...ln1.cubes,
    ]);

    for (let blk of layout.cubes) {
            if (!relevantBlocks.has(blk)) {
            blk.opacity = lerp(1.0, 0.15, t_hideExtra.t);
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
        frameTo(state, t_focusColumn, [aggMuCol, aggVarCol, residCol, inputCol], 275, 6, { fill: 1.15 });
        if (t_colSequence.t <= 0) splitGrid(layout, residCol, Dim.Y, c + 0.5, 0)!.highlight = 0.3;
        let inCell = splitGrid(layout, inputCol, Dim.Y, c + 0.5, 0)!;
        inCell.highlight = 0.6;
        if (tr) drawCellTag(state, inCell, `x[${c}]=${fmtA(tr.rms.x_c)}`, TraceColors.carry);
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

            // sweep down to the traced channel and stop there
            let cIdx = clamp(Math.floor(t_colSequence.t * (c + 1)), 0, c);
            let destIdx = new Vec3(exampleIdx, cIdx, 0);
            drawDependences(state, ln1.lnResid, destIdx);
            drawDataFlow(state, ln1.lnResid, destIdx, pinIdx);

            let targetCell = splitGrid(layout, residCol, Dim.Y, cIdx + 0.5, 0.0)!;
            targetCell.highlight = 0.3;
            if (tr && cIdx === c) drawCellTag(state, targetCell, `y[${c}]=${fmtA(tr.rms.y_c)}`, TraceColors.out);

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
