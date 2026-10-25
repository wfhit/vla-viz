import { duplicateGrid, splitGrid } from "../Annotations";
import { embedColForToken } from "../VlaModel";
import { IBlkDef } from "../VlaModelLayout";
import { lerp } from "@/src/utils/math";
import { Dim, Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, frameTo, initialFrame } from "./WalkthroughTools";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough02_Embedding(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, tools: { c_blockRef, c_dimRef, afterTime, cleanup, breakAfter }, layout } = args;

    if (wt.phase !== Phase.Input_Detail_Embedding) {
        return;
    }

    initialFrame(state, [args.layout.idxObj, args.layout.tokEmbedObj, args.layout.residual0], 287, 14.5, { fill: 1.05 });
    wt.dimHighlightBlocks = [layout.idxObj, layout.tokEmbedObj, layout.residual0];
    let cfg = layout.shape.full;
    let exampleIdx = layout.shape.T - 1;
    let exampleTok = cfg ? JSON.stringify(cfg.window_tokens[exampleIdx]) : '…';
    let exampleId = cfg ? cfg.window_token_ids[exampleIdx] : 0;

    commentary(wt)`
k005b 的前缀由三类 token 组成：${cfg?.n_visual_tokens ?? '…'} 个视觉 token（3 路相机经 Qwen3-VL 视觉塔）、语言 token（目标、"Last knack" 等提示词），以及 ${cfg?.n_state_tokens ?? 16} 个状态 token，共 ${cfg?.T_prefix ?? '…'} 个。
3D 中的${c_blockRef('_输入索引_', state.layout.idxObj, DimStyle.TokenIdx)}是真实提示词里 token ${cfg ? `${cfg.window.t0}–${cfg.window.t0 + cfg.window.Wt - 1}` : '…'} 的 id（窗口末尾就是 h* 边界 token）。

下面以窗口最后一个 token（${exampleTok}，id ${exampleId}）为例，看看它如何生成${c_blockRef('_输入嵌入_', state.layout.residual0)}的最后一列。`;
    breakAfter();

    let t_moveCamera = afterTime(null, 1.0);
    let t0_splitEmbedAnim = afterTime(null, 0.3);

    breakAfter();

    commentary(wt)`
用该 token 的 id（${c_dimRef(String(exampleId), DimStyle.TokenIdx)}）从左侧的${c_blockRef('_嵌入表_', state.layout.tokEmbedObj)}（真实形状 ${cfg?.vocab_size ?? 151936} × ${cfg?.n_embd ?? 2048}，图中显示第 ${cfg?.window.v0 ?? '…'}–${(cfg?.window.v0 ?? 0) + (cfg?.window.Wv ?? 64) - 1} 行、通道 0–${(cfg?.window.Wc ?? 64) - 1}）中取出对应的一行。

结果是一个长度为 ${c_dimRef(`_C_ = ${cfg?.n_embd ?? 2048}`, DimStyle.C)} 的向量，称为输入嵌入。Qwen3-VL 没有可学习的位置表：位置信息在注意力里通过 RoPE 旋转 Q/K 注入。
    `;
    breakAfter();

    let t1_fadeEmbedAnim = afterTime(null, 0.3);
    let t2_highlightTokenEmbed = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
嵌入表是训练得到的（蓝色表示可学习参数；k005b 的嵌入表与输出层权重共享）。取出的这一行直接成为输入嵌入的这一列。
`;

    breakAfter();

    let t3_moveTokenEmbed = afterTime(null, 0.8);
    let t7_addAnim = afterTime(null, 0.8);
    let t8_placeAnim = afterTime(null, 0.8);
    let t9_cleanupInstant = afterTime(null, 0.0);
    let t10_fadeAnim = afterTime(null, 0.8);

    breakAfter();

    commentary(wt)`
对窗口中的每个 token 重复同样的查表（id 不在显示行范围内的 token，其嵌入行没有画出，但输入嵌入列仍是真实值）。
`;

    breakAfter();

    let t11_fillRest = afterTime(null, 5.0);

    breakAfter();

    commentary(wt)`
可以把鼠标悬停在${c_blockRef('_输入嵌入_', state.layout.residual0)}矩阵的单元格上，查看计算过程和来源。

处理完所有 token 后得到一个 ${c_dimRef('_T_', DimStyle.T)} x ${c_dimRef('_C_', DimStyle.C)} 的矩阵。
${c_dimRef('_T_', DimStyle.T)} 是序列长度（${c_dimRef('_时间_', DimStyle.T)} / 位置维），${c_dimRef('_C_', DimStyle.C)} 是${c_dimRef('_通道_', DimStyle.C)}数，也叫特征维或嵌入维度，是模型的超参数。

这个${c_blockRef('_输入嵌入_', state.layout.residual0)}将被送入后面的 Qwen3-VL 主干（共 ${cfg?.n_layer ?? 28} 层 transformer）。
这种 ${c_dimRef('T', DimStyle.T)} 列、每列长 ${c_dimRef('C', DimStyle.C)} 的结构会在后文反复出现。
`;

    cleanup(t9_cleanupInstant, [t3_moveTokenEmbed, t7_addAnim, t8_placeAnim]);
    cleanup(t10_fadeAnim, [t0_splitEmbedAnim, t1_fadeEmbedAnim, t2_highlightTokenEmbed]);

    frameTo(state, t_moveCamera, [args.layout.idxObj, args.layout.residual0], 290, 15.5, { fill: 1.25 });

    let residCol: IBlkDef = null!;
    if ((t0_splitEmbedAnim.t > 0.0 || t10_fadeAnim.t > 0.0) && t11_fillRest.t === 0) {
        splitGrid(layout, layout.idxObj, Dim.X, exampleIdx + 0.5, t0_splitEmbedAnim.t * 4.0);

        layout.residual0.access!.disable = true;
        layout.residual0.opacity = lerp(1.0, 0.1, t1_fadeEmbedAnim.t);

        residCol = splitGrid(layout, layout.residual0, Dim.X, exampleIdx + 0.5, t0_splitEmbedAnim.t * 4.0)!;
        residCol.highlight = 0.3;

        residCol.opacity = lerp(1.0, 0.0, t1_fadeEmbedAnim.t);

    }

    let tokValue = Math.max(0, layout.model ? embedColForToken(layout.model, exampleIdx) : 0);


    let tokColDupe: IBlkDef | null = null;

    if (t2_highlightTokenEmbed.t > 0.0) {
        let tokEmbedCol = splitGrid(layout, layout.tokEmbedObj, Dim.X, tokValue + 0.5, t2_highlightTokenEmbed.t * 4.0)!;

        tokColDupe = duplicateGrid(layout, tokEmbedCol);
        tokColDupe.t = 'i';
        tokEmbedCol.highlight = 0.3;

        let startPos = new Vec3(tokEmbedCol.x, tokEmbedCol.y, tokEmbedCol.z);
        let targetPos = new Vec3(residCol.x, residCol.y, residCol.z).add(new Vec3(-2.0, 0, 3.0));

        let pos = startPos.lerp(targetPos, t3_moveTokenEmbed.t);

        tokColDupe.x = pos.x;
        tokColDupe.y = pos.y;
        tokColDupe.z = pos.z;
    }


    let origResidPos = residCol ? new Vec3(residCol.x, residCol.y, residCol.z) : new Vec3();
    let offsetResidPos = origResidPos.add(new Vec3(0.0, 0, 3.0));

    if (t7_addAnim.t > 0.0 && tokColDupe) {
        let targetPos = offsetResidPos;
        let tokStartPos = new Vec3(tokColDupe.x, tokColDupe.y, tokColDupe.z);

        let tokPos = tokStartPos.lerp(targetPos, t7_addAnim.t);

        tokColDupe.x = tokPos.x;
        tokColDupe.y = tokPos.y;
        tokColDupe.z = tokPos.z;

        if (t7_addAnim.t > 0.95) {
            tokColDupe.opacity = 0.0;
            residCol.opacity = 1.0;
            residCol.highlight = 0.0;
            residCol.access!.disable = false;
            residCol.x = targetPos.x;
            residCol.y = targetPos.y;
            residCol.z = targetPos.z;
        }
    }

    if (t8_placeAnim.t > 0.0) {
        let startPos = offsetResidPos;
        let targetPos = origResidPos;
        let pos = startPos.lerp(targetPos, t8_placeAnim.t);
        residCol.x = pos.x;
        residCol.y = pos.y;
        residCol.z = pos.z;
    }

    if (t9_cleanupInstant.t > 0.0 && residCol) {
        residCol.opacity = 1.0;
        residCol.highlight = 0.0;
        residCol.access!.disable = false;
    }

    if (t11_fillRest.t > 0.0) {
        layout.residual0.access!.disable = true;

        let prevInfo = startProcessBefore(state, layout.residual0);
        processUpTo(state, t11_fillRest, layout.residual0, prevInfo);
    }
    // new Vec3(-6.9, 0, -36.5), new Vec3(281.5, 5.5, 0.8)
}
