import { dimProps, duplicateGrid, findSubBlocks, splitGrid, splitGridAll } from "../Annotations";
import { drawDataFlow, getBlockValueAtIdx } from "../components/DataFlow";
import { getBlkDimensions, IBlkDef, setBlkPosition } from "../VlaModelLayout";
import { drawDependences } from "../Interaction";
import { IProgramState } from "../Program";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { clamp } from "@/src/utils/data";
import { inverseLerp, lerp, lerpSmoothstep } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";
import { embedInline } from "./Walkthrough01_Prelim";
import { Colors, commentary, DimStyle, IWalkthroughArgs, frameTo, initialFrame, drawIdxTag, TraceColors } from "./WalkthroughTools";
import { Box, fmtA, N, QkLedger, TermsTable } from "../components/TraceLedger";
import { BlockText } from '../components/CommentaryHelpers';

export function walkthrough04_SelfAttention(args: IWalkthroughArgs) {
    let { walkthrough: wt, layout, state, tools: { afterTime, c_blockRef, c_dimRef, breakAfter, cleanup } } = args;
    let { C, A, nHeads } = layout.shape;

    if (wt.phase !== Phase.Input_Detail_SelfAttention) {
        return;
    }

    let block0 = layout.blocks[0];
    let head0 = block0.heads[0];

    initialFrame(state, [block0.ln1.lnResid, ...block0.heads.flatMap(h => h.cubes)], 294, 12.8, { fill: 1.0 });
    wt.dimHighlightBlocks = [layout.residual0, block0.ln1.lnResid, ...head0.cubes];

    let f = state.page.facts!;
    let pi0 = f.id === 'pi0';
    let groot = f.id === 'groot';

    let tr = state.vlaModel?.trace;
    let at = tr?.attn;
    let T = layout.shape.T;
    let qi = T - 1;                                   // traced query = last window column
    let jw = at ? T - 1 - (at.p - at.j) : 0;          // traced key j, as a window column
    let tq = `t = ${qi}`;
    let traceIntro = embedInline(at && <span>本章接着跟踪上一章那个 token（窗口最后一列 {tq}，序列位置 {at.p}）。在 head {at.head} 上只追第 {at.d} 维（图中 Q/K/V 块最上面一行）：上一章的输出 y[{at.c}] = <N v={tr!.rms.y_c} carry /> 从这里进入 Q 投影，一路变成注意力输出 o[{at.d}]，再经输出投影回到通道 {at.c} 的残差。</span>);
    if (groot) commentary(wt, null, 0)`
注意力是 transformer 层的核心：输入嵌入矩阵中的各列在这里相互“交流”。在其他阶段，各列都是相互独立处理的。

GR00T 主干（${f.trunkName}）的语言模型层用 GQA：每层 ${f.nHead} 个 query head、${f.nKv} 个 key/value head（每 ${f.nHead / f.nKv} 个 query head 共享一组 K/V），head 维度 ${f.headDim}。图中只画 L0 的 head 0（真实数值，显示 ${layout.shape.T} 个 token × ${layout.shape.A} 维的窗口）：其余 head 结构相同、只是权重不同，所以不重复铺开，标注为 ×${f.nHead}。Q/K 在点积前还要经过 q_norm / k_norm（RMSNorm）和 M-RoPE 旋转。主干只做自注意力：这里没有动作 token 参与，DiT 是在主干跑完之后才读它的输出。

${traceIntro}`;
    else if (pi0) commentary(wt, null, 0)`
注意力是 transformer 层的核心：输入嵌入矩阵中的各列在这里相互“交流”。在其他阶段，各列都是相互独立处理的。

π0.5（Gemma-2B）的注意力是 MQA：每层 ${f.nHead} 个 query head 共享 _同一组_ key/value head（${f.nKv} 个 KV 头），head 维度 ${f.headDim}。图中只画 L0 的 head 0（真实数值，显示 ${layout.shape.T} 个 token × ${layout.shape.A} 维的窗口）：其余 ${f.nHead - 1} 个 head 与它结构相同、K/V 完全共用，只是 Q 权重不同，所以不重复铺开，标注为 ×${f.nHead}。Gemma 没有 q_norm / k_norm，Q/K 在点积前只做 RoPE 旋转。

${traceIntro}`;
    else commentary(wt, null, 0)`
注意力是 transformer 层的核心：输入嵌入矩阵中的各列在这里相互“交流”。在其他阶段，各列都是相互独立处理的。

k005b 的注意力是 GQA：每层 16 个 query head、8 个 key/value head（每 2 个 query head 共享一组 K/V），head 维度 128。图中只画 L0 的 head 0（真实数值，显示 32 个 token × 64 维的窗口）：其余 15 个 head 与 7 组 K/V 结构完全相同，只是权重不同，所以不重复铺开，标注为 ×16。Q/K 在点积前还要经过 q_norm / k_norm（RMSNorm）和 RoPE 旋转。

${traceIntro}`;
    breakAfter();
    let t_moveCamera = afterTime(null, 1.0);
    let t_highlightHeads = afterTime(null, 2.0);
    let t_moveCamera2 = afterTime(null, 1.0);
    let t_focusHeads = focusSelfAttentionHeadTimers(args, 3.0);

    breakAfter();
    commentary(wt)`
第一步，从${c_blockRef('归一化输入嵌入矩阵', block0.ln1.lnResid)}出发，为 ${c_dimRef('T', DimStyle.T)} 个列中的每一列生成三个向量，即 Q、K、V：

${embedInline(<ul>
    <li>Q: <BlockText blk={head0.qBlock}>查询向量 (Query)</BlockText></li>
    <li>K: <BlockText blk={head0.kBlock}>键向量 (Key)</BlockText></li>
    <li>V: <BlockText blk={head0.vBlock}>值向量 (Value)</BlockText></li>
</ul>)}

每个向量都由一次矩阵-向量乘法得到（${pi0 ? 'Gemma' : 'Qwen3'} 的 QKV 没有偏置），每个输出单元格都是输入向量的线性组合。我们追的那一格是${c_blockRef('Q 向量', head0.qBlock)}最后一列（${c_dimRef(tq, DimStyle.T)}）的第 0 行：它是${c_blockRef('Q 权重矩阵', head0.qWeightBlock)}的第 0 行与${c_blockRef('输入矩阵', block0.ln1.lnResid)}的最后一列做点积。`;
    breakAfter();

    let t_focusQCol = afterTime(null, 1.0);
    let t_qIterColDot = afterTime(null, 3.0);

    breakAfter();
    commentary(wt)`
点积很简单：把两个向量对应位置的元素两两相乘，再把结果相加。`;
    breakAfter();

    let t_moveDotCells = afterTime(null, 2.0, 0.5);
    let t_dotCellsZoomClose = afterTime(null, 1.0, 0.5);
    let t_collapseDotCellsA = afterTime(null, 2.0);
    let t_collapseDotCellsB = afterTime(null, 2.0, 0.5);
    let t_dotCellsZoomOut = afterTime(null, 1.0, 0.5);
    let t_addBias = afterTime(null, 2.0, 0.5);
    let t_moveToDest = afterTime(null, 0.5);

    breakAfter();
    commentary(wt)`

这是一种让每个输出元素都能受整个输入向量影响（影响程度由权重决定）的通用方式，因此在神经网络中随处可见。

这一格的真实账目（${f.dModel} 项全算，只列最大的几项；第一行 i = ${at?.c ?? '…'} 的输入就是上一章的 y）：
${embedInline(at && <QkLedger p={at.q} inName='y' carryIn normName='q_norm' />)}
图中 Q 格存的就是 RoPE 之后的 ${embedInline(at && <span>q[{at.d}] = <N v={at.q.rope.post[0]} out /></span>)}（RoPE 把第 ${at?.q.rope.pair[0] ?? 0} 维与第 ${at?.q.rope.pair[1] ?? 0} 维配成一对旋转，所以要同时算这两维的投影）。

我们对 Q、K、V 向量的每个输出单元格重复这一运算：`;
    breakAfter();

    let t_revertFocusCol = afterTime(null, 0.25, 0.5);
    cleanup(t_revertFocusCol, [t_focusQCol]);
    let t_processQkv = afterTime(null, 5.0);

    breakAfter();
    commentary(wt)`
Q、K、V 各司其职：用当前列的 Q 去和每个可见列的 K 做点积，得到相似度分数；softmax 把这一行分数变成和为 1 的权重；再用这些权重把各列的 V 加起来。

下面用真实数字走一遍我们这一列（${c_dimRef(tq, DimStyle.T)}）：`;
    breakAfter();

    let t_focusQKVCols = afterTime(null, 1.0);

    breakAfter();
// It would like to find relevant information from other columns and extract their values. The other
// columns each have a K (key) vector, which represents the information that that column has, and our
// Q (query) vector is what information is relevant to us.
    commentary(wt)`
当前列（${c_dimRef(tq, DimStyle.T)}）的${c_blockRef('Q 向量', head0.qBlock)}要与它可见的每一列的${c_blockRef('K 向量', head0.kBlock)}做点积${pi0 ? '（π0.5 的前缀是双向注意力，所以是窗口内的 _所有_ 列，不分先后）' : ''}，结果存入${c_blockRef('注意力矩阵', head0.attnMtx)}的第 ${qi} 行。

拿其中一列来对账：位置 ${at?.j ?? '…'}（图中第 ${jw} 列）。它的 k[0] 由同一套流程从它自己那一列输入算出：
${embedInline(at && <QkLedger p={at.k} inName={`y@${at.j}`} carryIn={false} normName='k_norm' />)}
`;
    breakAfter();

    let t_processAttnRow = afterTime(null, 3.0);

    breakAfter();
    commentary(wt)`
点积衡量两个向量的相似度：越相似，点积越大；差异越大，点积越小甚至为负。

${groot ? '在 GR00T 的语言模型层中，注意力是因果的：每个 token 只能看到它自己和它之前的 token，所以注意力矩阵是下三角，被掩码的位置是真实的 -inf（softmax 后为 0）。主干最后 4 块 VL 自注意力则是双向的。' : pi0 ? '在 π0.5 中，前缀内部是双向注意力：每个前缀 token 都能看到全部前缀 token，所以主干的注意力矩阵没有三角掩码。后缀流（动作专家）的查询可以关注全部前缀和全部后缀；前缀流不会“看到”后缀。' : '在 k005b 中，后缀流（动作专家）的查询可以关注前缀流的全部 K/V，也可以关注后缀自身；前缀流不会“看到”后缀。图中被掩码的位置是真实的 -inf（softmax 后为 0）。'}

点积之后还要除以 sqrt(${c_dimRef('A', DimStyle.A)}) = sqrt(${f.headDim})，其中 ${c_dimRef('A', DimStyle.A)} 是 Q/K/V 向量的长度。这样可避免较大的数值在下一步 softmax 中占据主导。我们这一对 (q, k_j) 的分数：
${embedInline(at && <Box>
    <TermsTable t={at.dot} wName='q_i' xHead='k_i' xName={i => `[${i}]`} carryFirst carryW carryRows={2} />
    <div>s_j = <N v={at.dot.sum} /> × 1/√{at.D} (<N v={at.scale} />) = <N v={at.s_j} out /></div>
</Box>)}
它就是注意力矩阵第 ${qi} 行第 ${jw} 列那一格。

然后 softmax 把这一行（${at?.n_keys ?? '…'} 个可见位置，图中只画窗口内 ${T} 个）归一化为总和为 1：
${embedInline(at && <Box>
    <div>行最大值 max = <N v={at.max} />（位置 {at.max_at}），Z = Σ exp(s − max) = <N v={at.Z} /></div>
    <div>p_j = exp(<N v={at.s_j} carry /> − <N v={at.max} />) / <N v={at.Z} /> = <N v={at.p_j} out /></div>
    <table className='font-mono text-xs mt-1'><thead><tr className='text-slate-400'><td className='pr-3'>前 3 名位置</td><td className='pr-3'>s</td><td>p</td></tr></thead>
        <tbody>{at.top.map(([i, sv, pv]) => <tr key={i}><td className='pr-3'>{i}{i === at!.j ? ' ← j' : ''}</td><td className='pr-3'><N v={sv} /></td><td><N v={pv} /></td></tr>)}</tbody></table>
</Box>)}
${at && at.max_at !== at.j ? (at.top[0][2] > 0.5
    ? `如实说明：这一行 ${fmtA(at.top[0][2] * 100)}% 的权重落在位置 ${at.max_at}（${at.max_at < at.p - qi ? '窗口之外，图中看不到' : '窗口内'}），它像一个“汇点”（attention sink）吸走了几乎全部注意力；窗口内各列只分到很小的份额，所以图中这一行看起来很暗。`
    : `如实说明：这一行的权重很分散，最大的位置 ${at.max_at} 也只占 ${fmtA(at.top[0][2] * 100)}%，j 只是其中一份。`) : ''}`;
    breakAfter();

    let t_processAttnSmAggRow = afterTime(null, 1.0);
    let t_processAttnSmRow = afterTime(null, 2.0);

    breakAfter();
    commentary(wt)`
最后生成当前列（${c_dimRef(tq, DimStyle.T)}）的输出向量：取${c_blockRef('归一化后的注意力矩阵', head0.attnMtxSm)}的第 ${qi} 行，用每个权重去逐元素乘以对应列的${c_blockRef('V 向量', head0.vBlock)}。`;
    breakAfter();

    let t_zoomVOutput = afterTime(null, 0.4, 0.5);
    // multi-step process here, displaying the multiplication of each element of the row with each column of V
    let t_expandVCols = afterTime(null, 1.0, 0.5);
    let t_moveAttnVals = afterTime(null, 1.0, 0.5);
    let t_applyMultiplies = afterTime(null, 1.0, 0.5);
    let t_applyAdds = afterTime(null, 1.0, 0.5);
    let t_placeVOutput = afterTime(null, 1.0, 0.5);
    let t_finalizeVOutput = afterTime(null, 0.5, 0.5);

    breakAfter();
    commentary(wt)`
把它们相加就得到输出向量，因此输出主要由得分高的列的 V 向量决定。第 0 维的账目（p 来自上面的 softmax）：
${embedInline(at && <Box>
    <TermsTable t={at.o} wName='p_j' xHead='v_j[0]' xName={i => `j=${i}`} carryW />
    <div>o[{at.d}] = <N v={at.o.sum} out /></div>
</Box>)}
它就是${c_blockRef('注意力输出', head0.vOutBlock)}最后一列的第 0 格，也是“输出投影与残差”一章 o_proj 点积中第 ${at?.o_slot ?? 0} 项的输入。

流程清楚了，现在对所有列都运行一遍。`;

    breakAfter();

    let t_processRemainZoom = afterTime(null, 0.5, 0.5);
    cleanup(t_processRemainZoom, [t_focusQKVCols]);
    let t_processRemain = afterTime(null, 8.0);

    breakAfter();
    commentary(wt)`
以上就是一个 head 的注意力过程：每一列通过用自己的 _query_ 去比较其他列的 _key_，找到相关信息并提取它们的值，并且只能看向允许的位置（${groot ? '因果掩码：只看自己和之前的 token' : '后缀可看前缀，前缀不看后缀'}）。注意力输出随后经过输出投影，再加回残差。
`;

// Running this process for all the columns produces our self-attention matrix, which is a square
// matrix, T x T, and due to the causal nature of the process, is a lower triangular matrix.

    frameTo(state, t_moveCamera, block0.heads.flatMap(h => h.cubes), 293.5, 49, { fill: 1.0 });
    frameTo(state, t_moveCamera2, [block0.ln1.lnResid, ...head0.cubes], 286, 12.8, { fill: 1.1 });


    if (t_highlightHeads.t > 0.0) {
        block0.selfAttendLabel.visible = lerp(0, 1, t_highlightHeads.t * 10);
        // one representative head (the other 15 share its slot and are not drawn)
        let headIdx = 0;
        let headFrac = clamp(t_highlightHeads.t * 3, 0, 1);

        let labelOpacity = lerp(0.0, 1.0, headFrac / 0.3); // * lerp(1.0, 0.0, (1.0 - headFrac) / 0.3);

        let head = block0.heads[headIdx];
        head.headLabel.visible = labelOpacity;

        for (let blk of head.headLabel.cubes) {
            blk.highlight = labelOpacity * 0.4;
        }
    }

    if (t_focusHeads.t0_dissolveHeads.t > 0.0) {
        let head = block0.heads[0];
        let t = t_focusHeads.t0_dissolveHeads.t;
        for (let blk of head.headLabel.cubes) {
            blk.highlight = lerp(1.0, 0.0, t * 4) * 0.4;
        }
        head0.qBlock.access!.disable = true;
        head0.kBlock.access!.disable = true;
        head0.vBlock.access!.disable = true;
    }

    focusSelfAttentionHead(args, t_focusHeads);

    frameTo(state, t_dotCellsZoomClose, [block0.ln1.lnResid, head0.qBlock], 274.1, 8.5, { fill: 1.1 });
    frameTo(state, t_dotCellsZoomOut, [block0.ln1.lnResid, ...head0.cubes], 286, 12.8, { fill: 1.1 });

    let exampleIdx = qi;
    if (t_focusQCol.t > 0) {
        let otherOpacity = lerp(1.0, 0.2, t_focusQCol.t);
        head0.qBlock.opacity = otherOpacity;
        head0.kBlock.opacity = otherOpacity;
        head0.vBlock.opacity = otherOpacity;
        block0.ln1.lnResid.opacity = otherOpacity;

        let splitAmt = lerp(0.0, 2.0, t_focusQCol.t);
        let qCol = splitGrid(layout, head0.qBlock, Dim.X, exampleIdx + 0.5, splitAmt)!;
        let inputCol = splitGrid(layout, block0.ln1.lnResid, Dim.X, exampleIdx + 0.5, splitAmt)!;

        qCol.opacity = 1.0;
        inputCol.opacity = 1.0;

        if (t_qIterColDot.t > 0) {
            // sweep bottom-up so it ends on row 0, the traced q[0]
            let aIdx = clamp(A - 1 - Math.floor(t_qIterColDot.t * A), 0, A - 1);
            let destIdx = new Vec3(exampleIdx, aIdx, 0);
            let pinIdx = new Vec3(exampleIdx, 0, 0);

            drawDependences(state, head0.qBlock, destIdx);
            drawDataFlow(state, head0.qBlock, destIdx, pinIdx);
            splitGrid(layout, qCol, Dim.Y, aIdx + 0.5, 0);
            for (let b of findSubBlocks(qCol, Dim.Y, aIdx, null)) {
                b.access!.disable = false;
            }
            inputCol.highlight = 0.3;
            if (at && t_moveDotCells.t === 0) drawIdxTag(state, block0.ln1.lnResid, qi, at.c, `y[${at.c}]=${fmtA(tr!.rms.y_c)}`, TraceColors.carry);
        }

        let targetTop = new Vec3(inputCol.x - 26, inputCol.y, inputCol.z + 5);
        let addTarget = new Vec3(targetTop.x + layout.cell, targetTop.y - layout.cell * 12, targetTop.z);
        let biasTarget = new Vec3(addTarget.x - layout.cell * 3, addTarget.y, addTarget.z);

        if (t_moveDotCells.t > 0 && t_moveToDest.t === 0.0) {
            let qWeightRow = findSubBlocks(head0.qWeightBlock, Dim.Y, 0, 0)[0];

            let qCells = splitGridAll(layout, qWeightRow, Dim.X);
            let inCells = splitGridAll(layout, inputCol, Dim.Y);

            let cellMovePct = 0.5;
            let prevCY = 0;
            for (let c = 0; c < C; c++) {
                let cPos = c / (C - 1);
                let startT = (1.0 - cellMovePct) * (1.0 - cPos);
                let cellMoveT = inverseLerp(startT, startT + cellMovePct, t_moveDotCells.t);

                let qInitial = getBlkDimensions(qCells[c]);
                let qFinal = targetTop.add(new Vec3(0, c * layout.cell * 1.2));

                let inInitial = getBlkDimensions(inCells[c]);
                let inFinal = targetTop.add(new Vec3(layout.cell * 2, c * layout.cell * 1.2));

                if (t_dotCellsZoomOut.t > 0) {
                    qFinal = inFinal = new Vec3(targetTop.x + layout.cell, targetTop.y - layout.cell * 12, qFinal.z);
                }

                // animate the cells from the initial grid position to be lined up next to each other
                setBlkPosition(qCells[c], qInitial.tl.lerp(qFinal, cellMoveT));
                setBlkPosition(inCells[c], inInitial.tl.lerp(inFinal, cellMoveT));

                let transitionPt = 0.15;
                let collapsDotCellsT = lerp(0.0, transitionPt, t_collapseDotCellsA.t) + lerp(0, 1-transitionPt, t_collapseDotCellsB.t);

                let startT2 = 0.9 * cPos;
                let cellTimesSymT = inverseLerp(startT2, startT2 + 0.1, collapsDotCellsT);

                // move the cells together in the times/add operation, with them ending up in the same place
                if (cellTimesSymT > 0.0 && t_dotCellsZoomOut.t == 0.0) {
                    let qCurr = getBlkDimensions(qCells[c]);
                    let inCurr = getBlkDimensions(inCells[c]);

                    let qCellPos = new Vec3(
                        lerp(qCurr.tl.x, addTarget.x, cellTimesSymT * 2.0),
                        lerp(qCurr.tl.y, addTarget.y, cellTimesSymT),
                        qCurr.tl.z
                    );
                    let inCellPos = new Vec3(
                        lerp(inCurr.tl.x, addTarget.x, cellTimesSymT * 2.0),
                        lerp(inCurr.tl.y, addTarget.y, cellTimesSymT),
                        qCurr.tl.z
                    );

                    setBlkPosition(qCells[c], qCellPos);
                    setBlkPosition(inCells[c], inCellPos);

                    if (c > 0 && cellTimesSymT > 0.0) {

                        let midPt = new Vec3(
                            lerp(qCurr.br.x, inCurr.tl.x, 0.5),
                            lerp(prevCY, qCellPos.y, 0.5),
                            qCurr.tl.z + layout.cell/2);

                        let mtx = Mat4f.fromTranslation(midPt);
                        let fontOpts: IFontOpts = { color: Colors.Black, size: 1.5, mtx };
                        let w = measureText(state.render.modelFontBuf, "+", fontOpts);
                        drawText(state.render.modelFontBuf, "+", -w/2, -fontOpts.size/2, fontOpts);
                    }
                    prevCY = qCellPos.y + layout.cell;
                }

                // draw times symbol between blocks (provided they're next to each other)
                // done after the other operations so that they are always positioned correctly
                if (cellMoveT >= 1.0) {
                    drawSymbolBetweenBlocks(args, qCells[c], inCells[c], Dim.X, "x", { size: 1.5, color: Colors.Black });
                }
            }

            // move the bias block next to the resulting dotprod cell, draw a plus between them, and then join them together
            if (t_addBias.t >= 0.0) {
                let qBiasCell = findSubBlocks(head0.qBiasBlock, Dim.Y, 0, 0)[0];

                let qBiasInitial = getBlkDimensions(qBiasCell);
                let qBiasPos = qBiasInitial.tl.lerp(biasTarget, inverseLerp(0, 0.4, t_addBias.t));
                setBlkPosition(qBiasCell, qBiasPos);

                let moveTogetherT = inverseLerp(0.6, 1.0, t_addBias.t);
                qBiasInitial = getBlkDimensions(qBiasCell);
                qBiasPos = qBiasInitial.tl.lerp(addTarget, moveTogetherT);
                setBlkPosition(qBiasCell, qBiasPos);

                if (t_addBias.t > 0.4) {
                    drawSymbolBetweenBlocks(args, qBiasCell, qCells[qCells.length - 1], Dim.X, "+", { size: 1.5, color: Colors.Black });
                }
            }
        }

        // move the resulting dotprod cell to the final destination
        if (t_moveToDest.t > 0) {
            let qWeightRow = findSubBlocks(head0.qWeightBlock, Dim.Y, 0, 0)[0];
            let qBiasCell = findSubBlocks(head0.qBiasBlock, Dim.Y, 0, 0)[0];
            qBiasCell.opacity = t_moveToDest.t;
            qWeightRow.opacity = t_moveToDest.t;
            inputCol.opacity = t_moveToDest.t;

            let qResultCell = findSubBlocks(qCol, Dim.Y, 0, 0)[0];
            let qResultInitial = getBlkDimensions(qResultCell);
            let qResultPos = qResultInitial.tl.lerp(addTarget, 1.0 - t_moveToDest.t);

            setBlkPosition(qResultCell, qResultPos);
        }
    }

    // simple run-through and process of each of the Q, K, V blocks
    if (t_processQkv.t > 0) {
        let processStart = startProcessBefore(state, head0.qBlock);
        processUpTo(state, t_processQkv, head0.vBlock, processStart);
    }

    // highlight the example index column for the Q block, and previous columns for K and V
    let attnExampleIdx = qi;
    if (t_focusQKVCols.t > 0 && t_processRemain.t <= 0) {
        let ignoreOpacity = lerp(1.0, 0.2, t_focusQKVCols.t);
        head0.qBlock.opacity = ignoreOpacity;
        head0.kBlock.opacity = ignoreOpacity;
        head0.vBlock.opacity = ignoreOpacity;

        let qCol = splitGrid(layout, head0.qBlock, Dim.X, attnExampleIdx + 0.5, 0)!;
        splitGrid(layout, head0.kBlock, Dim.X, attnExampleIdx + 0.5, 0)!;
        splitGrid(layout, head0.vBlock, Dim.X, attnExampleIdx + 0.5, 0)!;

        let kBeforeCols = findSubBlocks(head0.kBlock, Dim.X, null, attnExampleIdx);
        let vBeforeCols = findSubBlocks(head0.vBlock, Dim.X, null, attnExampleIdx);

        for (let col of [...kBeforeCols, ...vBeforeCols, qCol]) {
            col.opacity = 1.0;
        }

        head0.attnMtx.access!.disable = true;
        head0.attnMtxSm.access!.disable = true;
        head0.attnMtxAgg1.access!.disable = true;
        head0.attnMtxAgg2.access!.disable = true;

        head0.qBlock.opacity = 1.0;
        head0.kBlock.opacity = 1.0;
        head0.vBlock.opacity = 1.0;
    }

    frameTo(state, t_focusQKVCols, [head0.qBlock, head0.kBlock, head0.vBlock, head0.attnMtx, head0.attnMtxSm], 270.1, -20, { fill: 1.4 });

    if (at && t_moveToDest.t >= 1 && t_processRemain.t <= 0) {
        drawIdxTag(state, head0.qBlock, qi, 0, `q[0]=${fmtA(at.q.rope.post[0])}`, TraceColors.carry);
        if (t_focusQKVCols.t > 0) {
            drawIdxTag(state, head0.kBlock, jw, 0, `k[0] @j=${at.j}: ${fmtA(at.k.rope.post[0])}`, TraceColors.carry, 'left');
            drawIdxTag(state, head0.vBlock, jw, 0, `v[0] @j=${at.j}: ${fmtA(at.v_j)}`, TraceColors.carry, 'left');
        }
    }

    if (t_processAttnRow.t > 0 && t_processRemain.t <= 0) {
        let aIdx = clamp(Math.floor(t_processAttnRow.t * (attnExampleIdx + 1)), 0, attnExampleIdx);
        let destIdx = new Vec3(aIdx, attnExampleIdx, 0);
        let pinIdx = new Vec3(attnExampleIdx, 0, 0);

        if (t_processAttnSmAggRow.t <= 0) {
            drawDependences(state, head0.attnMtx, destIdx);
            drawDataFlow(state, head0.attnMtx, destIdx, pinIdx);
        }

        let attnRow = splitGrid(layout, head0.attnMtx, Dim.Y, attnExampleIdx, 0)!;
        splitGrid(layout, attnRow, Dim.X, aIdx, 0)!;
        let attnRowStart = findSubBlocks(attnRow, Dim.X, null, aIdx);

        for (let blk of attnRowStart) {
            blk.access!.disable = false;
        }
        if (at && aIdx >= jw && t_zoomVOutput.t <= 0) drawIdxTag(state, head0.attnMtx, jw, qi, `s[${at.j}]=${fmtA(at.s_j)}`, TraceColors.out);
    }

    if (t_processAttnSmAggRow.t > 0 && t_processRemain.t <= 0) {
        let agg0T = inverseLerp(0, 0.5, t_processAttnSmAggRow.t);
        let agg1T = inverseLerp(0.5, 1.0, t_processAttnSmAggRow.t);
        let hidePopup = t_processAttnSmRow.t > 0;
        processDim(state, head0.attnMtxAgg2, Dim.Y, attnExampleIdx, agg0T, { pinIdx: new Vec3(5, 0, 0), clamp: true, hidePopup });

        if (agg1T > 0) {
            processDim(state, head0.attnMtxAgg1, Dim.Y, attnExampleIdx, agg1T, { pinIdx: new Vec3(-12, 0, 0), clamp: true, hidePopup });
        }
    }

    if (t_processAttnSmRow.t > 0 && t_processRemain.t <= 0) {
        let hidePopup = t_zoomVOutput.t > 0;
        processDim(state, head0.attnMtxSm, Dim.Y, attnExampleIdx, t_processAttnSmRow.t, { pinIdx: new Vec3(5, 0, 0), clamp: true, maxIdx: attnExampleIdx + 1, hidePopup });
        if (at && t_processAttnSmRow.t * (attnExampleIdx + 1) >= jw && t_expandVCols.t <= 0) drawIdxTag(state, head0.attnMtxSm, jw, qi, `p[${at.j}]=${fmtA(at.p_j)}`, TraceColors.out, 'left');
    }

    if (t_zoomVOutput.t > 0 && t_processRemain.t <= 0) {
        head0.vOutBlock.access!.disable = true;
    }

    // Do the attn_sm times V vectors to get the output vectors animation
    {
        let topLeftPos = getBlkDimensions(head0.vBlock).tl.add(new Vec3(0, 4, 5));
        let midLeftPos = topLeftPos.add(new Vec3(0, layout.cell * (A / 2 - 0.5)));
        // many columns: pack them 2 cells apart and drop each weight in from above (no room for x/+ symbols)
        let nV = attnExampleIdx + 1;
        let dense = nV > 8;
        let sp = dense ? 2 : 5;
        let expandBox = { x: topLeftPos.x, y: topLeftPos.y - 3 * layout.cell, z: topLeftPos.z, dx: nV * sp * layout.cell, dy: (A + 3) * layout.cell, dz: layout.cell };
        frameTo(state, t_zoomVOutput, [head0.vBlock, head0.attnMtxSm, head0.vOutBlock, expandBox], 270.1, -7.5, { fill: 1.2 });

        if (t_expandVCols.t > 0 && t_placeVOutput.t <= 0) {
            let allVCols = [];
            let vBeforeCols = findSubBlocks(head0.vBlock, Dim.X, null, attnExampleIdx);
            for (let col of vBeforeCols) {
                allVCols.push(...splitGridAll(layout, col, Dim.X));
            }

            let allAttnCells = [];
            let attnRow = findSubBlocks(head0.attnMtxSm, Dim.Y, attnExampleIdx, attnExampleIdx)[0];
            let attnCellsBefore = findSubBlocks(attnRow, Dim.X, null, attnExampleIdx);
            for (let cell of attnCellsBefore) {
                for (let subCell of splitGridAll(layout, cell, Dim.X)) {
                    allAttnCells.push(duplicateGrid(layout, subCell));
                }
            }

            for (let i = 0; i < attnExampleIdx + 1; i++) {
                let attnVal = getBlockValueAtIdx(head0.attnMtxSm, new Vec3(i, attnExampleIdx, 0)) ?? 0.2;

                let initColPos = getBlkDimensions(allVCols[i]).tl;
                let destColPos = topLeftPos.add(new Vec3(i * layout.cell * sp, 0, 0));

                setBlkPosition(allVCols[i], initColPos.lerp(destColPos, t_expandVCols.t));

                let initAttnPos = getBlkDimensions(allAttnCells[i]).tl;
                let destAttnPos = dense ? topLeftPos.add(new Vec3(i * layout.cell * sp, -2 * layout.cell)) : midLeftPos.add(new Vec3(i * layout.cell * 5 - 2 * layout.cell, 0));

                setBlkPosition(allAttnCells[i], initAttnPos.lerp(destAttnPos, t_moveAttnVals.t));

                if (t_applyMultiplies.t > 0) {
                    initAttnPos = destAttnPos;
                    destAttnPos = initAttnPos.add(dense ? new Vec3(0, layout.cell * 2) : new Vec3(layout.cell * 2, 0));
                    setBlkPosition(allAttnCells[i], initAttnPos.lerp(destAttnPos, t_applyMultiplies.t));

                    allAttnCells[i].opacity = 1.0 - t_applyMultiplies.t;
                    allVCols[i].highlight = lerp(0.0, attnVal * 1.5, t_applyMultiplies.t);
                }

                if (at && !dense && i === jw && t_applyAdds.t <= 0) drawIdxTag(state, allVCols[i], i, 0, `${fmtA(at.p_j)} x ${fmtA(at.v_j)}`, TraceColors.carry);

                if (!dense && t_moveAttnVals.t > 0.8 && t_applyMultiplies.t < 0.7) {
                    drawSymbolBetweenBlocks(args, allVCols[i], allAttnCells[i], Dim.X, 'x', { color: Colors.Black, size: 1.5 });
                }

                if (t_applyAdds.t > 0) {
                    initColPos = destColPos;
                    destColPos = topLeftPos.add(new Vec3(0, 0, attnVal * 1.0));

                    setBlkPosition(allVCols[i], initColPos.lerp(destColPos, t_applyAdds.t));
                }

                if (!dense && t_applyMultiplies.t > 0.6 && i > 0 && t_applyAdds.t < 0.7) {
                    drawSymbolBetweenBlocks(args, allVCols[i - 1], allVCols[i], Dim.X, '+', { color: Colors.Black, size: 1.5 });
                }
            }
        }

        if (t_placeVOutput.t > 0 && t_finalizeVOutput.t <= 0) {
            let prepareT = inverseLerp(0, 0.5, t_placeVOutput.t);

            let vOutCol = splitGrid(layout, head0.vOutBlock, Dim.X, attnExampleIdx + 0.5, prepareT * 2.0)!;
            vOutCol.access!.disable = true;
            vOutCol.opacity = lerp(1.0, 0.0, prepareT);

            for (let col of findSubBlocks(head0.vBlock, Dim.X, null, attnExampleIdx)) {
                col.opacity = t_placeVOutput.t;
            }

            let vOutColDupe = duplicateGrid(layout, vOutCol);
            vOutColDupe.access!.disable = false;
            vOutColDupe.opacity = 1.0;

            let colInitialPos = topLeftPos;
            let colFinalPos = getBlkDimensions(vOutCol).tl;

            setBlkPosition(vOutColDupe, colInitialPos.lerp(colFinalPos, t_placeVOutput.t));
        }

        if (t_finalizeVOutput.t > 0) {
            let splitAmt = lerp(1.0, 0.0, t_finalizeVOutput.t) * 2.0;
            let vOutCol = splitGrid(layout, head0.vOutBlock, Dim.X, attnExampleIdx + 0.5, splitAmt)!;
            vOutCol.access!.disable = false;
        }
        if (at && t_placeVOutput.t >= 1) drawIdxTag(state, head0.vOutBlock, qi, 0, `o[0]=${fmtA(at.o.sum)}`, TraceColors.out);
    }

    frameTo(state, t_processRemainZoom, head0.cubes, 275.6, -4.4, { fill: 1.1 });

    if (t_processRemain.t > 0) {
        for (let blk of [head0.attnMtx, head0.attnMtxSm, head0.attnMtxAgg1, head0.attnMtxAgg2, head0.vOutBlock]) {
            blk.access!.disable = true;
        }

        let processStart = startProcessBefore(state, head0.attnMtx);
        processUpTo(state, t_processRemain, head0.vOutBlock, processStart);
    }


}

interface IProcessDimOpts {
    pinIdx?: Vec3;
    clamp?: boolean;
    maxIdx?: number;
    hidePopup?: boolean;
}

function processDim(state: IProgramState, block: IBlkDef, dim: Dim, destIdx: number, t: number, options: IProcessDimOpts = {}) {
    let { layout } = state;
    let { pinIdx, clamp: keep, maxIdx, hidePopup } = options;
    let otherDim = dim === Dim.X ? Dim.Y : Dim.X;

    let { cx: cxOther } = dimProps(block, otherDim);

    pinIdx ||= new Vec3(0, 0, 0);

    let rowCol = splitGrid(layout, block, dim, destIdx, 0);

    if (!rowCol) {
        return;
    }

    let maxPos = maxIdx ?? cxOther; // for attn matrix, reduce to the row number
    let cellPos = t * maxPos;

    if (keep) {
        cellPos = clamp(cellPos, 0, maxPos - 1);
    }

    let cellIdx = Math.floor(cellPos);

    if (cellIdx >= maxPos) {
        return;
    }

    splitGrid(layout, rowCol, otherDim, cellIdx + 0.5, 0);

    let destIdxVec = new Vec3(0, 0, 0);
    destIdxVec.setAt(dim, destIdx);
    destIdxVec.setAt(otherDim, cellIdx);

    if (rowCol && !hidePopup) {
        drawDependences(state, block, destIdxVec);
        drawDataFlow(state, block, destIdxVec, pinIdx);
    }

    for (let blk of findSubBlocks(rowCol, otherDim, null, cellIdx)) {
        blk.access!.disable = false;
    }
}

function focusSelfAttentionHeadTimers(args: IWalkthroughArgs, duration: number) {
    let afterTime = args.tools.afterTime;

    let totalTime = 1.5 * 2;
    let timeScale = duration / totalTime;

    let t0_dissolveHeads = afterTime(null, 1 * timeScale, 0.5 * timeScale);
    let t2_alignqkv = afterTime(null, 1.0 * timeScale, 0.5 * timeScale);

    return { t0_dissolveHeads, t2_alignqkv };
}

function focusSelfAttentionHead(args: IWalkthroughArgs, timers: ReturnType<typeof focusSelfAttentionHeadTimers>) {
    let { layout } = args;
    let { t0_dissolveHeads, t2_alignqkv } = timers;

    let targetHeadIdx = 0;
    let targetHead = layout.blocks[0].heads[targetHeadIdx];

    let block = layout.blocks[0];
    {
        for (let headIdx = 0; headIdx < block.heads.length; headIdx++) {
            if (headIdx == targetHeadIdx) {
                continue;
            }
            for (let cube of block.heads[headIdx].cubes) {
                cube.opacity = lerpSmoothstep(1.0, 0.0, t0_dissolveHeads.t);
            }
        }
    }

    {
        let headZ = targetHead.attnMtx.z;
        let targetHeadZ = block.ln1.lnResid.z;
        let deltaZ = lerpSmoothstep(0, (targetHeadZ - headZ), t2_alignqkv.t);
        for (let cube of targetHead.cubes) {
            cube.z += deltaZ;
        }
    }

    {
        let qkv = [
            [targetHead.qBlock, targetHead.qWeightBlock, targetHead.qBiasBlock],
            [targetHead.kBlock, targetHead.kWeightBlock, targetHead.kBiasBlock],
            [targetHead.vBlock, targetHead.vWeightBlock, targetHead.vBiasBlock],
        ];
        let targetZ = block.ln1.lnResid.z;
        let strideY = targetHead.qBlock.dy + layout.margin;
        let baseY = targetHead.qBlock.y;
        let qkvYPos = [-strideY * 2, -strideY, 0];

        for (let i = 0; i < 3; i++) {
            let y = lerpSmoothstep(qkv[i][0].y, baseY + qkvYPos[i], t2_alignqkv.t);
            let z = lerpSmoothstep(qkv[i][0].z, targetZ, t2_alignqkv.t);
            for (let cube of qkv[i]) {
                cube.y = y;
                cube.z = z;
            }
        }

        let blockMidY = (blk: IBlkDef) => blk.y + blk.dy / 2;

        let resid0Idx = layout.cubes.indexOf(block.ln1.lnResid);
        let yDelta = lerpSmoothstep(0, blockMidY(block.ln1.lnResid) - blockMidY(targetHead.kBlock), t2_alignqkv.t);

        for (let i = 0; i < resid0Idx; i++) {
            let targetOpacity = 0.2;
            layout.cubes[i].opacity = lerpSmoothstep(1.0, targetOpacity, t2_alignqkv.t);
        }

        let afterAttn = false;
        for (let i = resid0Idx + 1; i < layout.cubes.length; i++) {
            let cube = layout.cubes[i];
            cube.y += yDelta;
            if (afterAttn) {
                cube.opacity = Math.min(lerpSmoothstep(1.0, 0.2, t2_alignqkv.t), cube.opacity ?? 1.0);
            }
            afterAttn = afterAttn || cube === targetHead.vOutBlock;
        }
    }

}

function drawSymbolBetweenBlocks(args: IWalkthroughArgs, block1: IBlkDef, block2: IBlkDef, dim: Dim, symbol: string, opts: { color: Vec4, size: number }) {
    let { color, size } = opts;

    let block1Dim = getBlkDimensions(block1);
    let block2Dim = getBlkDimensions(block2);

    let midPt: Vec3;

    if (dim === Dim.X) {
        midPt = new Vec3(
            lerp(block1Dim.br.x, block2Dim.tl.x, 0.5),
            (block1Dim.tl.y + block1Dim.br.y + block2Dim.tl.y + block2Dim.br.y) * 0.25,
            block1Dim.tl.z + args.layout.cell / 2)
    } else {
        midPt = new Vec3(
            (block1Dim.tl.x + block1Dim.br.x + block2Dim.tl.x + block2Dim.br.x) * 0.25,
            lerp(block1Dim.br.y, block2Dim.tl.y, 0.5),
            block1Dim.tl.z + args.layout.cell / 2);
    }

    let mtx = Mat4f.fromTranslation(midPt);
    let fontOpts: IFontOpts = { color, size, mtx };
    let w = measureText(args.state.render.modelFontBuf, symbol, fontOpts);
    drawText(args.state.render.modelFontBuf, symbol, -w/2, -fontOpts.size/2, fontOpts);
}
