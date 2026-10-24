import { IWalkthrough, Phase } from "./Walkthrough";
import { Colors, commentary, DimStyle, dimStyleColor, embed, ITimeInfo, IWalkthroughArgs, moveCameraTo, phaseTools, setInitialCamera, frameTo, initialFrame } from "./WalkthroughTools";
import s from './Walkthrough.module.scss';
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { clamp, makeArray } from "@/src/utils/data";
import React, { useState } from "react";
import { useProgramState } from "../Sidebar";
import { dimProps, findSubBlocks, splitGrid } from "../Annotations";
import { useGlobalDrag } from "@/src/utils/pointer";
import { BlkSpecial, IBlkDef } from "../GptModelLayout";
import { IProgramState } from "../Program";
import { lerp } from "@/src/utils/math";
import { drawDependences } from "../Interaction";
import { drawDataFlow } from "../components/DataFlow";

/*
We're mostly on the right track here I think.

Main things that could do with improvement:

 - Make the camera movement more robust. It should basically be on rails when moving the slider.
 - We store the camera pos at the start of a given movement, and use that as a lerp, and also
   have a default so we can do a reversal

 - Have a highlight-region animation in both the text and the model, to bring attention to a specific
   point. Probably a rectangle with a rotating border. Could also include a shaded background on the text

 - Scroll the text to the next region to read

 - Have more pronounced (& consistent) delays between: camera movement/scroll -> higlight -> action
   - Can probably combine the camera movement and the highlight into a single action, but still have
     a delay between the highlight and the action

 - Generally lean on the new display features more. Might need to add a few more (e.g. animate the dot product)
 - Also probably need to do TeX style layout for maths in the html view :( Could probably reuse the code from the
   3d layout? Just need to get the glyphs in via css/ttf, and then maybe use canvas for positioning? Hmm should
   be able to use divs with abs positioning.

 - For stopping/starting, use t + dt to figure out when it was crossed, rather than storing the lastPauseTime etc

*/

interface IIntroState {

}

function getIntroState(walkthrough: IWalkthrough): IIntroState {
    return walkthrough.phaseData.get(Phase.Intro_Intro) as IIntroState;
}

export function walkthroughIntro(args: IWalkthroughArgs) {
    let { breakAfter, afterTime, c_str } = phaseTools(args.state);
    let { state, layout, walkthrough: wt } = args;
    let nTok = layout.shape.T;

    if (wt.phase !== Phase.Intro_Intro) {
        return;
    }

    initialFrame(state, args.layout.cubes, 296, 16, { fill: 1.0 });

    let cfg = layout.shape.full;
    let c0 = commentary(wt, null, 0)`欢迎来到 _k005b_ VLA 网络的可视化讲解。k005b 把相机画面、语言指令和机器人状态放进同一个 Qwen3-VL-2B 主干，再分出输出：knack 分类头（说"现在该做什么"）、done 头，以及动作专家（生成 16×23 的动作块）。

3D 场景里显示的全部是 _真实 k005b checkpoint（step ${cfg?.step ?? '…'}）_ 的权重，以及对一个真实训练样本（"${cfg?.sample.goal_text ?? '…'}"）做前向时用 hook 抓下来的真实激活。真实尺寸太大无法逐格画出，所以每个张量只画一个真实窗口：token ${cfg ? `${cfg.window.t0}–${cfg.window.t0 + cfg.window.Wt - 1}` : '…'}（以 h* 边界 token 结尾）、通道 0–${(cfg?.window.Wc ?? 64) - 1}（共 ${cfg?.n_embd ?? 2048}）、head 维 0–${(cfg?.window.Wd ?? 32) - 1}（共 ${cfg?.head_dim ?? 128}）、FF 0–${(cfg?.window.Wf ?? 64) - 1}（共 ${cfg?.intermediate ?? 6144}）。鼠标悬停任意格子会显示真实张量名、完整形状、真实下标、数值和整张量统计。整体流程：
1. 输入嵌入：${cfg?.n_visual_tokens ?? '…'} 个视觉 token + 语言 token + ${cfg?.n_state_tokens ?? 16} 个状态 token，共 ${cfg?.T_prefix ?? '…'} 个前缀 token。
2. Qwen3-VL 主干：${cfg?.n_layer ?? 28} 层 Transformer（16 个 query 头、8 个 KV 头的 GQA，head_dim 128）。
3. 展开一层：RMSNorm → 联合注意力（前缀 + 动作后缀）→ o_proj → 残差 → RMSNorm → SwiGLU MLP → 残差。
4. h* 边界读出：ln_f 在 p=${cfg?.p_star ?? '…'} 处的输出。
5. knack 头 + done 头；动作专家与主干逐层联合注意力，输出动作块。

窗口里的真实输入 token：${embed(ExampleInputOutput)}`;

    if (c0.t > 0) {
        for (let cube of layout.cubes) {
            if (cube.t === 'i' && cube.access) {
                cube.access.disable = true;
            }
        }
        state.display.tokenIdxModelOpacity = makeArray(nTok, 0);
    }

    let t4 = afterTime(null, 1.5, 0.4);

    frameTo(args.state, t4, [layout.idxObj, layout.tokEmbedObj, layout.residual0], 281.5, 12.5, { fill: 1.15 });
    let t6 = afterTime(null, 1.0, 0.2);

    if (t4.active) {
        state.display.topOutputOpacity = 0.2;
    }

    if (t6.active && t6.t < 1.0) {
        let mixes = makeArray(nTok, 0);
        for (let i = 0; i < nTok; i++) {
            // want to smoothly flash each token in turn (t6.t goes from 0-1, and each token should flash at 0.2, 0.4, 0.6, 0.8, 1.0 etc)
            let highT = (i + 1.5) / (nTok + 2);
            mixes[i] = 1.0 - clamp(Math.abs(t6.t - highT) * nTok, 0, 1);
        }
        state.display.tokenColors = { mixes, color2: dimStyleColor(DimStyle.Token) }; //  new Vec4(0.8, 0.2, 0.8) };
    }

    breakAfter();

    let tokenStr = c_str('_token_', 0, DimStyle.Token);
    let tokenIdxStr = c_str('_token index_', 0, DimStyle.TokenIdx);

    commentary(wt, t6)`其中每一项叫做一个 ${tokenStr}。k005b 使用 Qwen 的 BPE 词表，共 ${cfg?.vocab_size ?? 151936} 个 token。

    每个 token 对应一个编号，即 ${tokenIdxStr}。窗口内的真实编号如下，它们被送入模型：${embed(ExampleTokenValues)}\n`;
    breakAfter();

    let t7 = afterTime(null, 1.5, 0.5);

    if (t7.active) {
        let opacity = makeArray(nTok, 0);
        for (let i = 0; i < nTok; i++) {
            let highT = (i + 1.5) / (nTok + 2);
            opacity[i] = clamp((t7.t - highT) * nTok, 0, 1);
        }
        state.display.tokenIdxColors = { mixes: opacity, color2: dimStyleColor(DimStyle.TokenIdx) };

        let idxPos = t7.t * nTok;

        if (t7.t < 1.0) {
            splitGrid(layout, layout.idxObj, Dim.X, idxPos, clamp(nTok - idxPos, 0, 1));
            for (let blk of findSubBlocks(layout.idxObj, Dim.X, null, Math.min(nTok - 1, Math.floor(idxPos)))) {
                if (blk.access) {
                    blk.access.disable = false;
                }
            }
        } else {
            if (layout.idxObj.access) {
                layout.idxObj.access.disable = false;
            }
        }
    }

    breakAfter();

    let c5 = commentary(wt)`在 3D 视图中，每个绿色格子是正在处理的数值，每个蓝色格子是权重。${embed(GreenBlueCells)}
    序列中的每个编号首先查嵌入表变成一个 ${cfg?.n_embd ?? 2048} 维向量，这叫 _嵌入_（图中显示通道 0–${(cfg?.window.Wc ?? 64) - 1}）。`;
    breakAfter(c5);

    {
        let t_camMove = afterTime(null, 1.0, 0.5);
        let t_makeVecs = afterTime(null, 2.0, 0.5);

        frameTo(state, t_camMove, [layout.idxObj, layout.residual0], 286, 14.5, { fill: 1.25 });

        if (t_makeVecs.active) {
            let idxPos = t_makeVecs.t * nTok;
            let splitWidth = clamp(nTok - idxPos, 0, 2);
            let splitIdx = Math.min(nTok - 1, Math.floor(idxPos));
            if (t_makeVecs.t < 1.0) {
                splitGrid(layout, layout.idxObj, Dim.X, idxPos, splitWidth);
                for (let blk of findSubBlocks(layout.idxObj, Dim.X, null, splitIdx)) {
                    if (blk.access) {
                        blk.access.disable = false;
                    }
                }

                splitGrid(layout, layout.residual0, Dim.X, idxPos, splitWidth);
                for (let blk of findSubBlocks(layout.residual0, Dim.X, null, splitIdx)) {
                    if (blk.access) {
                        blk.access.disable = false;
                    }
                }
            } else {
                if (layout.residual0.access) {
                    layout.residual0.access.disable = false;
                }
            }
        }
    }

    breakAfter();
    commentary(wt)`嵌入向量随后依次穿过全部 ${cfg?.n_layer ?? 28} 层 Transformer（3D 中每层都是真实权重与激活），最终到达底部的读出位置 h*。`;
    breakAfter();

    {

        let t_firstResid = afterTime(null, 1.0, 0.5);
        frameTo(state, t_firstResid, [layout.residual0, ...layout.blocks[0].ln1.cubes, ...layout.blocks[0].heads.flatMap(h => h.cubes), layout.blocks[0].attnResidual], 292.3, 26.8, { fill: 1.0 });
        let t_firstResidWalk = afterTime(null, 5.0, 0.5);

        let processState = processUpTo(state, t_firstResidWalk, layout.blocks[0].attnResidual);

        let t_firstTransformer = afterTime(null, 1.0, 0.5);
        frameTo(state, t_firstTransformer, layout.blocks[0].cubes, 299.4, 14.7, { fill: 1.0 });
        let t_firstTransformerWalk = afterTime(null, 3.5, 0.5);
        processUpTo(state, t_firstTransformerWalk, layout.blocks[0].mlpResidual, processState);

        if (t_firstTransformer.active) {
            layout.blocks[0].transformerLabel.visible = t_firstTransformer.t;
        }

        let t_fullFrame = afterTime(null, 1.0, 0.5);
        frameTo(state, t_fullFrame, layout.cubes, 298.5, 23.4, { fill: 1.0 });
        let t_fullFrameWalk = afterTime(null, 5.0, 0.5);
        processUpTo(state, t_fullFrameWalk, layout.ln_f.lnResid, processState);


        // let t_endFrame = afterTime(null, 1.0, 0.5);
        // moveCameraTo(state, t_endFrame, new Vec3(-18.3, 0, -1576), new Vec3(280.6, 9.7, 1.9));
        // let t_endFrameWalk = afterTime(null, 2.0, 0.5);
        // processUpTo(state, t_endFrameWalk, layout.ln_f.lnResid, processState);

        let t_output = afterTime(null, 1.0, 0.5);
        frameTo(state, t_output, [args.layout.ln_f.lnResid, args.layout.hStar, args.layout.lmHeadWeight, args.layout.logits, args.layout.logitsSoftmax, args.layout.doneWeight, args.layout.doneProbs], 271.3, 6.4, { fill: 1.2 });
        // moveCameraTo(state, t_output, new Vec3(-53.9, 0, -1654.1), new Vec3(270.9, 6.2, 1.1));
        let t_outputWalk = afterTime(null, 2.0, 0.5);
        processUpTo(state, t_outputWalk, layout.logitsSoftmax, processState);

        let t_outputToks = afterTime(null, 1.0, 0.5);

        if (t_firstResid.active) {
            let arr = makeArray(nTok, 0);

            if (t_outputToks.active) {
                for (let i = 0; i < nTok; i++) {
                    let highT = (i + 1.5) / (nTok + 2);
                    arr[i] = clamp((t_outputToks.t - highT) * nTok, 0, 1);
                }
            }

            // state.display.lines.push(arr.map(a => a.toFixed(2).padStart(4)).join(', '));
            state.display.tokenOutputColors = { color1: new Vec4(0,0,0,0), color2: Vec4.fromHexColor('#000', 1), mixes: arr };
        }
    }

    commentary(wt)`输出是什么？在答案边界 h* 处读出：knack 头对 ${cfg?.knack_names.length ?? 17} 个 knack 做 softmax（本样本预测 ${cfg?.pred_knack ?? '…'}，目标 ${cfg?.sample.target_knack ?? '…'}），done 头输出 ${cfg?.pred_done ?? '…'}。`

    commentary(wt)`另一路是动作专家（d=${cfg?.expert_hidden ?? 768}）：它与主干逐层联合注意力，输出速度 v，动作块 = x1 − v（16×23）。模型只负责汇报，切换子任务由 Harness 决定。接下来逐步展开每个环节。`;

    breakAfter();
}

interface IProcessInfo {
    lastBlockIdx: number;
}

export function startProcessBefore(state: IProgramState, block: IBlkDef): IProcessInfo {
    let activeBlocks = state.layout.cubes.filter(a => a.t !== 'w');

    return {
        lastBlockIdx: activeBlocks.indexOf(block) - 1,
    };
}

export function processUpTo(state: IProgramState, timer: ITimeInfo, block: IBlkDef, prevInfo?: IProcessInfo): IProcessInfo {

    let activeBlocks = state.layout.cubes.filter(a => a.t !== 'w');

    let firstIdx = prevInfo ? prevInfo.lastBlockIdx + 1 : 0;
    let lastIdx = activeBlocks.indexOf(block);

    // we weight the time on each block by the number of cells in the block, times by how many dependent cells it has
    // although to make this less extreme, we take a fractional power of this
    let cellCounts = activeBlocks
        .filter((_, i) => i >= firstIdx && i <= lastIdx)
        .map(a => (a.cx * a.cy) * Math.pow(a.deps?.dotLen ?? 1, 0.25));
    let totalCells = cellCounts.reduce((a, b) => a + b, 0);

    let accCell = 0;
    let currIdx = firstIdx;
    let subPos = 0; // 0 -> 1
    for (let i = firstIdx; i <= lastIdx; i++) {
        let blockFract = cellCounts[i - firstIdx] / totalCells;
        accCell += blockFract;
        if (timer.t < accCell) {
            currIdx = i;
            subPos = (timer.t - (accCell - blockFract)) / blockFract;
            break;
        }
    }

    let blk = activeBlocks[currIdx];

    // default, but switched for attention matrix
    let dim0 = Dim.X;
    let dim1 = Dim.Y;
    if (blk.transpose) {
        dim0 = Dim.Y;
        dim1 = Dim.X;
    }

    let { cx } = dimProps(blk, dim0);
    let { cx: cy } = dimProps(blk, dim1);

    let horizPos = lerp(0, cx, subPos);
    let horizIdx = Math.floor(horizPos);

    let vertPos = lerp(0, cy, horizPos - horizIdx);
    let vertIdx = Math.floor(vertPos);

    let blockPos = new Vec3().withSetAt(dim0, horizIdx).withSetAt(dim1, vertIdx); // new Vec3(horizIdx, vertIdx, 0);
    let pinPos = new Vec3(Math.floor(cx / 2), 0, 0);

    if (blk === state.layout.residual0) {
        pinPos = new Vec3(cx * 2, -2, 0);
    }

    if (timer.t >= 1.0) {
        currIdx = lastIdx;
    }

    for (let i = firstIdx; i < currIdx; i++) {
        let blk = activeBlocks[i];
        if (blk.access) {
            blk.access.disable = false;
        }
    }

    if (timer.active && timer.t < 1.0) {
        drawDependences(state, blk, blockPos);
        drawDataFlow(state, blk, blockPos, pinPos);

        for (let label of state.layout.labels) {
            for (let c of label.cubes) {
                if (c === blk) {
                    label.visible = 1.0;
                }
            }
        }

        blk.highlight = 0.3;

        let column = splitGrid(state.layout, blk, dim0, horizPos, 0);
        if (column) {
            for (let col of findSubBlocks(blk, dim0, null, horizIdx)) {
                if (col.access) {
                    col.access.disable = false;
                    col.highlight = 0.1;
                }
            }
            column.highlight = 0.4;

            let curr = splitGrid(state.layout, column, dim1, vertPos, 0);
            for (let blk of findSubBlocks(column, dim1, null, vertIdx)) {
                if (blk.access) {
                    blk.access.disable = false;
                }
            }
            if (curr) {
                curr.highlight = 0.7;
            }
        }


    } else if (timer.active) {
        let blk = activeBlocks[lastIdx];
        if (blk.access) {
            blk.access.disable = false;
        }
    }

    let info = prevInfo ?? { lastBlockIdx: currIdx };
    info.lastBlockIdx = lastIdx;
    return info;
}

function tokenDisplay(tok: string) {
    return tok.replace(/\n/g, '⏎');
}

const ExampleInputOutput: React.FC = () => {
    let state = useProgramState();
    let cols = state.display.tokenColors;
    let toks = state.jsGptModel?.config.window_tokens ?? [];

    return <div className={s.tableWrap}>
        <div>{toks.map((c, i) => {
            let baseColor = dimStyleColor(DimStyle.Token);
            if (cols && cols.mixes[i]) {
                baseColor = Vec4.lerp(baseColor, cols.color2, cols.mixes[i]);
            }
            return <span key={i} style={{ color: baseColor.toHexColor(), whiteSpace: 'pre' }}>{tokenDisplay(c)}|</span>;
        })}</div>
    </div>;
};

const ExampleTokenValues: React.FC = () => {
    let state = useProgramState();
    let cols = state.display.tokenIdxColors;
    let ids = state.jsGptModel?.config.window_token_ids ?? [];

    return <div className={s.tableWrap}>
        <div>{ids.map((id, i) => {
            let baseColor = dimStyleColor(DimStyle.TokenIdx);
            if (cols && cols.mixes[i]) {
                baseColor = Vec4.lerp(baseColor, cols.color2, cols.mixes[i]);
            }
            return <span key={i} style={{ color: baseColor.toHexColor() }}>{id} </span>;
        })}</div>
    </div>;
};

const GreenBlueCells: React.FC = () => {

    // real values: embed_out[p*, c0..2] (activation) and embed[v0, c0..2] (weight), scaled by each window's colour scale
    let model = useProgramState().jsGptModel;
    let realGreen = model ? Array.from(model.embedOut.localBuffer.slice((model.embedOut.height - 1) * model.embedOut.width, (model.embedOut.height - 1) * model.embedOut.width + 3)).map(v => v * model!.embedOut.meta.absScale) : [];
    let realBlue = model ? Array.from(model.embedWeight.localBuffer.slice(0, 3)).map(v => v * model!.embedWeight.meta.absScale) : [];
    let [blueNums, setBlueNums] = useState<number[] | null>(null);
    let [greenNums, setGreenNums] = useState<number[] | null>(null);
    if (!model) {
        return null;
    }

    let blueColor = new Vec4(0.3, 0.3, 1.0);
    let greenColor = new Vec4(0.3, 0.9, 0.3);

    return <div className={s.tableWrap}>
        <div className={s.cellInfoCols}>
            <div className={s.cellInfoCol}>
                <Cell nums={greenNums ?? realGreen} color={greenColor} mul={0.5} />
                <Graph nums={greenNums ?? realGreen} color={greenColor} setNums={setGreenNums} />
                <div className={s.cellInfoText}>处理中的数值</div>
            </div>
            <div className={s.cellInfoCol}>
                <Cell nums={blueNums ?? realBlue} color={blueColor} mul={1} />
                <Graph nums={blueNums ?? realBlue} color={blueColor} setNums={setBlueNums} />
                <div className={s.cellInfoText}>权重</div>
            </div>
        </div>
    </div>
};

const Cell: React.FC<{ nums: number[], color: Vec4, mul?: number }> = ({ color, nums, mul }) => {

    let grey = new Vec4(0.5, 0.5, 0.5, 1.0);
    let cellLight = Vec4.lerp(color, grey, 0.9);
    let cellDark = cellLight.mul(0.98);
    cellDark.w = 1.0;

    let cellColor = (n: number) => {
        let weight = clamp(Math.abs(n), 0.0, 1.0);

        let negColor = new Vec4(0.0, 0.0, 0.0);
        let posColor = color;
        let zeroColor = new Vec4(0.5, 0.5, 0.5);
        if (n < 0.0) {
            return Vec4.lerp(zeroColor, negColor, weight).toHexColor();
        } else {
            return Vec4.lerp(zeroColor, posColor, weight).toHexColor();
        }
    };

    return <div className={s.cellArrayHoriz}>
        {nums.map((n, i) => {
            return <div className={s.cellRect} key={i} style={{ backgroundColor: (i % 2 === 0 ? cellLight : cellDark).toHexColor() }}>
                <div className={s.cellCircle} style={{ backgroundColor: cellColor(n * (mul ?? 1.0)) }} />
            </div>;
        })}
    </div>
};

const Graph: React.FC<{
    nums: number[],
    color: Vec4,
    max?: number,
    setNums?: (nums: number[]) => void,
}> = ({ color, nums, max, setNums }) => {
    let [graphEl, setGraphEl] = useState<HTMLDivElement | null>(null);

    let ticks = [-1, 0, 1];
    let cellW = 30;
    let dispColor = color.mul(1.0);
    dispColor.w = 0.5;

    interface IDragInitial {
        index: number;
        nums: number[];
    }

    let [, setDragStart] = useGlobalDrag<IDragInitial>(function handleMove(ev, ds) {
        let dy = ev.clientY - ds.clientY;
        let h = graphEl!.clientHeight * 0.5;
        let nums = [...ds.data.nums];
        nums[ds.data.index] = clamp(nums[ds.data.index] - dy / h, -1.0, 1.0);
        setNums?.(nums);
        ev.preventDefault();
        ev.stopImmediatePropagation();
    })

    return <div className={s.graph} style={{ width: cellW * nums.length }} ref={setGraphEl}>

        <div className={s.axisLeft} />

        <div className={s.axisZero} />

        {nums.map((n, i) => {
            let nScaled = n / (max ?? 1.0);

            return <div className={s.graphCol} key={i}>
                <div className={s.graphBar} style={{
                    backgroundColor: dispColor.toHexColor(),
                    top: nScaled < 0 ? '50%' : `${(0.5 - nScaled/2) * 100}%`,
                    height: `${(Math.abs(nScaled)/2) * 100}%`,
                }} />
                <div
                    className={s.graphBarHit}
                    onMouseDown={ev => {
                        setDragStart(ev, { index: i, nums });
                        ev.stopPropagation();
                        ev.preventDefault();
                    }}
                    style={{
                        top: `${(0.5 - nScaled/2) * 100}%`
                    }} />
                <div className={s.graphBarLabel} style={{
                     bottom: nScaled < 0 ? '50%' : undefined,
                        top: nScaled > 0 ? '50%' : undefined,
                }}>
                    {n.toFixed(1)}
                </div>
            </div>;
        })}

    </div>;

};
