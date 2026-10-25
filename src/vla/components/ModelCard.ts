import { cameraToMatrixView } from "../Camera";
import { cellPosition, IVlaModelLayout } from "../VlaModelLayout";
import { IProgramState, vlaLayout } from "../Program";
import { drawText, IFontOpts, measureText, measureTextWidth, writeTextToBuffer } from "../render/fontRender";
import { addLine2 as drawLine, drawLineSegs, ILineOpts, makeLineOpts } from "../render/lineRender";
import { IRenderState } from "../render/modelRender";
import { addQuad } from "../render/triRender";
import { lerp } from "@/src/utils/math";
import { Mat4f } from "@/src/utils/matrix";
import { Dim, Vec3, Vec4 } from "@/src/utils/vector";
import { lineHeight } from "./TextLayout";
import { IColorMix } from "../Annotations";
import { clamp } from "@/src/utils/data";

export function drawModelCard(state: IProgramState, layout: IVlaModelLayout, title: string, offset: Vec3) {
    let { render } = state;
    let { camPos } = cameraToMatrixView(state.camera);
    let dist = camPos.dist(new Vec3(0, 0, -30)); //.add(offset));

    let scale = clamp(dist / 500.0, 1.0, 800.0);

    let pinY = -60;
    let mtx = Mat4f.fromScaleTranslation(new Vec3(scale, scale, scale), new Vec3(0, pinY, 0).add(offset))
        .mul(Mat4f.fromTranslation(new Vec3(0, -pinY, 0)));

    let thick = 1.0 / 10.0 * scale;
    let borderColor = Vec4.fromHexColor("#555599", 0.8);
    let backgroundColor = Vec4.fromHexColor("#93c5fd", 0.3);
    let titleColor = Vec4.fromHexColor("#000000", 1.0);
    let n = new Vec3(0, 0, 1);

    let lineOpts: ILineOpts = { color: borderColor, mtx, thick, n };

    let tl = new Vec3(-75, -112, 0);
    let br = new Vec3( 75, -70, 0);
    drawLineRect(render, tl, br, lineOpts);

    addQuad(render.triRender, new Vec3(tl.x, tl.y, -0.1), new Vec3(br.x, br.y, -0.1), backgroundColor, mtx);

    let midX = (tl.x + br.x) / 2;
    let paramLineHeight = 1.3;
    let paramFontScale = 3;

    let titleFontScale = 11;
    let titleW = measureTextWidth(render.modelFontBuf, title, titleFontScale);
    let titleHeight = titleFontScale * paramLineHeight;
    writeTextToBuffer(render.modelFontBuf, title, titleColor, midX - titleW / 2, tl.y + 2, titleFontScale, mtx);

    let cfg = layout.shape.full;
    let hasHeads = !!layout.model && !layout.model.knackProbs.meta.absent;
    let paramOff = tl.y + titleHeight + 3;

    let nParamsText = `n_params = `;
    let weightCountText = cfg ? numberToCommaSep(layout.weightCount) : `loading real ${state.page.facts?.modelName ?? ''} checkpoint…`;
    let weightSize = 6;
    let weightTitleW = measureTextWidth(render.modelFontBuf, nParamsText, paramFontScale);
    let weightCountW = measureTextWidth(render.modelFontBuf, weightCountText, weightSize);
    let weightX = midX - (weightCountW + weightTitleW) / 2;
    writeTextToBuffer(render.modelFontBuf, nParamsText, titleColor, weightX, paramOff - paramFontScale / 2 + 1, paramFontScale, mtx);
    writeTextToBuffer(render.modelFontBuf, weightCountText, titleColor, weightX + weightTitleW, paramOff - weightSize / 2, weightSize, mtx);
    paramOff += weightSize * paramLineHeight - 1;

    if (cfg) {
        let w = cfg.window;
        let mlp = cfg.variant?.trunk_mlp === 'geglu' ? 'GeGLU' : 'SwiGLU';
        let lines = [
            cfg.step != null ? `${cfg.name} · step ${cfg.step}` : `${cfg.name} · ${cfg.train_config ?? ""}`,
            `trunk: ${cfg.n_layer} layers · d ${cfg.n_embd} · ${cfg.n_head} q / ${cfg.n_kv_head} kv heads × ${cfg.head_dim} · ${mlp} ${cfg.intermediate} · vocab ${cfg.vocab_size}`,
            `expert: d ${cfg.expert_hidden} · ${cfg.expert_heads} q / ${cfg.expert_kv_heads} kv heads · FF ${cfg.expert_intermediate} · ${cfg.n_suffix} suffix tokens → action ${cfg.action_horizon}×${cfg.action_dim}`,
            `sample: row ${cfg.sample.row} · "${cfg.sample.goal_text}" · prefix ${cfg.T_prefix} tok (${cfg.n_visual_tokens} visual)` + (hasHeads ? ` · h* at p=${cfg.p_star}` : ''),
            ...(hasHeads ? [`knack = ${cfg.pred_knack} (target ${cfg.sample.target_knack ?? '?'}) · done = ${cfg.pred_done} (target ${cfg.sample.target_done ?? '?'})`] : []),
            `3D cells = real windows: tokens ${w.t0}–${w.t0 + w.Wt - 1}, channels 0–${w.Wc - 1}, head dims 0–${w.Wd - 1}, FF 0–${w.Wf - 1}, embed rows ${w.v0}–${w.v0 + w.Wv - 1}`,
        ];
        for (let line of lines) {
            let lw = measureTextWidth(render.modelFontBuf, line, paramFontScale);
            writeTextToBuffer(render.modelFontBuf, line, titleColor, midX - lw / 2, paramOff, paramFontScale, mtx);
            paramOff += paramFontScale * paramLineHeight;
        }
    }

    if (hasHeads) renderOutputAtBottom(state);

    renderInputAtTop(state, hasHeads);
}

function tokenDisplayString(tok: string) {
    return tok.replace(/\n/g, '⏎').replace(/^ /, '·');
}

interface IInputBoxOpts {
    tokMixes?: IColorMix | null;
    idxMixes?: IColorMix | null;
}

function renderInputBoxes(state: IProgramState, layout: IVlaModelLayout, tl: Vec3, br: Vec3, cellW: number, fontSize: number, lineOpts: ILineOpts, opts?: IInputBoxOpts) {
    let render = state.render;
    let { T } = layout.shape;
    let inCellH = br.y - tl.y;

    let tokTextOpts: IFontOpts = { color: Vec4.fromHexColor("#000000", 1.0), mtx: lineOpts.mtx, size: fontSize };
    let idxTextOpts: IFontOpts = { color: Vec4.fromHexColor("#666666", 1.0), mtx: lineOpts.mtx, size: fontSize * 0.6 };


    drawLineRect(render, tl, br, lineOpts);

    let tokens = layout.model?.inputTokens.localBuffer;

    for (let i = 0; i < T; i++) {

        if (i > 0) {
            let lineX = tl.x + i * cellW;
            drawLine(render.lineRender, new Vec3(lineX, tl.y, 0), new Vec3(lineX, br.y, 0), lineOpts);
        }

        let tokStrs = layout.model?.config.window_tokens;
        if (tokens && tokStrs && i < tokens.length) {
            let cx = tl.x + (i + 0.5) * cellW;

            let tokOpts = { ...tokTextOpts, color: mixColorValues(opts?.tokMixes ?? null, tokTextOpts.color, i) };
            let tokIdxOpts = { ...idxTextOpts, color: mixColorValues(opts?.idxMixes ?? null, idxTextOpts.color, i) };
            let tokStr = tokenDisplayString(tokStrs[i]);
            let maxW = cellW * 0.95;
            let fullW = measureText(render.modelFontBuf, tokStr, tokTextOpts);
            if (fullW > maxW) {
                tokOpts = { ...tokOpts, size: tokOpts.size * maxW / fullW };
            }
            let tokW = measureText(render.modelFontBuf, tokStr, tokOpts);
            let idxW = measureText(render.modelFontBuf, tokens[i].toString(), idxTextOpts);
            let totalH = tokTextOpts.size + idxTextOpts.size;
            let top = tl.y + (inCellH - totalH) / 2;

            drawText(render.modelFontBuf, tokStr, cx - tokW / 2, top, tokOpts);
            drawText(render.modelFontBuf, tokens[i].toString(),  cx - idxW / 2, top + tokTextOpts.size, tokIdxOpts);
        }

    }
}

interface IOutputBoxOpts {
    opacity?: number;
    boldLast?: boolean;
    tokMixes?: IColorMix | null;
}

function renderOutputBoxes(state: IProgramState, layout: IVlaModelLayout, tl: Vec3, br: Vec3, cellW: number, fontSize: number, lineOpts: ILineOpts, opts?: IOutputBoxOpts) {
    let render = state.render;
    let { T } = layout.shape;
    let outCellH = br.y - tl.y;

    let opacity = opts?.opacity ?? 1.0;
    let boldLast = opts?.boldLast ?? true;

    lineOpts = { ...lineOpts, color: lineOpts.color.mul(opacity ?? 1.0) };
    let tokTextOpts: IFontOpts = { color: Vec4.fromHexColor("#000000", opacity), mtx: lineOpts.mtx, size: fontSize };
    let idxTextOpts: IFontOpts = { color: Vec4.fromHexColor("#666666", opacity), mtx: lineOpts.mtx, size: fontSize * 0.6 };


    drawLineRect(render, tl, br, lineOpts);

    // The trunk has no LM head here: the only output is the knack/done prediction
    // read from h* at the last (boundary) token. Show the real knack softmax there.
    let model = layout.model;

    for (let i = 0; i < T; i++) {
        if (i > 0) {
            let lineX = tl.x + i * cellW;
            drawLine(render.lineRender, new Vec3(lineX, tl.y, 0), new Vec3(lineX, br.y, 0), lineOpts);
        }
    }

    if (model) {
        let probs = model.knackProbs.localBuffer;
        let names = model.config.knack_names;
        let order = Array.from(probs).map((p, j) => ({ p, j })).sort((a, b) => b.p - a.p);
        let i = T - 1;
        let cx = tl.x + (i + 0.5) * cellW;
        let usedSoFar = 0.0;
        for (let { p, j } of order) {
            let partTop = tl.y + usedSoFar * outCellH;
            let partH = p * outCellH;
            let color = mixColorValues(opts?.tokMixes ?? null, tokTextOpts.color, i);
            if (!boldLast) color = color.mul(0.3);
            let tokOpts = { ...tokTextOpts, color, size: tokTextOpts.size * 0.5 };
            let idxOpts = { ...idxTextOpts, color: color.mul(0.6), size: idxTextOpts.size * 0.6 };
            let nameStr = names[j];
            let pStr = p.toFixed(3);
            let nameW = measureText(render.modelFontBuf, nameStr, tokOpts);
            let pW = measureText(render.modelFontBuf, pStr, idxOpts);
            let textH = tokOpts.size + idxOpts.size;
            if (partH > textH) {
                let top = partTop + (partH - textH) / 2;
                drawText(render.modelFontBuf, nameStr, cx - nameW / 2, top, tokOpts);
                drawText(render.modelFontBuf, pStr, cx - pW / 2, top + tokOpts.size, idxOpts);
            }
            usedSoFar += p;
            drawLine(render.lineRender, new Vec3(cx - cellW/2, partTop + partH, 0), new Vec3(cx + cellW/2, partTop + partH, 0), lineOpts);
            if (usedSoFar >= 1.0 - 1e-3) break;
        }
    }
}

function mixColorValues(mixes: IColorMix | null, baseColor: Vec4, idx: number) {
    if (!mixes) {
        return baseColor;
    }
    let mix = mixes.mixes[idx] ?? 0.0;
    return Vec4.lerp(mixes.color1 ?? baseColor, mixes.color2, mix);
}

let _lineRectArr = new Float32Array(3 * 4);
export function drawLineRect(render: IRenderState, tl: Vec3, br: Vec3, opts: ILineOpts) {

    _lineRectArr[0] = tl.x;
    _lineRectArr[1] = tl.y;
    _lineRectArr[2] = 0;
    _lineRectArr[3] = br.x;
    _lineRectArr[4] = tl.y;
    _lineRectArr[5] = 0;
    _lineRectArr[6] = br.x;
    _lineRectArr[7] = br.y;
    _lineRectArr[8] = 0;
    _lineRectArr[9] = tl.x;
    _lineRectArr[10] = br.y;
    _lineRectArr[11] = 0;

    drawLineSegs(render.lineRender, _lineRectArr, makeLineOpts({ ...opts, closed: true }));
}

function numberToCommaSep(a: number) {
    let s = a.toString();
    let out = "";
    for (let i = 0; i < s.length; i++) {
        if (i > 0 && (s.length - i) % 3 == 0) {
            out += ",";
        }
        out += s[i];
    }
    return out;
}

function renderInputAtTop(state: IProgramState, hasHeads: boolean) {
    let layout = vlaLayout(state);
    let render = state.render;

    let inputTokBlk = layout.idxObj;

    let topMid = new Vec3(inputTokBlk.x + inputTokBlk.dx/2, inputTokBlk.y - layout.margin);

    let inCellH = 10;
    let inCellW = 6;

    let nCells = layout.shape.T;
    let tl = new Vec3(topMid.x - inCellW * nCells / 2, topMid.y - inCellH);
    let br = new Vec3(topMid.x + inCellW * nCells / 2, topMid.y);

    let outputOpacity = state.display.topOutputOpacity ?? 1.0;

    let lineOpts = makeLineOpts({ color: Vec4.fromHexColor("#000000", 0.2), mtx: new Mat4f(), thick: 1.5 });
    let titleTextOpts: IFontOpts = { color: Vec4.fromHexColor("#666666", 1.0), mtx: lineOpts.mtx, size: 1.9 };

    renderInputBoxes(state, layout, tl, br, inCellW, 4, lineOpts, { tokMixes: state.display.tokenColors, idxMixes: state.display.tokenIdxColors });

    let inputTitle = layout.model ? `Prompt tokens ${layout.model.config.window.t0}–${layout.model.config.window.t0 + layout.shape.T - 1} (of ${layout.model.config.T_prefix}-token prefix)` : "Prompt tokens";
    drawText(render.modelFontBuf, inputTitle, tl.x, tl.y - lineHeight(titleTextOpts), titleTextOpts);

    if (hasHeads) {
        let outCellH = 12;
        let outBr = new Vec3(br.x, tl.y - 4);
        let outTl = new Vec3(tl.x, outBr.y - outCellH);
        renderOutputBoxes(state, layout, outTl, outBr, inCellW, 4, lineOpts, { opacity: outputOpacity, boldLast: outputOpacity < 1.0, tokMixes: state.display.tokenOutputColors });

        let outputTitle = "Knack prediction (softmax at h*)";
        let outputTextOpts = { ...titleTextOpts, color: titleTextOpts.color.mul(outputOpacity) };
        drawText(render.modelFontBuf, outputTitle, outTl.x, outTl.y - lineHeight(titleTextOpts), outputTextOpts);
    }

    for (let i = 0; i < nCells; i++) {
        let mixes = state.display.tokenIdxColors;

        let lineOptsLocal = { ...lineOpts, color: mixColorValues(mixes, lineOpts.color, i) };

        let tx = tl.x + (i + 0.5) * inCellW;
        let ty = tl.y + layout.cell + inCellH;
        let bx = cellPosition(layout, inputTokBlk, Dim.X, i) + 0.5 * layout.cell;
        let by = inputTokBlk.y - 0.5 * layout.cell;

        let midY1 = lerp(by, ty, 1/6);
        let midY2 = lerp(by, ty, 3/4);

        drawLine(state.render.lineRender, new Vec3(bx, by), new Vec3(bx, midY1), lineOptsLocal);
        drawLine(state.render.lineRender, new Vec3(bx, midY1), new Vec3(tx, midY2), lineOptsLocal);
        drawLine(state.render.lineRender, new Vec3(tx, midY2), new Vec3(tx, ty), lineOptsLocal);

        let arrLen = 0.6;
        let arrowLeft = new Vec3(bx - arrLen, by - arrLen);
        let arrowRight = new Vec3(bx + arrLen, by - arrLen);
        drawLine(state.render.lineRender, arrowLeft, new Vec3(bx, by), lineOptsLocal);
        drawLine(state.render.lineRender, arrowRight, new Vec3(bx, by), lineOptsLocal);
    }
}

function renderOutputAtBottom(state: IProgramState) {
    let layout = vlaLayout(state);

    let softmax = layout.logitsSoftmax;


    let topMid = new Vec3(softmax.x + softmax.dx/2, softmax.y + softmax.dy + layout.margin);

    let outCellH = 10;
    let outCellW = 6;

    let nCells = layout.shape.T;
    let tl = new Vec3(topMid.x - outCellW * nCells / 2, topMid.y);
    let br = new Vec3(topMid.x + outCellW * nCells / 2, topMid.y + outCellH);

    let lineOpts = makeLineOpts({ color: Vec4.fromHexColor("#000000", 0.2), mtx: new Mat4f(), thick: 1.5 });

    renderOutputBoxes(state, layout, tl, br, outCellW, 4, lineOpts, { boldLast: true, tokMixes: state.display.tokenOutputColors });

    for (let i = 0; i < nCells; i++) {
        let tx = cellPosition(layout, softmax, Dim.X, i) + 0.5 * layout.cell;
        let ty = softmax.y + softmax.dy + 0.5 * layout.cell;
        let bx = tl.x + (i + 0.5) * outCellW;
        let by = tl.y - layout.cell;

        let midY1 = lerp(ty, by, 1/6);
        let midY2 = lerp(ty, by, 3/4);

        drawLine(state.render.lineRender, new Vec3(tx, ty), new Vec3(tx, midY1), lineOpts);
        drawLine(state.render.lineRender, new Vec3(tx, midY1), new Vec3(bx, midY2), lineOpts);
        drawLine(state.render.lineRender, new Vec3(bx, midY2), new Vec3(bx, by), lineOpts);

        let arrLen = 0.6;
        let arrowLeft = new Vec3(bx - arrLen, by - arrLen);
        let arrowRight = new Vec3(bx + arrLen, by - arrLen);
        drawLine(state.render.lineRender, arrowLeft, new Vec3(bx, by), lineOpts);
        drawLine(state.render.lineRender, arrowRight, new Vec3(bx, by), lineOpts);
    }

}
