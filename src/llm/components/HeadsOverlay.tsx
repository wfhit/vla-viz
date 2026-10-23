import React, { useEffect, useRef } from 'react';
import { useProgramState } from '../Sidebar';
import { Phase } from '../walkthrough/Walkthrough';
import { IGptModelLink, IRealTex } from '../GptModel';
import s from './HeadsOverlay.module.scss';

const heads = [Phase.Head_Knack, Phase.Head_Done, Phase.Action_Expert, Phase.Action_Denoise, Phase.Harness_Law];

export const HeadsOverlay: React.FC = () => {
    let progState = useProgramState();
    let wt = progState.walkthrough;
    let canvasRef = useRef<HTMLCanvasElement | null>(null);
    let wrapRef = useRef<HTMLDivElement | null>(null);
    let visible = heads.includes(wt.phase);

    useEffect(() => {
        if (!visible) {
            return;
        }
        let canvas = canvasRef.current!;
        let wrap = wrapRef.current!;
        let raf = 0;
        let stopped = false;

        function frame(now: number) {
            if (stopped) {
                return;
            }
            let bcr = wrap.getBoundingClientRect();
            let dpr = window.devicePixelRatio || 1;
            let w = Math.max(1, Math.floor(bcr.width));
            let h = Math.max(1, Math.floor(bcr.height));
            if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
                canvas.width = w * dpr;
                canvas.height = h * dpr;
            }
            let ctx = canvas.getContext('2d')!;
            ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
            ctx.clearRect(0, 0, w, h);
            let t = progState.walkthrough.phaseLength > 0 ? progState.walkthrough.time : 0;
            drawScene(ctx, w, h, progState.walkthrough.phase, t, progState.jsGptModel);
            raf = requestAnimationFrame(frame);
        }
        raf = requestAnimationFrame(frame);
        return () => {
            stopped = true;
            cancelAnimationFrame(raf);
        };
    }, [visible, progState]);

    if (!visible) {
        return null;
    }

    return <div className={s.wrap} ref={wrapRef}>
        <canvas ref={canvasRef} className={s.canvas} />
    </div>;
};

// ---------- drawing helpers ----------

const palette = {
    bg: '#111827',
    text: '#e5e7eb',
    dim: '#94a3b8',
    cyan: '#22d3ee',
    blue: '#3b82f6',
    knack: '#a78bfa',
    object: '#f472b6',
    target: '#fb923c',
    facing: '#facc15',
    kind: '#34d399',
    direction: '#60a5fa',
    done: '#f87171',
    action: '#4ade80',
    time: '#fbbf24',
};

function clamp01(x: number) {
    return Math.max(0, Math.min(1, x));
}

function ease(x: number) {
    x = clamp01(x);
    return x * x * (3 - 2 * x);
}

function seg(t: number, a: number, b: number) {
    return ease((t - a) / (b - a));
}

function shade(hex: string, f: number, alpha = 1) {
    let n = parseInt(hex.slice(1), 16);
    let r = Math.min(255, Math.round(((n >> 16) & 255) * f));
    let g = Math.min(255, Math.round(((n >> 8) & 255) * f));
    let b = Math.min(255, Math.round((n & 255) * f));
    return `rgba(${r},${g},${b},${alpha})`;
}

function text(ctx: CanvasRenderingContext2D, str: string, x: number, y: number, opts: { size?: number; color?: string; align?: CanvasTextAlign; bold?: boolean } = {}) {
    ctx.font = `${opts.bold ? '600 ' : ''}${opts.size ?? 13}px system-ui, "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif`;
    ctx.fillStyle = opts.color ?? palette.text;
    ctx.textAlign = opts.align ?? 'left';
    ctx.fillText(str, x, y);
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, h: number, r: number) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

// curve with a travelling pulse; p in 0..1 is the pulse position (-1 => no pulse)
function flowCurve(ctx: CanvasRenderingContext2D, x0: number, y0: number, x1: number, y1: number, color: string, p: number, reveal = 1) {
    let cx = (x0 + x1) / 2;
    ctx.save();
    ctx.strokeStyle = shade(color, 0.7, 0.45 * reveal);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.bezierCurveTo(cx, y0, cx, y1, x1, y1);
    ctx.stroke();
    if (p >= 0 && p <= 1 && reveal > 0) {
        let u = p;
        let mt = 1 - u;
        let px = mt * mt * mt * x0 + 3 * mt * mt * u * cx + 3 * mt * u * u * cx + u * u * u * x1;
        let py = mt * mt * mt * y0 + 3 * mt * mt * u * y0 + 3 * mt * u * u * y1 + u * u * u * y1;
        let g = ctx.createRadialGradient(px, py, 0, px, py, 14);
        g.addColorStop(0, shade(color, 1.6, 0.95));
        g.addColorStop(1, shade(color, 1.0, 0));
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(px, py, 14, 0, Math.PI * 2);
        ctx.fill();
    }
    ctx.restore();
}

function title(ctx: CanvasRenderingContext2D, w: number, main: string, sub: string) {
    text(ctx, main, 20, 28, { size: 17, bold: true, color: palette.cyan });
    text(ctx, sub, 20, 48, { size: 12.5, color: palette.dim });
}

// ---------- real-data helpers ----------
// Everything below draws values straight from the exported k005b tensors
// (public/k005b-acts.json / k005b-weights.json); nothing is synthesised.

function fmt(v: number, d = 3) {
    return Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3) ? v.toExponential(2) : v.toFixed(d);
}

function divergingColor(v: number, scale: number) {
    let a = clamp01(Math.abs(v) * scale);
    if (v >= 0) {
        return `rgb(${Math.round(30 + 220 * a)},${Math.round(41 + 120 * a)},${Math.round(59 - 20 * a)})`;
    }
    return `rgb(${Math.round(30 - 10 * a)},${Math.round(41 + 120 * a)},${Math.round(59 + 196 * a)})`;
}

// heatmap of a real [rows, cols] window (row-major)
function heatmap(ctx: CanvasRenderingContext2D, x: number, y: number, data: ArrayLike<number>, rows: number, cols: number, cw: number, ch: number, reveal = 1) {
    let maxAbs = 0;
    for (let i = 0; i < rows * cols; i++) maxAbs = Math.max(maxAbs, Math.abs(data[i]));
    let scale = maxAbs > 0 ? 1 / maxAbs : 1;
    let shown = Math.floor(rows * reveal + 1e-6);
    for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
            ctx.fillStyle = r < shown ? divergingColor(data[r * cols + c], scale) : 'rgba(30,41,59,0.6)';
            ctx.fillRect(x + c * cw, y + r * ch, Math.max(1, cw - 1), Math.max(1, ch - 1));
        }
    }
    ctx.strokeStyle = 'rgba(148,163,184,0.4)';
    ctx.strokeRect(x - 0.5, y - 0.5, cols * cw + 1, rows * ch + 1);
    return maxAbs;
}

function texRows(tex: IRealTex) { return tex.height; }
function texCols(tex: IRealTex) { return tex.width; }

function metaLine(tex: IRealTex) {
    let st = tex.meta.stats;
    return `${tex.meta.name} [${tex.meta.fullShape.join('×')}]  min ${fmt(st.min)}  max ${fmt(st.max)}  mean ${fmt(st.mean)}`;
}

function drawHStar(ctx: CanvasRenderingContext2D, m: IGptModelLink, x: number, y: number, h: number) {
    let tex = m.hStar;
    let n = Math.min(64, tex.localBuffer.length);
    let ch = Math.max(2, Math.floor(h / n));
    heatmap(ctx, x, y, tex.localBuffer, n, 1, 18, ch);
    text(ctx, 'h*', x + 9, y - 8, { size: 13, bold: true, color: '#38bdf8', align: 'center' });
    text(ctx, `c 0–${n - 1} of ${tex.meta.fullShape[0]}`, x + 9, y + n * ch + 16, { size: 10.5, color: palette.dim, align: 'center' });
    text(ctx, `p=${m.config.p_star}`, x + 9, y + n * ch + 30, { size: 10.5, color: palette.dim, align: 'center' });
    return { right: x + 18, midY: y + n * ch / 2 };
}

function probBars(ctx: CanvasRenderingContext2D, x: number, y: number, w: number, rowH: number, names: string[], logits: ArrayLike<number>, probs: ArrayLike<number>, highlight: string | null, target: string | null, color: string, reveal: number) {
    let order = names.map((_, j) => j).sort((a, b) => probs[b] - probs[a]);
    let barX = x + 92;
    let barW = w - 92 - 150;
    order.forEach((j, k) => {
        let by = y + k * rowH;
        let isPred = names[j] === highlight;
        let isTgt = names[j] === target;
        text(ctx, names[j], barX - 8, by + rowH * 0.72, { size: 11.5, align: 'right', color: isPred ? palette.text : palette.dim, bold: isPred });
        ctx.fillStyle = 'rgba(15,23,42,0.8)';
        ctx.fillRect(barX, by + 2, barW, rowH - 4);
        ctx.fillStyle = shade(color, isPred ? 1.25 : 0.75);
        ctx.fillRect(barX, by + 2, barW * clamp01(probs[j]) * reveal, rowH - 4);
        text(ctx, `p=${fmt(probs[j], 4)}  z=${fmt(logits[j], 2)}${isTgt ? '  ← target' : ''}`, barX + barW + 8, by + rowH * 0.72, { size: 10.5, color: isTgt ? palette.time : palette.dim });
        if (isPred) {
            ctx.strokeStyle = 'rgba(255,255,255,0.9)';
            ctx.lineWidth = 1.5;
            ctx.strokeRect(barX - 1, by + 1, barW + 2, rowH - 2);
        }
    });
}

// ---------- scenes (real k005b data) ----------

function sceneKnack(ctx: CanvasRenderingContext2D, m: IGptModelLink, w: number, h: number, t: number) {
    let cfg = m.config;
    title(ctx, w, `knack 头  Linear(${cfg.n_embd} → ${cfg.knack_names.length})`, `真实 k005b step ${cfg.step}：h* = ln_f 输出在边界 token p=${cfg.p_star}；样本 "${cfg.sample.goal_text}"`);
    let hs = drawHStar(ctx, m, 40, 80, h - 150);
    let areaX = 120;
    let rowH = Math.min(22, (h - 120) / cfg.knack_names.length);
    flowCurve(ctx, hs.right + 4, hs.midY, areaX + 20, 80 + rowH * cfg.knack_names.length / 2, palette.knack, -1, 1);
    probBars(ctx, areaX, 76, w - areaX - 20, rowH, cfg.knack_names, m.knackLogits.localBuffer, m.knackProbs.localBuffer, cfg.pred_knack, cfg.sample.target_knack, palette.knack, seg(t, 0.05, 0.4));
    text(ctx, `预测 ${cfg.pred_knack}  ·  目标 ${cfg.sample.target_knack ?? '?'}  ·  ${metaLine(m.knackWeight)}`, 20, h - 18, { size: 12, color: palette.dim });
}

function sceneDone(ctx: CanvasRenderingContext2D, m: IGptModelLink, w: number, h: number, t: number) {
    let cfg = m.config;
    title(ctx, w, `done 头  Linear(${cfg.n_embd} → ${cfg.done_names.length})`, '读同一个 h*，只报告“本 subtask 是否完成”；不覆盖 knack 的答案');
    let hs = drawHStar(ctx, m, 40, 80, h - 150);
    let areaX = 120;
    flowCurve(ctx, hs.right + 4, hs.midY, areaX + 20, 110, palette.done, -1, 1);
    let tgt = cfg.sample.target_done === null ? null : (cfg.sample.target_done ? 'DONE' : 'NOT_DONE');
    probBars(ctx, areaX, 90, w - areaX - 20, 34, cfg.done_names, m.doneLogits.localBuffer, m.doneProbs.localBuffer, cfg.pred_done ? 'DONE' : 'NOT_DONE', tgt, palette.done, seg(t, 0.05, 0.4));
    let wd = m.doneWeight;
    let cols = Math.min(64, texCols(wd));
    text(ctx, `done_head 权重（行 0–1，列 c 0–${cols - 1} / ${wd.meta.fullShape[1]}）`, areaX, 196, { size: 12, color: palette.dim });
    let sub = new Float32Array(2 * cols);
    for (let r = 0; r < 2; r++) for (let c = 0; c < cols; c++) sub[r * cols + c] = wd.localBuffer[r * texCols(wd) + c];
    heatmap(ctx, areaX, 206, sub, 2, cols, Math.max(4, Math.floor((w - areaX - 40) / cols)), 16);
    text(ctx, `预测 done=${cfg.pred_done}  ·  目标 ${cfg.sample.target_done ?? '?'}  ·  ${metaLine(wd)}`, 20, h - 18, { size: 12, color: palette.dim });
}

function sceneExpert(ctx: CanvasRenderingContext2D, m: IGptModelLink, w: number, h: number, t: number) {
    let cfg = m.config;
    title(ctx, w, `动作专家  d=${cfg.expert_hidden} · ${cfg.expert_heads}q/${cfg.expert_kv_heads}kv 头 · FF ${cfg.expert_intermediate}`, `${cfg.n_suffix} 个后缀 token 与 ${cfg.T_prefix} 个前缀 token 逐层联合注意力（${cfg.n_layer} 层）；下图为最后一层后缀输出`);
    let eo = m.expertOut;
    let rows = texRows(eo), cols = Math.min(64, texCols(eo));
    let cw = Math.max(4, Math.floor((w - 80) / cols));
    let ch = Math.max(6, Math.floor((h - 150) / rows));
    heatmap(ctx, 40, 80, eo.localBuffer, rows, cols, cw, ch, seg(t, 0.0, 0.5));
    text(ctx, `expert_out：${rows} 个后缀 token × 通道 0–${cols - 1}（共 ${eo.meta.fullShape[1]}）`, 40, 80 + rows * ch + 20, { size: 12, color: palette.dim });
    text(ctx, metaLine(eo), 40, h - 18, { size: 12, color: palette.dim });
}

function sceneDenoise(ctx: CanvasRenderingContext2D, m: IGptModelLink, w: number, h: number, t: number) {
    let cfg = m.config;
    title(ctx, w, `速度场 → 动作块 ${cfg.action_horizon}×${cfg.action_dim}`, 'action_out_proj(expert_out) = v；action = x1 − v（导出时的真实前向）');
    let H = cfg.action_horizon, D = cfg.action_dim;
    let gap = 30;
    let cw = Math.max(4, Math.floor((w - 80 - 2 * gap) / (3 * D)));
    let ch = Math.max(6, Math.floor((h - 170) / H));
    let blocks: [string, IRealTex][] = [['x1 (噪声)', m.actionX1], ['v = 速度', m.actionVelocity], ['action = x1 − v', m.actionChunk]];
    blocks.forEach(([label, tex], i) => {
        let x = 40 + i * (D * cw + gap);
        let reveal = seg(t, 0.1 + i * 0.2, 0.3 + i * 0.2);
        heatmap(ctx, x, 90, tex.localBuffer, H, D, cw, ch, reveal);
        text(ctx, label, x, 80, { size: 12.5, bold: true, color: palette.action });
        let st = tex.meta.stats;
        text(ctx, `min ${fmt(st.min)}  max ${fmt(st.max)}`, x, 90 + H * ch + 16, { size: 10.5, color: palette.dim });
    });
    let a = m.actionChunk.localBuffer;
    text(ctx, `action[0, 0:5] = ${Array.from(a.slice(0, 5)).map(v => fmt(v, 4)).join(', ')}`, 40, h - 18, { size: 12, color: palette.text });
}

function sceneHarness(ctx: CanvasRenderingContext2D, m: IGptModelLink, w: number, h: number, t: number) {
    let cfg = m.config;
    title(ctx, w, 'Harness 法则', '模型只报告 knack / done，不自行切换 subtask；切换由服务端 harness 决定');
    let pDone = m.doneProbs.localBuffer[1];
    let steps = [
        `模型报告：knack=${cfg.pred_knack}, P(done)=${fmt(pDone, 4)}`,
        `done=${cfg.pred_done} → ${cfg.pred_done ? '播完当前 chunk 后由 harness 刹车并推进 subtask' : '继续执行 action chunk，不推进 subtask'}`,
        `执行 ${cfg.action_horizon}×${cfg.action_dim} 动作块（knack 边界内）`,
    ];
    steps.forEach((st, i) => {
        let y = 90 + i * 64;
        ctx.globalAlpha = 0.3 + 0.7 * seg(t, i * 0.2, i * 0.2 + 0.2);
        ctx.fillStyle = 'rgba(30,41,59,0.7)';
        roundRect(ctx, 40, y, w - 80, 46, 8);
        ctx.fill();
        ctx.strokeStyle = 'rgba(56,189,248,0.45)';
        ctx.stroke();
        text(ctx, st, 56, y + 29, { size: 14, color: palette.text });
        ctx.globalAlpha = 1;
    });
}

function drawScene(ctx: CanvasRenderingContext2D, w: number, h: number, phase: Phase, time: number, m: IGptModelLink | null) {
    ctx.fillStyle = 'rgba(17,24,39,0.98)';
    roundRect(ctx, 0.5, 0.5, w - 1, h - 1, 12);
    ctx.fill();
    ctx.strokeStyle = 'rgba(56,189,248,0.35)';
    ctx.lineWidth = 1;
    ctx.stroke();

    if (!m) {
        text(ctx, '正在加载真实 k005b 权重与激活…', 20, 30, { size: 14, color: palette.dim });
        return;
    }

    let t = clamp01(time / 12);

    switch (phase) {
        case Phase.Head_Knack: sceneKnack(ctx, m, w, h, t); break;
        case Phase.Head_Done: sceneDone(ctx, m, w, h, t); break;
        case Phase.Action_Expert: sceneExpert(ctx, m, w, h, t); break;
        case Phase.Action_Denoise: sceneDenoise(ctx, m, w, h, t); break;
        case Phase.Harness_Law: sceneHarness(ctx, m, w, h, t); break;
    }
}
