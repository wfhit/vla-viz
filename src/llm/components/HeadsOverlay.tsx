import React, { useEffect, useRef } from 'react';
import { useProgramState } from '../Sidebar';
import { Phase } from '../walkthrough/Walkthrough';
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
            drawScene(ctx, w, h, progState.walkthrough.phase, t, now / 1000);
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

function hash(n: number) {
    let x = Math.sin(n * 127.1 + 311.7) * 43758.5453;
    return x - Math.floor(x);
}

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

interface IIsoOpts {
    cols: number;
    rows: number;
    cell: number;
    depth?: number;
    color: string;
    flicker?: number; // 0..1: how many cells flicker with numbers
    clock?: number;
    seed?: number;
    fill?: number; // 0..1 how many cells (row-major) are "computed" (bright)
    highlight?: number;
    label?: string;
    labelPos?: 'above' | 'below';
}

// isometric-ish block: front face is the cell grid, top & right faces are shaded extrusions
function isoBlock(ctx: CanvasRenderingContext2D, x: number, y: number, o: IIsoOpts) {
    let { cols, rows, cell, color } = o;
    let depth = o.depth ?? Math.max(6, cell * 1.2);
    let w = cols * cell;
    let h = rows * cell;
    let dx = depth * 0.8;
    let dy = -depth * 0.5;

    // top face
    ctx.fillStyle = shade(color, 1.1);
    ctx.beginPath();
    ctx.moveTo(x, y);
    ctx.lineTo(x + dx, y + dy);
    ctx.lineTo(x + w + dx, y + dy);
    ctx.lineTo(x + w, y);
    ctx.closePath();
    ctx.fill();
    // right face
    ctx.fillStyle = shade(color, 0.55);
    ctx.beginPath();
    ctx.moveTo(x + w, y);
    ctx.lineTo(x + w + dx, y + dy);
    ctx.lineTo(x + w + dx, y + h + dy);
    ctx.lineTo(x + w, y + h);
    ctx.closePath();
    ctx.fill();

    // front face: cells
    let maxCells = 40 * 40;
    let step = Math.max(1, Math.ceil(Math.sqrt((cols * rows) / maxCells)));
    for (let r = 0; r < rows; r += step) {
        for (let c = 0; c < cols; c += step) {
            let idx = r * cols + c;
            let computed = o.fill === undefined || idx / (cols * rows) <= o.fill;
            let base = 0.42;
            let v = hash(idx + (o.seed ?? 0) * 977);
            if (o.flicker && o.clock !== undefined && hash(idx * 3.1 + Math.floor(o.clock * 8) + (o.seed ?? 0)) < o.flicker) {
                v = hash(idx + Math.floor(o.clock * 8) * 13);
            }
            let f = computed ? base + v * 0.55 : 0.22;
            ctx.fillStyle = shade(color, f);
            ctx.fillRect(x + c * cell, y + r * cell, cell * step - (cell > 5 ? 1 : 0), cell * step - (cell > 5 ? 1 : 0));
        }
    }
    ctx.strokeStyle = shade(color, 1.35, 0.9);
    ctx.lineWidth = 1;
    ctx.strokeRect(x + 0.5, y + 0.5, w, h);
    if (o.highlight) {
        ctx.strokeStyle = `rgba(255,255,255,${0.9 * o.highlight})`;
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 1, y - 1, w + 2, h + 2);
    }
    if (o.label) {
        ctx.fillStyle = palette.text;
        ctx.font = '12px system-ui, "Noto Sans CJK SC", sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(o.label, x + w / 2, o.labelPos === 'above' ? y + dy - 6 : y + h + 16);
    }
    return { x, y, w, h, right: x + w + dx, midY: y + h / 2 };
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

// ---------- scenes ----------

interface IField {
    key: string;
    name: string;
    color: string;
    size: string;
    options: string[];
    winner: number;
    logits: number[];
}

const fields: IField[] = [
    { key: 'knack', name: 'knack', color: palette.knack, size: 'vocab', options: ['pick_up', 'place_on', 'open', 'go_to', 'wait'], winner: 1, logits: [1.2, 3.4, 0.4, 1.9, -0.6] },
    { key: 'object', name: 'object', color: palette.object, size: '260', options: ['obj_031', 'obj_142', 'obj_007', 'obj_208', 'obj_054'], winner: 2, logits: [0.8, 1.5, 3.1, -0.2, 0.9] },
    { key: 'target', name: 'target', color: palette.target, size: '261', options: ['tgt_019', 'tgt_203', 'tgt_088', 'tgt_260', 'tgt_005'], winner: 0, logits: [3.3, 0.5, 1.1, 0.0, 1.6] },
    { key: 'facing', name: 'facing', color: palette.facing, size: '4', options: ['N', 'E', 'S', 'W'], winner: 3, logits: [0.3, 0.6, 1.0, 2.8] },
    { key: 'kind', name: 'kind', color: palette.kind, size: 'K', options: ['grasp', 'push', 'tap', 'hold'], winner: 0, logits: [2.7, 0.9, 0.2, 1.0] },
    { key: 'direction', name: 'direction', color: palette.direction, size: '6', options: ['up', 'down', 'left', 'right', 'fwd', 'back'], winner: 4, logits: [0.4, -0.3, 0.7, 1.1, 2.9, 0.2] },
];

function softmax(xs: number[]) {
    let m = Math.max(...xs);
    let es = xs.map(x => Math.exp(x - m));
    let sum = es.reduce((a, b) => a + b, 0);
    return es.map(e => e / sum);
}

function drawHStar(ctx: CanvasRenderingContext2D, x: number, y: number, clock: number, pulse: number) {
    isoBlock(ctx, x, y, { cols: 1, rows: 16, cell: 9, color: '#38bdf8', flicker: 0.5, clock, seed: 3, label: 'h*', labelPos: 'above', highlight: pulse > 0 ? 0.5 : 0 });
    text(ctx, '答案边界读出', x + 4, y + 16 * 9 + 34, { size: 12, color: palette.dim, align: 'center' });
}

function sceneKnack(ctx: CanvasRenderingContext2D, w: number, h: number, t: number, clock: number) {
    title(ctx, w, 'knack 并行分类头（System One / Jev 式）', 'h* 同时送入 6 个独立线性头；每个字段各自 softmax，各自只胜出一个值');
    let hx = 70;
    let hy = h / 2 - 70;
    let tFlow = seg(t, 0.0, 0.28);
    let tBars = seg(t, 0.25, 0.6);
    let tWin = seg(t, 0.6, 0.78);

    let cols = 2;
    let areaX = 190;
    let areaW = w - areaX - 24;
    let cellW = areaW / cols;
    let rowsN = 3;
    let top = 70;
    let cellH = (h - top - 16) / rowsN;

    drawHStar(ctx, hx, hy, clock, tFlow > 0 && tFlow < 1 ? 1 : 0);

    fields.forEach((f, i) => {
        let col = i % cols;
        let row = Math.floor(i / cols);
        let px = areaX + col * cellW;
        let py = top + row * cellH;
        let panelW = cellW - 14;
        let panelH = cellH - 12;

        flowCurve(ctx, hx + 18, hy + 70, px, py + panelH / 2, f.color, tFlow < 1 ? clamp01(tFlow * 1.2 - i * 0.04) : (Math.floor(clock * 0.6 + i * 0.17) % 1 === 0 ? ((clock * 0.35 + i * 0.13) % 1) : -1), 1);

        ctx.fillStyle = 'rgba(30,41,59,0.55)';
        roundRect(ctx, px, py, panelW, panelH, 8);
        ctx.fill();
        let won = tWin > 0.5;
        ctx.strokeStyle = won ? shade(f.color, 1.0, 0.9) : 'rgba(148,163,184,0.25)';
        ctx.lineWidth = won ? 1.5 : 1;
        ctx.stroke();

        // linear head block
        isoBlock(ctx, px + 12, py + 28, { cols: 6, rows: 4, cell: 7, color: f.color, flicker: 0.5, clock, seed: i, depth: 8 });
        text(ctx, `${f.name} (${f.size})`, px + 12, py + 18, { size: 12.5, bold: true, color: f.color });
        text(ctx, 'softmax', px + 12, py + 76, { size: 11, color: palette.dim });

        // probability bars
        let probs = softmax(f.logits.map((l, k) => l + (k === f.winner ? 0 : 0)));
        let barX = px + 112;
        let barW = panelW - 112 - 52;
        let n = f.options.length;
        let rowH = Math.min(17, (panelH - 14) / n);
        let barsTop = py + 8 + (panelH - 14 - rowH * n) / 2;
        for (let k = 0; k < n; k++) {
            let by = barsTop + k * rowH;
            let isWin = k === f.winner;
            let grow = tBars;
            let jitter = (1 - grow) * hash(k + i * 7 + Math.floor(clock * 6)) * 0.6;
            let pv = probs[k] * grow + jitter * (1 - grow);
            let dim = won && !isWin ? 0.35 : 1;
            text(ctx, f.options[k], barX - 6, by + rowH * 0.72, { size: 11, align: 'right', color: isWin && won ? palette.text : palette.dim });
            ctx.fillStyle = 'rgba(15,23,42,0.8)';
            ctx.fillRect(barX, by + 2, barW, rowH - 4);
            ctx.fillStyle = shade(f.color, isWin && won ? 1.25 : 0.8, dim);
            ctx.fillRect(barX, by + 2, barW * clamp01(pv), rowH - 4);
            if (grow > 0.9) {
                text(ctx, (probs[k] * 100).toFixed(0) + '%', barX + barW + 6, by + rowH * 0.72, { size: 10.5, color: isWin && won ? palette.text : palette.dim });
            }
            if (isWin && won) {
                ctx.strokeStyle = 'rgba(255,255,255,0.9)';
                ctx.lineWidth = 1.5;
                ctx.strokeRect(barX - 1, by + 1, barW + 2, rowH - 2);
            }
        }
    });
}

function sceneDone(ctx: CanvasRenderingContext2D, w: number, h: number, t: number, clock: number) {
    title(ctx, w, 'done 独立头  [B, 2]', 'subtask 级信号，作为叠加输出；绝不覆盖 knack 的答案');
    let hx = 70;
    let hy = h / 2 - 70;
    let tFlow = seg(t, 0.0, 0.3);
    let tBars = seg(t, 0.3, 0.7);
    drawHStar(ctx, hx, hy, clock, tFlow > 0 && tFlow < 1 ? 1 : 0);

    // upper branch: knack answer (locked)
    let kx = 260;
    let ky = 90;
    flowCurve(ctx, hx + 18, hy + 70, kx, ky + 60, palette.knack, tFlow < 1 ? tFlow : -1, 1);
    ctx.fillStyle = 'rgba(30,41,59,0.55)';
    roundRect(ctx, kx, ky, w - kx - 30, 120, 8);
    ctx.fill();
    ctx.strokeStyle = shade(palette.knack, 1, 0.7);
    ctx.stroke();
    text(ctx, 'knack 答案（已锁定）', kx + 14, ky + 22, { size: 13, bold: true, color: palette.knack });
    let chips = [['knack', 'place_on', palette.knack], ['object', 'obj_007', palette.object], ['target', 'tgt_019', palette.target], ['facing', 'W', palette.facing], ['kind', 'grasp', palette.kind], ['direction', 'fwd', palette.direction]];
    let chipX = kx + 14;
    let chipY = ky + 44;
    chips.forEach(([k, v, c], i) => {
        let cw = 118;
        let cx = chipX + (i % 3) * (cw + 10);
        let cy = chipY + Math.floor(i / 3) * 34;
        ctx.fillStyle = shade(c, 0.35, 0.9);
        roundRect(ctx, cx, cy, cw, 26, 6);
        ctx.fill();
        ctx.strokeStyle = shade(c, 1.0, 0.8);
        ctx.stroke();
        text(ctx, `${k}: ${v}`, cx + 8, cy + 17.5, { size: 12 });
    });
    // lock icon
    text(ctx, '🔒', w - 60, ky + 26, { size: 16 });

    // lower branch: done head
    let dx = 260;
    let dy = h / 2 + 20;
    flowCurve(ctx, hx + 18, hy + 70, dx, dy + 55, palette.done, tFlow < 1 ? tFlow : -1, 1);
    ctx.fillStyle = 'rgba(30,41,59,0.55)';
    roundRect(ctx, dx, dy, w - dx - 30, h - dy - 24, 8);
    ctx.fill();
    ctx.strokeStyle = shade(palette.done, 1, 0.7);
    ctx.stroke();
    isoBlock(ctx, dx + 16, dy + 36, { cols: 2, rows: 10, cell: 8, color: palette.done, flicker: 0.5, clock, seed: 9, depth: 10, label: '[B, 2]', labelPos: 'below' });
    text(ctx, 'done 线性头 → softmax', dx + 16, dy + 22, { size: 13, bold: true, color: palette.done });

    let pTrue = 0.08 + 0.87 * tBars;
    let labels = [['done = false', 1 - pTrue], ['done = true', pTrue]] as const;
    let bx = dx + 130;
    let bw = w - bx - 120;
    labels.forEach(([l, p], i) => {
        let by = dy + 44 + i * 38;
        text(ctx, l, bx - 10, by + 17, { size: 13, align: 'right', color: palette.dim });
        ctx.fillStyle = 'rgba(15,23,42,0.8)';
        ctx.fillRect(bx, by, bw, 24);
        ctx.fillStyle = shade(i === 1 ? palette.done : '#64748b', 1.1);
        ctx.fillRect(bx, by, bw * p, 24);
        text(ctx, (p * 100).toFixed(0) + '%', bx + bw + 10, by + 17, { size: 12.5 });
    });
    text(ctx, tBars > 0.9 ? 'done=true：本 subtask 的动作块已足够 → 交给 harness 处理' : '读取 h*，只判断“本 subtask 是否完成”', dx + 16, h - 40, { size: 12.5, color: tBars > 0.9 ? palette.done : palette.dim });
}

function sceneExpert(ctx: CanvasRenderingContext2D, w: number, h: number, t: number, clock: number) {
    title(ctx, w, '动作专家（action expert）', '状态 token + 16 个噪声动作 token x_t + 时间嵌入（adaLN）→ 联合去噪器，逐层与主干联合注意力');

    let tIn = seg(t, 0.0, 0.25);
    let tProj = seg(t, 0.2, 0.45);
    let tJoint = seg(t, 0.4, 1.0);

    // backbone slabs (left)
    let slabX = 28;
    let slabTop = 78;
    let nSlabs = 12;
    let slabH = (h - slabTop - 30) / nSlabs;
    text(ctx, 'Qwen3-VL 主干 ×36（示意 12 层）', slabX, slabTop - 8, { size: 12, color: palette.dim });
    let slabW = 150;
    let centersY: number[] = [];
    for (let i = 0; i < nSlabs; i++) {
        let y = slabTop + i * slabH;
        centersY.push(y + slabH / 2);
        let active = tJoint > 0 && ((clock * 2.2) % nSlabs | 0) === i;
        isoBlock(ctx, slabX, y + 2, { cols: 30, rows: 1, cell: slabW / 30, depth: 6, color: '#3b82f6', flicker: 0.3, clock, seed: i, highlight: active ? 0.7 : 0 });
    }

    // expert column
    let ex = w * 0.58;
    text(ctx, '联合去噪器（动作专家 ×36）', ex, slabTop - 8, { size: 12, color: palette.dim });
    for (let i = 0; i < nSlabs; i++) {
        let y = slabTop + i * slabH;
        isoBlock(ctx, ex, y + 2, { cols: 18, rows: 1, cell: 150 / 18, depth: 6, color: palette.action, flicker: 0.3, clock, seed: i + 50, highlight: tJoint > 0 && ((clock * 2.2) % nSlabs | 0) === i ? 0.7 : 0 });
        // joint attention link
        let p = tJoint > 0 ? (((clock * 0.9 + i * 0.11) % 1)) : -1;
        flowCurve(ctx, slabX + slabW + 8, y + slabH / 2, ex - 4, y + slabH / 2, palette.cyan, p, tJoint);
    }

    // inputs (middle)
    let mx = slabX + slabW + 70;
    let my = slabTop + 4;
    let boxW = ex - mx - 50;
    function inputRow(y: number, label: string, color: string, cols: number, rows: number, cell: number, sub: string, a: number) {
        ctx.globalAlpha = 0.25 + 0.75 * a;
        isoBlock(ctx, mx, y, { cols, rows, cell, color, flicker: 0.6, clock, seed: Math.floor(y), depth: 8 });
        text(ctx, label, mx, y - 12, { size: 12.5, bold: true, color });
        text(ctx, sub, mx, y + rows * cell + 16, { size: 11, color: palette.dim });
        ctx.globalAlpha = 1;
    }
    inputRow(my + 18, '状态 token', palette.cyan, 12, 1, 8, tProj < 0.5 ? '61+14 = 75 维' : '75 → 768（线性投影）', tIn);
    let rowsShown = 4;
    inputRow(my + 98, '噪声动作 token x_t (×16)', palette.action, 16, rowsShown, 7, 'x_t ~ N(0, I)，16 个 token', tIn);
    inputRow(my + 98 + 7 * rowsShown + 72, '时间嵌入 t → adaLN', palette.time, 8, 1, 8, '产生 scale / shift 调制每层', seg(t, 0.1, 0.35));

    // arrows to expert
    for (let k = 0; k < 3; k++) {
        let y0 = [my + 22, my + 112, my + 98 + 7 * rowsShown + 76][k];
        flowCurve(ctx, mx + 130, y0, ex - 6, slabTop + (k + 1) * 3 * slabH, [palette.cyan, palette.action, palette.time][k], tIn < 1 ? tIn : ((clock * 0.5 + k * 0.3) % 1), tIn);
    }
    text(ctx, 'attention 中后缀(动作)流 Q 看 前缀(VLM) K/V', ex, h - 10, { size: 12, color: palette.dim });
}

function sceneDenoise(ctx: CanvasRenderingContext2D, w: number, h: number, t: number, clock: number) {
    title(ctx, w, '速度场 v_t  →  Euler 积分  →  16×23 动作块', 'x_{t+dt} = x_t + dt · v_θ(x_t, t)，从 t=0 积分到 t=1；settle 行速度清零、位置保持');

    let steps = 10;
    let prog = seg(t, 0.05, 0.85);
    let stepIdx = Math.min(steps, Math.floor(prog * steps));
    let tau = prog;

    let cell = Math.min(16, Math.floor((w * 0.5) / 23), Math.floor((h - 200) / 16));
    let gx = 40;
    let gy = 100;
    let gw = 23 * cell;
    let gh = 16 * cell;

    // x_t chunk: noise -> clean
    let target = (r: number, c: number) => 0.5 + 0.45 * Math.sin(c * 0.45 + r * 0.12) * Math.cos(r * 0.2);
    let settleRows = 3;
    for (let r = 0; r < 16; r++) {
        for (let c = 0; c < 23; c++) {
            let noise = hash(r * 23 + c + 1);
            let v = (1 - tau) * noise + tau * target(r, c);
            let settle = r >= 16 - settleRows;
            if (settle) {
                v = (1 - Math.min(1, tau * 1.2)) * noise * 0.6 + target(16 - settleRows - 1, c) * Math.min(1, tau * 1.2) + 0 * v;
            }
            let hue = 190 + v * 60;
            let flickerNoise = tau < 1 && hash(r * 31 + c + Math.floor(clock * 8)) < 0.1 ? 0.15 : 0;
            ctx.fillStyle = `hsl(${hue}, 70%, ${18 + (v + flickerNoise) * 48}%)`;
            ctx.fillRect(gx + c * cell, gy + r * cell, cell - 1, cell - 1);
        }
    }
    // frame as iso slab hint (top & right faces)
    ctx.fillStyle = shade('#22d3ee', 0.9, 0.9);
    ctx.beginPath();
    ctx.moveTo(gx, gy);
    ctx.lineTo(gx + 8, gy - 5);
    ctx.lineTo(gx + gw + 8, gy - 5);
    ctx.lineTo(gx + gw, gy);
    ctx.fill();
    ctx.fillStyle = shade('#22d3ee', 0.45, 0.9);
    ctx.beginPath();
    ctx.moveTo(gx + gw, gy);
    ctx.lineTo(gx + gw + 8, gy - 5);
    ctx.lineTo(gx + gw + 8, gy + gh - 5);
    ctx.lineTo(gx + gw, gy + gh);
    ctx.fill();
    ctx.strokeStyle = 'rgba(165,243,252,0.8)';
    ctx.strokeRect(gx + 0.5, gy + 0.5, gw, gh);

    text(ctx, '23 维 × 16 步（动作块）', gx, gy - 14, { size: 12.5, bold: true, color: palette.action });
    // settle bracket
    let sy = gy + (16 - settleRows) * cell;
    ctx.strokeStyle = palette.time;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(gx - 3, sy, gw + 6, settleRows * cell);
    ctx.setLineDash([]);
    text(ctx, 'settle 行：v = 0，位置保持', gx + gw + 18, sy + settleRows * cell / 2 + 4, { size: 12, color: palette.time });
    // knack boundary
    let by = gy + gh + 14;
    ctx.fillStyle = shade(palette.knack, 0.6, 0.9);
    ctx.fillRect(gx, by, gw, 5);
    text(ctx, '▲ knack 边界：动作块永不跨越', gx, by + 22, { size: 12, color: palette.knack });

    // right: velocity arrows + step track
    let rx = gx + gw + 40 + 150;
    rx = Math.max(rx, w * 0.62);
    let rw = w - rx - 24;
    text(ctx, `Euler 步 ${stepIdx}/${steps}   t = ${tau.toFixed(2)}`, rx, 100, { size: 14, bold: true, color: palette.cyan });
    ctx.fillStyle = 'rgba(15,23,42,0.9)';
    ctx.fillRect(rx, 112, rw, 8);
    ctx.fillStyle = palette.cyan;
    ctx.fillRect(rx, 112, rw * tau, 8);
    for (let k = 0; k <= steps; k++) {
        ctx.fillStyle = k <= stepIdx ? palette.cyan : '#475569';
        ctx.fillRect(rx + (rw * k) / steps - 1, 108, 2, 16);
    }

    // velocity field mini plot
    let vy = 150;
    let vh = Math.min(130, h - vy - 120);
    ctx.strokeStyle = 'rgba(148,163,184,0.25)';
    ctx.strokeRect(rx + 0.5, vy + 0.5, rw, vh);
    text(ctx, 'v_θ(x_t, t)  速度场', rx + 8, vy + 16, { size: 12, color: palette.dim });
    let nx = 10;
    let ny = 4;
    for (let i = 0; i < nx; i++) {
        for (let j = 0; j < ny; j++) {
            let px = rx + 20 + (i + 0.5) * ((rw - 40) / nx);
            let py = vy + 28 + (j + 0.5) * ((vh - 36) / ny);
            let mag = (0.15 + 0.85 * (1 - tau)) * (0.6 + 0.4 * Math.sin(i * 0.9 + j + clock));
            let ang = -0.4 + 0.5 * Math.sin(i * 0.5 + j * 0.8);
            let len = 14 * mag;
            ctx.strokeStyle = shade(palette.cyan, 0.8 + mag * 0.5, 0.3 + 0.7 * mag);
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.moveTo(px, py);
            ctx.lineTo(px + Math.cos(ang) * len, py + Math.sin(ang) * len);
            ctx.stroke();
        }
    }
    text(ctx, tau >= 1 ? 't = 1：得到干净的 16×23 动作块 → 送给执行器' : '每步：网络预测 v_t，再沿速度方向前进 dt = 1/10', rx, vy + vh + 28, { size: 12.5, color: tau >= 1 ? palette.action : palette.dim });
}

function sceneHarness(ctx: CanvasRenderingContext2D, w: number, h: number, t: number, clock: number) {
    title(ctx, w, 'Harness 法则', '模型只报告，不切换：由 harness 在服务端决定何时刹车、何时推进 subtask');

    let lanes = [
        { name: '模型', color: palette.cyan },
        { name: 'Harness / 服务端', color: palette.time },
        { name: '机器人', color: palette.action },
    ];
    let lx = 40;
    let laneX = 190;
    let laneW = w - laneX - 30;
    let top = 80;
    let laneH = Math.min(96, (h - top - 130) / 3);

    lanes.forEach((l, i) => {
        let y = top + i * laneH;
        ctx.fillStyle = 'rgba(30,41,59,0.45)';
        roundRect(ctx, lx, y, w - lx * 2 + 6, laneH - 10, 8);
        ctx.fill();
        text(ctx, l.name, lx + 14, y + (laneH - 10) / 2 + 5, { size: 14, bold: true, color: l.color });
    });

    // events along time axis
    let ev = [
        { at: 0.10, lane: 0, label: 'done = true', color: palette.done },
        { at: 0.32, lane: 1, label: '收到 done → 等待 chunk 播完', color: palette.time },
        { at: 0.62, lane: 1, label: '刹车 · 定格', color: palette.time },
        { at: 0.78, lane: 1, label: '推进 subtask', color: palette.cyan },
    ];
    let pos = (a: number) => laneX + a * laneW;

    // chunk playback bar on robot lane
    let robotY = top + 2 * laneH;
    let barH = 24;
    let chunkStart = 0.14;
    let chunkEnd = 0.58;
    let play = clamp01((t * 0.9 - chunkStart) / (chunkEnd - chunkStart));
    let tNorm = seg(t, 0, 0.95);
    let cur = tNorm * 0.95 + 0.04;
    ctx.fillStyle = 'rgba(15,23,42,0.9)';
    ctx.fillRect(pos(chunkStart), robotY + 14, laneW * (chunkEnd - chunkStart), barH);
    let playFrac = clamp01((cur - chunkStart) / (chunkEnd - chunkStart));
    ctx.fillStyle = shade(palette.action, 0.9);
    ctx.fillRect(pos(chunkStart), robotY + 14, laneW * (chunkEnd - chunkStart) * playFrac, barH);
    text(ctx, '播放剩余 action chunk', pos(chunkStart) + 8, robotY + 31, { size: 12, color: '#052e16' });
    // freeze
    let frozenFrom = 0.6;
    if (cur >= frozenFrom) {
        ctx.fillStyle = shade('#60a5fa', 0.7);
        ctx.fillRect(pos(frozenFrom), robotY + 14, laneW * (0.98 - frozenFrom), barH);
        text(ctx, '定格（刹车）', pos(frozenFrom) + 8, robotY + 31, { size: 12, color: '#e0f2fe' });
    }

    ev.forEach(e => {
        let shown = cur >= e.at;
        let a = shown ? 1 : 0.25;
        let x = pos(e.at);
        let y = top + e.lane * laneH + (laneH - 10) / 2;
        ctx.globalAlpha = a;
        ctx.fillStyle = e.color;
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.fill();
        text(ctx, e.label, x + 12, y + 5, { size: 12.5, color: palette.text });
        ctx.globalAlpha = 1;
        if (shown && cur - e.at < 0.08) {
            ctx.strokeStyle = shade(e.color, 1.2, 0.5 * (1 - (cur - e.at) / 0.08));
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.arc(x, y, 7 + (cur - e.at) * 200, 0, Math.PI * 2);
            ctx.stroke();
        }
    });

    // time cursor
    ctx.strokeStyle = 'rgba(226,232,240,0.6)';
    ctx.setLineDash([3, 4]);
    ctx.beginPath();
    ctx.moveTo(pos(cur), top - 6);
    ctx.lineTo(pos(cur), top + 3 * laneH - 8);
    ctx.stroke();
    ctx.setLineDash([]);
    void play;

    // rule box
    let by = top + 3 * laneH + 6;
    ctx.fillStyle = 'rgba(14,116,144,0.25)';
    roundRect(ctx, lx, by, w - lx * 2 + 6, Math.max(44, h - by - 14), 8);
    ctx.fill();
    ctx.strokeStyle = 'rgba(34,211,238,0.5)';
    ctx.stroke();
    let steps = ['done = true', '播完 chunk', '服务端刹车 · 定格', '推进 subtask'];
    let stepAt = [0.1, 0.32, 0.62, 0.78];
    let sw = (w - lx * 2 - 30 - 3 * 28) / 4;
    steps.forEach((st, i) => {
        let sx = lx + 14 + i * (sw + 28);
        let on = cur >= stepAt[i];
        ctx.fillStyle = on ? 'rgba(14,165,233,0.35)' : 'rgba(30,41,59,0.8)';
        roundRect(ctx, sx, by + 14, sw, 36, 8);
        ctx.fill();
        ctx.strokeStyle = on ? '#38bdf8' : 'rgba(148,163,184,0.3)';
        ctx.stroke();
        text(ctx, st, sx + sw / 2, by + 37, { size: 13.5, align: 'center', color: on ? '#e0f2fe' : palette.dim });
        if (i < 3) {
            text(ctx, '→', sx + sw + 14, by + 38, { size: 16, align: 'center', color: palette.cyan });
        }
    });
    text(ctx, '模型只报告 done，从不自行切换 subtask；切换由 harness 在服务端完成', lx + 14, by + 76, { size: 13, color: '#cffafe' });
}

function drawScene(ctx: CanvasRenderingContext2D, w: number, h: number, phase: Phase, time: number, clock: number) {
    ctx.fillStyle = 'rgba(17,24,39,0.98)';
    roundRect(ctx, 0.5, 0.5, w - 1, h - 1, 12);
    ctx.fill();
    ctx.strokeStyle = 'rgba(56,189,248,0.35)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // normalised scene time (phases animate over ~12s)
    let t = clamp01(time / 12);

    switch (phase) {
        case Phase.Head_Knack: sceneKnack(ctx, w, h, t, clock); break;
        case Phase.Head_Done: sceneDone(ctx, w, h, t, clock); break;
        case Phase.Action_Expert: sceneExpert(ctx, w, h, t, clock); break;
        case Phase.Action_Denoise: sceneDenoise(ctx, w, h, t, clock); break;
        case Phase.Harness_Law: sceneHarness(ctx, w, h, t, clock); break;
    }
}
