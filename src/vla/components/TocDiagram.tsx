import React, { memo, useCallback, useLayoutEffect, useState } from 'react';
import { BoundingBox3d, Vec3 } from '@/src/utils/vector';
import s from './TocDiagram.module.scss';
import { isNotNil } from '@/src/utils/data';
import { Phase } from '../walkthrough/Walkthrough';
import clsx from 'clsx';
import { jumpToPhase } from '../Commentary';
import { useProgramState } from '../Sidebar';
import { Subscriptions } from '@/src/utils/hooks';

interface INode {
    id: string;
    x: number;
    y: number;
    w: number;
    h: number;
    label: string[];
    color: string;
    dashed?: boolean;
    round?: boolean;
}

interface IEntryGroup {
    groupName: string;
    entries: IEntryInfo[];
}

interface IEntryInfo {
    id: Phase;
    title: string;
    ids: string[];
    groupIds: boolean;
}

const colors = {
    input: '#b48cf0',
    embed: '#e58ad8',
    norm: '#e0e56f',
    attn: '#f2b76b',
    mlp: '#5fd6d6',
    proj: '#7aa7ff',
    backbone: '#5ea0ff',
    head: '#6fe38a',
    done: '#ff8f8f',
    expert: '#4cc9f0',
    harness: '#ffd166',
    line: '#9db4d0',
    focus: '#4cc9f0',
    text: '#e6edf7',
};

const SVG_W = 360;
const SVG_H = 612;
const BUS_X = 312;

const nodes: INode[] = [
    { id: 'inCam', x: 6, y: 8, w: 94, h: 40, label: ['3×相机', '850 视觉 token'], color: colors.input },
    { id: 'inLang', x: 106, y: 8, w: 94, h: 40, label: ['语言 token', '提示词'], color: colors.input },
    { id: 'inState', x: 206, y: 8, w: 94, h: 40, label: ['状态', '16 token'], color: colors.input },
    { id: 'embed', x: 6, y: 62, w: 294, h: 28, label: ['Embedding (→ 2048)'], color: colors.embed },
    { id: 'backbone', x: 6, y: 104, w: 200, h: 330, label: [], color: colors.backbone, dashed: true },
    { id: 'ln1', x: 46, y: 134, w: 120, h: 22, label: ['RMSNorm'], color: colors.norm },
    { id: 'attn', x: 31, y: 172, w: 150, h: 44, label: ['GQA 联合注意力', '16Q / 8KV head'], color: colors.attn },
    { id: 'proj', x: 46, y: 230, w: 120, h: 22, label: ['输出投影'], color: colors.proj },
    { id: 'ln2', x: 46, y: 284, w: 120, h: 22, label: ['RMSNorm'], color: colors.norm },
    { id: 'mlp', x: 46, y: 320, w: 120, h: 30, label: ['MLP (SwiGLU)'], color: colors.mlp },
    { id: 'hstar', x: 88, y: 444, w: 36, h: 28, label: ['h*'], color: colors.harness, round: true },
    { id: 'knack', x: 6, y: 500, w: 140, h: 44, label: ['knack 头', '17 类 softmax'], color: colors.head },
    { id: 'done', x: 152, y: 500, w: 84, h: 44, label: ['done 头', '2 类'], color: colors.done },
    { id: 'exIn', x: 214, y: 104, w: 86, h: 52, label: ['后缀 17 token', '+ 噪声 x1', '宽度 768'], color: colors.expert },
    { id: 'denoiser', x: 214, y: 172, w: 86, h: 44, label: ['动作专家', '逐层联合注意力'], color: colors.expert },
    { id: 'vt', x: 214, y: 236, w: 86, h: 24, label: ['速度 v'], color: colors.expert },
    { id: 'euler', x: 214, y: 276, w: 86, h: 24, label: ['Euler: x1 − v'], color: colors.expert },
    { id: 'chunk', x: 214, y: 316, w: 86, h: 36, label: ['16×23', '动作块'], color: colors.expert },
    { id: 'harness', x: 6, y: 570, w: 294, h: 30, label: ['Harness：done→播完→刹车→切换'], color: colors.harness },
];

const plusNodes = [{ id: 'add1', x: 106, y: 268 }, { id: 'add2', x: 106, y: 372 }];

function findNode(id: string): BoundingBox3d | null {
    let n = nodes.find(a => a.id === id);
    if (n) {
        return new BoundingBox3d(new Vec3(n.x, n.y), new Vec3(n.x + n.w, n.y + n.h));
    }
    let p = plusNodes.find(a => a.id === id);
    if (p) {
        return new BoundingBox3d(new Vec3(p.x - 8, p.y - 8), new Vec3(p.x + 8, p.y + 8));
    }
    return null;
}

export const TocDiagram: React.FC<{
    activePhase: Phase | null;
    onEnterPhase?: (phase: Phase) => void;
}> = ({ activePhase, onEnterPhase }) => {
    let [entryManager] = useState(() => new EntryManager());
    let [diagramEl, setDiagramEl] = useState<SVGElement | null>(null);
    let progState = useProgramState();
    // re-render once mounted so the leader lines can measure the TOC entries
    let [, setMounted] = useState(false);
    useLayoutEffect(() => {
        setMounted(true);
    }, []);

    let [hoverId, setHoverId] = useState<Phase | null>(null);

    let setHover = useCallback((id: Phase, hover: boolean) => {
         setHoverId(hover ? id : null);
    }, []);

    let [activeId, setActiveId] = useState<Phase | null>(null);

    let setActive = useCallback((id: Phase, active: boolean) => {
        jumpToPhase(progState.walkthrough, id);
        onEnterPhase?.(id);
        setActiveId(active ? id : null);
    }, [progState.walkthrough, onEnterPhase]);

    activeId = activePhase ?? activeId;

    let entries: IEntryInfo[] = [];
    let entryGroups: IEntryGroup[] = [];

    function makeEntry(id: Phase, title: string, ids: string[] = [], groupIds: boolean = false) {
        let entry = { id, title, ids, groupIds };
        entries.push(entry);
        entryGroups[entryGroups.length - 1].entries.push(entry);
    }

    entryGroups.push({ groupName: '概览', entries: [] });
    makeEntry(Phase.Intro_Intro, '总览', []);
    makeEntry(Phase.Intro_Prelim, '预备知识', ['inCam', 'inLang', 'inState'], true);

    entryGroups.push({ groupName: '主干：一层展开', entries: [] });
    makeEntry(Phase.Input_Detail_Embedding, '输入嵌入', ['inCam', 'inLang', 'inState', 'embed'], true);
    makeEntry(Phase.Input_Detail_LayerNorm, 'RMSNorm', ['ln1', 'ln2']);
    makeEntry(Phase.Input_Detail_SelfAttention, 'GQA 联合注意力', ['attn']);
    makeEntry(Phase.Input_Detail_Softmax, '前缀 Softmax', ['attn']);
    makeEntry(Phase.Input_Detail_Projection, '输出投影与残差', ['proj', 'add1'], true);
    makeEntry(Phase.Input_Detail_Mlp, 'MLP (SwiGLU)', ['mlp', 'add2'], true);
    makeEntry(Phase.Input_Detail_Transformer, '28 层主干堆叠', ['backbone']);
    makeEntry(Phase.Input_Detail_Output, 'h* 答案边界读出', ['hstar']);

    entryGroups.push({ groupName: '输出头', entries: [] });
    makeEntry(Phase.Head_Knack, 'knack 分类头', ['knack']);
    makeEntry(Phase.Head_Done, 'done 头', ['done']);
    makeEntry(Phase.Harness_Law, 'Harness 法则', ['harness']);

    entryGroups.push({ groupName: 'DiT 动作专家与流匹配', entries: [] });
    makeEntry(Phase.D0_Towers, 'D0 两座塔', ['exIn', 'denoiser'], true);
    makeEntry(Phase.D1_SuffixInput, 'D1 后缀输入嵌入', ['exIn'], true);
    makeEntry(Phase.D2_AdaNorm, 'D2 AdaRMSNorm', ['denoiser'], true);
    makeEntry(Phase.D3_JointAttn, 'D3 联合注意力', ['denoiser'], true);
    makeEntry(Phase.D4_Projection, 'D4 输出投影与残差', ['denoiser'], true);
    makeEntry(Phase.D5_Mlp, 'D5 MLP (SwiGLU)', ['denoiser'], true);
    makeEntry(Phase.D6_Stack, 'D6 28 层堆叠与输出', ['denoiser', 'vt'], true);
    makeEntry(Phase.D7_Euler, 'D7 10 步 Euler 与动作块', ['vt', 'euler', 'chunk'], true);

    let focusEntry = entries.find(e => e.id === (hoverId ?? activeId));
    let hasFocus = isNotNil(hoverId) || isNotNil(activePhase);

    function opacityOf(id: string): number {
        if (!hasFocus || !focusEntry || focusEntry.ids.length === 0) {
            return 1.0;
        }
        return focusEntry.ids.includes(id) ? 1.0 : 0.4;
    }

    function arrowHead(x: number, y: number, dir: 'down' | 'left' | 'right' | 'up', key: string, color = colors.line) {
        let d = {
            down: `M${x - 3},${y - 5} L${x},${y} L${x + 3},${y - 5}Z`,
            up: `M${x - 3},${y + 5} L${x},${y} L${x + 3},${y + 5}Z`,
            left: `M${x + 5},${y - 3} L${x},${y} L${x + 5},${y + 3}Z`,
            right: `M${x - 5},${y - 3} L${x},${y} L${x - 5},${y + 3}Z`,
        }[dir];
        return <path key={key} d={d} fill={color} />;
    }

    function vArrow(x: number, y0: number, y1: number, key: string, color = colors.line) {
        return <g key={key}>
            <line x1={x} y1={y0} x2={x} y2={y1 - 3} stroke={color} strokeWidth={1.5} />
            {arrowHead(x, y1, 'down', key + 'h', color)}
        </g>;
    }

    function renderNode(n: INode) {
        let cx = n.x + n.w / 2;
        let lineH = 12;
        let top = n.y + n.h / 2 - ((n.label.length - 1) * lineH) / 2 + 4;
        let fontSize = n.label.length > 1 ? 10.5 : 11.5;
        return <g key={n.id} opacity={opacityOf(n.id)}>
            <rect x={n.x} y={n.y} width={n.w} height={n.h} rx={n.round ? n.h / 2 : 4} ry={n.round ? n.h / 2 : 4}
                fill={n.color} fillOpacity={n.dashed ? 0.06 : 0.2} stroke={n.color} strokeWidth={1.2}
                strokeDasharray={n.dashed ? '3 3' : undefined} />
            {n.label.map((l, i) => <text key={i} x={cx} y={top + i * lineH} fontSize={fontSize}
                textAnchor='middle' fill={colors.text}>{l}</text>)}
        </g>;
    }

    function renderPlus(p: { id: string, x: number, y: number }) {
        return <g key={p.id} opacity={opacityOf(p.id)}>
            <circle cx={p.x} cy={p.y} r={7} stroke={colors.line} fill='#1b2433' />
            <line x1={p.x - 4} y1={p.y} x2={p.x + 4} y2={p.y} stroke={colors.text} />
            <line x1={p.x} y1={p.y - 4} x2={p.x} y2={p.y + 4} stroke={colors.text} />
        </g>;
    }

    function renderConnections() {
        let ex = colors.expert;
        return <>
            {[48, 156, 252].map((x, i) => vArrow(x, 48, 62, 'in' + i))}
            {vArrow(106, 90, 134, 'e-ln1')}
            {vArrow(106, 156, 172, 'ln1-attn')}
            {vArrow(106, 216, 230, 'attn-proj')}
            {vArrow(106, 252, 261, 'proj-add1')}
            {vArrow(106, 275, 284, 'add1-ln2')}
            {vArrow(106, 306, 320, 'ln2-mlp')}
            {vArrow(106, 350, 365, 'mlp-add2')}
            {vArrow(106, 379, 396, 'add2-dots', colors.line)}
            <text x={106} y={410} fontSize={10.5} textAnchor='middle' fill={colors.text} opacity={0.8}>… 重复 ×28 层 …</text>
            {vArrow(106, 434, 444, 'bb-hstar', colors.harness)}
            <text x={12} y={120} fontSize={11.5} fill={colors.text}>Qwen3-VL ×28</text>
            <path d='M106,122 L192,122 L192,268 L116,268' fill='none' stroke={colors.line} strokeWidth={1.2} />
            {arrowHead(114, 268, 'left', 'res1')}
            <path d='M106,278 L192,278 L192,372 L116,372' fill='none' stroke={colors.line} strokeWidth={1.2} />
            {arrowHead(114, 372, 'left', 'res2')}
            <line x1={181} y1={194} x2={214} y2={194} stroke={ex} strokeWidth={1.5} strokeDasharray='3 2' />
            {arrowHead(181, 194, 'left', 'jl', ex)}
            {arrowHead(214, 194, 'right', 'jr', ex)}
            {vArrow(257, 156, 172, 'ex1', ex)}
            {vArrow(257, 216, 236, 'ex2', ex)}
            {vArrow(257, 260, 276, 'ex3', ex)}
            {vArrow(257, 300, 316, 'ex4', ex)}
            <path d='M106,472 L106,486 M76,486 L194,486 M76,486 L76,500 M194,486 L194,500' fill='none' stroke={colors.line} strokeWidth={1.2} />
            {arrowHead(76, 500, 'down', 'k')}
            {arrowHead(194, 500, 'down', 'd')}
            {vArrow(194, 544, 570, 'd-h', colors.harness)}
            {vArrow(257, 352, 570, 'c-h', ex)}
        </>;
    }

    function getFocusBounds(entry: IEntryInfo): BoundingBox3d[] {
        let idBounds = entry.ids.map(findNode).filter(isNotNil);
        if (entry.groupIds && idBounds.length > 0) {
            let box = new BoundingBox3d();
            idBounds.forEach(b => box.combineInPlace(b));
            return [box];
        }
        return idBounds;
    }

    function renderFocusBoxes() {
        return focusEntry ? getFocusBounds(focusEntry).map((bb, i) => {
            let tl = bb.min.sub(new Vec3(3, 3));
            let br = bb.max.add(new Vec3(3, 3));
            return <rect key={i} x={tl.x} y={tl.y} width={br.x - tl.x} height={br.y - tl.y} rx={5}
                fill={colors.focus} fillOpacity={0.08} stroke={colors.focus} strokeWidth={1.5} strokeDasharray='6 3'
                style={{ filter: `drop-shadow(0px 0px 4px ${colors.focus})` }} />;
        }) : [];
    }

    function renderTocToDigramLines() {
        if (!diagramEl) {
            return null;
        }

        let svgRect = diagramEl.getBoundingClientRect();
        let scale = svgRect.width / SVG_W || 1;
        let offsetLeft = BUS_X;
        let result: React.ReactNode[] = [];
        let i = 0;

        for (let entry of entries) {
            let tocEl = entryManager.entries.get(entry.id);
            if (!tocEl) {
                continue;
            }
            let tocRect = tocEl.getBoundingClientRect();
            let endPt = new Vec3((tocRect.left - svgRect.left) / scale - 4, (tocRect.top - svgRect.top + tocRect.height / 2) / scale);
            let allBounds = getFocusBounds(entry);
            if (allBounds.length === 0) {
                continue;
            }

            let yPts = [...allBounds.map(b => b.center().y), endPt.y];
            let midX = offsetLeft;
            let mainPath = `M${midX},${Math.min(...yPts)} L${midX},${Math.max(...yPts)} M${midX},${endPt.y} L${endPt.x},${endPt.y}`;
            let isFocusEntry = hoverId ? entry.id === hoverId : entry.id === activeId;
            if (isFocusEntry) {
                let opts = { stroke: colors.focus, strokeWidth: 1, fill: 'none' };
                result.push(<path key={i++} {...opts} d={mainPath} />);
                for (let bound of allBounds) {
                    let startPt = new Vec3(bound.max.x + 6, bound.center().y);
                    result.push(<path key={i++} {...opts} d={`M${startPt.x},${startPt.y} L${midX},${startPt.y}`} />);
                }
            }
            offsetLeft += 3;
        }

        return result;
    }

    return <div>
        <div className={s.tocDiagram}>
            <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} ref={setDiagramEl} style={{ flex: '0 1 ' + SVG_W + 'px', minWidth: 0, width: SVG_W, maxWidth: '100%', height: 'auto', overflow: 'visible' }}>
                {nodes.map(renderNode)}
                {plusNodes.map(renderPlus)}
                {renderConnections()}
                {renderFocusBoxes()}
                {renderTocToDigramLines()}
            </svg>
            <div className={s.toc}>
                {entryGroups.map((group, i) => {
                    return <React.Fragment key={i}>
                        <div className={s.tocGroupTitle}>{group.groupName}</div>
                        {group.entries.map((entry, j) => {
                            return <MenuEntry
                                key={j}
                                entryManager={entryManager}
                                title={entry.title}
                                id={entry.id}
                                active={entry.id === activeId}
                                hover={entry.id === hoverId}
                                setHover={setHover}
                                setActive={setActive}
                            />;
                        })}
                    </React.Fragment>;
                })}
            </div>
        </div>
    </div>;
};

class EntryManager {
    subscriptions = new Subscriptions();
    entries = new Map<Phase, HTMLDivElement | null>();
    setEl(id: Phase, ref: HTMLDivElement | null) {
        if (ref) {
            this.entries.set(id, ref);
            return () => {
                this.entries.delete(id);
            }
        } else {
            this.entries.delete(id);
        }
    }
}

const MenuEntry: React.FC<{
    entryManager: EntryManager,
    title: string,
    id: Phase,
    active: boolean,
    hover: boolean,
    setHover: (id: Phase, hover: boolean) => void,
    setActive: (id: Phase, active: boolean) => void,
}> = memo(function MenuEntry({ entryManager, id, title, active, hover, setHover, setActive }) {

    let setDivRef = useCallback((div: HTMLDivElement | null) => {
        entryManager.setEl(id, div);
    }, [entryManager, id]);


    function handleClick() {
        setActive(id, !active);
    }

    function handleHover(hover: boolean) {
        setHover(id, hover);
    }

    return <div
        ref={setDivRef}
        className={clsx(s.menuEntry, active && s.active, hover && s.hover)}
        onClick={handleClick}
        onMouseEnter={() => handleHover(true)}
        onMouseLeave={() => handleHover(false)}
    >
        {title}
    </div>;
});
