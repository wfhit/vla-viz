import React from 'react';

// Path-continuity ledgers (public/<prefix>-trace.json, tools/trace_common.py): one real object is
// followed link by link; each link shows the number it takes in (yellow = the previous link's output)
// and the number it gives out (bold), so the reader can reconcile the whole chain.

export type ITerm = [number, number, number, number]; // [i, w_i, x_i, w_i·x_i]
export interface ITerms { n: number; terms: ITerm[]; rest: number; sum: number }
export interface ILin extends ITerms { k: 'lin'; name: string; out: string; b: number | null; y: number; xnames?: [number, string][] }
export interface IAct { k: 'act'; name: string; x: number; y: number; out: string }
export interface IAdd { k: 'add'; name: string; a: number; b: number; y: number; out: string }
export interface INorm { k: 'norm'; name: string; kind: 'rms' | 'ln'; x: number; mu: number; ms: number; eps: number; rsqrt: number; mult: number; add: number; y: number; out: string }
export type ILink = ILin | IAct | IAdd | INorm;

export interface IQkPath {
    name: string;
    raw: ILin[];
    norm?: { mean: number; eps: number; rsqrt: number; g: number[]; y: number[] };
    rope: { d: number; pair: [number, number]; pre: number[]; cos: number; sin: number; angle: number; post: number[] };
}

export interface ITrace {
    model: string;
    p: number;
    win_idx: number;
    embed: { id: number; row_c: number; mult: number; mult_name: string; x_c: number };
    rms: { c: number; C: number; x_c: number; sumsq: ITerms; mean: number; eps: number; rsqrt: number; gain: string; g_c: number; y_c: number };
    attn: {
        p: number; head: number; d: number; c: number; D: number; j: number;
        q: IQkPath; k: IQkPath; k_y_c: number;
        dot: ITerms; scale: number; s_j: number; max: number; max_at: number; Z: number; n_keys: number; p_j: number;
        top: [number, number, number][]; v_j: number; o: ITerms; o_slot: number; o_proj: ILin; x_c: number; resid: number;
    };
    euler: {
        step: number; t: number; dt: number; i: number; tok: number; j: number; c: number; x: number;
        enc: ILink[]; layers: [number, number, number][]; dec: ILink[]; v: number; x_next: number;
    };
}

export function fmt(v: number | null | undefined) {
    if (v === null || v === undefined || !isFinite(v)) return '…';
    let a = Math.abs(v);
    let s = a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(3) : v.toPrecision(4);
    if (s.includes('.') && !s.includes('e')) s = s.replace(/0+$/, '').replace(/\.$/, '');
    return s.replace('-', '−');
}

/** ASCII form for 3D tags. */
export const fmtA = (v: number | null | undefined) => fmt(v).replace('−', '-');

/** A number in the chain: `carry` = it is the previous link's output; `out` = this link's output. */
export const N: React.FC<{ v: number | null | undefined; carry?: boolean; out?: boolean }> = ({ v, carry, out }) =>
    <span className={'font-mono whitespace-nowrap ' + (carry ? 'text-yellow-300 border border-yellow-400/70 rounded px-0.5 ' : '') + (out ? 'text-green-300 font-bold underline decoration-green-400 ' : '')}
        title={carry ? '上一步的输出' : out ? '本步输出 → 下一步的输入' : undefined}>{fmt(v)}</span>;

export const Box: React.FC<{ children?: React.ReactNode }> = ({ children }) =>
    <div className='my-2 px-2 py-1 border-l-4 border-green-500 bg-slate-800/60 text-sm'>{children}</div>;

/** Σ w_i·x_i shown as the carried term + the largest terms + the remainder = the real sum. */
export const TermsTable: React.FC<{ t: ITerms; wName: string; xName: (i: number) => string; carryFirst?: boolean; square?: boolean; xHead?: string; carryW?: boolean; carryRows?: number }> = ({ t, wName, xName, carryFirst, square, xHead = 'x_i', carryW, carryRows = 1 }) => {
    let shown = t.terms.length;
    return <table className='font-mono text-xs my-1'>
        <thead><tr className='text-slate-400'>
            <td className='pr-3'>i</td>{!square && <td className='pr-3'>{wName}</td>}<td className='pr-3'>{xHead}</td><td>{square ? `${xHead}²` : `${wName}·${xHead}`}</td>
        </tr></thead>
        <tbody>
            {t.terms.map(([i, w, x, wx], r) => <tr key={r}>
                <td className='pr-3'>{xName(i)}</td>
                {!square && <td className='pr-3'><N v={w} carry={carryW && r < carryRows} /></td>}
                <td className='pr-3'><N v={x} carry={carryFirst && r < carryRows} /></td>
                <td><N v={wx} /></td>
            </tr>)}
            {t.n > shown && <tr className='text-slate-400'><td colSpan={square ? 2 : 3} className='pr-3'>其余 {t.n - shown} 项之和</td><td><N v={t.rest} /></td></tr>}
            <tr className='border-t border-slate-500'><td colSpan={square ? 2 : 3} className='pr-3'>Σ（{t.n} 项）</td><td><N v={t.sum} /></td></tr>
        </tbody>
    </table>;
};

function segName(xnames: [number, string][] | undefined, inName: string, i: number) {
    if (!xnames) return `${inName}[${i}]`;
    let seg = [...xnames].reverse().find(([o]) => i >= o)!;
    return `${seg[1]}[${i - seg[0]}]`;
}

export const LinLedger: React.FC<{ l: ILin; inName: string; carry?: boolean }> = ({ l, inName, carry = true }) => <Box>
    <div><span className='font-mono'>{l.out}</span> = Σ<sub>i</sub> {l.name}[·, i] · {l.xnames ? l.xnames.map(s => s[1]).join(' ‖ ') : inName}[i]{l.b !== null ? ' + b' : ''}</div>
    <TermsTable t={l} wName='w_i' xName={i => segName(l.xnames, inName, i)} carryFirst={carry} />
    <div>{l.b !== null ? <><N v={l.sum} /> + b (<N v={l.b} />) = </> : '= '}<N v={l.y} out /></div>
</Box>;

export const LinkLedger: React.FC<{ l: ILink; inName: string; carry?: boolean }> = ({ l, inName, carry = true }) => {
    if (l.k === 'lin') return <LinLedger l={l} inName={inName} carry={carry} />;
    if (l.k === 'act') return <Box><span className='font-mono'>{l.out}</span> = {l.name}(<N v={l.x} carry={carry} />) = <N v={l.y} out /></Box>;
    if (l.k === 'add') return <Box><span className='font-mono'>{l.out}</span> = <N v={l.a} carry={carry} /> + {l.name.replace(/^\+/, '')} (<N v={l.b} />) = <N v={l.y} out /></Box>;
    let ln = l.kind === 'ln';
    return <Box>
        <div><span className='font-mono'>{l.out}</span> = {ln ? '(x − μ)' : 'x'} · rsqrt(mean({ln ? '(x − μ)²' : 'x²'}) + ε) · m{l.add !== 0 ? ' + a' : ''}</div>
        <div>= {ln ? <>(<N v={l.x} carry={carry} /> − <N v={l.mu} />)</> : <N v={l.x} carry={carry} />} · rsqrt(<N v={l.ms} /> + ε) · <N v={l.mult} />{l.add !== 0 ? <> + <N v={l.add} /></> : null}</div>
        <div>= {ln ? <N v={l.x - l.mu} /> : <N v={l.x} />} · <N v={l.rsqrt} /> · <N v={l.mult} />{l.add !== 0 ? <> + <N v={l.add} /></> : null} = <N v={l.y} out /></div>
    </Box>;
};

/** The chain of links, each consuming the previous output. */
export const ChainLedger: React.FC<{ links: ILink[]; firstIn: string }> = ({ links, firstIn }) => <>
    {links.map((l, i) => <LinkLedger key={i} l={l} inName={i === 0 ? firstIn : (links[i - 1].out.replace(/\[\d+\]$/, ''))} />)}
</>;

/** q/k path: the two raw projections of a RoPE pair, optional q_norm/k_norm, then the rotation. */
export const QkLedger: React.FC<{ p: IQkPath; inName: string; carryIn: boolean; normName: string }> = ({ p, inName, carryIn, normName }) => {
    let [d0, d1] = p.rope.pair;
    let pre = p.rope.pre;
    return <>
        {p.raw.map((l, i) => <LinLedger key={i} l={l} inName={inName} carry={carryIn} />)}
        {p.norm && <Box>
            <div>{normName}（对 {p.name} 的全部 head 维做 RMSNorm）：rsqrt(mean(x²) + ε) = rsqrt(<N v={p.norm.mean} /> + ε) = <N v={p.norm.rsqrt} /></div>
            <div>{p.name}[{d0}] = <N v={p.raw[0].y} carry /> · <N v={p.norm.rsqrt} /> · g (<N v={p.norm.g[0]} />) = <N v={p.norm.y[0]} out /></div>
            <div>{p.name}[{d1}] = <N v={p.raw[1].y} carry /> · <N v={p.norm.rsqrt} /> · g (<N v={p.norm.g[1]} />) = <N v={p.norm.y[1]} out /></div>
        </Box>}
        <Box>
            <div>RoPE 把第 {d0} 与第 {d1} 维当作一对坐标，按该 token 的位置旋转角度 θ = <N v={p.rope.angle} /> rad（cos <N v={p.rope.cos} />，sin <N v={p.rope.sin} />）：</div>
            <div>{p.name}[{d0}] = <N v={pre[0]} carry /> · <N v={p.rope.cos} /> − <N v={pre[1]} carry /> · <N v={p.rope.sin} /> = <N v={p.rope.post[0]} out /></div>
            <div className='text-slate-400'>（配对维 {p.name}[{d1}] = <N v={pre[1]} /> · cos + <N v={pre[0]} /> · sin = <N v={p.rope.post[1]} />）</div>
        </Box>
    </>;
};

/** One Euler step for one element of the action chunk: x → input chain → every layer's residual → output chain → v → x_next. */
export const EulerLedger: React.FC<{ e: ITrace['euler']; layerName: string }> = ({ e, layerName }) => {
    let nL = e.layers.length;
    return <>
        <Box>起点：x_t[{e.i}][{e.j}] = <N v={e.x} out />（t = {e.t}，纯噪声）</Box>
        <ChainLedger links={e.enc} firstIn={`x_t[${e.i}]`} />
        <Box>
            <div>通道 {e.c} 沿残差流穿过 {nL} 个{layerName}（每行的输入 = 上一行的输出）：</div>
            <table className='font-mono text-xs my-1'>
                <thead><tr className='text-slate-400'><td className='pr-3'>层</td><td className='pr-3'>输入</td><td className='pr-3'>+ 注意力</td><td>+ MLP</td></tr></thead>
                <tbody>{e.layers.map(([a, b, c], l) => <tr key={l}>
                    <td className='pr-3'>{l}</td><td className='pr-3'><N v={a} carry /></td><td className='pr-3'><N v={b} /></td><td><N v={c} out={l === nL - 1} /></td>
                </tr>)}</tbody>
            </table>
        </Box>
        <ChainLedger links={e.dec} firstIn={`h${nL}`} />
        <Box>Euler：x_next = x (<N v={e.x} carry />) + dt (<N v={e.dt} />) · v (<N v={e.v} carry />) = <N v={e.x_next} out /></Box>
    </>;
};
