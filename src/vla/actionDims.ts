import { IK005bConfig } from "./VlaModel";

// The 23 native action dims (behavior_challenge data/schema.py ACTION_DIM; names exported in
// config.action_dim_names). Dims 0-2 are base velocities, 3-22 joint/gripper positions.
interface IActionGroup { label: string; start: number; end: number; color: string; }

export const actionGroups: IActionGroup[] = [
    { label: '底盘速度', start: 0, end: 3, color: '#f59e0b' },
    { label: '躯干', start: 3, end: 7, color: '#a78bfa' },
    { label: '左臂', start: 7, end: 14, color: '#38bdf8' },
    { label: '左夹爪', start: 14, end: 15, color: '#22d3ee' },
    { label: '右臂', start: 15, end: 22, color: '#4ade80' },
    { label: '右夹爪', start: 22, end: 23, color: '#86efac' },
];

// bounds99 inverse (vla/prepare_input.py bounds99_unnormalize): normalized → robot units
export function unnormalizeAction(cfg: IK005bConfig, d: number, v: number) {
    let q01 = cfg.action_q01[d], q99 = cfg.action_q99[d];
    return (v + 1) / 2 * (q99 - q01 + 1e-6) + q01;
}

// robot-unit value → bounds99-normalized
export function normalizeAction(cfg: IK005bConfig, d: number, v: number) {
    let q01 = cfg.action_q01[d], q99 = cfg.action_q99[d];
    return (v - q01) / (q99 - q01 + 1e-6) * 2 - 1;
}

export function fmtSigned(v: number, digits = 3) {
    return (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(digits);
}

function gripperState(robotVal: number) {
    return robotVal < 0 ? '闭合' : '张开';
}

// "name = normalized (robot units)" for each dim of one chunk row, grouped
export function describeActionRow(cfg: IK005bConfig, chunk: ArrayLike<number>, row: number) {
    let D = cfg.action_dim;
    return actionGroups.map(g => {
        let parts: string[] = [];
        for (let d = g.start; d < g.end; d++) {
            let v = chunk[row * D + d];
            let r = unnormalizeAction(cfg, d, v);
            let extra = g.label.endsWith('夹爪') ? `，${gripperState(r)}` : '';
            parts.push(`${d} ${cfg.action_dim_names[d]} ${fmtSigned(v)}（${fmtSigned(r)}${extra}）`);
        }
        return `${g.label}：${parts.join('；')}`;
    });
}
