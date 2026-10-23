import React from 'react';
import clsx from 'clsx';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import { faBackwardStep, faForwardStep, faPause, faPlay } from '@fortawesome/free-solid-svg-icons';
import { useProgramState } from '../Sidebar';
import { jumpPhase } from '../Commentary';
import { phaseToGroup } from '../walkthrough/Walkthrough';
import { isCommentary } from '../walkthrough/WalkthroughTools';
import { ICommentary } from '../walkthrough/WalkthroughTools';
import s from './PlayBar.module.scss';

const speeds = [0.5, 1, 2, 4];

function commentaryToText(c: ICommentary) {
    let out = '';
    for (let i = 0; i < c.strings.length; i++) {
        out += c.strings[i];
        let v = c.values[i];
        if (v && typeof v === 'object' && typeof v.str === 'string') {
            out += v.str;
        }
    }
    return out.replace(/[_*]/g, '').replace(/\s+/g, ' ').trim();
}

export const PlayBar: React.FC = () => {
    let progState = useProgramState();
    let wt = progState.walkthrough;
    let group = phaseToGroup(wt);
    let phase = group.phases.find(p => p.id === wt.phase)!;

    let narration = '';
    for (let t of wt.times) {
        if (isCommentary(t) && t.start <= wt.time + 1e-6) {
            let text = commentaryToText(t);
            if (text) {
                narration = text;
            }
        }
    }
    // show the last sentence group so the single line stays short
    let sentences = narration.split(/(?<=[。！？!?])/).map(x => x.trim()).filter(Boolean);
    let shown = sentences.length > 2 ? sentences.slice(-2).join('') : narration;

    let allPhases = wt.phaseList.flatMap(g => g.phases);
    let stepIdx = allPhases.findIndex(p => p.id === wt.phase);

    function step(delta: number) {
        jumpPhase(wt, delta);
        progState.markDirty();
    }

    function togglePlay() {
        if (wt.time >= wt.phaseLength && wt.phaseLength > 0) {
            jumpPhase(wt, 1);
        } else {
            wt.running = !wt.running;
        }
        progState.markDirty();
    }

    function setSpeed(v: number) {
        wt.speed = v;
        progState.markDirty();
    }

    return <div className={s.wrap} onPointerDown={ev => ev.stopPropagation()}>
        <div className={s.narration}>
            <span className={s.stepTag}>步骤 {stepIdx + 1}/{allPhases.length} · {phase.title}</span>
            <span className={s.text}>{shown || '点击播放开始本步骤的讲解'}</span>
        </div>
        <div className={s.controls}>
            <button className={s.btn} onClick={() => step(-1)} title="上一步" aria-label="上一步">
                <FontAwesomeIcon icon={faBackwardStep} />
            </button>
            <button className={clsx(s.btn, s.play)} onClick={togglePlay} title="播放/暂停" aria-label="播放/暂停">
                <FontAwesomeIcon icon={wt.running ? faPause : faPlay} />
            </button>
            <button className={s.btn} onClick={() => step(1)} title="下一步" aria-label="下一步">
                <FontAwesomeIcon icon={faForwardStep} />
            </button>
            <div className={s.speeds}>
                {speeds.map(v => <button key={v} className={clsx(s.speed, wt.speed === v && s.active)} onClick={() => setSpeed(v)}>{v}×</button>)}
            </div>
        </div>
    </div>;
};
