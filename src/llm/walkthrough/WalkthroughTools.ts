import { IBlkDef, IGptModelLayout } from "../GptModelLayout";
import { IRenderState } from "../render/modelRender";
import { clamp } from "@/src/utils/data";
import { measureTextWidth } from "../render/fontRender";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { IWalkthrough, Phase, PhaseGroup } from "./Walkthrough";
import { IProgramState } from "../Program";
import { ICameraPos } from "../Camera";

export interface IWalkthroughArgs {
    state: IProgramState;
    layout: IGptModelLayout;
    walkthrough: IWalkthrough;
    tools: ReturnType<typeof phaseTools>;
}

export function embed(fc: React.FC) {
    return { insert: () => fc };
}

export function phaseTools(state: IProgramState) {
    let phaseState = state.walkthrough;

    function c_str(str: string, duration: number = 0.3, style: DimStyle = DimStyle.T) {
        return { str, duration, start: 0, t: 0.0, color: dimStyleColor(style) };
    }

    function c_blockRef(str: string, blk: IBlkDef | IBlkDef[], style?: DimStyle) {
        let firstBlk = Array.isArray(blk) ? blk[0] : blk;
        style ??= firstBlk.t === 'i' ? DimStyle.Intermediates : firstBlk.t === "w" ? DimStyle.Weights : DimStyle.Aggregates;
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), blk };
    }

    function c_dimRef(str: string, style: DimStyle) {
        return { str, duration: 0, start: 0, t: 0.0, color: dimStyleColor(style), dim: style };
    }

    function atTime(start: number, duration?: number, wait?: number): ITimeInfo {
        return createAtTime(phaseState, start, duration, wait);
    }

    function atEvent(evt: { str: string, duration: number, t: number, start: number }): ITimeInfo {
        return atTime(evt.start, evt.duration);
    }

    function afterTime(prev: ITimeInfo | null, duration: number, wait?: number): ITimeInfo {
        prev = prev ?? phaseState.times[phaseState.times.length - 1];
        return atTime(prev.start + prev.duration + prev.wait, duration, wait);
    }

    function cleanup(t: ITimeInfo, times: ITimeInfo[] = phaseState.times) {
        if (t.t > 0.0) {
            for (let prevTime of times) {
                prevTime.t = 1.0 - t.t;
                if (t.t >= 1.0) {
                    prevTime.active = false;
                }
            }
        }
    }

    function breakAfter(evt?: ITimeInfo) {
        evt = evt ?? phaseState.times[phaseState.times.length - 1];
        if (!evt) {
            return;
        }
        let breakEvt = afterTime(evt, 0.001);
        if (phaseState.running && phaseState.time - phaseState.dt < breakEvt.start && phaseState.time >= breakEvt.start) {
            phaseState.running = false;
            phaseState.speed = 1.0;
            phaseState.time = breakEvt.start + breakEvt.duration;
        }
        breakEvt.isBreak = true;
    }

    function commentary(stringsArr: TemplateStringsArray, ...values: any[]) {
        return writeCommentary(state, null, stringsArr, ...values);
    }

    function commentaryPara(c: ICommentaryRes) {
        return (stringsArr: TemplateStringsArray, ...values: any[]) => {
            return writeCommentary(state, c, stringsArr, ...values);
        };
    }

    return { atTime, atEvent, afterTime, cleanup, commentary, commentaryPara, c_str, c_blockRef, c_dimRef, breakAfter };
}

function createAtTime(wt: IWalkthrough, start: number, duration?: number, wait?: number): ITimeInfo {
    duration = duration ?? 0;
    wait = wait ?? 0;
    let info: ITimeInfo = {
        name: '',
        start,
        duration,
        wait,
        t: duration === 0 ? (wt.time > start ? 1 : 0) : clamp((wt.time - start) / duration, 0, 1),
        active: wt.time > start,
    };
    wt.times.push(info);
    wt.phaseLength = Math.max(wt.phaseLength, start + duration + wait);
    return info;
}

export function eventEndTime(evt: ITimeInfo) {
    return evt.start + evt.duration + evt.wait;
}

export interface ICommentary extends ITimeInfo {
    strings: readonly string[];
    values: any[];
}

export function isCommentary(evt: ITimeInfo): evt is ICommentary {
    return 'strings' in evt;
}

export function commentary(wt: IWalkthrough, prev?: ITimeInfo | null, duration?: number) {
    return (stringsArr: TemplateStringsArray, ...values: any[]): ICommentary => {
        let t = 0;
        prev = prev ?? wt.times[wt.times.length - 1];

        if (prev) {
            t = prev.start + prev.duration + prev.wait;
        }

        let commentaryT = createAtTime(wt, prev ? eventEndTime(prev) : 0, duration ?? 0.2);

        // plain strings / numbers are spliced into the text so they render (and take part in markup parsing)
        let strings: string[] = [stringsArr[0]];
        let objValues: any[] = [];
        values.forEach((v, i) => {
            if (v === null || v === undefined || typeof v !== 'object' && typeof v !== 'function') {
                strings[strings.length - 1] += String(v ?? '') + stringsArr[i + 1];
            } else {
                objValues.push(v);
                strings.push(stringsArr[i + 1]);
            }
        });

        let res: ICommentary = {
            ...commentaryT,
            strings,
            values: objValues,
        };

        wt.times[wt.times.length - 1] = res; // replace the time info with the commentary

        return res;
    }
}

export function writeCommentary(state: IProgramState, prev: ICommentaryRes | null, stringsArrRaw: TemplateStringsArray, ...values: any[]): ICommentaryRes {
    let t = prev?.duration ?? 0;
    let colNum = 0;
    let fontSize = 17;
    let maxWidth = 500;
    let charsPerSecond = 400;
    let lineHeight = fontSize * 1.2;

    let lineOffset = prev ? prev.lineOffset + lineHeight * 1.5 : 10;
    let stringsArr = stringsArrRaw.map(s => s.replace(/([ \n])+/g, ' '));

    for (let i = 0; i < values.length + 1; i++) {
        let str = stringsArr[i];

        t += str.length / charsPerSecond;

        if (i < values.length && 't' in values[i]) {
            // calculate the t value of this point
            values[i].start = t;
            t += values[i].duration;
        }
    }

    let targetT = state.walkthrough.time;

    function writeWord(str: string, tStart: number, tEnd: number, colOverride?: Vec4, fontOverride?: string) {

        while (str.startsWith('\n\n')) {
            lineOffset += lineHeight;
            colNum = 0;
            str = str.substring(2);
        }
        str = str.replace(/([ \n])+/g, ' ');

        let strToDraw = str;
        let nextOff = 0;
        let w = measureTextWidth(state.render.modelFontBuf, str, fontSize);
        if (colNum + w > maxWidth) {
            lineOffset += lineHeight;
            colNum = 0;
            strToDraw = str.trimStart();
            w = measureTextWidth(state.render.modelFontBuf, strToDraw, fontSize);
            if (w > maxWidth) {
                // ignore for now; single word longer than line: should break at the character level
            }
            nextOff = w;
        } else {
            nextOff = colNum + w;
        }

        let color = new Vec4(0.5, 0.5, 0.5, 1).mul(0.5);
        if (targetT > tStart) {
            let targetColor = colOverride ?? new Vec4(0.1, 0.1, 0.1, 1);
            // lerp is 0 at tStart, 1 at tEnd
            let x = clamp((targetT - tStart) / (tEnd - tStart), 0, 1);
            color = Vec4.lerp(color, targetColor, x);
        }
        // writeTextToBuffer(state.overlayFontBuf, strToDraw, color, 10 + colNum, lineOffset, fontSize, undefined, fontOverride);

        colNum = nextOff;
    }

    t = prev?.duration ?? 0;
    for (let i = 0; i < values.length + 1; i++) {
        let words = stringsArr[i].split(/(?=[ \n]+)/).filter(a => a !== ' ');

        for (let word of words) {
            let tEnd = t + word.length / charsPerSecond;
            writeWord(word, t, tEnd);
            t = tEnd;
        }

        if (i < values.length && 't' in values[i]) {
            let val = values[i];
            // calculate the t value of this point
            val.start = t;
            writeWord(values[i].str, t, val.color, val.fontFace);
            t += val.duration;
        }
    }

    let commentryT = createAtTime(state.walkthrough, 0, t, 0);

    let res: ICommentaryRes = {
        ...commentryT,
        stringsArr: stringsArrRaw,
        values,
        lineOffset,
        colNum,
        commentaryList: [],
    };
    res.commentaryList = [res];

    if (prev) {
        prev.lineOffset = lineOffset;
        prev.colNum = colNum;
        prev.duration = t;
        prev.commentaryList = [...prev.commentaryList, res];
    } else {
        state.walkthrough.commentary = res;
    }

    return res;
}

export interface ICommentaryRes extends ITimeInfo {
    stringsArr: TemplateStringsArray;
    values: any[];
    commentaryList: ICommentaryRes[];
    duration: number;
    lineOffset: number;
    colNum: number;
}

export interface ITimeInfo {
    name: string;
    start: number;
    duration: number;
    wait: number;

    // will change over the course of a phase: used to lerp
    t: number; // 0 - 1
    active: boolean;

    isBreak?: boolean;
}

function getPhaseTransitiveData(wt: IWalkthrough) {
    wt.phaseTransitiveData ??= {};
    return wt.phaseTransitiveData;
}

export function setInitialCamera(state: IProgramState, target: Vec3, rot: Vec3) {
    let wt = state.walkthrough;
    wt.cameraInitial = { angle: rot, center: target };

    let data = getPhaseTransitiveData(wt);

    if (wt.time === 0) {
        data.cameraSrc ??= { angle: state.camera.angle, center: state.camera.center };
        data.cameraT ??= 0;

        if (data.cameraT < 1) {
            let src = data.cameraSrc;
            let dest = wt.cameraInitial;
            // clamp so a slow frame can't overshoot 1 and leave the camera short of its target
            let t = data.cameraT = Math.min(1, data.cameraT + wt.viewDt / 1000 * 1.5);
            state.camera.angle = src.angle.lerp(dest.angle, t);
            state.camera.center = src.center.lerp(dest.center, t);
            wt.markDirty();
        }
    }
}

export function moveCameraTo(state: IProgramState, time: ITimeInfo, target: Vec3, rot: Vec3) {

    let wt = state.walkthrough;
    let phaseData = wt.phaseData.get(wt.phase);
    if (!phaseData) {
        wt.phaseData.set(wt.phase, phaseData = { cameraData: null });
    }
    if (!phaseData.cameraData) {
        phaseData.cameraData = new Map<number, ICameraPos>();
    }

    let prevTime = [...phaseData.cameraData.entries()].filter(([t, _]) => t < time.start).pop()?.[1];

    let camData = phaseData.cameraData.get(time.start);
    if (!camData) {
         phaseData.cameraData.set(time.start, camData = {
            initialCaptured: prevTime ? undefined : wt.cameraInitial ?? {
                angle: state.camera.angle,
                center: state.camera.center,
            },
            target: { angle: rot, center: target },
        });
    }

    // if we transition from before the ITimeInfo to the start of it, we capture the camera position
    // we store the camera position in a map, keyed by the ITimeInfo name
    // then we can use that position to lerp from its initial value to the target values


    // if we don't get to the start via running (e.g. clicking on a link), we use the camera position
    // of the last moveCameraTo call (so need to keep track of that!)

    // if (wt.running && wt.time - wt.dt < time.start && wt.time >= time.start && !prevTime && !camData.initialCaptured) {
    //     camData.initialCaptured = {
    //         angle: state.camera.angle,
    //         center: state.camera.center,
    //     };
    // }

    let src = prevTime?.target ?? wt.cameraInitial ?? camData.initialCaptured;

    let dest: ICameraPos = {
        center: target,
        angle: rot,
    };

    let isMoving = wt.running || wt.time !== wt.prevTime;
    let prevWasActive = wt.prevTime >= time.start && wt.prevTime <= time.start + time.duration;

    if (src && isMoving && (time.active || prevWasActive)) {
        let t = time.t;
        state.camera.angle = src.angle.lerp(dest.angle, t);
        state.camera.center = src.center.lerp(dest.center, t);
    }
}




export enum DimStyle {
    None,
    t,
    T,
    C,
    B,
    A,
    n_vocab,
    n_heads,
    n_layers,
    Token,
    TokenIdx,
    C4,
    Intermediates,
    Weights,
    Aggregates,
}

export function dimStyleColor(style: DimStyle) {
     switch (style) {
        case DimStyle.t:
        case DimStyle.T:
            return Vec4.fromHexColor('#359da8');
        case DimStyle.A:
            return Vec4.fromHexColor('#d368a4');
        case DimStyle.C:
        case DimStyle.C4:
            return Vec4.fromHexColor('#ce2983');
        case DimStyle.Token:
            return new Vec4(0.3, 0.7, 0.3, 1);
        case DimStyle.TokenIdx:
            return Vec4.fromHexColor('#1b495d');
        case DimStyle.n_vocab:
            return Vec4.fromHexColor('#7c3c8d'); // new Vec4(0.8, 0.6, 0.3, 1);
        case DimStyle.Intermediates:
            return Vec4.fromHexColor('#00ad00');
        case DimStyle.Weights:
            return Colors.Weights;
        case DimStyle.Aggregates:
            return Vec4.fromHexColor('#e3a300');
    }
    return new Vec4(0,0,0);
}

export function dimStyleText(style: DimStyle) {
    switch (style) {
        case DimStyle.TokenIdx: return 'Token Index';
        case DimStyle.C4: return 'C * 4';
        default: return DimStyle[style];
    }
}

export function dimStyleTextShort(style: DimStyle) {
    switch (style) {
        case DimStyle.B: return 'b';
        case DimStyle.T: return 't';
        case DimStyle.A: return 'a';
        case DimStyle.C: return 'c';
        case DimStyle.C4: return 'c';
        default: return DimStyle[style];
    }
}

export const Colors = {
    Weights: new Vec4(0.3, 0.3, 1.0),
    Intermediates: new Vec4(0.4, 0.8, 0.4),
    Aggregates: new Vec4(1.0, 0.8, 0.3), // a yellowish color

    Black: new Vec4(0, 0, 0),
}

export function hideFromBlock(state: IRenderState, layout: IGptModelLayout, targetBlk: IBlkDef) {
    let seen = false;
    for (let blk of layout.cubes) {
        if (!seen && blk === targetBlk) {
            seen = true;
        }
        seen && blk.t === 'i' && hideBlock(blk);
    }
    function hideBlock(b: IBlkDef) {
        if (b.access) {
            b.access.disable = true;
        }
        b.subs?.forEach(hideBlock);
    }
}

export interface IPhaseGroup {
    groupId: PhaseGroup;
    title: string;
    phases: IPhaseDef[];
}

export interface IPhaseDef {
    id: Phase;
    title: string;
}

// Camera that frames a set of blocks (layout coords) from a given orbit angle. The FOV is
// the fixed 40° of Camera.ts — only the distance (zoom) and look-at point are derived.
// `shiftX` moves the content left on screen by that fraction of its half-width (to leave
// room for a side panel).
export function frameCubes(cubes: (IBlkDef | { x: number, y: number, z: number, dx: number, dy: number, dz: number })[],
        angleX: number, angleY: number, opts: { fill?: number, aspect?: number, shiftX?: number } = {}): ICameraPos {
    let mn = new Vec3(Infinity, Infinity, Infinity), mx = new Vec3(-Infinity, -Infinity, -Infinity);
    for (let c of cubes) {
        if ('opacity' in c && c.opacity === 0) continue;
        mn = new Vec3(Math.min(mn.x, c.x), Math.min(mn.y, c.y), Math.min(mn.z, c.z));
        mx = new Vec3(Math.max(mx.x, c.x + c.dx), Math.max(mx.y, c.y + c.dy), Math.max(mx.z, c.z + c.dz));
    }
    let fill = opts.fill ?? 1.15;
    let aspect = opts.aspect ?? 1.0; // the 3D canvas is about square (the text panel takes the left half)
    let ax = angleX * Math.PI / 180, ay = angleY * Math.PI / 180;
    let hx = (mx.x - mn.x) / 2, hy = (mx.y - mn.y) / 2, hz = (mx.z - mn.z) / 2;
    // world = (x, −z, −y): extents seen across the screen / up the screen from this orbit angle
    let across = Math.abs(Math.sin(ax)) * hx + Math.abs(Math.cos(ax)) * hz;
    let depth = Math.abs(Math.cos(ax)) * hx + Math.abs(Math.sin(ax)) * hz;
    let up = Math.cos(ay) * hy + Math.sin(ay) * depth;
    let tanHalf = Math.tan(20 * Math.PI / 180);
    let dist = Math.max(up, across / aspect) / tanHalf * fill + depth * Math.cos(ay);
    let center = new Vec3((mn.x + mx.x) / 2, -(mn.z + mx.z) / 2, -(mn.y + mx.y) / 2);
    if (opts.shiftX) {
        let right = new Vec3(-Math.sin(ax), Math.cos(ax), 0);
        center = center.add(right.mul(across * opts.shiftX));
    }
    return { center, angle: new Vec3(angleX, angleY, dist / 200) };
}

// Keyframed camera on a phase-local clock; stops steering after the last key so the user
// can orbit freely. Key 0 eases in from wherever the camera was when the phase began.
export function cameraKeyframes(state: IProgramState, time: number, keys: { t: number, cam: ICameraPos }[]) {
    let wt = state.walkthrough;
    let data = getPhaseTransitiveData(wt);
    data.kfSrc ??= { angle: state.camera.angle, center: state.camera.center };
    let last = keys[keys.length - 1];
    if (data.kfDone && time >= last.t) return;
    let prev: ICameraPos = data.kfSrc, prevT = 0;
    for (let k of keys) {
        if (time < k.t) {
            let u = clamp((time - prevT) / Math.max(1e-6, k.t - prevT), 0, 1);
            u = u * u * (3 - 2 * u);
            state.camera.angle = prev.angle.lerp(k.cam.angle, u);
            state.camera.center = prev.center.lerp(k.cam.center, u);
            wt.markDirty();
            return;
        }
        prev = k.cam;
        prevT = k.t;
    }
    state.camera.angle = last.cam.angle;
    state.camera.center = last.cam.center;
    data.kfDone = true;
}

// A phase-local clock that keeps advancing while the walkthrough is paused (so a chapter
// opened from the table of contents still plays its 3D animation).
export function phaseClock(state: IProgramState, rate: number = 1) {
    let wt = state.walkthrough;
    let data = getPhaseTransitiveData(wt);
    let now = performance.now();
    data.clockStartMs ??= now;
    let t = Math.max(wt.time, (now - data.clockStartMs) / 1000 * rate);
    return t;
}

type IFrameCube = IBlkDef | { x: number, y: number, z: number, dx: number, dy: number, dz: number };
type IFrameOpts = { fill?: number, aspect?: number, shiftX?: number };

// layout-relative replacements for the hard-coded camera positions
export function initialFrame(state: IProgramState, cubes: IFrameCube[], angleX: number, angleY: number, opts: IFrameOpts = {}) {
    let c = frameCubes(cubes, angleX, angleY, opts);
    setInitialCamera(state, c.center, c.angle);
}

export function frameTo(state: IProgramState, time: ITimeInfo, cubes: IFrameCube[], angleX: number, angleY: number, opts: IFrameOpts = {}) {
    let c = frameCubes(cubes, angleX, angleY, opts);
    moveCameraTo(state, time, c.center, c.angle);
}
