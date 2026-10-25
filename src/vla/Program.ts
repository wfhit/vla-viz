import { genModelViewMatrices, ICamera, ICameraPos, updateCamera } from "./Camera";
import { auditArrows, drawAllArrows } from "./components/Arrow";
import { defaultActionViz, drawActionViz, IActionVizState } from "./components/ActionViz";
import { drawBlockLabels } from "./components/SectionLabels";
import { drawModelCard } from "./components/ModelCard";
import { IVlaModelLink, IModelShape } from "./VlaModel";
import { genVlaModelLayout, IBlkDef, IVlaModelLayout } from "./VlaModelLayout";
import { drawText, IFontAtlasData, IFontOpts, measureText } from "./render/fontRender";
import { initRender, IRenderState, IRenderView, renderModel, resetRenderBuffers } from "./render/modelRender";
import { beginQueryAndGetPrevMs, endQuery } from "./render/queryManager";
import { SavedState } from "./SavedState";
import { isNotNil } from "@/src/utils/data";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { initWalkthrough, runWalkthrough } from "./walkthrough/Walkthrough";
import { IColorMix } from "./Annotations";
import { Mat4f } from "@/src/utils/matrix";
import { runMouseHitTesting } from "./Interaction";
import { RenderPhase } from "./render/sharedRender";
import { drawBlockInfo } from "./components/BlockInfo";
import { IBlockRender } from "./render/blockRender";
import { ILayout } from "../utils/layout";
import { DimStyle } from "./walkthrough/WalkthroughTools";
import { Subscriptions } from "../utils/hooks";

export interface IProgramState {
    mouse: IMouseState;
    render: IRenderState;
    inWalkthrough: boolean;
    walkthrough: ReturnType<typeof initWalkthrough>;
    camera: ICamera;
    htmlSubs: Subscriptions;
    layout: IVlaModelLayout;
    mainExample: IModelExample;
    examples: IModelExample[];
    currExampleId: number;
    shape: IModelShape;
    vlaModel: IVlaModelLink | null;
    display: IDisplayState;
    pageLayout: ILayout;
    markDirty: () => void;
}

export interface IModelExample {
    name: string;
    shape: IModelShape;
    enabled: boolean;
    layout?: IVlaModelLayout;
    blockRender: IBlockRender;
    offset: Vec3;
    modelCardOffset: Vec3;
    camera?: ICameraPos;
}

export interface IMouseState {
    mousePos: Vec3;
}

export interface IDisplayState {
    tokenColors: IColorMix | null;
    tokenIdxColors: IColorMix | null;
    tokenOutputColors: IColorMix | null;
    tokenIdxModelOpacity?: number[];
    topOutputOpacity?: number;
    lines: string[];
    hoverTarget: IHoverTarget | null;
    blkIdxHover: number[] | null;
    dimHover: DimStyle | null;
    actionViz?: IActionVizState;
}

interface IHoverTarget {
    subCube: IBlkDef;
    mainCube: IBlkDef;
    mainIdx: Vec3;
}

export function initProgramState(canvasEl: HTMLCanvasElement, fontAtlasData: IFontAtlasData): IProgramState {

    let render = initRender(canvasEl, fontAtlasData);
    let walkthrough = initWalkthrough();

    let prevState = SavedState.state;
    let camera: ICamera = {
        angle: prevState?.camera.angle ?? new Vec3(296, 16, 13.5),
        center: prevState?.camera.center ?? new Vec3(-8.4, 0, -481.5),
        transition: {},
        modelMtx: new Mat4f(),
        viewMtx: new Mat4f(),
        lookAtMtx: new Mat4f(),
        camPos: new Vec3(),
        camPosModel: new Vec3(),
    }

    // Geometry before the real data arrives: same window dims the exporter writes
    // (tools/export_k005b.py); replaced by shapeFromConfig(config) once loaded.
    // No data is attached until then (blocks render blank).
    let shape: IModelShape = {
        B: 1, T: 32, C: 64, nHeads: 16, nKvHeads: 8, A: 32, F: 64, nBlocks: 28,
        vocabSize: 64, nKnack: 17, nDone: 2, actH: 16, actD: 23, nSuffix: 17,
        full: null,
    };

    function makeCamera(center: Vec3, angle: Vec3): ICameraPos {
        return { center, angle };
    }

    return {
        render: render!,
        inWalkthrough: true,
        walkthrough,
        camera,
        shape: shape,
        layout: genVlaModelLayout(shape),
        currExampleId: -1,
        mainExample: {
            name: 'k005b',
            enabled: true,
            shape: shape,
            offset: new Vec3(),
            modelCardOffset: new Vec3(),
            blockRender: null!,
            camera: makeCamera(new Vec3(42.771, 0.000, -569.287), new Vec3(284.959, 26.501, 12.867)),
        },
        examples: [],
        vlaModel: null,
        markDirty: () => { },
        htmlSubs: new Subscriptions(),
        mouse: {
            mousePos: new Vec3(),
        },
        display: {
            tokenColors: null,
            tokenIdxColors: null,
            tokenOutputColors: null,
            lines: [],
            hoverTarget: null,
            dimHover: null,
            blkIdxHover: null,
        },
        pageLayout: {
            height: 0,
            width: 0,
            isDesktop: true,
            isPhone: true,
        }
    };
}

export function runProgram(view: IRenderView, state: IProgramState) {
    let timer0 = performance.now();

    if (!state.render) {
        return;
    }

    resetRenderBuffers(state.render);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    state.display.lines = [];
    state.display.hoverTarget = null;
    state.display.tokenColors = null;
    state.display.tokenIdxColors = null;
    state.display.actionViz = defaultActionViz();

    // generate the base model, incorporating the gpu-side model if available
    state.layout = genVlaModelLayout(state.shape, state.vlaModel);

    // @TODO: handle different models in the same scene.
    // Maybe need to copy a lot of different things like the entire render state per model?
    for (let example of state.examples) {
        if (example.enabled && !example.layout) {
            let layout = genVlaModelLayout(example.shape, null, example.offset);
            example.layout = layout;
        }
    }

    genModelViewMatrices(state);

    let queryRes = beginQueryAndGetPrevMs(state.render.queryManager, 'render');
    if (isNotNil(queryRes)) {
        state.render.lastGpuMs = queryRes;
    }

    state.render.renderTiming = false; // state.pageLayout.isDesktop;

    // will modify layout; view; render a few things.
    if (state.inWalkthrough) {
        runWalkthrough(state, view);
    }

    updateCamera(state, view);

    drawBlockInfo(state);
    // these will get modified by the walkthrough (stored where?)
    drawAllArrows(state.render, state.layout, cubes => cubesInView(state, cubes));
    drawActionViz(state, cubes => cubesInView(state, cubes));
    // self-check hook for the headless probes: visible data blocks without an in / out arrow
    if (typeof window !== 'undefined') {
        (window as any).__arrowAudit = () => auditArrows(state.layout);
        (window as any).__k005bState = state;
    }

    drawModelCard(state, state.layout, 'k005b VLA', new Vec3());

    for (let example of state.examples) {
        if (example.enabled && example.layout) {
            drawModelCard(state, example.layout, example.name, example.offset.add(example.modelCardOffset));
        }
    }

    runMouseHitTesting(state);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    drawBlockLabels(state.render, state.layout);

    let lineNo = 1;
    let tw = state.render.size.x;
    state.render.sharedRender.activePhase = RenderPhase.Overlay2D;
    for (let line of state.display.lines) {
        let opts: IFontOpts = { color: new Vec4(0.05, 0.05, 0.1, 1), size: 14 };
        let w = measureText(state.render.modelFontBuf, line, opts);
        drawText(state.render.modelFontBuf, line, tw - w - 4, lineNo * opts.size * 1.3 + 4, opts)
        lineNo++;
    }

    // render everything; i.e. here's where we actually do gl draw calls
    // up until now, we've just been putting data in cpu-side buffers
    renderModel(state);

    endQuery(state.render.queryManager, 'render');
    state.render.gl.flush();

    state.render.lastJsMs = performance.now() - timer0;
}

// conservative frustum test of the bounding box of `cubes` (layout space)
function cubesInView(state: IProgramState, cubes: IBlkDef[]) {
    let mn = new Vec3(Infinity, Infinity, Infinity), mx = new Vec3(-Infinity, -Infinity, -Infinity);
    for (let c of cubes) {
        mn = new Vec3(Math.min(mn.x, c.x), Math.min(mn.y, c.y), Math.min(mn.z, c.z));
        mx = new Vec3(Math.max(mx.x, c.x + c.dx), Math.max(mx.y, c.y + c.dy), Math.max(mx.z, c.z + c.dz));
    }
    let mtx = state.camera.viewMtx.mul(state.camera.modelMtx);
    let out = [0, 0, 0, 0, 0]; // x<-w, x>w, y<-w, y>w, behind
    for (let i = 0; i < 8; i++) {
        let p = new Vec3(i & 1 ? mx.x : mn.x, i & 2 ? mx.y : mn.y, i & 4 ? mx.z : mn.z);
        let v = mtx.mulVec4(new Vec4(p.x, p.y, p.z, 1));
        if (v.x < -v.w) out[0]++;
        if (v.x > v.w) out[1]++;
        if (v.y < -v.w) out[2]++;
        if (v.y > v.w) out[3]++;
        if (v.w <= 0) out[4]++;
    }
    return !out.some(n => n === 8);
}
