import { genModelViewMatrices, ICamera, ICameraPos, updateCamera } from "./Camera";
import { defaultActionViz, IActionVizState } from "./components/ActionViz";
import { IVlaModelLink, IModelShape } from "./VlaModel";
import { IBlkDef, ISceneLayout, IVlaModelLayout } from "./VlaModelLayout";
import { drawText, IFontAtlasData, IFontOpts, measureText } from "./render/fontRender";
import { initRender, IRenderState, IRenderView, renderModel, resetRenderBuffers } from "./render/modelRender";
import { beginQueryAndGetPrevMs, endQuery } from "./render/queryManager";
import { SavedState } from "./SavedState";
import { isNotNil } from "@/src/utils/data";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { initWalkthrough } from "./walkthrough/Walkthrough";
import { IColorMix } from "./Annotations";
import { Mat4f } from "@/src/utils/matrix";
import { runMouseHitTesting } from "./Interaction";
import { RenderPhase } from "./render/sharedRender";
import { drawBlockInfo } from "./components/BlockInfo";
import { IBlockRender } from "./render/blockRender";
import { ILayout } from "../utils/layout";
import { DimStyle } from "./walkthrough/WalkthroughTools";
import { Subscriptions } from "../utils/hooks";
import { IPageSpec } from "./PageSpec";

export interface IProgramState {
    page: IPageSpec;
    mouse: IMouseState;
    render: IRenderState;
    inWalkthrough: boolean;
    walkthrough: ReturnType<typeof initWalkthrough>;
    camera: ICamera;
    htmlSubs: Subscriptions;
    // the current page's scene; the /vla page's chapters read it as IVlaModelLayout (vlaLayout)
    layout: ISceneLayout;
    mainExample: IModelExample;
    examples: IModelExample[];
    currExampleId: number;
    shape: IModelShape;
    vlaModel: IVlaModelLink | null;
    // per-page data that only that page's code reads (e.g. the /groot tensors)
    pageData: any;
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

export function initProgramState(canvasEl: HTMLCanvasElement, fontAtlasData: IFontAtlasData, page: IPageSpec): IProgramState {

    let render = initRender(canvasEl, fontAtlasData);
    let walkthrough = initWalkthrough(page);

    let prevState = SavedState.state?.page === page.id ? SavedState.state : null;
    let camera: ICamera = {
        angle: prevState?.camera.angle ?? page.initialCamera.angle,
        center: prevState?.camera.center ?? page.initialCamera.center,
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
        page,
        render: render!,
        inWalkthrough: true,
        walkthrough,
        camera,
        shape: shape,
        layout: { cell: 1.5, height: 0, margin: 0, cubes: [], labels: [] },
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
        pageData: null,
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

    genModelViewMatrices(state);

    let queryRes = beginQueryAndGetPrevMs(state.render.queryManager, 'render');
    if (isNotNil(queryRes)) {
        state.render.lastGpuMs = queryRes;
    }

    state.render.renderTiming = false; // state.pageLayout.isDesktop;

    // the page builds its scene (state.layout) and runs its walkthrough (moves the camera)
    state.page.prepare(state, view);

    updateCamera(state, view);

    state.page.drawScene(state);
    drawBlockInfo(state);
    if (typeof window !== 'undefined') {
        (window as any).__vizState = state;
    }

    runMouseHitTesting(state);
    state.render.sharedRender.activePhase = RenderPhase.Opaque;
    state.page.drawLabels(state);

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

// the /vla page's scene (only valid while that page is shown)
export function vlaLayout(state: IProgramState): IVlaModelLayout {
    if (state.page.id !== 'vla') throw new Error(`vlaLayout() on page ${state.page.id}`);
    return state.layout as IVlaModelLayout;
}

// conservative frustum test of the bounding box of `cubes` (layout space)
export function cubesInView(state: IProgramState, cubes: IBlkDef[]) {
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
