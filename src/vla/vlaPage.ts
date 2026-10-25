import { Vec3 } from "@/src/utils/vector";
import { IPageSpec } from "./PageSpec";
import { cubesInView, IProgramState, vlaLayout } from "./Program";
import { createModelGpuLink, fetchModelData, fetchModelFm, shapeFromConfig } from "./VlaModel";
import { genVlaModelLayout } from "./VlaModelLayout";
import { enginePhaseList, Phase, runWalkthrough } from "./walkthrough/Walkthrough";
import { auditArrows, drawAllArrows } from "./components/Arrow";
import { drawActionViz } from "./components/ActionViz";
import { drawModelCard } from "./components/ModelCard";
import { drawBlockLabels } from "./components/SectionLabels";
import { cameraToMatrixView } from "./Camera";
import { clamp } from "@/src/utils/data";
import { IVlaModelLayout } from "./VlaModelLayout";
import { IModelFacts, modelFacts } from "./ModelFacts";
import { ICameraPos } from "./Camera";
import { PageId } from "./PageSpec";

function cameraBasis(state: IProgramState) {
    let { camPos } = cameraToMatrixView(state.camera);
    let c = state.camera.center;
    let cam = new Vec3(camPos.x, -camPos.z, -camPos.y);
    let fwd = new Vec3(c.x, -c.z, -c.y).sub(cam);
    let dist = fwd.len();
    fwd = fwd.mul(1 / Math.max(dist, 1e-3));
    let right = Vec3.cross(fwd, new Vec3(0, -1, 0)).normalize(); // layout −y is world up
    let up = Vec3.cross(right, fwd);
    return { cam, fwd, right, up, dist };
}

// does a layout-space box lie entirely inside the view frustum (half-angle 20°, screen aspect)?
function fitsView(state: IProgramState) {
    let { cam, fwd, right, up } = cameraBasis(state);
    let ty = Math.tan(20 * Math.PI / 180);
    let cv = state.render.canvasEl;
    let tx = ty * (cv && cv.height > 0 ? cv.width / cv.height : 1);
    return (mn: Vec3, mx: Vec3) => {
        for (let i = 0; i < 8; i++) {
            let p = new Vec3(i & 1 ? mx.x : mn.x, i & 2 ? mx.y : mn.y, i & 4 ? mx.z : mn.z).sub(cam);
            let d = p.dot(fwd);
            if (d <= 1) return false;
            if (Math.abs(p.dot(right) / d) > tx || Math.abs(p.dot(up) / d) > ty) return false;
        }
        return true;
    };
}

// blocks entirely in front of the framed working set (wt.focus) that cover it on screen fade
// out: with 16 heads fanned ±237 in z, a close-up otherwise puts the front heads over the subject
function fadeOccluders(state: IProgramState, layout: IVlaModelLayout) {
    let focus = state.inWalkthrough ? state.walkthrough.focus : null;
    if (!focus) return;
    let { camPos } = cameraToMatrixView(state.camera);
    let c = state.camera.center;
    let cam = new Vec3(camPos.x, -camPos.z, -camPos.y);
    let fwd = new Vec3(c.x, -c.z, -c.y).sub(cam);
    let dist = fwd.len();
    if (dist < 1e-3) return;
    fwd = fwd.mul(1 / dist);
    let right = Vec3.cross(fwd, new Vec3(0, -1, 0)).normalize(); // layout −y is world up
    let up = Vec3.cross(right, fwd);
    // camera-space box: depth range + screen rect (lateral / depth)
    function proj(mn: Vec3, mx: Vec3) {
        let r = { d0: Infinity, d1: -Infinity, x0: Infinity, x1: -Infinity, y0: Infinity, y1: -Infinity };
        for (let i = 0; i < 8; i++) {
            let p = new Vec3(i & 1 ? mx.x : mn.x, i & 2 ? mx.y : mn.y, i & 4 ? mx.z : mn.z).sub(cam);
            let d = p.dot(fwd), dd = Math.max(d, 1);
            let sx = p.dot(right) / dd, sy = p.dot(up) / dd;
            r.d0 = Math.min(r.d0, d); r.d1 = Math.max(r.d1, d);
            r.x0 = Math.min(r.x0, sx); r.x1 = Math.max(r.x1, sx);
            r.y0 = Math.min(r.y0, sy); r.y1 = Math.max(r.y1, sy);
        }
        return r;
    }
    let F = proj(focus.mn, focus.mx);
    let margin = 0.08 * dist;
    let ex = 0.25 * (F.x1 - F.x0), ey = 0.25 * (F.y1 - F.y0);
    F.x0 -= ex; F.x1 += ex; F.y0 -= ey; F.y1 += ey;
    for (let b of layout.cubes) {
        if (b.opacity <= 0) continue;
        let P = proj(new Vec3(b.x, b.y, b.z), new Vec3(b.x + b.dx, b.y + b.dy, b.z + b.dz));
        if (P.d1 > F.d0) continue;
        if (P.x1 < F.x0 || P.x0 > F.x1 || P.y1 < F.y0 || P.y0 > F.y1) continue;
        let f = clamp((P.d1 - (F.d0 - margin)) / margin, 0.04, 1);
        b.opacity *= f;
    }
}

// one head / kv group stands for all of them (they share its slot; labelled "x16 heads")
function hideRepeatedHeads(layout: IVlaModelLayout) {
    for (let L of [...layout.blocks, ...layout.expertLayers]) {
        for (let h of L.heads.slice(1)) for (let c of h.ownCubes) c.opacity = 0;
        for (let g of L.kvGroups.slice(1)) for (let c of g.cubes) c.opacity = 0;
    }
}

// blocks of tensors this model does not have (absent placeholders) are never drawn
function hideAbsent(layout: IVlaModelLayout) {
    for (let c of layout.cubes) if (c.absent) c.opacity = 0;
}

// One instance of the shared VLA engine: the same scene layout, walkthrough chapters, arrows
// and play bar for every model; `prefix` selects the exported tensors (public/<prefix>-*.json)
// and `facts` the model's structural numbers the chapter texts quote.
export function makeEnginePage(id: PageId, prefix: string, facts: IModelFacts, initialCamera: ICameraPos): IPageSpec {
    return {
        id,
        facts,
        phaseList: enginePhaseList(facts),
        firstPhase: Phase.Intro_Intro,
        initialCamera,

        async loadData(state: IProgramState) {
            let data = await fetchModelData(prefix);
            let model = createModelGpuLink(state.render.gl, data);
            state.vlaModel = model;
            state.shape = shapeFromConfig(model.config);
            state.mainExample.shape = state.shape;
            (window as any).__model = model;
            state.markDirty();
            // the per-Euler-step capture is only needed by the expert decode chapters: load it after
            model.fmSteps = await fetchModelFm(state.render.gl, prefix);
        },

        prepare(state, view) {
            state.layout = genVlaModelLayout(state.shape, state.vlaModel);
            for (let example of state.examples) {
                if (example.enabled && !example.layout) {
                    example.layout = genVlaModelLayout(example.shape, null, example.offset);
                }
            }
            if (state.inWalkthrough) {
                runWalkthrough(state, view);
            }
            hideRepeatedHeads(state.layout as IVlaModelLayout);
            hideAbsent(state.layout as IVlaModelLayout);
        },

        drawScene(state) {
            let layout = vlaLayout(state);
            fadeOccluders(state, layout);
            drawAllArrows(state.render, layout, cubes => cubesInView(state, cubes));
            drawActionViz(state, cubes => cubesInView(state, cubes));
            // self-check hooks for the headless probes: visible data blocks without an in / out arrow
            if (typeof window !== 'undefined') {
                (window as any).__arrowAudit = () => auditArrows(layout);
                (window as any).__engineState = state;
            }
            drawModelCard(state, layout, facts.modelName, new Vec3());
            for (let example of state.examples) {
                if (example.enabled && example.layout) {
                    drawModelCard(state, example.layout, example.name, example.offset.add(example.modelCardOffset));
                }
            }
        },

        drawLabels(state) {
            let c = state.camera.center;
            drawBlockLabels(state.render, vlaLayout(state), facts, new Vec3(c.x, -c.z, -c.y), fitsView(state));
        },
    };
}

export const vlaPage = makeEnginePage('vla', 'k005b', modelFacts.k005b,
    { angle: new Vec3(296, 16, 13.5), center: new Vec3(-8.4, 0, -481.5) });
