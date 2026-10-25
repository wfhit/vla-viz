import { Vec3 } from "@/src/utils/vector";
import { IPageSpec } from "./PageSpec";
import { cubesInView, IProgramState, vlaLayout } from "./Program";
import { createK005bGpuModel, fetchK005bData, fetchK005bFm, shapeFromConfig } from "./VlaModel";
import { genVlaModelLayout } from "./VlaModelLayout";
import { Phase, runWalkthrough, vlaPhaseList } from "./walkthrough/Walkthrough";
import { auditArrows, drawAllArrows } from "./components/Arrow";
import { drawActionViz } from "./components/ActionViz";
import { drawModelCard } from "./components/ModelCard";
import { drawBlockLabels } from "./components/SectionLabels";

// /vla: k005b (Qwen3-VL trunk + DiT action expert), real weights + real activations
export const vlaPage: IPageSpec = {
    id: 'vla',
    phaseList: vlaPhaseList,
    firstPhase: Phase.Intro_Intro,
    initialCamera: { angle: new Vec3(296, 16, 13.5), center: new Vec3(-8.4, 0, -481.5) },

    async loadData(state: IProgramState) {
        let data = await fetchK005bData();
        let model = createK005bGpuModel(state.render.gl, data);
        state.vlaModel = model;
        state.shape = shapeFromConfig(model.config);
        state.mainExample.shape = state.shape;
        (window as any).__k005b = model;
        state.markDirty();
        // the per-Euler-step capture is only needed by the DiT decode chapters: load it after
        model.fmSteps = await fetchK005bFm(state.render.gl);
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
    },

    drawScene(state) {
        let layout = vlaLayout(state);
        drawAllArrows(state.render, layout, cubes => cubesInView(state, cubes));
        drawActionViz(state, cubes => cubesInView(state, cubes));
        // self-check hooks for the headless probes: visible data blocks without an in / out arrow
        if (typeof window !== 'undefined') {
            (window as any).__arrowAudit = () => auditArrows(layout);
            (window as any).__k005bState = state;
        }
        drawModelCard(state, layout, 'k005b VLA', new Vec3());
        for (let example of state.examples) {
            if (example.enabled && example.layout) {
                drawModelCard(state, example.layout, example.name, example.offset.add(example.modelCardOffset));
            }
        }
    },

    drawLabels(state) {
        drawBlockLabels(state.render, vlaLayout(state));
    },
};
