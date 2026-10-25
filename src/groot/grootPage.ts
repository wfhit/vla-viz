import { Vec3 } from "@/src/utils/vector";
import { IPageSpec } from "@/src/vla/PageSpec";
import { genGrootLayout } from "./GrootLayout";
import { IGrootModel, loadGrootModel } from "./GrootModel";
import { drawGrootScene } from "./GrootScene";
import { GPhase, grootPhaseList, runGrootWalkthrough } from "./GrootWalkthrough";

// /groot: NVIDIA GR00T N1.7 (System 2 VLM + System 1 DiT action head), real weights + real
// activations on a BEHAVIOR demo frame (tools/export_groot.py)
export const grootPage: IPageSpec = {
    id: 'groot',
    phaseList: grootPhaseList,
    firstPhase: GPhase.G0_Overview,
    tagScaleMax: 5,
    initialCamera: { angle: new Vec3(300, 28, 12), center: new Vec3(900, 0, -500) },

    async loadData(state) {
        state.pageData = await loadGrootModel(state.render.gl);
        state.markDirty();
    },

    prepare(state, view) {
        state.layout = genGrootLayout(state.pageData as IGrootModel | null);
        if (state.inWalkthrough) runGrootWalkthrough(state, view);
    },

    drawScene(state) {
        drawGrootScene(state, state.layout as any, state.pageData as IGrootModel | null);
    },

    drawLabels() { },
};
