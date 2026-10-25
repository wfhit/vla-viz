import { Vec3 } from "@/src/utils/vector";
import { IPageSpec } from "@/src/vla/PageSpec";
import { genPi0Layout } from "./Pi0Layout";
import { IPi0Model, loadPi0Model } from "./Pi0Model";
import { drawPi0Scene } from "./Pi0Scene";
import { PPhase, pi0PhaseList, runPi0Walkthrough } from "./Pi0Walkthrough";

// /pi0: π0.5 (PaliGemma prefix + Gemma-300M action expert, joint attention, adaRMS time
// conditioning), real openpi-comet weights + real activations on a BEHAVIOR demo frame
// (tools/export_pi0.py)
export const pi0Page: IPageSpec = {
    id: 'pi0',
    phaseList: pi0PhaseList,
    firstPhase: PPhase.P0_Overview,
    tagScaleMax: 5,
    initialCamera: { angle: new Vec3(278, 6, 12), center: new Vec3(600, 600, 0) },

    async loadData(state) {
        state.pageData = await loadPi0Model(state.render.gl);
        state.markDirty();
    },

    prepare(state, view) {
        state.layout = genPi0Layout(state.pageData as IPi0Model | null);
        if (state.inWalkthrough) runPi0Walkthrough(state, view);
    },

    drawScene(state) {
        drawPi0Scene(state, state.layout as any, state.pageData as IPi0Model | null);
    },

    drawLabels() { },
};
