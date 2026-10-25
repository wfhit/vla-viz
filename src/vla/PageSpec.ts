import type { ICameraPos } from "./Camera";
import type { IProgramState } from "./Program";
import type { IRenderView } from "./render/modelRender";
import type { IPhaseGroup } from "./walkthrough/WalkthroughTools";

export type PageId = 'vla' | 'groot' | 'pi0';

// One visualization page: its own model data, scene layout, walkthrough and scene drawing on
// top of the shared renderer (blocks, arrows, text, camera, commentary panel).
export interface IPageSpec {
    id: PageId;
    phaseList: IPhaseGroup[];
    firstPhase: number;
    initialCamera: ICameraPos;
    // max world-size factor of the block name tags (they grow with camera distance up to this)
    tagScaleMax?: number;
    // once, after the GL context exists: fetch the exported tensors and attach them to `state`
    loadData(state: IProgramState): Promise<void>;
    // per frame, before the camera update: rebuild state.layout and run the walkthrough
    prepare(state: IProgramState, view: IRenderView): void;
    // per frame, after the camera update: arrows and in-scene annotations
    drawScene(state: IProgramState): void;
    // per frame, after hover hit-testing (which decides which section labels show)
    drawLabels(state: IProgramState): void;
}
