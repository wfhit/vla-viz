import { ICamera } from "./Camera";

interface ISavedState {
    page: string;
    phase: number;
    phaseTime: number;
    camera: ICamera;
}

export let SavedState = {
    state: null as ISavedState | null,
};
