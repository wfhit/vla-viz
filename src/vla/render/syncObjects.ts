
export interface ISyncObject {
    sync: WebGLSync;
    isReady: boolean;
    startTime: number;
    elapsedMs: number;
}

/* To use a sync object, store it in the tree with a specific name. In subsequent frames, check if
   the sync object is ready. */
