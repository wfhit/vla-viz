'use client';

import { useContext, useEffect, useLayoutEffect, useState } from 'react';
import s from './LayerView.module.scss';
import { IRenderState, IRenderView } from './render/modelRender';
import { fetchFontAtlasData, IFontAtlasData } from './render/fontRender';
import { ProgramStateContext, WalkthroughSidebar } from './Sidebar';
import { initProgramState, IProgramState, runProgram } from './Program';
import { CanvasEventSurface } from './CanvasEventSurface';
import { Vec3 } from '@/src/utils/vector';
import { useScreenLayout } from '@/src/utils/layout';
import { jumpPhase } from './Commentary';
import { KeyboardManagerContext, KeyboardOrder, useGlobalKeyboard } from '@/src/utils/keyboard';
import { Resizer } from '../utils/Resizer';
import { ModelSelectorToolbar } from './components/ModelSelectorToolbar';
import { PlayBar } from './components/PlayBar';
import { HeadsOverlay } from './components/HeadsOverlay';
import { IPageSpec, PageId } from './PageSpec';
import { getPageSpec } from './pages';

export function LayerView({ pageId = 'vla' }: { pageId?: PageId }) {
    let [canvasEl, setCanvasEl] = useState<HTMLCanvasElement | null>(null);
    let [canvasRender, setCanvasRender] = useState<CanvasRender | null>(null);
    let [fontAtlasData, setFontAtlasData] = useState<IFontAtlasData | null>(null);
    let layout = useScreenLayout();
    let keyboardManager = useContext(KeyboardManagerContext);

    useGlobalKeyboard(KeyboardOrder.MainPage, (ev: KeyboardEvent) => {
        if (!canvasRender?.progState) {
            return;
        }
        let wt = canvasRender.progState.walkthrough;
        if (ev.key === ' ') {
            if (wt.time >= wt.phaseLength) {
                jumpPhase(wt, 1);
                wt.time = 0;
            } else {
                wt.running = !wt.running;
            }
            canvasRender.markDirty();
        }
        if (ev.key === 'Backspace' || ev.key === 'Delete') {
            wt.running = false;
            wt.time = 0;
            canvasRender.markDirty();
        }

        if (ev.key === ' ') {
            ev.preventDefault();
        }
    });

    useEffect(() => {
        document.addEventListener('keydown', keyboardManager.handleKey);
        return () => {
            document.removeEventListener('keydown', keyboardManager.handleKey);
        };
    }, [keyboardManager]);

    useEffect(() => {
        let stale = false;
        async function getData() {
            let data = await fetchFontAtlasData();
            if (stale) return;
            setFontAtlasData(data);
        }

        getData();

        return () => { stale = true; };
    }, []);

    useEffect(() => {
        if (canvasEl && fontAtlasData) {
            let canvasRenderLocal = new CanvasRender(canvasEl, fontAtlasData, getPageSpec(pageId));
            let resizeObserver = new ResizeObserver(() => {
                canvasRenderLocal.canvasSizeDirty = true;
                canvasRenderLocal.markDirty();
            });
            let handleWheel = (ev: WheelEvent) => ev.preventDefault();
            setCanvasRender(canvasRenderLocal);
            resizeObserver.observe(canvasEl);
            canvasEl.addEventListener('wheel', handleWheel, { passive: false });
            return () => {
                canvasEl!.removeEventListener('wheel', handleWheel);
                canvasRenderLocal.destroy();
                resizeObserver.disconnect();
            };
        } else {
            setCanvasRender(null);
        }
    }, [canvasEl, fontAtlasData, pageId]);

    useLayoutEffect(() => {
        if (canvasRender) {
            canvasRender.progState.pageLayout = layout;
            canvasRender.markDirty();
        }
    }, [canvasRender, layout]);

    let sidebar = canvasRender && <div className={s.sidebar}>
        <ProgramStateContext.Provider value={canvasRender.progState}>
            <WalkthroughSidebar />
        </ProgramStateContext.Provider>
    </div>;

    let mainView = <div className={s.canvasWrap}>
        <canvas
            className={s.canvas}
            ref={setCanvasEl}
        />
        {canvasRender && !canvasRender.progState.render && <div className='absolute flex flex-col items-center w-full h-full justify-center'>
            <div className='text-2xl'>This application requires a WebGL2 capable browser.</div>
            <div className='text-lg mt-2'>Please try the latest version of Chrome or Firefox.</div>
        </div>}
        {canvasRender && <ProgramStateContext.Provider value={canvasRender.progState}>
            <CanvasEventSurface />
            {pageId === 'vla' && <ModelSelectorToolbar />}
            {pageId === 'vla' && <HeadsOverlay />}
            <PlayBar />
        </ProgramStateContext.Provider>}
    </div>;

    return <div className={s.view}>
        <Resizer id={"vla-sidebar"} className={"flex-1"} vertical={!layout.isDesktop} defaultAmt={0.4}>
            {layout.isDesktop && sidebar}
            {mainView}
            {!layout.isDesktop && sidebar}
        </Resizer>
    </div>;
}

class CanvasRender {
    renderState: IRenderState;
    progState: IProgramState;
    stopped = false;
    canvasSizeDirty = true;

    constructor(canvasEl: HTMLCanvasElement, fontAtlasData: IFontAtlasData, page: IPageSpec) {
        this.progState = initProgramState(canvasEl, fontAtlasData, page);
        this.progState.markDirty = this.markDirty;
        this.progState.walkthrough.markDirty = this.markDirty;
        this.renderState = this.progState.render;
        if (this.progState.render) {
            page.loadData(this.progState)
                .catch(e => console.error(`${page.id} data`, e))
                .finally(() => this.markDirty());
        }
        this.markDirty();
    }

    destroy() {
        this.stopped = true;
    }

    prevTime: number = performance.now();
    rafHandle: number = 0;
    isDirty = false;
    isWaitingForSync = false;

    markDirty = () => {
        if (this.stopped) {
            return;
        }

        this.isDirty = true;
        if (!this.rafHandle) {
            this.prevTime = performance.now();
            this.rafHandle = requestAnimationFrame(this.loop);
        }
    }

    loop = (time: number) => {
        if (!(this.isDirty || this.isWaitingForSync) || this.stopped) {
            this.rafHandle = 0;
            return;
        }
        let wasDirty = this.isDirty;

        this.isDirty = false;
        this.isWaitingForSync = false;

        let dt = time - this.prevTime;
        this.prevTime = time;
        if (dt < 8) dt = 16; // sometimes we get -ve dt due to perf.now() vs requestAnimationFrame() timing, so put to 16ms in that case

        // we separate waitingForSync from dirty, so we don't have to render if we're only waiting for sync
        this.checkSyncObjects();
        let prevSyncCount = this.progState.render?.syncObjects.length ?? 0;

        if (wasDirty || this.isDirty) {
            this.render(time, dt);
        }

        let newSyncCount = this.progState.render?.syncObjects.length ?? 0;
        if (newSyncCount !== prevSyncCount) {
            this.isWaitingForSync = true;
        }

        this.rafHandle = requestAnimationFrame(this.loop);
    }

    checkSyncObjects() {
        if (!this.progState.render) {
            return;
        }

        let gl = this.renderState.gl;
        let objs = this.progState.render.syncObjects;
        let anyToRemove = false;

        for (let i = 0; i < objs.length; i++) {
            let obj = objs[i];
            if (obj.isReady) {
                anyToRemove = true;
                continue;
            }
            let syncStatus = gl.clientWaitSync(obj.sync, 0, 0);
            if (syncStatus === gl.TIMEOUT_EXPIRED) {
                this.isWaitingForSync = true;
            } else {
                obj.isReady = true;
                obj.elapsedMs = performance.now() - obj.startTime;
                gl.deleteSync(obj.sync);
                anyToRemove = true;
            }
        }
        if (anyToRemove) {
            this.progState.render.syncObjects = objs.filter(o => !o.isReady);
            this.markDirty();
        }
    }

    render(time: number, dt: number) {
        if (!this.progState.render) {
            return;
        }

        let canvasEl = this.renderState.canvasEl;

        if (this.canvasSizeDirty) {
            let bcr = canvasEl.getBoundingClientRect();
            let scale = window.devicePixelRatio;
            canvasEl.width = bcr.width * scale;
            canvasEl.height = bcr.height * scale;
            this.progState.render.size = new Vec3(bcr.width, bcr.height);
            this.canvasSizeDirty = false;
        }

        let view: IRenderView = { time, dt, markDirty: this.markDirty };
        runProgram(view, this.progState);
        this.progState.htmlSubs.notify();
    }

}
