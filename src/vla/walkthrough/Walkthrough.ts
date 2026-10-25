import { ICameraPos } from "../Camera";
import { IBlkDef } from "../VlaModelLayout";
import { IProgramState } from "../Program";
import { IRenderView } from "../render/modelRender";
import { SavedState } from "../SavedState";
import { ICommentary, IPhaseGroup, ITimeInfo, IWalkthroughArgs, phaseTools } from "./WalkthroughTools";
import { walkthroughIntro } from "./Walkthrough00_Intro";
import { walkthrough01_Prelim } from "./Walkthrough01_Prelim";
import { walkthrough02_Embedding } from "./Walkthrough02_Embedding";
import { walkthrough03_LayerNorm } from "./Walkthrough03_LayerNorm";
import { walkthrough04_SelfAttention } from "./Walkthrough04_SelfAttention";
import { walkthrough05_Softmax } from "./Walkthrough05_Softmax";
import { walkthrough06_Projection } from "./Walkthrough06_Projection";
import { walkthrough07_Mlp } from "./Walkthrough07_Mlp";
import { walkthrough08_Transformer } from "./Walkthrough08_Transformer";
import { walkthrough09_Output } from "./Walkthrough09_Output";
import { walkthrough10_Heads } from "./Walkthrough10_Heads";
import { walkthrough11_FlowMatching } from "./Walkthrough11_FlowMatching";


/**

Thoughts about the walkthrough:

- Linking text events to the visual actions doesn't work well.
- Attention needs to flip between them directly. Much better to do a chunk of text, then [Spacebar], then action.
  Can still do aligned highlights though, but focused on the model, and linking back to the text.
  This also works well with hover on either. The color-coding helps a lot anyway.

- Better to do all this text in html, it adds searchability, and supports more things. _However_, probably
  difficult to do TeX fonts. Although maybe that's a bit excessive anyway.

- Can show a reasonable amount of text: Limit to half a screen on mobile, say.

- Fast but smooth transitions, with a pause in between, is much better.
- Slow, linked transitions is kinda nauseating.

- Need more controls! Mainly jumping between phases, and selecting times on phases.
- Can compute the latest time on each phase by keeping track of the events. Can keep this in an array for
  display as well.
- Need to push back into react-land at the end of each frame. (commentary; stats; etc)
- Probably need to query all the phases up front somehow. Could have a pass where we don't actually
  update anything, but read off info for each one. A pain for the in-place updates though. Likely to break
  things. Easier to just have a list of them and info like titles.
- Hmm, do want to expand to total time, hmm.

- Need a bunch more annotations/effects yet
  - Get the trails working well
  - Want a descriptor of how a cell is computed (x, y from these; mul by a, b from these, plus this; all added, [t])
  - It applies to everything, and could allow for 'draw for this 1 cell', 'draw for this col/row of cells'.
  - Curved arrows and flows with thickness
    - Arrows for showing embedding mapping, and for linking src & sink
    - Flows for the residual pathway: wide transparent no-width path, with glowing more-solid sides
      - Basically wide arrows
      - Also need side bit coming off of it for when it's copied
      - Not as wide as the actual block; keep it maybe half-T thick
      - Some indication of flow
  - Highlight block or row/column

 */


export type IWalkthrough = ReturnType<typeof initWalkthrough>;

export function initWalkthrough() {
    return {
        phase: SavedState.state?.phase ?? Phase.Intro_Intro,
        time: SavedState.state?.phaseTime ?? 0,
        viewDt: 0,
        dt: 0,
        prevTime: 0,
        prevPhase: Phase.None,
        running: false,
        speed: 1,
        cameraInitial: null as ICameraPos | null,
        times: [] as (ITimeInfo | ICommentary)[],
        phaseLength: 0,
        dimHighlightBlocks: null as IBlkDef[] | null,
        markDirty: () => { }, // bit of a hack to get it to WalkthroughSidebar
        phaseData: new Map<Phase, any>(),
        phaseTransitiveData: null as any,
        phaseList: [{
            groupId: PhaseGroup.Intro,
            title: '概览',
            phases: [
                { id: Phase.Intro_Intro, title: '总览' },
                { id: Phase.Intro_Prelim, title: '预备知识' },
            ],
        }, {
            groupId: PhaseGroup.Detailed_Input,
            title: '主干：一层展开',
            phases: [
                { id: Phase.Input_Detail_Embedding, title: '输入嵌入' },
                { id: Phase.Input_Detail_LayerNorm, title: 'RMSNorm' },
                { id: Phase.Input_Detail_SelfAttention, title: 'GQA 联合注意力' },
                { id: Phase.Input_Detail_Softmax, title: '前缀 Softmax' },
                { id: Phase.Input_Detail_Projection, title: '输出投影与残差' },
                { id: Phase.Input_Detail_Mlp, title: 'MLP (SwiGLU)' },
                { id: Phase.Input_Detail_Transformer, title: '28 层主干堆叠' },
                { id: Phase.Input_Detail_Output, title: 'h* 答案边界读出' },
            ],
        }, {
            groupId: PhaseGroup.Heads,
            title: '输出头',
            phases: [
                { id: Phase.Head_Knack, title: 'knack 分类头' },
                { id: Phase.Head_Done, title: 'done 头' },
                { id: Phase.Harness_Law, title: 'Harness 法则' },
            ],
        }, {
            groupId: PhaseGroup.FlowMatching,
            title: 'DiT 动作专家与流匹配',
            phases: [
                { id: Phase.D0_TwoTransformers, title: 'D0 两个 Transformer' },
                { id: Phase.D1_DitBlock, title: 'D1 DiT 块解剖' },
                { id: Phase.D2_Concat, title: 'D2 拼接成一条序列' },
                { id: Phase.D3_JointAttn, title: 'D3 联合注意力 = 耦合' },
                { id: Phase.D4_Forward, title: 'D4 一次前向穿过 28 层' },
                { id: Phase.D5_PrefixCache, title: 'D5 前缀只算一次' },
                { id: Phase.D6_Decode, title: 'D6 10 次 DiT 穿行' },
                { id: Phase.D7_ActionChunk, title: 'D7 动作块' },
            ],
        }] as IPhaseGroup[],
    };
}

export enum PhaseGroup {
    Intro,
    Detailed_Input,
    Heads,
    FlowMatching,
}

export enum Phase {
    None,

    Intro_Intro,
    Intro_Prelim,
    Input_Detail_Embedding,
    Input_Detail_LayerNorm,
    Input_Detail_SelfAttention,
    Input_Detail_Softmax,
    Input_Detail_Projection,
    Input_Detail_Mlp,
    Input_Detail_Transformer,
    Input_Detail_Output,
    Head_Knack,
    Head_Done,
    Harness_Law,
    D0_TwoTransformers,
    D1_DitBlock,
    D2_Concat,
    D3_JointAttn,
    D4_Forward,
    D5_PrefixCache,
    D6_Decode,
    D7_ActionChunk,
}

export function phaseToGroup(wt: IWalkthrough) {
    return wt.phaseList.find(g => g.phases.find(p => p.id === wt.phase))!;
}


export function runWalkthrough(state: IProgramState, view: IRenderView) {
    let wt = state.walkthrough;
    wt.viewDt = view.dt;

    if (wt.running) {
        let dtSeconds = view.dt / 1000 * wt.speed;
        wt.time += dtSeconds;
        wt.dt = dtSeconds;

        if (wt.time > wt.phaseLength) {
            wt.running = false;
            wt.time = wt.phaseLength;
        }

        view.markDirty();
    }

    SavedState.state = { phase: wt.phase, phaseTime: wt.time, camera: state.camera };

    if (wt.prevPhase !== wt.phase) {
        wt.phaseTransitiveData = null;
    }

    wt.cameraInitial = null;
    wt.times = [];
    wt.phaseLength = 0;
    wt.dimHighlightBlocks = null;

    let wtArgs: IWalkthroughArgs = { state, layout: state.layout, tools: phaseTools(state), walkthrough: wt };

    let groupId = phaseToGroup(wt).groupId;
    // the single-layer trunk chapters explain one mechanism: the DiT tails appear from D0 on
    if (groupId === PhaseGroup.Detailed_Input && state.layout) {
        for (let c of state.layout.ditCubes) c.opacity = 0;
    }
    if (groupId === PhaseGroup.Intro) {
        walkthroughIntro(wtArgs);
        walkthrough01_Prelim(wtArgs);
    } else if (groupId === PhaseGroup.Detailed_Input) {
        walkthrough02_Embedding(wtArgs);
        walkthrough03_LayerNorm(wtArgs);
        walkthrough04_SelfAttention(wtArgs);
        walkthrough05_Softmax(wtArgs);
        walkthrough06_Projection(wtArgs);
        walkthrough07_Mlp(wtArgs);
        walkthrough08_Transformer(wtArgs);
        walkthrough09_Output(wtArgs);
    } else if (groupId === PhaseGroup.Heads) {
        walkthrough10_Heads(wtArgs);
    } else if (groupId === PhaseGroup.FlowMatching) {
        walkthrough11_FlowMatching(wtArgs);
    }

    wt.prevPhase = wt.phase;
    wt.prevTime = wt.time;
}
