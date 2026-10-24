import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { cameraKeyframes, commentary, frameCubes, IWalkthroughArgs, phaseClock, setInitialCamera } from "./WalkthroughTools";
import { clamp } from "@/src/utils/data";
import { describeActionRow, fmtSigned, normalizeAction } from "../actionDims";

// the head-specific visuals are drawn by components/HeadsOverlay.tsx, driven by wt.time over [0, 12]
export const headsAnimLength = 12;

export function walkthrough10_Heads(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { afterTime } } = args;
    let cfg = layout.shape.full;
    let m = state.jsGptModel;

    let L = layout;
    let headBlocks = [L.ln_f.lnResid, L.hStar, L.lmHeadWeight, L.logits, L.logitsSoftmax, L.doneWeight, L.doneLogits, L.doneProbs];
    let fieldBox = (() => {
        let f = L.actionField;
        let h = f.heightScale * 2.6;
        return { x: f.x - 8, y: f.y - h, z: f.z, dx: f.nDims * f.pitch + 30, dy: 2 * h + 20, dz: f.nSteps * f.pitch + 4 };
    })();
    let actionBlocks = [L.expertOut, L.actionWeight, L.actionVelocity, L.actionX1, L.actionChunk, fieldBox];
    // the 2D panel sits on the right during the action chapters: keep the 3D content left of it
    let side = { shiftX: 0.9, aspect: 0.8 };

    if (wt.phase === Phase.Head_Knack || wt.phase === Phase.Head_Done || wt.phase === Phase.Harness_Law) {
        let cam = frameCubes(headBlocks, 282, 10, { fill: 1.3 });
        setInitialCamera(state, cam.center, cam.angle);
    }

    switch (wt.phase) {
    case Phase.Head_Knack: {
        let c = commentary(wt, null, 0)`

从 h* 出发，第一路是 knack 分类头：一个 17 × 2048 的线性层（knack_head），对 17 个 knack（${cfg?.knack_names.join(' / ') ?? '…'}）各输出一个 logit，再做 softmax。

右侧的条形图是本样本的真实 logits / 概率：argmax 为 ${cfg?.pred_knack ?? '…'}，标注目标为 ${cfg?.sample.target_knack ?? '…'}。

因为是一次分类，没有逐 token 的自回归解码，也不需要采样温度。

`;
        afterTime(c, headsAnimLength);
    } break;
    case Phase.Head_Done: {
        let c = commentary(wt, null, 0)`

第二路是 done 头：一个 2 × 2048 的线性层（done_head），输出 [NOT_DONE, DONE] 两个 logit。

done 只回答“当前 subtask 是否已经完成”，它与 knack 头共享 h*，但各自独立做 softmax。本样本预测 done = ${String(cfg?.pred_done ?? '…')}，目标 ${String(cfg?.sample.target_done ?? '…')}。

`;
        afterTime(c, headsAnimLength);
    } break;
    case Phase.Harness_Law: {
        let c = commentary(wt, null, 0)`

最后是 Harness 法则，它规定了模型与系统之间的分工：

done = true → 播完当前 chunk → 服务端刹车并定格 → 推进到下一个 subtask。

模型只负责报告 done，绝不自己切换 subtask。本样本中模型的真实输出为 knack = ${cfg?.pred_knack ?? '…'}、done = ${String(cfg?.pred_done ?? '…')}。

`;
        afterTime(c, headsAnimLength);
    } break;
    }
}
