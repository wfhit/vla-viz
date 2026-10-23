import { Vec3 } from "@/src/utils/vector";
import { Phase } from "./Walkthrough";
import { commentary, DimStyle, IWalkthroughArgs, moveCameraTo, setInitialCamera } from "./WalkthroughTools";
import { lerp, lerpSmoothstep } from "@/src/utils/math";
import { processUpTo, startProcessBefore } from "./Walkthrough00_Intro";

export function walkthrough06_Projection(args: IWalkthroughArgs) {
    let { walkthrough: wt, state, layout, tools: { breakAfter, afterTime, c_blockRef, c_dimRef, cleanup } } = args;

    if (wt.phase !== Phase.Input_Detail_Projection) {
        return;
    }

    setInitialCamera(state, new Vec3(-73.167, 0.000, -270.725), new Vec3(293.606, 2.613, 1.366));
    let block = layout.blocks[0];
    wt.dimHighlightBlocks = [...block.heads.map(h => h.vOutBlock), block.projBias, block.projWeight, block.attnOut];

    let outBlocks = block.heads.map(h => h.vOutBlock);

    commentary(wt, null, 0)`

注意力计算完成后，每个 head 都得到了自己的输出：由 Q/K 决定权重、对 V 向量加权混合后的结果。

要合并各 head 的${c_blockRef('输出向量', outBlocks)}，只需把它们首尾相接（concat）。以时间位置
${c_dimRef('t = 4', DimStyle.T)} 为例，3 个长度为 ${c_dimRef('A = 16', DimStyle.A)} 的向量被拼成 1 个长度为 ${c_dimRef('C = 48', DimStyle.C)} 的向量。`;

    breakAfter();

    let t_fadeOut = afterTime(null, 1.0, 0.5);
    // let t_zoomToStack = afterTime(null, 1.0);
    let t_stack = afterTime(null, 1.0);

    breakAfter();

    commentary(wt)`

每个 head 内的向量长度（${c_dimRef('A = 16', DimStyle.A)}）等于 ${c_dimRef('C', DimStyle.C)} / head 数，
因此拼回去后长度恰好还是 ${c_dimRef('C', DimStyle.C)}。（这里是 3 层的示意小模型；k005b 真实尺寸要大得多。）

接着用输出投影矩阵（output projection）对每一列做矩阵-向量乘法并加上 bias，把各 head 的信息混合起来，
得到这一层注意力的输出。`;

    breakAfter();

    let t_process = afterTime(null, 3.0);

    breakAfter();

    commentary(wt)`

现在得到了注意力层的输出。我们不直接把它交给下一阶段，而是逐元素加回到这一层的输入上，
即绿色竖直箭头所示的_残差连接_（residual connection / residual pathway）。
在 k005b 中，prefix（VLM）流和 suffix（action expert）流各自经过自己的输出投影，并分别加回各自的残差流。
`;

    breakAfter();

    let t_zoomOut = afterTime(null, 1.0, 0.5);
    let t_processResid = afterTime(null, 3.0);

    cleanup(t_zoomOut, [t_fadeOut, t_stack]);

    breakAfter();

    commentary(wt)`

与 RMSNorm 一样，残差通路对深层网络的有效训练至关重要：每一层只需学习“增量”，信息和梯度可以沿残差流直通。

注意力部分到此结束，接下来进入这一层的后半部分：MLP（前馈网络）。
`;

    breakAfter();

    if (t_fadeOut.active) {
        for (let head of block.heads) {
            for (let blk of head.cubes) {
                if (blk !== head.vOutBlock) {
                    blk.opacity = lerpSmoothstep(1, 0, t_fadeOut.t);
                }
            }
        }
    }

    if (t_stack.active) {
        let targetZ = block.attnOut.z;
        for (let headIdx = 0; headIdx < block.heads.length; headIdx++) {
            let head = block.heads[headIdx];
            let targetY = head.vOutBlock.y + head.vOutBlock.dy * (headIdx - block.heads.length + 1);
            head.vOutBlock.y = lerp(head.vOutBlock.y, targetY, t_stack.t);
            head.vOutBlock.z = lerp(head.vOutBlock.z, targetZ, t_stack.t);
        }
    }

    let processInfo = startProcessBefore(state, block.attnOut);

    if (t_process.active) {
        processUpTo(state, t_process, block.attnOut, processInfo);
    }

    moveCameraTo(state, t_zoomOut, new Vec3(-8.304, 0.000, -175.482), new Vec3(293.606, 2.623, 2.618));

    if (t_processResid.active) {
        processUpTo(state, t_processResid, block.attnResidual, processInfo);
    }
}
