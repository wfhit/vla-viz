import { camScaleToScreen } from "../Camera";
import { IProgramState } from "../Program";
import { drawText, IFontOpts, measureText } from "../render/fontRender";
import { RenderPhase } from "../render/sharedRender";
import { Mat4f } from "@/src/utils/matrix";
import { Vec3, Vec4 } from "@/src/utils/vector";
import { drawRoundedRect } from "./DataFlow";

export function drawBlockInfo(state: IProgramState) {

    // same-name tags: one per screen, on the block nearest the camera's focus point
    let c = state.camera.center;
    let focus = new Vec3(c.x, -c.z, -c.y);
    let best = new Map<string, { d: number, blk: typeof state.layout.cubes[0] }>();
    for (let blk of state.layout.cubes) {
        if (blk.opacity === 0 || !blk.name || blk.noTag) continue;
        let d = new Vec3(blk.x + blk.dx / 2, blk.y + blk.dy / 2, blk.z + blk.dz / 2).sub(focus).len();
        let cur = best.get(blk.name);
        if (!cur || d < cur.d) best.set(blk.name, { d, blk });
    }

    for (let { blk } of best.values()) {

        let blkTopMid = new Vec3(blk.x + blk.dx / 2, blk.y, blk.z + blk.dz / 2);

        let scale = camScaleToScreen(state, blkTopMid);

        scale = Math.min(scale, state.page.tagScaleMax ?? 1.45);

        let textColor = new Vec4(1, 1, 1, 1).mul(blk.opacity);
        let bgColor = new Vec4(0, 0, 0, 1).mul(blk.opacity);

        if (blk.opacity === 0 || !blk.name || blk.noTag) {
            continue;
        }

        // draw text, centered on top of the block
        let text = blk.name;
        let mtx = Mat4f.fromTranslation(blkTopMid);
        let textOpts: IFontOpts = { color: textColor, size: scale * 2.5, mtx };
        let textW = measureText(state.render.modelFontBuf, text, textOpts);

        let pad = 0.4;
        state.render.sharedRender.activePhase = RenderPhase.Opaque;
        drawRoundedRect(state.render, new Vec3(-textW / 2 - pad, -textOpts.size - pad * 2, 0), new Vec3(textW / 2 + pad, 0, 0), bgColor, mtx, scale * 0.4);

        state.render.sharedRender.activePhase = RenderPhase.Overlay;
        drawText(state.render.modelFontBuf, text, -textW / 2, -textOpts.size - pad, textOpts);
    }
}
