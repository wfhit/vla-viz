import { Vec3 } from "@/src/utils/vector";
import { IPageSpec, PageId } from "./PageSpec";
import { makeEnginePage, vlaPage } from "./vlaPage";
import { modelFacts } from "./ModelFacts";

export const pageSpecs: Record<string, IPageSpec> = {
    vla: vlaPage,
    groot: makeEnginePage('groot', 'groot', modelFacts.groot,
        { angle: new Vec3(296, 16, 13.5), center: new Vec3(-8.4, 0, -481.5) }),
    pi0: makeEnginePage('pi0', 'pi0', modelFacts.pi0,
        { angle: new Vec3(296, 16, 13.5), center: new Vec3(-8.4, 0, -481.5) }),
};

export function getPageSpec(id: PageId): IPageSpec {
    let p = pageSpecs[id];
    if (!p) throw new Error(`unknown page ${id}`);
    return p;
}
