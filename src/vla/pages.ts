import { IPageSpec, PageId } from "./PageSpec";
import { vlaPage } from "./vlaPage";
import { grootPage } from "../groot/grootPage";
import { pi0Page } from "../pi0/pi0Page";

export const pageSpecs: Record<string, IPageSpec> = {
    vla: vlaPage,
    groot: grootPage,
    pi0: pi0Page,
};

export function getPageSpec(id: PageId): IPageSpec {
    let p = pageSpecs[id];
    if (!p) throw new Error(`unknown page ${id}`);
    return p;
}
