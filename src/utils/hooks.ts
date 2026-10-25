import { useEffect, useReducer, useRef } from "react";

export function useFunctionRef<T extends ((...args: any[]) => any) | undefined>(fn: T): React.MutableRefObject<T> {
    let ref = useRef<T>(fn);
    useEffect(() => {
        ref.current = fn;
    }, [fn]);
    return ref;
}

export function useRequestAnimationFrame(active: boolean, cb: (dtSeconds: number) => void) {
    let cbRef = useFunctionRef(cb);
    useEffect(() => {
        let stale = false;
        let handle: number;
        let prevTimeMs: number | undefined;

        function loop(timeMs: number) {
            let dtSeconds = (prevTimeMs === undefined ? 16 : (timeMs - prevTimeMs)) / 1000;
            prevTimeMs = timeMs;
            cbRef.current(dtSeconds);
            if (!stale) {
                handle = requestAnimationFrame(loop);
            }
        }

        if (active) {
            handle = requestAnimationFrame(loop);
            return () => {
                stale = true;
                cancelAnimationFrame(handle);
            };
        }
    }, [active, cbRef]);
}


export class Subscriptions {
    subs = new Set<() => void>();
    subscribe = (fn: () => void): (() => void) => {
        this.subs.add(fn);
        return () => this.subs.delete(fn);
    }
    notify = () => {
        for (let sub of this.subs) {
            sub();
        }
    }
}

export function useSubscriptions(subscription: Subscriptions | null) {
    let [, refresh] = useReducer(a => a + 1, 0);
    useEffect(() => subscription?.subscribe(refresh), [subscription]);
}
