import { Vec3 } from "./vector";

/* Cubic bezier curve defined by 4 control points.  */

interface IBezierQueueItem {
    p0: Vec3;
    p1: Vec3;
    p2: Vec3;
    p3: Vec3;
}

let _cacheRes = new Float32Array(1024 * 3);

export function bezierCurveBuild(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, threshold: number) {
    let res = _cacheRes;
    let resOff = 0;

    let queue: IBezierQueueItem[] = [];
    function push(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3) {
        queue.push({ p0, p1, p2, p3 });
    }
    push(p0, p1, p2, p3);
    p3.writeToBuf(res, resOff);
    resOff += 3;
    let iter = 0;

    while (queue.length > 0) {
        let { p0, p1, p2, p3 } = queue.pop()!;

        let q0 = p0.mid(p1);
        let q1 = p1.mid(p2);
        let q2 = p2.mid(p3);
        let r0 = q0.mid(q1);
        let r1 = q1.mid(q2);
        let s0 = r0.mid(r1); // = mid

        let d03 = p3.sub(p0);
        let d31 = p1.sub(p3);
        let d32 = p2.sub(p3);
        let d2a = Math.abs(d31.y * d03.z - d31.z * d03.y);
        let d2b = Math.abs(d32.y * d03.z - d32.z * d03.y);

        let needsSubdivion = ((d2a + d2b) * (d2a + d2b) > threshold * d03.lenSq());

        if (needsSubdivion) {
            push(p0, q0, r0, s0);
            push(s0, r1, q2, p3);
        } else {
            if (resOff + 6 > res.length) {
                let newRes = new Float32Array(res.length * 2);
                newRes.set(res);
                res = newRes;
            }
            p0.writeToBuf(res, resOff);
            resOff += 3;
        }
        iter++;
    }

    _cacheRes = res;
    return res.slice(0, resOff);
}

