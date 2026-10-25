
export function makeArray<T = number>(length: number, val?: T): T[] {
    return new Array(length).fill(val ?? 0);
}

export function makeArrayRange(length: number, min: number, max: number): number[] {
    return new Array(length).fill(0).map((_, i) => min + i * (max - min) / (length - 1));
}

export function isNil(a: any): a is null | undefined {
    return a === null || a === undefined;
}

export function isNotNil<T>(a: T | null | undefined): a is T {
    return a !== null && a !== undefined;
}

export function clamp(num: number, min: number, max: number) {
    if (num < min) {
        return min;
    } else if (num > max) {
        return max;
    }
    return num;
}

export function base64ToArrayBuffer(base64: string) {
    let binaryString = window.atob(base64);
    let len = binaryString.length;
    let bytes = new Uint8Array(len);
    for (let i = 0; i < len; i++) {
        bytes[i] = binaryString.charCodeAt(i);
    }
    return bytes.buffer;
}
