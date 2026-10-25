export interface IBufferTex {
    width: number;
    height: number;
    channels: number;
    texture: WebGLTexture;
    localBuffer?: Float32Array;
}

export function createBufferTex(gl: WebGL2RenderingContext, width: number, height: number, channels: number): IBufferTex {
    let texture = gl.createTexture()!;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    let [format, iformat] = channelsToFormat(gl, channels);
    gl.texImage2D(gl.TEXTURE_2D, 0, iformat, width, height, 0, format, gl.FLOAT, null);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

    return {
        width,
        height,
        texture,
        channels,
    };
}

export function writeToBufferTex(gl: WebGL2RenderingContext, buffer: IBufferTex, data: Float32Array) {
    if (data.length !== buffer.width * buffer.height * buffer.channels) {
        throw new Error('Data length does not match buffer size');
    }
    gl.bindTexture(gl.TEXTURE_2D, buffer.texture);
    let [format] = channelsToFormat(gl, buffer.channels);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, buffer.width, buffer.height, format, gl.FLOAT, data);
}

function channelsToFormat(gl: WebGL2RenderingContext, channels: number): [GlFormat, GlInternalFormat] {
    switch (channels) {
        case 1: return [gl.RED, gl.R32F];
        case 2: return [gl.RG, gl.RG32F];
        case 3: return [gl.RGB, gl.RGB32F];
        case 4: return [gl.RGBA, gl.RGBA32F];
        default: throw new Error(`Invalid number of channels: ${channels}. Must be 1, 2, 3, or 4.`);
    }
}

enum GlFormat {
    RED = 0x1903,
    RG = 0x8227,
    RGB = 0x1907,
    RGBA = 0x1908,
}

enum GlInternalFormat {
    R32F = 0x822E,
    RG32F = 0x8230,
    RGB32F = 0x8815,
    RGBA32F = 0x8814,
}
