// The structural numbers and names each engine instance's chapter texts quote. They are the
// checkpoints' real config values (tools/export_*.py write the same values into config).

export interface IModelFacts {
    id: 'k005b' | 'pi0' | 'groot';
    modelName: string;        // ASCII (3D model card)
    title: string;            // page / overview title
    trunkName: string;        // the VLM language trunk
    visionName: string;
    nLayer: number;           // trunk layers (k005b / pi0: = expert layers = layer pairs)
    nExpert: number;          // expert / DiT layers
    nHead: number;
    nKv: number;
    headDim: number;
    dModel: number;
    ffn: number;
    expertD: number;
    expertHeads: number;
    expertKv: number;
    expertFfn: number;
    nSuffix: number;          // expert tokens
    actH: number;             // action horizon
    actD: number;             // action dims of the model's output (robot dims are actDReal)
    actDReal: number;
    K: number;                // Euler steps of the decode
    attnName: string;         // 'GQA' | 'MQA'
    mlpName: string;          // 'SwiGLU' | 'GeGLU'
    mlpAct: string;           // 'SiLU' | 'GELU'
    adaName: string;          // the expert's time-conditioned norm
    stateToken: boolean;      // suffix token 0 = state_proj(state)
    heads: boolean;           // knack / done heads on the trunk
    causal: boolean;          // the prefix attention is causal
    qkNorm: boolean;
    expertName: string;       // what the page calls the action expert
    trunkLabel: string;       // ASCII 3D tower labels
    expertLabel: string;
}

export const modelFacts: Record<IModelFacts['id'], IModelFacts> = {
    k005b: {
        id: 'k005b', modelName: 'k005b VLA', title: 'k005b VLA',
        trunkName: 'Qwen3-VL-2B', visionName: 'Qwen3-VL ViT',
        nLayer: 28, nExpert: 28, nHead: 16, nKv: 8, headDim: 128, dModel: 2048, ffn: 6144,
        expertD: 768, expertHeads: 16, expertKv: 2, expertFfn: 2048,
        nSuffix: 17, actH: 16, actD: 23, actDReal: 23, K: 10,
        attnName: 'GQA', mlpName: 'SwiGLU', mlpAct: 'SiLU', adaName: 'AdaRMSNorm',
        stateToken: true, heads: true, causal: true, qkNorm: true, expertName: 'DiT 动作专家',
        trunkLabel: 'Qwen3-VL trunk', expertLabel: 'DiT action expert',
    },
    pi0: {
        id: 'pi0', modelName: 'pi0.5 b1k', title: 'π0.5（BEHAVIOR 微调）',
        trunkName: 'PaliGemma (Gemma-2B)', visionName: 'SigLIP So400m/14',
        nLayer: 18, nExpert: 18, nHead: 8, nKv: 1, headDim: 256, dModel: 2048, ffn: 16384,
        expertD: 1024, expertHeads: 8, expertKv: 1, expertFfn: 4096,
        nSuffix: 32, actH: 32, actD: 32, actDReal: 23, K: 10,
        attnName: 'MQA', mlpName: 'GeGLU', mlpAct: 'GELU', adaName: '门控 adaRMSNorm',
        stateToken: false, heads: false, causal: false, qkNorm: false, expertName: '动作专家',
        trunkLabel: 'PaliGemma trunk', expertLabel: 'Gemma action expert',
    },
    groot: {
        id: 'groot', modelName: 'GR00T N1.7-3B', title: 'GR00T N1.7',
        trunkName: 'Cosmos-Reason2-2B（Qwen3-VL 结构）', visionName: 'Qwen3-VL ViT (24 blocks)',
        nLayer: 20, nExpert: 32, nHead: 16, nKv: 8, headDim: 128, dModel: 2048, ffn: 6144,
        expertD: 1536, expertHeads: 32, expertKv: 32, expertFfn: 6144,
        nSuffix: 41, actH: 40, actD: 23, actDReal: 23, K: 4,
        attnName: 'GQA', mlpName: 'SwiGLU', mlpAct: 'SiLU', adaName: 'AdaLayerNorm',
        stateToken: true, heads: false, causal: true, qkNorm: true, expertName: 'DiT 动作头',
        trunkLabel: 'Cosmos-Reason2 (Qwen3-VL) trunk', expertLabel: 'GR00T DiT action head',
    },
};
