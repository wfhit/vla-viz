# 三套 VLA 可视化总体方案：/groot · /pi0 · /vla

> 状态：**第三版（已批，施工中）**。§3 起仍是设计，未下载任何权重（只用 HTTP Range 读了 safetensors 头和 config）；§2 的改名与遗留清理已按法则 5 实施完毕。
>
> **已裁决（2026-10-25）**
> 1. 顺序：**π0 先做，GR00T 后做**。
> 2. 输入：π0 用 **BEHAVIOR challenge 原始数据集（`behavior-1k-knack` 索引 → `2026-challenge-demos`）的真实 demo 帧**喂真权重做激活捕获，**不用** π0 自己的训练数据。
> 3. 许可证：**纯内部使用，不公开部署**，不再作为阻塞项。
> 4. 改名：~~可见层现在改、内部类名最后统一清~~ → **被法则 5 取代**，可见层与内部一次清完（§2，已完成）。
> 5. **不拖延法则（Frank）**：问题不往后拖——能删就删、能改就改，再小也改；判定有问题就立即改，不留到"最后统一"。前提是不破坏三套页面渲染，改完跑页面自测。本方案所有阶段都按此执行：发现的遗留命名、死代码、注释掉的代码、无效功能，当场处理，不进待办。
> 数据规则沿用 k005b 标准：真权重 + 真激活 + 窗口化；实在拿不到的部件才允许示意，并在 3D 里标 `toy`。

---

## 0. 一页结论

| | /groot | /pi0 | /vla（原 /llm） |
|---|---|---|---|
| 模型 | GR00T N1.6-3B（HF `nvidia/GR00T-N1.6-3B`） | π0 base（HF `lerobot/pi0_base`，PyTorch 版） | k005b（我们自己的） |
| 主干 | Eagle-Block2A-2B：SigLIP2 27 层 + Qwen3 **截断到 16 层** | PaliGemma：SigLIP So400m 27 层 + Gemma-2B 18 层 | Qwen3-VL-2B 28 层 |
| 动作头 | 32 块 DiT（偶数块交叉注意力 / 奇数块自注意力） | Gemma-300M 专家 18 层 | DiT 专家（跑 28 层） |
| **主干↔专家耦合** | **交叉注意力**：专家只读主干**第 16 层一次性输出**（单一来源） | **联合自注意力**：专家第 i 层与主干第 i 层拼接成一个序列 | **联合注意力**：专家第 i 层读主干第 i 层缓存的 K/V |
| 时间条件 | AdaLN（每块 scale/shift）+ 动作编码器里拼 τ | 只在动作 token 里拼 [a ‖ τ] 过 MLP（**无 adaLN**） | AdaRMSNorm（每层）+ 动作 token 里拼 [a ‖ τ] |
| 解码步数 | **4 步**，t 从 0 升到 1（`a += dt·v`） | **10 步**，t 从 1 降到 0 | **10 步**，t 从 1 降到 0 |
| 动作块 | 50 × 128（按 embodiment 截取有效维） | 50 × 32 | 16 × 23 |
| 本体处理 | 每个 embodiment 一套编码器和解码器（32 槽），含 `behavior_r1_pro`=24 | 统一 32 维 padding | 单一本体 |
| 权重体积 | 6.57 GB bf16 | 14.0 GB fp32 | 本地已有 |

一句话：三者是 **三种耦合拓扑**。GR00T 是"串联加单总线交叉注意力"，π0 是"并排、逐层拼接"，k005b 是"并排、逐层借 KV"。场景结构必须让这一差异一眼可见，不能套一个模板换数字。

---

## 1. 事实核对（全部来自 config、权重头、源码，非记忆）

### 1.1 GR00T N1.6

**来源**
- HF `nvidia/GR00T-N1.6-3B`：config.json、`model.safetensors.index.json`。
- 源码：Isaac-GR00T 仓 `n1.6-release` 分支。

**主干：Eagle**
- 视觉：SigLIP2，27 层，宽 1152，patch 14，256 个位置。
- 投影：pixel-shuffle 2×2，即 4608 → `mlp1`（LN, 4608→2048→2048）。
- 语言：Qwen3（带 q_norm/k_norm），hidden 2048，16 Q 头 × 128，8 KV 头，MLP 6144。
- `select_layer=16`：`eagle_backbone.py:52-53` 把第 16 层以后的 LLM 层直接 `pop` 掉，所以**主干只有 16 层**。
- 主干输出过 `vlln`，即 LayerNorm(2048)（`gr00t_n1d6.py:144`）。

**状态编码**
- `state_encoder` 是按 embodiment 分槽的 MLP：128 → 1024 → 1536（ReLU）。
- config 里 `apply_sincos_state_encoding=true`。

**动作编码**
- `action_encoder`（`embodiment_conditioned_mlp.py:176-206`）：
  - W1：128 → 1536。
  - 与 τ 的正弦编码拼接，经 W2：3072 → 1536，再过 swish。
  - W3：1536 → 1536。
- 之后加 `position_embedding`[1024, 1536]。

**序列**
- `sa_embs = cat(state_features, action_features)`（`gr00t_n1d6.py:335`）。
- 专家序列**只含 state + action**，VL token 不在里面。

**DiT：`AlternateVLDiT`**（`dit.py`，`n1.6-release`）
- 32 块，内部宽 1536（32 头 × 48），FF 为 GELU，1536 → 6144 → 1536。
- 每块 `norm1` 是 AdaLayerNorm：`x = LN(x)·(1+scale)+shift`，scale/shift = Linear(SiLU(temb))，[3072, 1536]。
- `idx` 为奇数：自注意力，只在 state+action 内部。
- `idx` 为偶数：交叉注意力，K/V = 主干输出 [1536, 2048]。
  - `idx % 4 == 0`：只看**文本（非图像）token**。
  - `idx % 4 == 2`：只看**图像 token**。
  - 即 attend_text_every_n_blocks=2。
- 收尾：`proj_out_1(SiLU(temb))` → shift/scale → `norm_out` → `proj_out_2`（→ 1024）。
- 最后是 `action_decoder`（按 embodiment 分槽，1024 → 1024 → 128）。

**解码**（`gr00t_n1d6.py:307-357`）
- `actions = randn(B, 50, 128)`，`dt = 1/4`。
- 循环 t = 0, 1, 2, 3：
  - `t_cont = t/4`，`t_disc = int(t_cont·1000)`，即 **0, 250, 500, 750**。
  - 计算 `actions = actions + dt · v`。
- **方向与 π0/k005b 相反**：t=0 是噪声，t=1 是数据，向上积分。

**训练**（`:200-205`）
- `x_t = (1−t)·noise + t·a`，目标 `v = a − noise`。
- t 由 `sample_time` 给出：`(1 − Beta(1.5, 1)) · 0.999`（`:137-140`）。
- 这里有一个可讲的点：GR00T 和 k005b v2 用的是同一个 Beta(1.5,1) 律，但时间轴方向相反。

**为什么主干只算一次**
- 主干对一次 `get_action` 只运行一次，DiT 运行 4 次。这是代码事实。
- "System 2 10Hz / System 1 120Hz"出自 N1 论文。场景里只标"论文声称"，不当实测值画。

### 1.2 π0（lerobot/pi0_base）

**来源**
- config.json：`paligemma_variant=gemma_2b`，`action_expert_variant=gemma_300m`，`chunk_size=50`，`max_action_dim=32`，`max_state_dim=32`，`num_inference_steps=10`，time ~ Beta(1.5,1)。
- 输入为 3 路 224² 相机：base、left_wrist、right_wrist。
- 权重头：单个 model.safetensors，F32，14.0 GB。

**结构**
- 视觉：SigLIP，27 层，宽 1152，MLP 4304，256 位置。
- `multi_modal_projector`：1152 → 2048。
- Gemma-2B：18 层，hidden 2048，q 2048（8 头 × 256），**k/v 256，即 MQA，1 个 KV 头**，MLP 16384。
- 专家 Gemma-300M：18 层，hidden 1024，q **2048**（8 头 × 256），k/v 256，o 2048 → 1024，MLP 4096。
- 专家与主干**头数和头宽完全一致**，所以两条流的 Q/K/V 能在每层拼成一个序列做一次注意力。这就是"联合自注意力"的张量前提，可以直接画出来。

**后缀**（`modeling_pi0.py:550-596`）
- token 0：`state_proj`（32 → 1024），att_mask 为 1，作为块边界。
- token 1..50：`action_in_proj`（32 → 1024）与正弦 τ 拼接（`:588`），经 `action_time_mlp_in`（2048 → 1024），swish，再经 `action_time_mlp_out`。
- `adarms_cond=None`：π0（非 π0.5）**没有 adaLN**，t 只从动作 token 进入。

**掩码**
- 前缀（图像 + 文本）内部双向。
- state 看前缀和自己。
- 动作看全部。
- 前缀不看后缀。
- 结构与 k005b 的块因果掩码同型。

**流约定**
- `FlowConvention.NOISE_AT_ONE`（`:615`）：t=1 是噪声，10 步向下积分。与 k005b v2 一致。

### 1.3 k005b

不变，见 `flow-matching-understanding.md` 和 `dit-section-design.md`：
- 前缀一次加 28 层 KV 缓存。
- 专家 768 维，16Q/2KV，运行 28 层。
- AdaRMSNorm、SwiGLU。
- 10 步 Euler 向下积分。
- 动作块 16 × 23。

---

## 2. LLM → VLA 改名与遗留清理（✅ 已完成，按法则 5 一次清完）

原计划分两批（可见层阶段 1、内部阶段 7），Frank 否决了"内部最后清"，改为立即全清。已做：

| 层 | 改动 |
|---|---|
| 路由 | `src/app/llm/` → `src/app/vla/`；**不留 /llm 跳转桩**（纯内部使用，无外链需要兼容） |
| 目录 | `git mv src/llm src/vla`；外部 import（`utils/shader.ts`、`codec/*`、`cpu/comps/Alu.tsx`）全部修正 |
| 类名/文件 | `GptModel.ts`→`VlaModel.ts`，`GptModelLayout.ts`→`VlaModelLayout.ts`，`IGptModelLayout`/`genGptModelLayout`/`IGptModelLink`/`IGptLayerNormLayout`→`IVlaModelLayout`/`genVlaModelLayout`/`IVlaModelLink`/`IVlaNormLayout`；`jsGptModel`→`vlaModel`，`gptGpuModel`→`vlaGpuModel`；`renderPhases.ts`→`bufferTex.ts`（只剩 buffer 纹理工具） |
| UI 文案 | 首页卡 "VLA Visualization"、k005b 描述、`/vla` 链接、新截图 `vla-viz-screenshot.jpg`；页头 "k005b VLA 可视化"；`package.json` 名 `vla-viz` |
| 删除的死代码 | 弹窗残留 `WelcomePopup`+`InfoButton`（弹窗早已注释，按钮是空操作）；GPT 示意图 `intro-image*.svg`、旧截图；`utils/tensor.ts`（`IGptModelConfig`）、`utils/basic.ts`；**整套键盘导航 `MovementControls`**（`manageMovement` 早被注释，WASD/QE/R/F 按键无任何效果）；bbycroft 英文旧解说 `walkthroughDetailed` 及其 4 个不在任何章节里的 Phase、画布文字解说通道 `writeCommentary`；约 70 个无引用函数/类型/导出、约 90 个无用 import、~30 个无用参数、~80 行注释掉的代码、`.idea/` 出库 |

**验收（已过）**
- `tsc` 零错误；`noUnusedLocals+noUnusedParameters` 下 `src/vla`、`src/utils` 零告警（`_exhaustCheck` 类型穷尽惯用法除外）；knip 在 `src/vla`/`src/utils` 无未用文件/导出（仅剩 2 个被 `src/cpu` 使用的）。
- `src/`、`public/`、`tools/` 里 `grep -ri "llm\|gpt"` 除第三方/base64 误报外为零。
- `next build` 通过；/vla 截图与改名前 /llm 对比：3D 场景一致，仅页头少了已失效的 "?" 按钮。

**第二轮（D7 批准后）**：bbycroft 其他应用、首页个人信息、yarn 与无用依赖全部删除（见 §10 D7）；`src/utils` 随之再清一轮无引用文件/导出。首页现为单卡（/vla），/pi0、/groot 卡随页面落地再加。

---

## 3. 代码架构：共享渲染器，模型各自独立

现状：k005b 专用代码约 5.4k 行（`export_k005b.py` + `VlaModelLayout.ts` + `VlaModel.ts` + walkthrough），布局和模型强耦合。

建议**不做"通用 VLA 布局引擎"**。三种耦合拓扑差异太大，抽象过早会把三页都拖坏。

```
src/vla/        k005b（路由 /vla）
src/groot/      GrootLayout.ts  GrootModel.ts  walkthrough/G*.tsx
src/pi0/        Pi0Layout.ts    Pi0Model.ts    walkthrough/P*.tsx
src/vizcore/    从 src/vla 抽出的纯渲染件：块/箭头/文字绘制、相机、播放条、
                窗口化 tensor 纹理、审计(audit) 钩子 —— 只抽"已被两页复用"的东西
tools/export_groot.py  tools/export_pi0.py   （以 export_k005b.py 为模板）
public/data/groot/*  public/data/pi0/*
```

- 抽 `vizcore` 是**按需**进行：做 /pi0 时发现要复制的渲染件才搬过去，并保证 /vla 截图不变。
- 每页沿用 k005b 的硬约束：
  - 箭头必须有起止块（audit 的 noIn/noOut 为空）。
  - 3D 文字只用 ASCII。
  - 无弹窗。
  - FOV 不动。
  - 真值标注。

---

## 4. /vla（k005b）

保持不动。只做改名，并在每页尾部加"对照"章（见 §7）。

---

## 5. /pi0 场景与章组

### 5.1 场景结构：并排双塔、逐层拼接

```
   [3 相机 224²]            [文本 prompt]        [state 32] [噪声 50×32]
        │                        │                  │           │
   SigLIP 27 层 ──proj──┐   Gemma 词嵌入             state_proj  action_in_proj ‖ τ → MLP
                        ▼        ▼                  ▼           ▼
                 ┌── 前缀序列 (图像 768 + 文本) ──┐  ┌─ 后缀 (1 + 50) ─┐
   层 0..17      │   Gemma-2B 层 i  (2048 宽)     │⇄│ 专家层 i (1024) │   ← 每层一道"拼接带"
                 └────────────────────────────────┘  └────────────────┘
                                                            │ action_out_proj
                                                       v (50×32) → Euler → 动作块
```

- 主干塔（宽）和专家塔（窄）**并排**，各 18 层，一一对齐。
- **每层画一道"拼接带"**，跨在两塔之间：
  - 两侧的 Q/K/V 在带上**物理拼成一条序列**。
  - 带上画块因果掩码，三块：前缀↔前缀，后缀→前缀，后缀↔后缀。
  - 注意力后**再拆回**两侧，各自走 o_proj 和 MLP。
- 与 k005b 的视觉差别：k005b 是专家"借"主干 KV，箭头单向（主干→专家）；π0 是两侧真正拼成一次注意力，但前缀行被掩码挡住不看后缀，所以箭头仍是单向。两页都要把这一点用掩码热力图画出来。
- 宽度差异直接可见：主干 2048，专家 1024；头宽相同（256），这是能拼接的原因。

### 5.2 章组（播放顺序）

| 章 | 内容 | 真值 |
|---|---|---|
| P0 总览 | 双塔、拼接带、10 步回路的一屏地图 | 层数、宽度、参数量（来自权重头） |
| P1 前缀 | 3 路图像经 SigLIP（窗口化 patch）、projector，得到图像 token；文本经 Gemma 嵌入 | 真 patch 嵌入窗口、真 token id |
| P2 后缀 | state_proj；动作 token = MLP([a ‖ τ])；**指出无 adaLN**，t 只从这里进 | 真 W 窗口、真 τ 正弦向量 |
| P3 一层解剖 | RMSNorm → Q/K/V（两侧各自投影）→ **拼接** → 掩码注意力 → 拆回 → o_proj → 残差 → GeGLU MLP → 残差 | 第 0 层真激活窗口 |
| P4 逐层穿行 | 前缀和后缀同时穿过 18 层，拼接带逐层点亮 | 每层残差范数 |
| P5 前缀 KV 复用 | 推理时前缀只算一次，后缀 10 次复用 18 层 KV | 真 KV 缓存窗口 |
| P6 10 步 Euler | 噪声 → 10 次专家穿行（t = 1.0, 0.9, …, 0.1）→ 轨迹带 → 动作块 50×32 | 每步真 x_t、v、‖v‖ |
| P7 对照 | 与 /vla、/groot 的差异条目（§7） | — |

### 5.3 数据：`tools/export_pi0.py`

- **权重**：`lerobot/pi0_base`，14 GB fp32。导出时用 bf16 跑前向，约 7 GB 显存，5090 足够；窗口切片保留 fp32 原值。
- **输入样本（已裁决）**：BEHAVIOR challenge 原始 demo 帧，不用 π0 自己的训练数据。
  - **数据位置**：
    - 索引：`/home/frank/workspace/behavior_challenge_data/dataset/behavior-1k-knack/index`（schema.yaml、episodes.parquet、tasks.parquet、stats.json）。
    - 原始数据：`bytes_pointer.json` 指向容器路径 `/workspace/dataset/2026-challenge-demos/{data,videos}`，宿主机对应 `.../behavior_challenge_data/dataset/2026-challenge-demos/`。
    - 数据为 R1Pro，30 fps，20000 条 episode；state 61 维，action 23 维；相机为 HEVC 视频。
  - **选帧**：默认与 k005b 导出**同一 episode、同一 frame**。`export_k005b.py` 已记录 row、episode、frame，三页可同帧对照。帧直接从 demos 的 parquet 和 hevc 视频解码，不走 k005b 的训练样本表。
  - **相机映射**（3 路 RGB → π0 的 3 个输入槽，各 resize/pad 到 224²）：
    - `observation.rgb.zed_link_camera_0`（头部）→ `base_0_rgb`
    - `observation.rgb.left_realsense_link_camera_0` → `left_wrist_0_rgb`
    - `observation.rgb.right_realsense_link_camera_0` → `right_wrist_0_rgb`
    - 深度通道不用。
  - **语言**：该 episode 的任务描述来自 `tasks.parquet`，经 PaliGemma tokenizer 编码。
  - **R1 Pro embodiment 配置（D6 已撤销，非问题）**：state 维度与 action 维度本来独立，各自经自己的投影进隐空间（k005b 即 `state_proj` 61→768 与 `action_in_proj` 23→768 互不相干；BEHAVIOR 官方 GR00T N1.7 baseline 也是写一份 R1 Pro embodiment 配置：state 61→隐层、action 23→隐层）。π0 同理：给它的状态/动作编解码层写一份 R1 Pro 配置（`tools/pi0_r1pro.yaml`：state 61、action 23、三相机键），**不做任何跨维映射设计**。
    - 若 `pi0_base` 里没有 R1 Pro 已训练的编解码权重，这几层是唯一非真值部件，3D 中标 `toy: R1 Pro adapter (untrained)`；主干与专家权重、全部激活仍是真值。
  - **归一化**：pi0_base 没有 R1Pro 统计量。state 用 `index/stats.json` 做 mean/std 归一化，并在导出 JSON 中记录所用统计量。
- **真值标注**：pi0_base 是预训练底模，没在 BEHAVIOR 上微调。激活和权重是真的，但输出动作对 R1Pro **无语义保证**。
  - 场景里标一行 `pi0_base zero-shot on BEHAVIOR frame: activations real, actions not calibrated for R1Pro`。
  - 动作块按 R1 Pro 配置解码为 50×23，按 R1Pro 名称标注。
- **hook**：
  - 每层两侧 RMSNorm 输入输出、Q/K/V、注意力权重（窗口化）、MLP。
  - 10 步循环里的 x_t、v、t。
  - 前缀 KV 缓存。
- **自检**：
  - 导出器内部重放 Euler 更新，与 hook 值逐元素对照。
  - 残差恒等式精确成立（同 k005b）。

---

## 6. /groot 场景与章组

### 6.1 场景结构：串联、单总线交叉注意力

```
 [图像] → SigLIP2 27 层 → pixel-shuffle → mlp1 ┐
 [文本] → Qwen3 嵌入 ─────────────────────────┴→ Qwen3 16 层（第 17..28 层画成"被截掉的虚影"）
                                                     │ vlln (LayerNorm)
                                                     ▼
                                   ══ VL 特征总线 (图像段 | 文本段) ══════════════╗
                                                                                  ║
 [state] → state_enc[emb=24] ┐                                                    ║
 [噪声 50×128] → action_enc[emb=24](a ‖ τ) + pos ┴→ sa 序列 (state + 50 action)   ║
                                                     │                            ║
   DiT 块 0  交叉 ← 读总线·文本段 ═══════════════════════════════════════════════╣
   DiT 块 1  自注意力 (只在 sa 内部)                                               ║
   DiT 块 2  交叉 ← 读总线·图像段 ═══════════════════════════════════════════════╣
   DiT 块 3  自注意力                                                              ║
   …… ×8 组（文/自/图/自）                                                        ║
   final AdaLN → proj_out_2 → action_dec[emb=24] → v (50×128) → Euler(4 步，向上)
```

- **主干和 DiT 是串联的，不是并排**：DiT 堆放在主干之后（场景纵向延伸）。
- 主干只把最后（第 16 层）的输出铺成一条 **VL 特征总线**，与 DiT 堆平行。
- 交叉块从总线**取 K/V**：
  - 文本块的箭头连到总线文本段，图像块的连到图像段。
  - 两种块用颜色区分，"每 4 块一轮"的节律可见。
- 自注意力块没有外连箭头，只在 sa 序列内部。
- 截断的 LLM 第 17..28 层画成**淡色虚影**，标 `pruned (select_layer=16)`，说明 GR00T 只用 Eagle 的中层特征。这是真事实，不是 toy。
- **embodiment 槽**：
  - 状态编码器、动作编码器、动作解码器画成 32 槽的抽屉。
  - 高亮第 24 槽 `behavior_r1_pro`，其余槽淡显。
  - 这是本页独有的看点，而且正好是我们的机器人。
- **AdaLN**：每块侧面一个小 γβ（scale/shift）读数，来自 temb；4 步之间这些值随 t 变化。

### 6.2 章组

| 章 | 内容 | 真值 |
|---|---|---|
| G0 总览 | 串联拓扑、VL 总线、32 块节律、4 步回路 | 层数、宽度、截断位置 |
| G1 System 2：Eagle 主干 | 图像 → SigLIP2 → pixel-shuffle → mlp1；文本嵌入；Qwen3 16 层逐层穿行；截断虚影；vlln | 真 patch 窗口、每层残差范数 |
| G2 embodiment 抽屉 | 32 槽编码器和解码器，选中 24 = behavior_r1_pro；state 编码 | 真 W 槽切片 |
| G3 动作 token | W1(a) ‖ sincos(τ) → W2 → swish → W3 → +pos | 真 τ 编码向量 |
| G4 DiT 块解剖（交叉块） | AdaLN（temb → scale/shift，作用于 LN 后）→ 交叉注意力（Q 来自 sa，K/V 来自总线）→ 残差 → AdaLN/LN → GELU FF → 残差 | 块 0 真激活、真 scale/shift |
| G5 自注意力块与交替节律 | 块 1 自注意力；块 0/2/4… 文本/图像交替；32 块逐块穿行 | 每块注意力在文本段与图像段上的质量 |
| G6 4 步流匹配 | 噪声 → 4 次 DiT 穿行（t_disc = 0, 250, 500, 750）→ `a += ¼·v` → 动作块；**标出时间方向与 π0/k005b 相反** | 每步真 x_t、v |
| G7 对照 | §7 | — |

### 6.3 数据：`tools/export_groot.py`

- **权重**：`nvidia/GR00T-N1.6-3B`，6.57 GB bf16，跑前向约 8 GB 显存。
- **代码**：clone Isaac-GR00T 的 `n1.6-release`，用它的 processor 和 model 类加载。只用代码，不改模型。
- **输入**：同一帧 BEHAVIOR 观测，embodiment = `behavior_r1_pro`（24）。
  - `statistics.json` 里有该本体的归一化统计，所以输出动作**有语义**，可与 k005b 同帧对比。这是三页里最强的并列对照。
  - 前提：需确认 R1Pro 的 modality 配置（相机键、state/action 维度切片）能从仓库的 embodiment 配置里取到。确认不了时退回仓库自带示例数据（如 GR1 或 G1），见决策点 D3。
- **hook**：
  - SigLIP2 和 Qwen3 每层残差。
  - vlln 输出。
  - DiT 每块的 temb→scale/shift、注意力（交叉块按文本段和图像段分别窗口化）、FF。
  - 4 步的 x_t、v、t_disc。
- **自检**：
  - Euler 重放对照。
  - 交叉块的掩码与 image_mask 一致性检查。
  - AdaLN 公式重算对照（同 D1 的 γβ 验收）。

### 6.4 可能的 toy 部件

- 预计**无**。全部部件权重都在 checkpoint 里，源码完整。
- 唯一示意项是 "System 2 10Hz / System 1 120Hz" 的频率标注：来自论文，以文字标 `paper claim`，不做成动画时间轴。

---

## 7. 并列对照

不另建 /compare 页。做法如下：
1. **首页**：三张卡片并列（/groot、/pi0、/vla），各配一张场景截图和一句耦合拓扑描述。
2. **每页末章（G7/P7/V-对照）**：同一张差异表（即 §0 的表），当前模型那列高亮。每一行可点击，跳到另一页对应的章，例如"耦合"跳到对方的 G5/P4/D 组。
3. **同帧同机器人**：/pi0 和 /vla 已确定用同一帧 BEHAVIOR demo；/groot 视 D3 而定。最后一章并列画动作块，注明 π0 是 zero-shot，k005b（以及 GR00T 若用 embodiment 24）针对 R1Pro。

---

## 8. 工作量与阶段

| 阶段 | 内容 | 估计 |
|---|---|---|
| 1 | ✅ LLM → VLA 改名与遗留清理（全部，§2）；首页三卡占位 | 已完成（卡位随 2/4 阶段补） |
| 2 | export_pi0.py（下载 14 GB；BEHAVIOR demo 帧解码与相机映射；R1 Pro embodiment 配置；hook；自检） | 1–1.5 天 |
| 3 | /pi0 布局 + 模型连接 + P0–P7（按需抽 vizcore） | 3–4 天 |
| 4 | export_groot.py（clone n1.6-release、R1Pro 配置、hook、自检） | 1–1.5 天 |
| 5 | /groot 布局（串联 + 总线 + 抽屉）+ G0–G7 | 4–5 天 |
| 6 | 对照章、首页定稿、三页审计和截图验收 | 1 天 |
| **合计** | | **约 11–13 个工作日**（剩余） |

**顺序理由**
- π0 和 k005b 拓扑最近（并排、逐层、10 步向下），可以复用最多，也最先逼出 vizcore 的边界。
- GR00T 拓扑最异，放最后。
- step_160000 落盘后穿插执行 /vla 的 D 组换数据（与本计划独立）。

---

## 9. 风险

| 风险 | 影响 | 对策 |
|---|---|---|
| 许可证：GR00T 为 NVIDIA OneWay Noncommercial；π0 权重受 Gemma 条款约束 | **已裁决：纯内部使用，不公开部署**，非阻塞 | 页面角落注明来源和许可证名称即可 |
| GR00T 的 R1Pro modality 配置不在公开仓 | 不能用 BEHAVIOR 帧做出有语义的动作 | 退回仓库示例本体（D3） |
| pi0_base 对 R1Pro 是 zero-shot | 动作无语义 | 已裁决仍用 BEHAVIOR 帧；3D 内明确标注 |
| demo 视频为 HEVC | 导出器需要能解码 HEVC 的环境 | 复用 behavior_challenge 现有数据加载代码；不行就用 ffmpeg 抽单帧 |
| 显存/内存 | 14 GB fp32 加载需要约 28 GB 内存峰值（60 GB 够） | bf16 前向；按需 mmap 读取窗口 |
| 导出数据体积 | 三页 public/data 膨胀 | 只存窗口；单页目标 < 30 MB（与 k005b 同量级） |
| GR00T 主分支已是 N1.7 | 代码与 N1.6 权重不匹配 | 固定 `n1.6-release` 的 commit hash |

---

## 10. 裁决记录与剩余问题

| # | 问题 | 结论 |
|---|---|---|
| D1 | 页面顺序 | ✅ π0 先，GR00T 后 |
| D2 | π0 输入帧 | ✅ BEHAVIOR 原始 demo 真实帧（`behavior-1k-knack` → `2026-challenge-demos`），不用 π0 训练数据 |
| D3 | GR00T 输入帧 | ⏳ 待定，GR00T 阶段再议。推荐 BEHAVIOR 同帧 + embodiment 24，R1Pro 配置拿不到时退回示例本体 |
| D4 | 许可证 | ✅ 纯内部使用，不公开部署 |
| D5 | 改名范围 | ✅ 法则 5：可见层与内部一次清完，已完成（§2） |
| D6 | π0 的 state 映射 | ✅ **撤销（伪问题）**：state/action 各自投影进隐层，写一份 R1 Pro embodiment 配置即可（§5.3）。此类本体适配问题以后按此模式直接处理，不再提问 |
| D7 | bbycroft 其他应用与首页个人信息 | ✅ 已全部删除：`src/cpu`、`src/codec`、`src/fluidsim`、`src/config` 及路由、`public/riscv`、`me.jpeg`、个人简介/捐助链接、`progress/`、`ols.json`、空 `.gitmodules`、yarn（统一 npm）、13 个无用依赖、死的 `eslintConfig`/`resolutions`、`create-symbol-font.jsm` |
