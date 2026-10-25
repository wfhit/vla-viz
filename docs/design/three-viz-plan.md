# 三套 VLA 可视化总体方案：/groot · /pi0 · /vla

> 状态：**第三版（已批，施工中）**。§3 起仍是设计，未下载任何权重（只用 HTTP Range 读了 safetensors 头和 config）；§2 的改名与遗留清理已按法则 5 实施完毕。
>
> **已裁决（2026-10-25）**
> 1. 顺序：**π0 先做，GR00T 后做**。
> 2. 输入：π0 用 **BEHAVIOR challenge 原始数据集（`behavior-1k-knack` 索引 → `2026-challenge-demos`）的真实 demo 帧**喂真权重做激活捕获，**不用** π0 自己的训练数据。
> 2b. **（补充裁决）/pi0 与 /groot 两页输入帧统一用本地 BEHAVIOR 原始数据**：`behavior-1k-knack`（元数据/索引）+ `2026-challenge-demos`（真实 parquet + HEVC 视频）。**权重照旧下载开源真权重**（`lerobot/pi0_base`、`nvidia/GR00T-N1.6-3B`），不改用本地已有的 π0.5 COMET / GR00T N1.7 微调权重。
> 2c. **（权重裁决，取代 2b 的"下载开源权重"）零下载，全部用 NAS 本地模型**（`~/workspace/model/` 即 NAS 挂载，只读）：
>     - π0 页：`openpi_comet/pi05-b1kpt12-cs32`、`openpi_comet/pi05-b1kpt50-cs32`（π0.5 在 BEHAVIOR-1K 上的微调版，各 12 GB）+ HF 缓存 `models--lerobot--pi0_base`（π0 base）。
>     - GR00T 页：`nvidia/GR00T-N1.7-3B`（6.5 GB，BEHAVIOR 官方 baseline）。
>     - 输入帧不变：本地 `behavior-1k-knack` + `2026-challenge-demos`。详见 §5.3、§6.3。
> 3. 许可证：**纯内部使用，不公开部署**，不再作为阻塞项。
> 4. 改名：~~可见层现在改、内部类名最后统一清~~ → **被法则 5 取代**，可见层与内部一次清完（§2，已完成）。
> 5. **不拖延法则（Frank）**：问题不往后拖——能删就删、能改就改，再小也改；判定有问题就立即改，不留到"最后统一"。前提是不破坏三套页面渲染，改完跑页面自测。本方案所有阶段都按此执行：发现的遗留命名、死代码、注释掉的代码、无效功能，当场处理，不进待办。
> 数据规则沿用 k005b 标准：真权重 + 真激活 + 窗口化；实在拿不到的部件才允许示意，并在 3D 里标 `toy`。

---

## 0. 一页结论

| | /groot | /pi0 | /vla（原 /llm） |
|---|---|---|---|
| 模型 | GR00T N1.7-3B（NAS 本地，BEHAVIOR baseline） | π0.5 COMET b1k 微调（NAS 本地）；π0 base（HF 缓存）作对照 | k005b（我们自己的） |
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

### 1.1 GR00T N1.6（历史核对；实际用 N1.7，以 `groot-understanding.md` 为准）

> ⚠ 页面改用本地 N1.7-3B 后，主干已换成 Cosmos-Reason2-2B（Qwen3-VL 结构，视觉塔 24 块/1024/patch 16，新增 vl_self_attention 4 块），动作维 132、horizon 40。下文 N1.6 事实中 DiT 部分（32 块、交替交叉/自注意力、AdaLN、4 步向上积分）经权重形状核对在 N1.7 中仍成立，其余以 `docs/design/groot-understanding.md` 为准。

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

## 5. /pi0 场景与章组（按 π0.5 实现，2026-10-25 定稿）

> 早先按 π0 base 写的草图（state token、时间并进动作 token、50×32 动作块、pi0_base zero-shot）已作废：本地 `pi0_base` 缓存不完整，零下载约束下主线改用 π0.5 b1k 微调版。下面是落地版本。

### 5.1 场景结构：并排双塔、逐层 K/V 耦合

```
 3 相机 224² → SigLIP 27 块 → 768 图像 token ┐
 指令 + 离散化 state → Gemma 词嵌入 → 117 文本 token ┴→ 前缀 885 token
                                                   │
   前缀塔 18 层（Gemma 2B，2048 宽）  ──K/V──▶  专家塔 18 层（Gemma 300M，1024 宽）  ◀── 右侧时间总线 cond → 每层 adaRMS scale/shift/gate
                                                   │ final adaRMS → action_out_proj
   x_t(32×32) → action_in_proj → 32 动作 token      v(32×32) → Euler 轨迹带 x(t=1.0) … x(t=0)（10 步）→ 反归一化 → R1 Pro 23 维动作
```

- 全部块画成**竖墙**：token 沿 x，通道沿 y；显示窗口 16（堆叠）/ 32（单板）/ 64（解剖）通道。
- 两塔各 18 层一一对齐，每层一根紫红 `kv` 箭头（前缀 L_l → 专家 L_l），标该层动作 query 投在前缀上的注意力份额。
- π0.5 与 π0 的结构差异按真实结构画：时间经 `time_mlp` 走 **adaRMSNorm**（每层 scale/shift/gate 三根小棒），state 离散化进文本前缀，**没有单独 state token**。
- 右侧两块第 0 层解剖板（前缀流、专家流）：norm → Q/K/V → 注意力（联合掩码块同时喂两边）→ o_proj → 残差 → norm → GeGLU → 残差；前缀 K/V 以 `kv` 箭头进专家注意力。

### 5.2 章组（实现版）

| 章 | 内容 | 真值 |
|---|---|---|
| P0 全貌 | 墙的约定、显示窗口、双塔拓扑 | 层数、宽度、参数量 |
| P1 三路相机 → SigLIP → 768 图像 token | 3 路 patch 嵌入、27 块、投影 | 块范数 41.9 / 100.1 / 455.7 |
| P2 指令 + 离散化状态 = 117 文本 token | 词嵌入；state 段高亮 | 真 token id |
| P3 18 层前缀塔 | 逐层穿行；L0 head0 注意力 | token 范数 38.5 → 4051.2 |
| P4 动作 token 与时间条件 | action_in_proj；time_mlp → cond → 总线 | 真 cond |
| P5 第 0 层解剖 | 两流联合注意力 + 掩码；门控残差 | 前缀份额 95%；残差核对 3.8e-6 |
| P6 两座塔逐层耦合 | 18 根 K/V 耦合箭头 | 每层前缀份额（L17 降到 22%） |
| P7 10 步 Euler 去噪 | 10 次专家穿行（缓存前缀 K/V），轨迹带逐格出现 | 每步 x_t、v、|v|≈36；最终 23 维动作 |
| P8 三个模型对照 | §7 | — |

### 5.3 数据：`tools/export_pi0.py`

- **权重（零下载）**：`~/workspace/model/openpi_comet/pi05-b1kpt12-cs32`（openpi orbax/zarr），直接读数组，在 PyTorch fp32 里按 openpi 结构重建前向（官方运行时为 bf16）。`action_in/out_proj`、`time_mlp_*` 与 R1 Pro 归一化统计都是训练过的真权重——**无 toy 部件**。唯一外部取用是 PaliGemma tokenizer 资产（4 MB，非权重）。
- **输入**：BEHAVIOR 2026-challenge-demos 第 0 条 episode 第 0 帧，三相机（head / left_wrist / right_wrist）解码自 HEVC；任务文本来自 `tasks.parquet`；state 按 checkpoint 的 norm_stats 归一化后离散化进前缀。注：demos 的 `base_qvel` 可能是世界坐标系。
- **输出**：`public/pi0-weights.json`（0.8 MB）、`pi0-acts.json`（3.0 MB，含 L0 head0 前缀注意力 f16）、`pi0-fm.json`（2.1 MB，10 步全塔激活）、`pi0-check.json`（抽查点）。
- **自检**：残差/门控恒等式、Euler 恒等式（1.2e-7）、掩码核对（改动作 token 前缀激活不变）；`tools/ref_pi0_jax.py` 用官方 openpi JAX `sample_actions` 同输入同噪声对照，归一化动作最大差 2.6e-5。

---

## 6. /groot 场景与章组

### 6.1 场景结构（按 N1.7 实现，2026-10-25 定稿）

> 早先按 N1.6 写的草图（SigLIP2/pixel-shuffle、50×128、槽 24、G0–G7）已作废，下面是落地版本；结构事实以 `groot-understanding.md` 为准。

```
 3 相机 → 视觉塔 24 块 → merger → 192 图像 token ┐
 指令 → Qwen3 嵌入 → 24 文本 token ──────────────┴→ LM 16 层（S = 216）→ VL 头 (vlln + 4 层 vl_self_attention)
                                                             │  按 image_mask 分成两根 K/V 竖条
 state(61) → state_enc[slot 23] → 1 个状态 token ┐            │
 x_t(40×132) → action_enc[slot 23](W1 a ‖ tau, W2, swish, W3) + pos_emb ┴→ sa 序列 41 token
                                                             ▼
   DiT D0 交叉(文本)  D1 自注意力  D2 交叉(图像)  D3 自注意力   …×8，共 32 块，右侧时间总线 temb → 每块 scale/shift
   final AdaLN → proj_out_2 → action_dec[slot 23] → v(40×132) → Euler 轨迹带 x_t0 … x_t4（4 步，t: 0 → 1）
```

- 全部块按 llm-viz 惯例画成**竖墙**：token 沿 x，通道沿 y，z 一格厚；显示窗口 16（堆叠）/ 32（单板）/ 64（解剖）通道，标注于 G0。
- 主干与 DiT **串联**；DiT 左侧两根 K/V 竖条（文本 24 / 图像 192），交叉块从对应竖条取 K/V（粉 = 文本、紫 = 图像），自注意力块无外连。
- slot 23 = `sim_behavior_r1_pro`：编解码权重是**未训练的初始化值**，场景和文案标 toy；DiT/主干为真权重真激活。
- 右侧两块解剖板：D0（交叉注意力）和 D1（自注意力），逐件画 AdaLN → 注意力 → 残差 → LN → GELU FF → 残差。

### 6.2 章组（实现版）

| 章 | 内容 | 真值 |
|---|---|---|
| G0 全貌 | 墙的约定、显示窗口、串联拓扑 | 层数、宽度 |
| G1 三路相机 + 指令 = 216 token | 视觉塔 → merger；文本嵌入 | 真 patch / token 激活 |
| G2 16 层 LM 与 VL 头 | LM 逐层穿行，L0/L15 头均值注意力图；VL 头 | 每层激活、注意力 |
| G3 本体适配器（slot 23） | state 61 → 隐层 → 状态 token（toy 标注） | 真槽切片 |
| G4 动作 token 与时间嵌入 | W1(a) ‖ tau → W2 → swish → W3 → +pos；Timesteps(256) → temb | temb 范数 |
| G5 D0 解剖：交叉注意力 | AdaLN → cross-attn（K/V = 文本）→ FF | scale/shift 真值 |
| G6 D1 解剖：自注意力 | 41 token 内部自注意力 | 真激活 |
| G7 32 块交替堆叠 | 文/自/图/自 ×8；图像块在三相机上的注意力份额 | 份额、平均 token 范数 |
| G8 4 步 Euler 去噪 | 4 次 DiT 穿行，轨迹带逐格出现；时间方向与 π0/k005b 相反 | 每步 x_t、v、|v| |
| G9 三个模型对照 | §7 | — |

### 6.3 数据：`tools/export_groot.py`

- **权重（零下载，NAS 本地）**：`/home/frank/workspace/model/nvidia/GR00T-N1.7-3B`（6.5 GB，safetensors；含 `experiment_cfg/`、`embodiment_id.json`、`dataset_statistics.json`）。跑前向约 8 GB 显存。
  - §1.1 的结构事实按 N1.6 核对，实现前须按 N1.7 的 `config.json` / `final_model_config.json` 重新核对（DiT 块数、交替模式、Euler 步数、主干截断层数），章组 G1–G7 随之修正。
- **代码**：用与 N1.7 匹配的 Isaac-GR00T 版本的 processor 和 model 类加载。只用代码，不改模型。
- **输入**：同一帧 BEHAVIOR 观测，按 checkpoint 内注册的 `sim_behavior_r1_pro` modality 模板组装（3 相机 head/left_wrist/right_wrist 256²、state 键表、23 维动作 base/torso/双臂/双夹爪、32 步）。
  - **核查结论（`groot-understanding.md` §3）**：该 checkpoint 的训练混合里没有 BEHAVIOR 数据；R1 Pro 槽的 state/action 编解码权重是初始化值（std 与未参训槽相同）。真正训练过的是 `real_r1_pro_sharpa_*`（真机 + Sharpa 灵巧手，62 维 EEF+手关节），不借用。
  - 因此：主干与 DiT 真权重、真激活；本体适配槽标 `toy`；**输出动作无语义，不与 k005b 做数值对比**（原"最强并列对照"一说作废，对照改为结构与激活层面）。
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
3. **同帧同机器人**：三页都用同一帧 BEHAVIOR demo。最后一章并列画动作块：k005b 与 π0.5 COMET 在 BEHAVIOR 上训练过、动作有语义；GR00T N1.7 的 R1 Pro 槽未训练，只并列结构与激活，动作块标 toy。

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
| GR00T N1.7 本地版的 R1 Pro 槽未在 BEHAVIOR 上训练 | 动作无语义 | 模板齐全可直接跑；适配槽标 toy，对照只比结构与激活（`groot-understanding.md` §3） |
| π0.5 COMET 为 openpi JAX/orbax 格式 | 无现成 PyTorch 加载 | 直接读 zarr 数组，PyTorch 重建前向，与 openpi 参考输出逐项比对自检 |
| demo 视频为 HEVC | 导出器需要能解码 HEVC 的环境 | 复用 behavior_challenge 现有数据加载代码；不行就用 ffmpeg 抽单帧 |
| 显存/内存 | 14 GB fp32 加载需要约 28 GB 内存峰值（60 GB 够） | bf16 前向；按需 mmap 读取窗口 |
| 导出数据体积 | 三页 public/data 膨胀 | 只存窗口；单页目标 < 30 MB（与 k005b 同量级） |
| 权重为 N1.7，§1.1 事实按 N1.6 核对 | 章组细节可能偏差 | 实现前按 N1.7 config 重核 §1.1 |

---

## 10. 裁决记录与剩余问题

| # | 问题 | 结论 |
|---|---|---|
| D1 | 页面顺序 | ✅ π0 先，GR00T 后 |
| D2 | π0 输入帧 | ✅ BEHAVIOR 原始 demo 真实帧（`behavior-1k-knack` → `2026-challenge-demos`），不用 π0 训练数据 |
| D3 | GR00T 输入帧 | ✅ BEHAVIOR 本地同帧 + checkpoint 内的 `sim_behavior_r1_pro` 模板；该槽未训练，标 toy（`groot-understanding.md` §3） |
| D4 | 许可证 | ✅ 纯内部使用，不公开部署 |
| D5 | 改名范围 | ✅ 法则 5：可见层与内部一次清完，已完成（§2） |
| D6 | π0 的 state 映射 | ✅ **撤销（伪问题）**：state/action 各自投影进隐层，写一份 R1 Pro embodiment 配置即可（§5.3）。此类本体适配问题以后按此模式直接处理，不再提问 |
| D7 | bbycroft 其他应用与首页个人信息 | ✅ 已全部删除：`src/cpu`、`src/codec`、`src/fluidsim`、`src/config` 及路由、`public/riscv`、`me.jpeg`、个人简介/捐助链接、`progress/`、`ols.json`、空 `.gitmodules`、yarn（统一 npm）、13 个无用依赖、死的 `eslintConfig`/`resolutions`、`create-symbol-font.jsm` |
