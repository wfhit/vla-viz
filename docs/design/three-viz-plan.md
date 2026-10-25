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

一句话：三者是 **三种耦合拓扑**。GR00T 是"串联加单总线交叉注意力"，π0 是"并排、逐层拼接"，k005b 是"并排、逐层借 KV"。三页用同一引擎和同一视觉语言（§3），但耦合拓扑由 ModelSpec 的逐层注意力类型（joint / cross / self）真实生成，所以差异仍然一眼可见——不是换数字的模板，也不是三套独立实现。

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

## 3. 代码架构：同一引擎，多模型实例（Frank 裁决 2026-10-25，取代旧"各自独立"方案）

**裁决**：/groot 与 /pi0 必须与 /vla 风格完全相同——3D 场景 + 章组步进 + 播放条 + 真权重真激活的 bbycroft 式逐层 walkthrough；不允许平铺静态页/文字页/"墙"式独立实现。三页是**同一引擎的三个模型实例**，只换模型结构与数据。

### 3.1 分层

```
src/vla/                     唯一引擎（层视图、层对竖向对叠、章节导航、播放条、箭头+审计、标签去重）
src/vla/ModelFacts.ts        三个模型实例的事实表（k005b / pi0 / groot：层数、宽度、头数、K、名称）
src/vla/pages.ts             makeEnginePage(id, 数据前缀, facts, 初始相机) → /vla、/pi0、/groot
src/vla/VlaModelLayout.ts    按 config 生成：joint（k005b/π0.5）或 crossDit（GR00T，L.crossDit）
src/vla/walkthrough/*        章节骨架共用；文案按 f.id 分支取真值
tools/export_{k005b,pi0,groot}.py  → 同一 "VLA schema" 的 JSON（public/{,pi0-,groot-}*.json）
```

`src/pi0/*`、`src/groot/*` 的墙式实现已删除（111ec10 / 本次 groot 提交）。

### 3.2 ModelSpec（驱动 createLayer / createExpertLayer）

| 字段 | k005b | π0.5 (/pi0) | GR00T N1.7 (/groot) |
|---|---|---|---|
| 主干 | Qwen3-VL-2B, 28 层, C=2048, 16 头/8 kv, hd128, SwiGLU, RMSNorm | Gemma 2B, 18 层, C=2048, 8 头/1 kv, hd256, GeGLU, RMSNorm | Cosmos-Reason2(Qwen3-VL) 取 16 层 + vlln + 4 层 vlsa 自注意力 |
| 专家 | 28 层, 768, 16 头/2 kv, SwiGLU, AdaRMSNorm | 18 层, 1024, AdaRMSNorm + gate_a/gate_f 门控残差, GeGLU | DiT 32 层, 32 头×48, AdaLN(scale/shift), GELU FF |
| 专家注意力类型（逐层） | joint（前缀+后缀） | joint（逐层 KV 拼接） | 偶数层 cross（只看 216 VL token）/ 奇数层 self（41 token） |
| 层对 | 主干 Li ↔ 专家 Ei，28 对 | 主干 Li ↔ 专家 Ei，18 对 | VL 主干堆 → 32 层 DiT（cross 层箭头接 VL 输出，不是逐层配对） |
| 后缀 token | 1 state + 16 action | 32 action × 32 维 | 1 state + 40 action |
| Euler 步数 K | 10 | 10 | 4 |
| 输出头 | knack/done（仅 k005b） | 无 | 无 |

专家注意力块按 kind 生成：joint = 前缀段 + 后缀段两部分；cross = 只有前缀段；self = 只有后缀段。布局统一采用"竖向对叠"（单列，见 dit-section-design §1.1）。

### 3.3 数据：统一 VLA schema

以 k005b 导出格式为标准：每层 `L#.q` 逐头窗口、`ln` 聚合、MLP、`E#.scp/scs/pp/ps/vo`、`ada` 等。现有 pi0/groot 导出只覆盖 L0 / D0–D1，无逐头、无逐层权重 → 扩展 `export_pi0.py`、`export_groot.py`（纯 torch 复现，已与官方核对）输出全层、全头的同构 schema。数据源不变：NAS 本地权重 + 本地 BEHAVIOR 帧，零下载。

### 3.4 硬约束（三页一致）

- 箭头必须有起止块（`__arrowAudit` 的 noIn/noOut 为空）。
- 3D 文字只用 ASCII。
- 无弹窗。
- FOV 不动。
- 真值标注；toy 必须标明。

### 3.5 施工顺序

1. ✅ 事实表抽取：k005b 走 facts 路径，/vla 审计不变（2332 边）。
2. ✅ export_pi0.py → 全层 schema；/pi0 切到引擎实例，删除 `src/pi0` 墙（111ec10，审计 1494 边）。
3. ✅ /groot 同理（主干 20 = LM 16 + vlsa 4；DiT 32 块 cross 文本/self/cross 图像/self 交替；AdaLN；4 步 Euler；删除 `src/groot` 墙；审计 1809 边，主干单层章 720 边）。
4. 每步：tsc、截图逐章自测、审计为空、`next build`，然后提交。

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

- 引擎实例（§3）：与 /vla 同一 3D 场景、章组步进、播放条；单列竖向对叠 18 个层对（主干 Li 绿块 ‖ 专家 Ei 橙块并排）。
- 每层对共用一次联合注意力：专家 query 看前缀 K/V + 自身后缀 K/V；前缀不看后缀。
- π0.5 的结构差异按真实结构画：时间经 `time_mlp` 走**门控 adaRMSNorm**（scale/shift/gate），state 离散化进文本前缀，**没有单独 state token**；主干无 final norm、无输出头。

### 5.2 章组（引擎版，与 /vla 同一骨架）

概览（总览、预备知识）→ 主干：一层展开（输入嵌入、RMSNorm、MQA 联合注意力、前缀 Softmax、输出投影与残差、MLP (GeGLU)、18 层主干堆叠）→ 动作专家与流匹配（D0 两座塔 … D6 18 层堆叠与输出、D7 10 步 Euler 与动作块）。每章文案按 `f.id === 'pi0'` 取真值。

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

- 引擎实例（§3，`L.crossDit`）：与 /vla 同一 3D 场景、章组步进、播放条。单列竖向对叠：左列主干 20 层（16 层 Qwen3 LM + 4 块 vlsa），右列 DiT 32 块；**不逐层配对**——每层一个独立标题（`trunk Li (LM|VL self-attention)`、`DiT i (cross: text|self|cross: image)`）。
- 主干最后输出 = VL token（`L.vlOut`），经两塔间的竖直总线（`drawVlBus`）接到每个交叉块的 K/V；自注意力块无外连。主干单层章（DiT 隐藏）时 VL token 是合法终点（审计 sink）。
- DiT 每块：AdaLayerNorm（无仿射 LN + 由 temb 线性层给出的 scale/shift）→ 交叉/自注意力（32 头 × 48，只画 head 0，标 ×32）→ o_proj → ⊕ → 无参 LN → GELU FF → ⊕；时间总线把 temb 送到每块。
- slot 23 = `sim_behavior_r1_pro`：state/action 编解码权重是**未训练的初始化值**，文案如实说明（动作无物理语义，不反归一化，不与 k005b 数值对比）；主干与 DiT 为真权重真激活。

### 6.2 章组（引擎版，与 /vla 同一骨架）

概览（总览、预备知识）→ 主干：一层展开（输入嵌入、RMSNorm、GQA 自注意力、Softmax、输出投影与残差、MLP (SwiGLU)、20 层主干堆叠）→ DiT 动作头与流匹配（D0 两座塔、D1 DiT 输入嵌入、D2 AdaLayerNorm、D3 交叉 / 自注意力、D4 输出投影与残差、D5 MLP (GELU)、D6 32 块堆叠与输出、D7 4 步 Euler 与动作块，t: 0 → 1）。每章文案按 `f.id === 'groot'` 取真值。

### 6.3 数据：`tools/export_groot.py`

- **权重（零下载，NAS 本地）**：`/home/frank/workspace/model/nvidia/GR00T-N1.7-3B`（6.5 GB，safetensors；含 `experiment_cfg/`、`embodiment_id.json`、`dataset_statistics.json`）。跑前向约 8 GB 显存。
  - 结构事实已按 N1.7 config 与 Isaac-GR00T 源码核对（`groot-understanding.md`）。
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
