# GR00T N1.7-3B（本地 NAS 版）：理解笔记

> 本文是 /groot 页（`three-viz-plan.md` §6）的事实基础。
> 模型目录 `G = /home/frank/workspace/model/nvidia/GR00T-N1.7-3B`（NAS，只读），`latest = global_step196000`。
> **证据等级**：
> - **[W]** 由权重头（safetensors 形状 / 逐槽统计）直接证实；
> - **[C]** 由本地 config 文件证实（`config.json`、`experiment_cfg/*`、`processor_config.json`、`embodiment_id.json`）；
> - **[S6]** 来自 Isaac-GR00T `n1.6-release` 源码（见 `three-viz-plan.md` §1.1）。N1.7 源码本地没有、零下载约束下不拉取，**实现导出器时必须逐条复核**。
>
> 已读 `G/EXPLAINABILITY.md`：它是 NVIDIA 模型卡式的可解释性表（用途、输入输出、风险、指标），**不含任何内部结构信息**，只确认"视觉+语言+本体感知 → 连续动作向量"这一层。结构事实全部来自下面的权重和 config。

---

## 0. 一句话

System 2（VLM 主干，只跑一次）把 3 路相机图像和指令编码成 2048 宽的 token 序列，截到第 16 层输出；System 1（DiT 动作头，32 块，宽 1536）拿 state token + 带噪动作 token 做序列，偶数块交叉注意力去读主干 token（文本段、图像段交替），奇数块只在自身序列里自注意力，时间 t 经 AdaLN 注入每块；DiT 跑 **4 次**（Euler 4 步，t 从 0 往 1 走），得到 40×132（按本体截取有效维）的动作块。

```
3 相机 256² ──▶ 视觉塔 24 块(1024) ──▶ merger(+3 deepstack) ──┐
指令 ─────────▶ tokenizer ────────────────────────────────────┤
                                                              ▼
                       语言模型 16 层 (2048, GQA)   ← select_layer=16，一次
                                                              ▼
                       vlln LayerNorm(2048) → vl_self_attention 4 块(2048)
                                                              ▼  VL token (K/V 源)
state 132 ─▶ state_encoder[槽] ─▶ 1 个 state token(1536) ─┐
x_t 40×132 ─▶ action_encoder[槽](+τ 正弦) + pos_emb ─────┤
                                                          ▼
for k = 0..3:   τ_k = k/4 → t_disc = 0,250,500,750
    DiT 32 块：偶数块 = 交叉注意力(VL)，奇数块 = 自注意力；每块 AdaLN(temb)
    proj_out(AdaLN) → action_decoder[槽] → v_k
    x ← x + ¼·v_k
x_1 = 动作块（取后 40 个动作 token 的输出）
```

---

## 1. 与 N1.6 / 原计划的差异（必须改的地方）

| 项 | 原计划（N1.6） | N1.7 本地版 | 证据 |
|---|---|---|---|
| VLM 主干 | Eagle（SigLIP2 27 层 1152 + Qwen3） | **Cosmos-Reason2-2B**（Qwen3-VL 结构） | [C] `model_name`、[W] `backbone.model.model.visual.*` |
| 视觉塔 | SigLIP2，patch 14，宽 1152 | **24 块，宽 1024，patch 16，时间 patch 2**（`patch_embed.proj [1024,3,2,16,16]`），位置表 2304 | [W] |
| 视觉→语言投影 | pixel-shuffle + mlp1 | **merger**（LN 1024 → 4096 → 2048）+ **3 个 deepstack merger**（注入语言层） | [W] |
| 语言模型 | Qwen3 16 层 | Qwen3 **16 层**（截断），2048 宽，16 Q 头×128，**8 KV 头**，q/k_norm，SwiGLU 6144 | [W] 仅 layers.0–15 |
| VL 自注意力 | N1.6 无 | **新增 `vl_self_attention` 4 块**（2048 宽，32 头×64，FF 8192，LayerNorm 带仿射） | [W][C] |
| 动作维 | 128 | **132**（`max_state_dim = max_action_dim = 132`） | [W] encoder/decoder 形状 |
| 动作 horizon | 50 | **40** | [C] `config.json`（`final_model_config.json` 写 50/128 是训练起点配置，已被权重形状否定） |
| state 正弦编码 | true | **false** | [C] `config.json` |
| embodiment 槽 | 32 | 32 | [W] `[32, …]` |

`config.json` 与 `experiment_cfg/final_model_config.json` 有冲突（horizon 40 vs 50，维度 132 vs 128，sincos false vs true）。**以 `config.json` 为准**，因为维度 132 与权重形状一致；horizon 40 需在导出时用 `position_embedding` 实际使用长度复核。

---

## 2. 逐部件结构

### 2.1 System 2：VLM 主干（只跑一次）
- **视觉塔**（24 块）：`norm1 → attn(qkv 1024→3072, proj) → norm2 → MLP(1024→4096→1024)`，LayerNorm 带偏置。[W]
- **merger**：2×2 空间合并后 LN(1024)·4 → 4096 → 2048。另有 3 个 deepstack merger，把视觉塔中间层特征加到语言模型前几层（Qwen3-VL 机制）。[W] 注入的具体层号 [S6 无，需复核 Qwen3-VL 源码]。
- **语言模型**（16 层）：标准 Qwen3 块 `RMSNorm → GQA(q/k_norm, RoPE) → 残差 → RMSNorm → SwiGLU → 残差`。`lm_head` 权重还在文件里但前向用不到（select_layer=16 取的是第 16 层隐状态）。[W][C]
- **vlln**：LayerNorm(2048)。[W]
- **vl_self_attention**（4 块）：DiT 风格的 BasicTransformerBlock，但**不带时间条件**（norm1/norm3 是普通 LayerNorm），对整段 VL token 做自注意力后作为 DiT 的 K/V 源。[W]

### 2.2 本体适配（按 embodiment 分槽，32 槽）
- `state_encoder`：132 → 1024 → 1536（两层 MLP，[32, in, out] 批量权重，按槽索引）。[W]
- `action_encoder`：W1 132→1536；与时间 τ 的正弦编码拼接 → W2 3072→1536（swish）；W3 1536→1536。[W] 结构 [S6]
- `position_embedding [1024, 1536]`：加到动作 token 上。[W]
- `action_decoder`：1024 → 1024 → 132。[W]
- 这就是 BEHAVIOR 官方 baseline 用来接 R1 Pro（state 61 → 隐层、action 23 → 隐层）的机制：每个本体占一个槽，不同维度在槽内各自投影。[C] 见 §3。

### 2.3 System 1：DiT 动作头（32 块，跑 4 次）
- **时间编码**：`timestep_embedder` 256（正弦）→ 1536 → 1536 = temb。[W]
- **每块**：`norm1` = AdaLN（`linear [3072,1536]` 从 SiLU(temb) 出 scale/shift，作用在无仿射的 LayerNorm 之后）→ `attn1` → 残差 → 无仿射 LayerNorm → FF（GELU，1536→6144→1536）→ 残差。只有 norm1 有权重，FF 前的 norm 没有参数。[W]
- **交替**（权重已证实）：`attn1.to_k/to_v` 输入宽度逐块为 `2048, 1536, 2048, 1536, …` [W]
  - **偶数块**（0,2,…,30）：**交叉注意力**，Q 来自动作序列（1536），K/V 来自 VL token（2048）。
  - **奇数块**（1,3,…,31）：**自注意力**，只在 state+action 序列内部。
  - 偶数块里再交替：`attend_text_every_n_blocks=2` → idx%4==0 只看文本 token，idx%4==2 只看图像 token。[C] `final_model_config.json` + [S6]
- **收尾**：`proj_out_1`（SiLU(temb) → 3072 → shift/scale）→ LayerNorm → `proj_out_2`（1536→1024）→ `action_decoder[槽]`。[W]
- 专家序列只有 state token + 动作 token，**VL token 不进序列**，只经交叉注意力读。[S6]

### 2.4 流匹配解码
- `num_inference_timesteps = 4`，`num_timestep_buckets = 1000`。[C]
- 循环 k=0..3：τ=k/4，t_disc = 0, 250, 500, 750；`x ← x + ¼·v`；**t=0 是噪声，向上积分**，与 k005b/π0 方向相反。[S6]
- 训练时间律：Beta(1.5, 1)，`noise_s = 0.999`。[C] 与 k005b v2 同律、方向相反——对照章的看点。
- 主干对一次推理只跑一次，DiT 跑 4 次。[S6]

---

## 3. 关于 `sim_behavior_r1_pro`：核查结果（与"训练时含 BEHAVIOR R1 Pro"的前提不符）

**结论：本 checkpoint 注册了 BEHAVIOR R1 Pro 的 embodiment 模板，但没有在 BEHAVIOR 数据上训练过；该槽的编解码权重是初始化值。**

证据：
1. **槽号本身不一致**：`embodiment_id.json` 写 `sim_behavior_r1_pro: 23`（同表里 `xdof: 24`），而 `experiment_cfg/final_processor_config.json` 里的映射写 `sim_behavior_r1_pro: 24`（`xdof: 23`）。两张表都是代码库的全局名表，也都列了 libero_sim、droid_sim、simpler_env 等未参训本体。[C]
2. **训练混合里没有它**：`experiment_cfg/conf.yaml` 的数据集只有 8 个 embodiment tag——`xdof_relative_eef_relative_joint(_subtask)`、`oxe_droid_relative_eef_relative_joint`、`real_g1_relative_eef_relative_joints`、`real_r1_pro_sharpa_relative_eef(_mecka/_maxinsights/_human)`；`grep behavior` 命中 0。`statistics.json` 与 `processor_config.json` 的 modality_configs 也只有这 8 个。wandb run 名为 `pretrain_n17_qwen3vl_2b_…human_robot_cross_embodiment…`。[C]
3. **权重统计**：槽 23 在 `state_encoder.layer1.W`、`action_encoder.W1.W`、`action_decoder.layer2.W` 的 std 分别为 0.01998 / 0.02000 / 0.01999，与从未参训的槽 0/2/13/17 完全相同（≈ 初始化 std 0.02）；参训槽 24–27 明显偏离（action_encoder.W1 达 0.028–0.041）。[W]

**模型里真正训练过的 R1 Pro** 是 `real_r1_pro_sharpa_*`：Galaxea R1 Pro 真机 + Sharpa 灵巧手，state/action = 左右腕 EEF 9 维 ×2 + 左右手关节 22 维 ×2（62 维，相对 EEF 动作）。与 BEHAVIOR 的模拟 R1 Pro（夹爪、关节空间、state 61 / action 23）是**同一机身、不同末端和动作空间**。

**仍然成立的部分**：`sim_behavior_r1_pro` 的 modality 模板完整存在于 processor 配置中，即 BEHAVIOR 适配的机制先例 [C]：
- video：`head_256_256`、`left_wrist_256_256`、`right_wrist_256_256`（与我们的 zed 头 + 左右 realsense 一一对应）；
- state：`robot_pos, robot_ori_cos/sin, robot_2d_ori(_cos/_sin), robot_lin_vel, robot_ang_vel, arm_left_qpos(_sin/_cos), eef_left_pos, eef_left_quat, gripper_left_qpos, arm_right_qpos(_sin/_cos), eef_right_pos, eef_right_quat, gripper_right_qpos, trunk_qpos`；
- action（32 步）：`base, torso, left_arm, left_gripper, right_arm, right_gripper`（= 3+4+7+1+7+1 = 23），torso/双臂为相对动作，其余绝对；
- language：`annotation.human.coarse_action`。

### 3.1 对 /groot 页的影响与处理（按"本体适配=写配置"规则直接定）
- 输入：BEHAVIOR 本地 demo 帧，按上面的 `sim_behavior_r1_pro` 模板组装 3 相机 + state + 指令。
- 主干（Cosmos-Reason2 16 层 + vl_self_attention）与 DiT 32 块：**真权重、真激活**。
- 本体适配槽（state_encoder / action_encoder / action_decoder 的 R1 Pro 槽）：**未训练的初始化权重**，3D 中标 `toy: sim_behavior_r1_pro slot untrained in this checkpoint`；输出动作块**无语义**，只展示数值流动，不与 k005b 动作做数值对比。
- 不借用 `real_r1_pro_sharpa` 槽：那需要把 BEHAVIOR 状态硬映射到 Sharpa 手的 62 维空间，属于伪造映射。
- 原计划 §6.3 "输出动作有语义、可与 k005b 同帧对比"的说法因此**不成立**，已在 `three-viz-plan.md` 修正。

---

## 4. 实现前待复核清单（N1.7 源码可用时）
1. DiT 偶数块的文本/图像交替规则（idx%4）与 vl_self_attention 的位置（vlln 之前还是之后）。
2. 解码方向与 t_disc 取值（N1.6：0,250,500,750，向上积分）。
3. action_encoder 中 τ 的编码方式与 W2 拼接顺序。
4. deepstack 注入的语言层号；视频 2 帧时间 patch 在单帧推理时的处理（run 名含 `2frame`）。
5. 槽号用 23 还是 24（以加载代码实际读取的映射表为准；两者都未训练 BEHAVIOR，结论不变）。
6. horizon 40 的实际取值。
