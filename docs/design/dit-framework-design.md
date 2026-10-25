# DiT 框架本身怎么讲（修订版，按 Frank 2026-10-25 裁决）

裁决：
1. 不做独立章组，也不做 /dit 页面。DiT 框架的讲解直接融进现有 D 组：**D1 扩为完整结构讲解**。
2. flow matching 按**真 10 步**画（D6 现有结构：10 次 DiT 穿行 + Euler 轨迹带），**不用玩具模型**。

因此全程只用 k005b 真权重和真激活，不再出现任何 toy 数据。"教科书 DiT"不单独搭场景，而是在讲 k005b 每个部件时，用一行"教科书做法 vs k005b"对照带出来。

## 1. D1 扩展：一个 DiT 块从头到尾（D1a–D1h）

场景仍是现在的 28 层对大场景，镜头逐个部件推近第 0 层的 DiT 窄条。每个子步：**被讲的块点亮，其余块淡出到 0.2**，箭头只保留当前部件的进出线，避免看不清。

| 子步 | 场景里动什么（镜头 / 点亮） | 讲什么：教科书 → k005b | 真值 |
|---|---|---|---|
| D1a 全貌 | 镜头沿第 0 层 DiT 窄条自上而下扫一遍，各部件依次闪一下，再拉远露出 28 层窄条 | DiT = 时间条件化的 Transformer：块内是 Norm→注意力→残差→Norm→MLP→残差，堆 N 层，最后 norm + 投影 | 结构 |
| D1b 时间条件 | te 总线：t → sin 嵌入（fm_time_emb）→ 竖直总线 → 每层的 ada 小块 | 教科书：t + 类别 c 嵌入相加；k005b：**只有 t**，任务和场景信息走注意力读主干（D3 讲） | fm_time_emb [64 窗口]；sin 参数 min 4e-3 / max 4 |
| D1c adaLN 调制 | e0.ln1 → e0.ada：先画 RMSNorm 后的柱状分布，再画 (1+γ)x+β 之后的分布，两者并排对比 | 教科书 adaLN-Zero：LayerNorm + γβα 共 6 组，α 门控残差；k005b：**AdaRMSNorm**，每层 2 处 γβ，零初始化，**没有 α**，残差直接相加 | E0.ada [4,64] 真 γβ；值域 [−8.99, +23.70]，第 27 层 [−11.57, +20.14] |
| D1d 注意力 | e0 的 Q 列 → scP / scS → softmax → vo → o_proj | 教科书：自注意力只看自己的 token；k005b：**联合注意力**，Q 来自 17 个后缀 token，K/V = 主干前缀 + 后缀（细节留给 D3） | E0.attn [17,49]、E0.mass |
| D1e 残差 1 | o → res1：主线 + 增量，两段 ⊕ 节点点亮 | 残差：每块只学增量 | E0.o、E0.res1 |
| D1f MLP | ln2 → ada2 → gate / up → silu·乘 → down → out | 教科书：点式 MLP（GELU）；k005b：**SwiGLU** 768→2752→768，逐 token 独立 | E0.ln2、E0.mlp、E0.out |
| D1g 堆叠 | 镜头沿 DiT 窄条从第 0 层降到第 1、2 层，再快速拉远到 28 层全貌；窄条逐层点亮 | 教科书 N 层（DiT-XL = 28）；k005b **28 层**，与主干逐层配对 | 结构 + 每层 ada 值域 |
| D1h 输出 | 列 2 底部：expertOut → 最终 RMSNorm → action_out_proj → 速度场 v（16×23） | 教科书：final adaLN + Linear + unpatchify；k005b：final 是**普通 RMSNorm**（final_norm_adanorm=False），丢掉 state token 后投影 768→23 | expert_out、action_velocity 真值 |

**D1 末尾的映射总表**（只在文案里，不弹窗、不加覆盖层）：

| # | 教科书 DiT | k005b 动作专家 | 代码 |
|---|---|---|---|
| ① | patchify → token | 16 步 × 23 维，每步一行就是一个 token（action_in_proj + 时间 MLP），再加 1 个 state token，共 17 个（D2 讲拼接） | fused_behavior_action.py:356-386 |
| ② | 自注意力 | 联合注意力，K/V 含主干前缀 | :86-140 |
| ③ | 条件 = t + 类别 c | 只有 t | :378, ada_cond=time_emb |
| ④ | LayerNorm + adaLN-Zero（γβα） | AdaRMSNorm（γβ 零初始化，无 α） | shared_attention.py:60-83 |
| ⑤ | GELU MLP | SwiGLU 768→2752→768 | shared_attention.py:91 |
| ⑥ | N 块 | 28 层，与主干配对 | 导出元数据 |
| ⑦ | final adaLN + Linear + unpatchify | RMSNorm → action_out_proj → 16×23 | :214, :441 |
| ⑧ | K 步采样 | Euler 10 步，x ← x − v/10 | :526-549（D6） |

## 2. 其余 D 章怎么调整

- **D0**：保留。末尾一句话预告"D1 先拆开 DiT 本身"。
- **D2–D5**：不变。D2 承接映射表第 ①条，D3 承接第 ②条。
- **D6**：不变，用真 10 步。补一句"这就是映射表第 ⑧ 条"，并补上每次穿行时 te 总线 t 值的变化（1.0 → 0.1）。目前仍是 step_130000 pre-stamp 预览，等 step_160000 换数据。
- **D7**：不变。

## 3. 实现要点

- 只改 Walkthrough11_FlowMatching.tsx 的 D1（拆成多段 commentary + afterTime，按子步分别设置镜头和点亮）。再给 ActionViz 加一个 "norm 前后分布对比" 小柱图，锚定在 e0.ada 旁，ASCII 标签。
- 子步镜头用紧凑的 cube 列表 + fill ≈ 1.05–1.7，避开前景注意力头的遮挡（上一版 D1 截图的教训）。
- 不需要重跑导出器：E0.ada、E0.attn、E0.* 激活、expert_out、action_velocity 都已在 k005b-fm.json / acts.json 里。
- 验收：D1a–D1h 每个子步截图一张；在 D1 跑一遍箭头审计，要求无孤悬箭头。

## 实现记录（D1a–D1h）

- D1b 与 D1c 共用 γβ 的近景正视镜头。原因：更宽的取景会让镜头落到第 0 层各注意力头（z ≈ −450）的后方，头块挡在 DiT 泳道前面。te 总线的全貌在 D0 已展示。
- D1c 的“调制前 / 调制后”条形图：橙条是捕获的 AdaRMSNorm 输出（真值）。灰条是由它逐通道反解得到的 n = (out − β)/(1 + γ)，因为调制是逐元素的，这个反解是精确的。
- γβ 数值对照（验收）：E0.ada 第 0 通道，acts.json 中 γ₁ = 5.8770、β₁ = 0.8275；fm.json 第 0 步（t = 1.0，fp16）中 γ₁ = 5.8789、β₁ = 0.8276。整块 [4,64] 的最大绝对差为 0.0077，最大相对差为 4.6e-4，差异来自 fp16 量化。
- 残差恒等式在 acts.json 上逐元素精确成立：expert_in + o = res1，res1 + mlp = out，误差为 0。
- D1 箭头审计：noIn 和 noOut 均为空，共 17450 条边。
