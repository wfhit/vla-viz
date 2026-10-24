# k005b 的 Flow Matching 机制：理解笔记

> 本文是可视化设计（`flow-matching-viz.md`）的事实基础。所有结论都对应到代码行号。凡是"实测"数字，都来自 k005b step_130000、训练行 row 0（episode 0, frame 0），也就是 `tools/export_k005b.py` 导出的那个样本。
> 代码根目录：`BC = /home/frank/behavior_challenge_ws/behavior_challenge/src`

---

## 0. 一句话

k005b 的动作头**在形式上是流匹配**：噪声 x1、速度场 v、Euler 解码 x0 = x1 − v。但它**只在 t = 1 这一个时刻训练和运行，只走 1 步 Euler**。整个"去噪"就是**一次前向**：专家的 17 个后缀 token 和主干的 1240 个前缀 token 一起穿过 28 对层，每层做一次联合注意力，得到 16×23 的速度，再用 `x1 − v` 解出动作块。

这和 π0 不同。π0 在 t∈(0,1] 上随机采样训练，推理走 10 步 Euler，并复用前缀 KV cache。

> **✅ 已验证（Frank 复核源码确认，2026-10-24）：k005b 是 t=1 单步，没有多步 Euler 迭代。**
> - 训练：`target_velocity = action_noise − action_target`，即 u = x1 − a，不对 t 采样（`BC/training/behavior_losses.py:219`）。
> - 部署：`action = x1 − action_velocity`，一次减法（`BC/model/fused_behavior_action.py:209`，`FusedOutput` 契约）。
> - t：`fm_time = torch.ones(b)`（`fused_behavior_action.py:270`），时间嵌入作为 `ada_cond` 送入专家各层 AdaRMSNorm（:262、:315）；专家 final norm 不受调制（`final_norm_adanorm=False`，:31、:199-202）。
>
> 因此**不存在可以 hook 的 Euler 迭代步**，此前"逐迭代步导出 x_t"的要求作废。真实机制只有一条单步流水线：
>
> **噪声板 x1 [16×23] → 专家单次前向（ada_cond = te(t=1)；28 层逐层联合注意力，K/V 来自主干前缀）→ 速度场 v [16×23] → 一次减法 x1 − v → 16×23 动作块**

---

## 1. 通用流匹配：用来对照的标准形式

| 量 | 定义（rectified-flow 约定，t=1 为噪声端） |
|---|---|
| 噪声 | x1 ~ N(0, I) |
| 数据 | x0 = 真实动作块 a |
| 插值路径 | x_t = t·x1 + (1−t)·a，是一条直线 |
| 条件速度（目标） | u = dx_t/dt = x1 − a，与 t 无关 |
| 训练损失 | E_t ‖v_θ(x_t, t, ctx) − u‖² |
| 生成 | 解 ODE dx/dt = v_θ，从 t=1 积分到 t=0 |
| Euler（K 步，δ=1/K） | x_{t−δ} = x_t − δ·v_θ(x_t, t) |

**Euler 步就是残差连接**（借用 Cambridge MLG 的视角）：每一步 `x ← x + (−δ)·v_θ(x)`，和 Transformer 里的 `h ← h + f(h)` 形式完全相同。K 步 Euler 等价于一个 K 层、各层共享权重、靠 t 区分的残差网络。我们的 viz 里已经有"残差流"的视觉语言，可以直接复用。

**分布视角**（借用 drscotthawley 的说法）：速度场搬运的是一整个分布，不是一个点。不同的噪声样本 x1 被同一个场推到不同的终点，多模态就体现在这里。

---

## 2. k005b 的具体实现

### 2.1 时间 t：固定为 1

- `BC/model/fused_behavior_action.py:270` 写死为 `fm_time = torch.ones(b)`，部署和训练都是这样。
- `BC/data/schema.py:20`：`ACTION_HORIZON = 16  # the fixed t=1 generation horizon`。
- 推论：路径插值 x_t = t·x1 + (1−t)·a 在 t=1 时**就是噪声 x1 本身**（`fused_behavior_action.py:262`："At t=1, x1 IS the noise"）。模型从来看不到"半去噪"的输入。

### 2.2 噪声初始化

| 场景 | 代码 | 规则 |
|---|---|---|
| 训练 | `BC/data/fused_batch.py:334-338` | `seed = flow_noise_seed(global_seed, episode, frame, occurrence)`，即每行一个确定种子；`randn(16, 23)` 后转成 bf16 |
| 部署 | `BC/harness/fused_adapter.py:271-272` | `torch.Generator().manual_seed(noise_seed=0)`，`randn(1, 16, 23)`。同一请求一定得到同一个输出 |
| 导出 | `tools/export_k005b.py`（经 `fused_train_tensors`） | 用训练行自己的 `action_noise`。实测 `action_noise == action_x1` ✓ |

### 2.3 训练目标

`BC/training/behavior_losses.py:209-221`：

```
target_velocity = action_noise − action_target        # u = x1 − a
loss = mean((v − u)²)  over 16×23, 再对 batch 取均值
```

- 没有对 t 采样，也没有 Beta 时间分布。`one_step_actions.py:26` 里虽然有 `sample_beta`，但这条路径不调用它。
- done 行和 settle 行**照样参与训练**（:213-215）。

**关键推论：t=1 的最优解只是条件均值。**
在 t=1 时，输入 x1 与 a 独立，所以 L2 意义下的最优解是
v* = x1 − E[a | ctx]，于是 x1 − v* = E[a | ctx]。
单步解码在理论上等价于"回归条件均值"，噪声不应该影响输出。实测发现网络**没有完全学会抵消噪声**，见 §4。

### 2.4 速度场 v_θ 的输入：17 个后缀 token

`BC/model/fused_behavior_action.py:259-279` 中的 `_suffix_embeds`，张量流如下：

```
action_state75 [1,75] ──state_proj(75→768)────────────────────────────▶ s  [1,1,768]   ← 后缀第 0 个 token
x1 [1,16,23] ──action_in_proj(23→768)──▶ a_emb [1,16,768] ┐
t=1 ──sinusoid(768, min_period=4e-3, max_period=4)──▶ te [1,768] ─expand─▶ [1,16,768] ┘
          cat → [1,16,1536] ─action_time_mlp_in(1536→768)─ SiLU ─action_time_mlp_out(768→768)─▶ [1,16,768]  ← 后缀第 1..16 个 token
suffix = cat[s, ·] → [1,17,768]；同时返回 te，作为 ada_cond
```

- 投影层定义在 `BC/model/action_state_token.py:12-18`。
- 时间编码在 `BC/model/one_step_actions.py:14-24`：384 个频率，周期从 0.004 到 4，按几何级数排列；输出是 `[sin, cos]` 拼起来的 768 维。
- 状态槽：61 维原始本体感知先做 bounds99 归一化，再补 14 个 0，凑成 75 维（`BC/data/fused_batch.py:114-121`，`schema.py:23-25`）。
- 实测：`expert_in` 是 [17, 768]，已导出窗口 [17, 64]。

### 2.5 t 的第二个入口：AdaRMSNorm

专家每一层的 `input_layernorm` 和 `post_attention_layernorm` 都是 AdaRMSNorm（`BC/model/shared_attention.py:60-85`，用法在 :119-142）：

```
h = RMSNorm(x) · weight
h = (1 + γ(te)) · h + β(te)          # γ, β 都是 768→768 的 Linear
```

t 因此从两条路进入专家：**拼进每个动作 token**（§2.4），以及**调制每层的归一化**（FiLM）。专家的 final norm 是普通 RMSNorm（`final_norm_adanorm=False`，`fused_behavior_action.py:31`，:199-202）。
由于 t 恒为 1，te 是常量，γ(te) 和 β(te) 也就退化成了**固定的逐通道仿射**。

### 2.6 专家与主干的联合注意力：逐层配对

主循环在 `BC/model/fused_behavior_action.py:183-192`，单层逻辑在 `_layer_pair :85-164`。

```
for i in range(28):                         # num_layers = len(trunk.layers) = 28   (:177)
  阶段 1 (:94-110)
    主干 L_i:  RMSNorm → q,k,v   q:[1,1240,16,128]  k,v:[1,1240,8,128]
    专家 E_i:  AdaRMSNorm(te) → q,k,v   q:[1,17,16,128]  k,v:[1,17,2,128]
  KV 头对齐 (:118-123): 专家 k,v 用 repeat_interleave ×4 扩到 8 个头
  拼接 + RoPE (:124-126): q,k,v 沿 token 轴拼成 L=1257；position_ids = cumsum(validity)−1 (:306-311)
                                            ⇒ 后缀的位置紧接在前缀之后
  一次共享注意力 (:127): softmax(q·kᵀ/√128 + mask) · v     （q 头 h 对应 kv 头 h//2）
  阶段 2 (:130-151): 输出按 token 切回两段
    主干段 [0:1240]    → L_i 的 o_proj + 残差 + MLP
    专家段 [1240:1257] → E_i 的 o_proj + 残差 + AdaRMSNorm(te) + MLP
  Deepstack (:152-163): 主干第 0..2 层后，把视觉塔的多层特征加到视觉位置上
最终: 专家 final RMSNorm → [1,17,768]，取后 16 个 token → action_out_proj(768→23) → v [1,16,23]   (:332)
```

- **checkpoint 里的专家有 36 层，但只有前 28 层会运行**：循环次数由主干层数决定。E28 到 E35 是不运行的权重。
- **专家没有自己独立的 KV cache 接口**。π0 是"把 VLM 的 KV 拼到专家 KV 后面"；我们的做法是**两条流在同一个 attention 调用里**，效果相同：专家 Q 能看到主干第 i 层的 K/V。

### 2.7 掩码：后缀能看到什么

`BC/model/attention_contract.py:13-48` 中的 `build_fused_mask`，序列为 `[16 个状态 token | 1224 个文本 token | 17 个后缀 token]`，总长 L=1257。

| Query ↓ / Key → | 16 个状态 token | 提示（视觉 850 + 文本 361 = 1211） | 答案列（13） | 后缀 17 |
|---|---|---|---|---|
| 前缀 | 因果 ✓ | 因果 ✓ | 因果 ✓ | ✗ |
| 后缀 | **✗**（:40） | ✓（:42） | **✗**（`~answer_mask`） | ✓ 全互看（:43） |

后果：
1. 前缀永远看不到后缀，所以**主干的 K/V 与 x1、t 完全无关**。如果做多步，前缀 KV 可以原样复用。
2. 改写答案文本**不会**影响动作输出（`fused_behavior_action.py:225-227`）。
3. 后缀内部是双向注意力：16 个动作步之间互相可见，这让动作在时间上保持平滑；还能看到状态 token。

实测每层后缀注意力的质量分布（已导出为 `E{i}.attn_mass`）：

| 层 | 文本 | 视觉 | 后缀 |
|---|---|---|---|
| L0 | 87.8% | — | 12.1% |
| L14 | 81.6% | 18.3% | 0.1% |
| L27 | 72.7% | 2.3% | 25.0% |

### 2.8 解码、反归一化与 settle

- 解码在 `BC/harness/fused_adapter.py:176-197`：`rows = (x1 − v)[0]`，形状 [16, 23]；检查数值有限后，用 bounds99 的逆变换还原到机器人单位。
- 23 维的排布见 `BC/data/schema.py:21-22`：

  | 维度 | 含义 |
  |---|---|
  | 0-2 | 底盘 vx / vy / yaw，**速度量** |
  | 3-6 | 躯干 1-4 |
  | 7-13 | 左臂 7 个关节 |
  | 14 | 左夹爪 |
  | 15-21 | 右臂 7 个关节 |
  | 22 | 右夹爪 |

  夹爪 +1 表示张开，−1 表示闭合。
- Settle 规则在 `BC/training/build_train_data.py:29-34`：动作块中越过当前 knack 结束帧 E 的那些行，**速度维度 0-2 置 0（按机器人单位），位置维度和夹爪维度保持最后一个真实值**；之后才做 bounds99 归一化。done 行也遵循这个"停 / 保持"的模式。

---

## 3. 统一记号：一次前向里的全部张量

| 记号 | 形状 | 来源 | 已导出键 |
|---|---|---|---|
| x1 | 16×23 | 噪声 | `action_x1` |
| t, te | 标量 1；768 | 正弦编码 | — **（新增）** |
| state75 | 75 | bounds99(state61) 补 0 | — **（新增）** |
| suffix_in | 17×768 | §2.4 | `expert_in` |
| E_i.attn | 17×1257 | 后缀 Q 对全部 K | `E{i}.attn`（窗口） |
| E_i.mass | 17×4 | 状态 / 视觉 / 文本 / 后缀 | `E{i}.attn_mass` |
| E_i.out | 17×768 | 第 i 层专家输出 | `E{i}.out` |
| expert_out | 17×768 | final norm | `expert_out` |
| v | 16×23 | action_out_proj | `action_velocity` |
| x0 = x1 − v | 16×23 | Euler 1 步 | `action_chunk` |
| a（真值） | 16×23 | 训练标签 | — **（新增）** |
| u = x1 − a | 16×23 | 训练目标速度 | — **（新增，可推导）** |

---

## 4. 实测：这 1 步究竟做了什么

脚本在 `docs/design/fm_experiment.py`，结果在 `fm_experiment_result.json`，使用真权重、float32、CPU，单次前向约 3 秒。

### 4.1 单步解码对比真值

- MSE(x0, a) = **0.370**。
- 一个恒等式：x0 − a = (x1 − v) − a = **u − v**。所以"解码误差"在代数上**就是训练损失**，实测两者同为 0.3705 ✓。
- 对照基线：MSE(0, a) = 0.337。也就是说，这个样本上单步解码**不比"全零"更好**。这需要在 viz 里如实呈现，不做美化。
- 第 0 步真值 a[0] 中，夹爪维 14 和 22 为 +1.0（张开），躯干 4（维 6）为 −1.0。预测 x0[0] 在 14 维 = 1.39（接近），在 22 维 = 0.11（偏差大）。
- 各维 MSE 最好的是 8、9 维（左臂 J2、J3，约 0.14 / 0.19），最差的是 4、6 维（躯干，约 0.58）。

### 4.2 噪声敏感性：用真实数据呈现"分布"

同一个 ctx，换 4 个噪声（原始 x1，以及种子 1001-1003），每个都是 t=1 的真实部署式前向：

| | 值 |
|---|---|
| 各噪声的 MSE | 0.370 / 0.363 / 0.469 / 0.425 |
| 4 个动作块之间的逐元素标准差均值 | **0.52**（作为对照，x1 自身的标准差是 1.0） |
| 4 块取平均后的 MSE | **0.156** |

**解读**：网络没有完全抵消噪声，约一半幅度的噪声泄漏到了动作里。4 次平均会逼近条件均值 E[a|ctx]，误差减半。这正是第 1 节"分布视角"的真实体现，而且**完全在训练分布内**（t=1），可以放心可视化。

### 4.3 反事实：强行走 10 步 Euler（t<1 不在训练分布内）

把 `_suffix_embeds` 的 t 改成 1.0, 0.9, …, 0.1，每步做一次真实前向，按 x ← x − 0.1·v 更新：

| k | t | ‖v‖ | MSE(x, a) |
|---|---|---|---|
| 0 | 1.0 | 11.76 | 1.167 |
| 1 | 0.9 | 6.58 | 1.127 |
| 5 | 0.5 | 4.09 | 1.037 |
| 9 | 0.1 | 2.55 | **1.008** |

**结论**：模型在 t<1 时输出的速度明显偏小，没有学过这些时刻。10 步走完 x 几乎还停在噪声附近，MSE 1.008，远差于真实单步的 0.370。
所以**"逐迭代步真值动画"如果用改 t 的多步来做，展示的是一个失败的、非部署的过程，会误导观众**。设计上不采用多步作为主线，详见 viz 文档 §1。（已验证：部署路径没有多步，见 §0 的验证框。）

---

## 5. 与 π0 的对照（只对照事实，不借用其视觉）

| | π0（视频字幕） | k005b |
|---|---|---|
| 主干 | PaliGemma / Gemma，18 层，2048 维 | Qwen3-VL-2B，28 层，2048 维 |
| 专家 | Gemma 同构，1024 维 | 768 维，16 个 Q 头、2 个 KV 头，checkpoint 36 层、运行 28 层 |
| 专家输入 | 1 个状态 + 50 个动作，14 维 | 1 个状态（75 维槽）+ 16 个动作，23 维 |
| t 训练 | 随机采样 | **固定为 1** |
| 推理 | 10 步 Euler，复用前缀 KV cache | **1 步**（x1 − v），一次联合前向 |
| 耦合 | 专家 Q 看 [VLM K/V ∥ 专家 K/V] | 同上，在一个共享 attention 中完成，并用掩码屏蔽状态 token 和答案列 |

---

## 6. 对可视化的直接约束（交给 viz 文档）

1. **主线只讲真实存在的这 1 步**，不画虚构的中间速度。
2. **x(t) = x1 − (1−t)·v 这条直线**可以作为几何解释画出来（端点都是真值），但必须标注"插值示意，非模型输出"。
3. **"分布 / 粒子"用 §4.2 的多噪声真实前向来表达**，全部是真值。
4. **Euler 步 = 残差**：复用 viz 已有的残差箭头语言，`x1 —(−v)→ x0`。
5. **耦合**：每层画"主干 L_i 的 K/V → 专家 E_i 的 Q"，其中 1211 列可见，状态和答案列被遮挡；用 `E{i}.attn_mass` 驱动视觉、文本、后缀三股的粗细。
6. **诚实指标**：在场景里给出 MSE 0.370 与"全零基线"0.337 的对比，以及 4 噪声平均后的 0.156。
