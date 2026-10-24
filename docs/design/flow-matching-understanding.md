# k005b 动作头的 Flow Matching 机制：理解笔记（10 步版）

> 本文是可视化设计（`flow-matching-viz.md`）的事实基础，所有结论都对应到代码行号。
> 代码根目录 `BC = /home/frank/behavior_challenge_ws/behavior_challenge/src`；行号对应主仓 **commit 20b39c2**（PR #638 合入后，含 dtype 修复）。定稿设计：`docs/design/action-10step-final.md`（主仓）。
>
> **版本说明（2026-10-25 更新）**
> - **v2：10 步流匹配（当前主线）**。PR #638 合入，新训练从 step_150000 起跑（t ~ Beta(1.5,1)，损失公式不变）。第一个带 `action_decode` 印章的检查点是 step_160000（尚未落盘）。§0–§5 都按 v2 写。
> - **v1：t=1 单步（旧血统，step_130000 及以前）**。此前"t=1 单步、没有迭代可 hook"的结论**对 v1 仍然成立**（Frank 2026-10-24 复核），但已被 v2 取代。v1 的实测数据保留在附录 A，作为 **K=1 回归基线**。
> - vla-viz 当前已提交的 3D 章节（1f7ca0f）展示的是 v1 / step_130000 的数据；等 step_160000 落盘后按本文重新导出。

---

## 0. 一句话

动作头是一个真正的流匹配采样器：主干对前缀**只跑一次**（`encode_prefix`），把每层 K/V 缓存下来；动作专家拿着这份缓存跑 **10 次**（`predict_velocity`），每次输入当前的 x_t 和 t，输出 16×23 的速度 v；每次 Euler 更新 `x ← x − 0.1·v`，t 从 1 走到 0。走完 10 步得到的 x_0 就是动作块。

```
noise x_1 [16×23] ─┐
                   ▼
encode_prefix (主干 28 层，一次) ──▶ PrefixEncoding{knack/done 头输出, 28 层 (K,V) 缓存, 后缀掩码, 后缀位置}
                   │
for k = 0..9:  t_k = 1 − 0.1·k   (1.0, 0.9, …, 0.1)
    v_k = predict_velocity(缓存, state75, x_{t_k}, t_k)     # 专家 28 层，联合注意力读缓存
    x_{t_k − 0.1} = x_{t_k} − 0.1 · v_k                    # Euler，向下积分
                   ▼
x_0 [16×23] = 动作块 ──▶ bounds99 逆变换 ──▶ 机器人
```

---

## 1. 公式：训练与解码

### 1.1 路径、目标、损失

| 量 | 定义 | 代码 |
|---|---|---|
| 噪声端 | x_1 = noise ~ N(0, I)，16×23 | 训练 `BC/data/fused_batch.py:334-336`（按 episode/帧/出现次数确定种子）；部署 `BC/harness/fused_adapter.py:273`（固定 seed 0） |
| 数据端 | x_0 = a，标签动作块（bounds99 归一化） | `batch["action_target"]` |
| 插值路径 | **x_t = (1−t)·a + t·x_1**（直线；t=1 是纯噪声） | `BC/training/fused_dual_head_sft.py:239` |
| 时间采样 | **t = 0.001 + 0.999·Beta(1.5, 1)**，用逆 CDF 实现：u ~ U(0,1)，s = u^(2/3) | `fused_dual_head_sft.py:194-202`（`sample_flow_t`） |
| 目标速度 | **u = x_1 − a**（= dx_t/dt，与 t 无关）——与 v1 逐字节相同 | `BC/training/behavior_losses.py:219` |
| 损失 | mean((v_θ(x_t, t, ctx) − u)²) over 16×23，再对 batch 平均 | `behavior_losses.py:220-221` |

- Beta(1.5,1) 的密度是 1.5·√t，向 t=1 倾斜，均值 0.6。这样暖启动来的专家最熟悉的区域（t≈1）仍然被密集训练（`fused_dual_head_sft.py:197-199` 注释）。
- **每行每次访问都重新抽 t**（数据增强，不写进数据表；`:199-200`）。
- 训练仍是**一次联合前向**（`forward`，`BC/model/fused_behavior_action.py:388-442`），只是后缀输入换成了 x_t，ada_cond 换成了 te(t)。
- 训练日志按 t 分五档统计 action loss（`fused_dual_head_sft.py:935-944`，`action_loss_t_bands`），并记录 K=10 解码相对 K=1 的偏移。

### 1.2 Euler 解码（向下积分）

`sample_action_chunk`，`fused_behavior_action.py:526-550`：

```
δ = 1/K   (K = flow_steps = 10)
x = noise (float32)
for step in 0..K−1:
    t = 1 − step·δ                       # 1.0, 0.9, …, 0.1
    v = predict_velocity(enc, state75, x, t)
    x = x − δ·v                          # 减号：v* = noise − target，减去它才是走向 target
return x (= x_0), v (最后一步的速度，只用于诊断)
```

- **符号**：定稿文档专门纠正了父规格里的 `+δ`（action-10step-final.md "Rejected ideas"）。
- **K=1 就是 v1**：t=1，一步 x_0 = x_1 − v。主仓测试 `test_k1_decode_reproduces_presurgery_golden`（`tests/test_action_10step_flow.py:126`）锁定了这一点。
- **Euler 的精确性**：如果 v_θ 正好等于直线路径的场 (x − a)/t，任意 K 都会精确落在 a 上（`test_euler_oracle_straight_path_field_lands_on_target`，:105）。

### 1.3 为什么要 10 步（相对 v1 的本质变化）

- v1 只在 t=1 训练。此时输入 x_1 与 a 独立，L2 最优解是 v* = x_1 − E[a|ctx]，所以一步解码只能得到**条件均值**。附录 A 的实测印证了这一点：8 个噪声的均值 MSE 0.108，单个是 0.37。
- v2 在 t∈(0,1] 上训练，最优场是 v*(x_t, t) = E[x_1 − a | x_t, t, ctx]。t 越小，x_t 里关于 a 的信息越多：
  - 前几步（t≈1）先推向均值方向；
  - 后几步看着"已经部分去噪"的 x_t，可以收敛到某一个具体的模态。
- 所以 10 步解码可以**采样**一个动作（不同噪声落到不同模态），而不是只输出平均。**这正是要在 3D 里展示的核心现象。**

---

## 2. 一次解码里的数据流（逐张量，v2）

### 2.1 Pass A：`encode_prefix`（主干一次）

`fused_behavior_action.py:446-502`

```
prefix = [16 个状态 token (StateProjector 61→2048) | text_embeds (视觉 850 + 文本 …)]   (:466-467)
position_ids = cumsum(validity) − 1，覆盖前缀 + 17 个后缀位置                              (:472-478)
out_prefix, layer_kv = trunk.forward_prefix_stream(prefix, mask[:npfx,:npfx], …)        (:481-483, 实现 :222-242)
   └─ 在层循环内部捕获每层 post-RoPE 的 (K, V)，保持原生 8 个 KV 头                         (_layer_pair 的 kv_capture, :117-127)
      必须在循环内捕获：deepstack 在第 0..2 层之间往视觉位置注入特征，会改变下一层的 K
behavior_fields = heads(out_prefix[boundary_column])                                     (:494-497)
return PrefixEncoding{behavior_fields, layer_kv[28], suffix_key_mask[17×npfx], suffix_positions[17]}  (:498-502)
```

- 前缀里**没有 t，也没有 x_t**。所以 knack/done 在 10 步中逐位相同（测试 `test_behavior_outputs_bit_identical_between_k1_and_k10`，:262）。主干的 28 层 K/V 也只算一次。
- 这就是 π0 说的"前缀 KV cache 复用"，在我们的代码里是**显式的数据结构**（`PrefixEncoding`，:289-299）。

### 2.2 Pass B：`predict_velocity`（专家一次，循环 10 次）

`fused_behavior_action.py:504-524`

```
后缀 token（_suffix_embeds, :356-386）：
  token 0     = state_proj(state75)                                    75→768
  token 1..16 = time_mlp_out(SiLU(time_mlp_in([action_in_proj(x_t) ‖ te(t)])))
                action_in_proj 23→768；te 是 768 维正弦（周期 0.004–4）；time_mlp_in 1536→768
  ada_cond    = te(t)                                                  ← 每步不同
后缀掩码 = [suffix_key_mask (前缀可见列) | 17×17 全 1]                    (:514-518)
trunk.forward_suffix_stream(后缀, layer_kv, 掩码, 位置, ada_cond=te)    (实现 :244-273)
  for i in 0..27:                                                      (:256)
     q,k,v = E_i(AdaRMSNorm(h; te))                                     (:258-259)
     RoPE(q,k, suffix_positions)                                        (:262-263)
     专家 2 个 KV 头 tile 到 8 个                                        (:264-265, tile_kv_heads = shared_attention.py:241-252)
     K = [缓存 K_i ‖ k_suffix]，V = [缓存 V_i ‖ v_suffix]                (:266-267)
     att = softmax(q Kᵀ/√d + mask) V                                    (:268)
     h = E_i(o_proj + 残差 + AdaRMSNorm(te) + MLP)                       (:269-270)
  final norm（普通 RMSNorm，final_norm_adanorm=False, :31, :271-273）
v = action_out_proj(后 16 个 token)                                     768→23    (:524)
```

- 循环层数由缓存长度决定，也就是主干的 28 层。checkpoint 里专家有 36 层，E28–E35 不运行。
- 注意力只算后缀的 17 行 query，对 [前缀可见列 ‖ 17 个后缀列]。前缀的 1240 行在 pass A 里已经算完。
- 这条拆分路径与训练时的联合前向**数值等价**（≤1e-5；测试 :181、:192、:237，覆盖右侧 padding、deepstack、专家层数多于主干的情况）。

### 2.3 t 的两个入口（每步都变）

1. **拼进每个动作 token**：te(t) 与 action_in_proj(x_t) 拼接后过 time MLP。
2. **调制每层归一化**：专家每层的 input / post-attention norm 都是 AdaRMSNorm（`shared_attention.py:60-85`）：
   `h = (1 + γ(te))·RMSNorm(x)·w + β(te)`。

v1 里 t 恒为 1，γβ 退化成固定仿射。**v2 里 γ(te_k)、β(te_k) 每一步都不同**，这是可以导出、可以动画的真实量。迁移时重新学习的正是这条时间通路：`action_time_mlp_*` 和 AdaRMSNorm 的 γ/β（定稿文档 Migration 行）。

### 2.4 掩码：后缀能看到什么（不变）

`BC/model/attention_contract.py:13-48` 中的 `build_fused_mask`。序列为 [16 状态 | 文本（视觉+提示+答案）| 17 后缀]。

| Query ↓ / Key → | 16 个状态 token | 提示（视觉 + 文本） | 答案列 | 后缀 17 |
|---|---|---|---|---|
| 前缀 | 因果 ✓ | 因果 ✓ | 因果 ✓ | ✗ |
| 后缀 | ✗ | ✓ | ✗ | ✓（全互看） |

- 前缀看不到后缀，这是 pass A 能独立出来的**结构性证明**。
- 后缀只能看到可见的提示列（v1 样本中是 1211 列 = 850 视觉 + 361 文本），看不到状态 token 和答案列。专家拿本体状态的唯一入口是后缀 token 0（state75）。

### 2.5 印章与加载规则（决定什么能导出）

| 规则 | 代码 |
|---|---|
| 解码法则字符串 `ACTION_DECODE_LAW = "euler-t-down-v-star-noise-minus-target"` | `fused_dual_head_sft.py:77` |
| 新检查点写入 `action_decode = {law, flow_steps}` | `checkpoint_payload`，:333-343 |
| **无印章 = v1 血统**：只能在 flow_steps=1 下加载；要求 K>1 时直接报错 | `check_action_decode_stamp`，:401-432 |
| 服务端：无印章用 K=1，有印章用 stamp.flow_steps（默认 10） | `BC/harness/fused_serve.py:172-190` |
| 服务端调用 `model.decode`，然后 `unnormalize_action_chunk` | `fused_adapter.py:239-244`，:177 |

对 vla-viz 的含义：**逐 Euler 步导出只对带印章的检查点有意义**（step_160000 起）。用 step_130000 跑 K=10 是分布外（附录 A.3 的实测 MSE 1.008），而且 loader 会拒绝。

### 2.6 解码之后：反归一化与 settle（不变）

- 23 维：0–2 底盘 vx/vy/yaw（速度量）；3–6 躯干 1–4；7–13 左臂；14 左夹爪；15–21 右臂；22 右夹爪（`BC/data/schema.py:21-22`）。夹爪 +1 张开、−1 闭合。
- bounds99：norm = (x − q01)/(q99 − q01)·2 − 1。服务端求逆（`unnormalize_action_chunk`，`fused_adapter.py:177`）。
- settle（`BC/training/build_train_data.py:29-34`）：越过 knack 结束帧的行，速度维 0–2 在机器人单位下清零，位置和夹爪保持最后的真实值。done 行和 settle 行照常参与训练。

---

## 3. 统一记号：一次 10 步解码的全部真实张量

| 记号 | 形状 | 每步不同？ | 来源 |
|---|---|---|---|
| x_1 | 16×23 | — | 噪声（部署 seed 0 / 训练行 action_noise） |
| state75 | 75 | 否 | 后缀 token 0 的输入 |
| prefix (K_i, V_i) | 28 × [npfx, 8, 128] | **否**（pass A 一次） | `PrefixEncoding.layer_kv` |
| knack / done | — | **否** | `PrefixEncoding.behavior_fields` |
| t_k | 标量，k=0..9 | 是 | 1.0, 0.9, …, 0.1 |
| te_k | 768 | 是 | 正弦嵌入 |
| γβ_{k,i} | 28 层 × 4 × 768 | 是 | AdaRMSNorm(te_k) |
| suffix_in_k | 17×768 | 是 | `_suffix_embeds(x_{t_k}, t_k)` |
| E_{k,i}.attn | 17 × (npfx+17) | 是 | 后缀 Q 对 [缓存 K ‖ 后缀 K] |
| E_{k,i}.mass | 17 × 4（状态/视觉/文本/后缀） | 是 | 由上面聚合 |
| E_{k,i}.out | 17×768 | 是 | 第 i 层专家输出 |
| v_k | 16×23 | 是 | action_out_proj |
| x_{t_k − δ} | 16×23 | 是 | x_{t_k} − 0.1·v_k；共 11 个状态 x_{1.0} … x_{0.0} |
| a | 16×23 | — | 标签（导出样本是训练行，所以有） |
| u = x_1 − a | 16×23 | 否（直线路径上为常数） | 训练目标速度 |
| x*_t = (1−t)a + t·x_1 | 16×23 × 11 | 是 | **理想直线路径**，可与真实 Euler 轨迹对照 |

可推导的诊断量（都基于真值）：
- 每步速度误差 ‖v_k − u‖²；
- 每步位置误差 MSE(x_{t_k}, a) 和偏离理想路径的距离 ‖x_{t_k} − x*_{t_k}‖；
- K=10 与 K=1 的最终 MSE 对比（训练日志里也有这一项）。

---

## 4. 讲清楚的手法（参考库吸收，不照搬视觉）

| 手法来源 | 讲法 | 落到我们的 10 步 |
|---|---|---|
| Cambridge MLG：Euler 步 = 残差 | x ← x + (−δ)·f(x) 与 h ← h + f(h) 同形 | 10 步解码 = 一个"10 层、共享权重、靠 t 区分"的残差网络，每一层本身又是专家的 28 层。深度（28）× 时间（10）两个方向在场景里正交摆放 |
| Roelants 三联画 + 速度场热力图 | 路径 / 场 / 采样并排 | 理想直线 x*_t（幽灵）、真实 Euler 轨迹 x_{t_k}、每步速度板 v_k 三者同框 |
| nbahador 粒子搬运 | 场搬运一群点 | 8 个噪声 × 10 步的真实轨迹；对照 v1 的"全部收缩到均值" |
| drscotthawley：搬运分布 | 多模态 | 看末端 x_0 是否分簇（step_160000 的真实结果，不预设） |
| π0 字幕：KV cache 复用 | 前缀算一次 | pass A 的缓存块在 10 步中保持不变，有断言作证 |

---

## 5. 与 π0 的对照（只对照事实）

| | π0 | k005b v1（step_130000） | k005b v2（step_160000 起） |
|---|---|---|---|
| t 训练 | 随机采样 | 固定为 1 | 0.001 + 0.999·Beta(1.5,1) |
| 推理 | 10 步 Euler + KV 复用 | 1 步 x_1 − v | **10 步 Euler（向下积分）+ 显式 PrefixEncoding 复用** |
| 专家 | 1024 维 | 768 维，16Q/2KV，运行 28 层 | 同 v1 |
| 动作 | 50×14 | 16×23 | 16×23 |

---

## 附录 A：v1 / step_130000 实测（K=1 回归基线）

> 已验证（Frank，2026-10-24）：v1 训练 u = x_1 − a，不采样 t（`fm_time = ones`）；部署一步 x_1 − v。当时 vla-viz 的导出与 3D 章节都基于此。数据来自 row 0（episode 0，frame 0）；脚本 `docs/design/fm_experiment.py`，结果见 `fm_experiment_result.json` 和 `public/k005b-check.json`。

### A.1 单步解码对比真值
- MSE(x_0, a) = **0.370** = MSE(v, u)，恒等式 x_0 − a = u − v。全零基线 0.337。
- 逐维最好的是 8、9 维（约 0.14），最差的是 4、6 维（约 0.58）。

### A.2 8 个噪声
- 单个 MSE 在 0.363–0.529 之间；块间标准差 0.584；**8 块均值的 MSE 为 0.108**。
- 结论：单步 ≈ 条件均值 + 泄漏进来的噪声。这正是 v2 要用多步改进的地方。
- 前缀 K/V 在 8 次前向中的最大差异 = 0.0，证明前缀与噪声无关。

### A.3 反事实：在 v1 权重上强行走 10 步
| k | t | ‖v‖ | MSE(x, a) |
|---|---|---|---|
| 0 | 1.0 | 11.76 | 1.167 |
| 5 | 0.5 | 4.09 | 1.037 |
| 9 | 0.1 | 2.55 | **1.008** |

v1 没学过 t<1，速度偏小，x 停在噪声附近。v2 的加载器因此**禁止**在无印章检查点上使用 K>1。

> 注：这里的前向用的是导出时 v1 的联合路径。v2 导出时须用 `encode_prefix` + `predict_velocity`，并断言它等于 `sample_action_chunk`。
