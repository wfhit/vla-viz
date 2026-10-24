# Flow Matching 3D 可视化：分镜设计方案

> 事实基础见 `flow-matching-understanding.md`（下文简称"理解 §x"）。本文只做设计，**不包含实现**。
> 约束：只用 k005b 的真实权重和激活；不弹窗，不做大面积覆盖层，所有演示都在 3D 场景内进行（与 Qwen3 章节同一风格）；3D 文字只用 ASCII；不改 FOV。
> 参考了几个外部来源（Roelants 的速度场图、nbahador 的粒子搬运、Cambridge MLG 的"残差"视角、π0 字幕），**只借鉴讲法，不照搬画面**。如何转译成我们的 3D 语言，见 §6。

---

## 1. 核心决策：讲"真实的 1 步"，不伪造多步

| 方案 | 内容 | 结论 |
|---|---|---|
| **A（推荐）** | 主线讲透 t=1 的这一次真实 Euler 步。"迭代"的直觉改用两种真实数据表达：① 28 对层逐层推进（深度方向的迭代）；② 多噪声真实前向（分布 / 粒子） | ✅ 全是真值，并且都在训练分布内 |
| B | 把 t 改成 1.0, 0.9, …, 0.1，做 10 步反事实前向 | ❌ 实测 10 步后 MSE 为 1.008，远差于真实 1 步的 0.370（理解 §4.3）。模型没学过 t<1，展示出来的是一个"失败的过程"，会误导观众 |

**✅ 已确认（Frank，2026-10-24）**：k005b 没有多步 Euler 迭代，"hook 每个 Euler 迭代步"的要求作废（没有迭代可 hook）。训练 u = x1 − a（`behavior_losses.py:219`），部署 action = x1 − v 一次减法（`fused_behavior_action.py:209`），详见理解 §0 验证框。

分镜主线按真实单步机制画：

**噪声板 x1 → 专家单次前向（ada_cond = te(t=1)，逐层联合注意力到主干前缀）→ 速度场 v → 一次减法 → 16×23 动作块**

对应分镜：FM-1/FM-2（噪声板与后缀 token）→ FM-3/FM-4（单次前向：联合注意力 + t=1 调制）→ FM-5（速度场）→ FM-6（一次减法）→ FM-7..FM-9（真值对照、多噪声、落到机器人）。

场景中能看到的"推进感"只来自真实数据：逐层 E_i 输出（28 个层对，属于同一次前向的深度方向）和多噪声的独立前向。设计会明确告诉观众：π0 走 10 步，k005b 是 1 步。FM-6 的 τ 动画只是 x1 → x1−v 的直线插值，场内标注 `interp only`；B 的实测结果只作为一段解说文字出现在 FM-6，不做动画。

---

## 2. 新导出数据（扩展 `tools/export_k005b.py`）

| 键 | 形状 | 含义 / 来源 | 大小估算 |
|---|---|---|---|
| `fm_state75` | 75 | `tensors["action_state"]`，即 bounds99 归一化后补 0 的状态槽 | <1 KB |
| `fm_time_emb` | 768 | `_suffix_embeds` 返回的 te（sin/cos 各 384 维） | 4 KB |
| `fm_state_emb` | 768 | `state_proj(state75)`，即后缀第 0 个 token | 4 KB |
| `fm_action_emb` | 16×768（窗口 16×64，附完整 stats） | `action_in_proj(x1)` | 6 KB |
| `fm_time_mlp_hidden` | 16×768（窗口） | `silu(action_time_mlp_in(cat))` | 6 KB |
| `E{i}.ada_gamma` / `E{i}.ada_beta` | 每层 2×(2×768)（input / post 两个 norm），窗口 64 | 真实 γ(te)、β(te)，即 t 对第 i 层的调制 | 约 60 KB |
| `E{i}.q_suffix` | 16 头×17×64 窗口 | 专家后缀的 Q（经 RoPE） | 约 1 MB，可只导出 head 0 以降到 70 KB |
| `E{i}.attn_full_mass_cols` | 17×1257 的分块汇总 | 把 1257 列按 [状态 16 \| 视觉 850 \| 文本 361 \| 答案 13 \| 后缀 17] 聚合成每个 query 的 5 段质量（已有 4 段，补上答案段，应恒为 0） | 小 |
| `fm_target` | 16×23 | 训练标签 a（`batch["action_target"]`） | 2 KB |
| `fm_u` | 16×23 | x1 − a，训练目标速度 | 2 KB |
| `fm_particles` | K=8 组 × {x1, v, chunk}，每个 16×23 | 8 个噪声种子（原 x1，以及种子 1001-1007）各做一次 t=1 真实前向 | 约 50 KB |
| `fm_particles_mass` | 8×28×17×4 | 每个噪声下逐层的后缀注意力质量 | 约 80 KB |
| `fm_cf_euler10` | 10×{t, ‖v‖, mse} | 反事实实验的标量摘要，只作为文字素材 | <1 KB |
| `config.fm` | — | `{t:1, steps:1, sigma_min:4e-3, sigma_max:4, n_seeds:8, mse:0.370, mse_zero:0.337, mse_mean8:…}` | — |

- 新增总量 < 0.5 MB；`k005b-acts.json` 当前 16 MB。
- 导出器内置断言：
  1. 8 个噪声下，**前缀各层 K/V 逐位相等**（验证前缀不依赖 x1，可复用 KV cache）；
  2. `x1 − v == chunk`；
  3. `mse(chunk, a) == mse(v, u)`。

  断言结果写进 `k005b-check.json`。

---

## 3. 场景布局改动（只动动作专家这一列）

```
 [state75 1x75]  [x1 16x23]  [t=1]→[te 768]          ← 新增：专家输入簇，位于专家列顶部、与 residual0 同高
      │ state_proj W   │ action_in_proj W   │
      └──────┬─────────┴── time_mlp_in/out W ┘
          [Expert Suffix In 17x768]                  ← 已有 expertIn，现在有了明确来源
              │
   ┌──────────┼──────────────┐ 每层 i：
   │ trunk L_i  K/V ──▶ [Joint Attn E_i 17x1257] ◀── Q(E_i)
   │                         │   [γβ(te)] 小条 ─▶ 本层 norm
   │                    [Expert i Out 17x768]
   └──── ×28（随主干分列换列，换列处只放文字标记）
              │
      [Expert Final Norm] ─▶ [action_out_proj W] ─▶ [v 16x23 速度板]
 x1 残差轨 ════════════════════════════════════════▶ [x0 = x1 − v 动作块] ─▶ [3D 柱场 / 粒子轴]
```

- **x1 残差轨**：从顶部 x1 块出发，沿专家列外侧向下，一直到动作块。它表达的是"Euler 步就是一次残差连接"。跨列时沿用主干换列的文字标记（`x1 continues ->`），不画回头箭头。底部现有的 `actionX1` 块改名为 `x1 (skip)`，作为这条轨的终点，**不再是一个独立的来源**。
  - 备选方案：保留底部块，只用文字标注"与顶部是同一张量"。**请 Frank 选择。**
- **速度板**：把 `actionVelocity` 块改成 23×16 的平板。格子颜色采用红负蓝正的对称色阶，格子高度表示真实的 v 值。

---

## 4. 箭头清理原则

1. **注册表制**：所有箭头统一写成 `{src, srcFace, dst, dstFace, width, phases}` 的形式。src 和 dst 都必须是 `layout.cubes` 里**可见**的块（不能 hidden，opacity 必须大于 0）。不满足条件的箭头不绘制；dev 模式下在控制台报错。
2. **启动时自检**：列出"没有入箭头的数据块"（白名单只包括 idxObj、各权重块、FM 输入簇）和"没有出箭头的数据块"（白名单只包括终端块）。自测时这两个列表都必须为空。
3. **不画没有箭头的线**：联合注意力连线一律改用带箭头的 `drawArrow`，不再使用 `addLine`。
4. **默认不显示**：联合注意力连线目前在总览中以 0.18 的不透明度常驻，看起来像从一个点辐射出去的一扇线，这就是 Frank 说的"飞的箭头"。改为只在 FM-3 和 FM-4 显示，其他阶段为 0。

### 第一块（入口）箭头清单

| # | 箭头 | 源块 | 终块 | 处置 |
|---|---|---|---|---|
| 1 | token id 选行 | `idxObj`（Prompt token ids） | `residual0` 顶面 | 保留 |
| 2 | 查嵌入表 | `tokEmbedObj` 右面 | `residual0` 左面 | 保留 |
| 3 | 位置嵌入 | `posEmbedObj`（hidden 占位，无数据） | `residual0` 右面 | **删除**（Qwen3 用 RoPE，没有位置表），占位块也一并删除 |
| 4 | 新增：专家输入 ×3 | state75 / x1 / te | 各自的投影权重，再到 `expertIn` | 新增 |
| — | 视觉 token（850）和 16 个状态 token | 窗口外，场景里没有对应的块 | — | 不画箭头；在 `residual0` 下方加一行 ASCII 标注 `window: text cols [t0w..p*]; 850 visual + 16 state tokens outside` |

---

## 5. 分镜表（新章节组"动作专家与流匹配"，替代现有的 Action_Expert / Action_Denoise）

| 步 | 标题 | 相机 | 3D 场景画什么 | 箭头（源 → 终） | 展示的真值 | 播放条（breakAfter 处） |
|---|---|---|---|---|---|---|
| FM-0 | 一次前向，两条流 | 远景，主干列加专家列 | 其他块淡化到 0.3，两列高亮；专家列标注 `28 layer pairs, 1 forward` | 现有残差链 | 层数 28、专家宽 768 / 主干宽 2048 | ① 两条流 ② "k005b 只走 1 步，π0 走 10 步"（附理解 §5 表） |
| FM-1 | 三个输入 | 推近专家列顶部的输入簇 | state75 柱条（75 格，后 14 格灰，表示补 0）；x1 的 23×16 格（噪声）；t=1 标量块展开为 te 768 维的 sin/cos 条纹 | — | `fm_state75`、`action_x1`（如 x1[0][0]=−0.953）、`fm_time_emb` | ① 状态槽 ② 噪声种子（训练按 episode/帧，部署固定 seed 0）③ t 恒为 1 |
| FM-2 | 拼成 17 个后缀 token | 同上，略拉远 | 依次点亮：state_proj → token 0；action_in_proj(x1) 与 te 拼接 → time_mlp_in → SiLU → time_mlp_out → token 1..16；`expertIn` 逐行填充 | state75 → state_proj W → expertIn[0]；x1 → action_in_proj W → time MLP → expertIn[1..16]；te → time MLP | `fm_state_emb`、`fm_action_emb`、`fm_time_mlp_hidden`、`expert_in` | ① token 0 ② token 1..16 ③ te 还会进入每一层的 norm |
| FM-3 | 联合注意力：Q 来自专家，K/V 来自主干 | 侧视第 activeLayer 层，主干 kvGroups 与 E_i 同框 | E_i.attn 块按列分段着色：状态 16 列和答案 13 列显示为**灰色并加删除线**（被掩码），视觉 / 文本 / 后缀三段用真实概率着色；块旁是三段质量条 | 主干 L_i 的 K 组、V 组 → E_i.attn 左面（箭头宽度 ∝ 前缀质量）；E_{i−1}.out → E_i.attn（提供 Q） | `E{i}.attn`、`E{i}.attn_mass`（L0 文本 87.8% / L14 视觉 18.3% / L27 后缀 25.0%） | ① 一层的拆解 ② 掩码规则 ③ 自动扫层 L0 → L27（activeLayer 递增，质量条随之变化）④ "前缀看不到后缀，所以主干 K/V 与噪声无关" |
| FM-4 | t 调制每层 | 推近当前层的 expert norm | 每层旁边一对细条 γ(te)、β(te)，显示 (1+γ)·h+β 的作用 | te → 各层 γβ 小条 → E_i norm（只画当前层，其余层淡化） | `E{i}.ada_gamma/beta` | ① AdaRMSNorm ② t 固定，所以这只是一组固定的逐通道仿射 |
| FM-5 | 28 层 → 速度场 v | 跟拍专家列下行，到动作组 | 逐层 E_i.out 块依次亮起（深度方向的迭代）；final norm → action_out_proj → **速度板** 23×16 升起 | E27.out → final norm → action_out_proj W → v | `E{i}.out`、`expert_out`、`action_velocity`（如 v[0][15]=−1.216） | ① 深度方向的迭代 ② 速度板，每格对应 维度×时间步 ③ 23 维分组（底盘 / 躯干 / 左臂 / 左夹爪 / 右臂 / 右夹爪） |
| FM-6 | Euler 一步就是残差 | 动作组斜俯视 | x1 残差轨点亮；柱场从 x1 高度出发，按 τ 从 0 到 1 线性移动到 x1−v，每根柱上画 −v 的竖向小箭头；角标 `x(t)=x1-(1-t)v  [interp only]` | x1（残差轨）→ 动作块；v → 动作块（带减号） | `action_x1`、`action_velocity`、`action_chunk` | ① 公式 ② τ 动画（可拖动）③ "中间帧只是直线插值，不是模型输出" ④ 反事实 10 步实验的数值（解说文字表格，不做动画） |
| FM-7 | 对照真值：误差就是损失 | 同上 | 真值 a 以线框幽灵柱的形式叠加；预测与真值之间画误差竖线；场内 ASCII 文本显示 `MSE 0.370 vs zero 0.337`；逐维误差小条 | 动作块 ↔ 真值，用误差线连接，不加箭头头部（误差不是数据流） | `fm_target`、`fm_u`；逐维 MSE（8、9 维最好约 0.14，4、6 维最差约 0.58） | ① x0−a = u−v ② 如实给出 MSE ③ 夹爪维 14 / 22 |
| FM-8 | 分布：8 个噪声，8 个动作 | 动作组正视，切到"粒子轴"模式 | 23 根竖轴，每根轴对应一个维度，以 step 0 为例（可切换 step）。每根轴上 8 个粒子，从各自的 x1 被各自的 v 推到 chunk（同步做 τ 动画）；终点画均值标记和真值标记 | 每个粒子的轨迹本身就是箭头：起点 x1_k，终点 chunk_k | `fm_particles`、`fm_particles_mass`；块间标准差 0.52，8 块平均 MSE（4 块时为 0.156） | ① 同一个场搬运不同噪声 ② 均值更接近真值，说明 t=1 单步本质上是在回归条件均值 ③ 前缀 K/V 在 8 次里完全相同（断言结果） |
| FM-9 | 落到机器人 | 柱场 | 柱场标签从归一化值切换为机器人单位（q01/q99 逆变换）；按组着色；底盘 0-2 标注"速度量"；settle 规则用文字说明 | 动作块 → 柱场 | `config.action_q01/q99`、`action_chunk` | ① 23 维含义 ② bounds99 逆变换 ③ settle：完成后速度清零、位置和夹爪保持（本样本是 frame 0，不处于 settle，如实说明） |

---

## 6. 参考手法 → 我们的转译（不照搬）

| 参考 | 它的讲法 | 我们的 3D 转译 |
|---|---|---|
| Roelants 速度场热力图（t × x） | 一张图看出"每一步往哪推" | 我们只有 t=1 一个时刻，所以把图轴换成 **维度 × 时间步** 的速度板（FM-5），颜色和高度都是真实的 v |
| nbahador 粒子搬运 | 场搬运粒子 | FM-8：8 个真实噪声粒子，全部是 t=1 的部署式前向 |
| Cambridge MLG：Euler = 残差 | 一步 Euler 等于一次残差连接 | FM-6：x1 残差轨，与主干的残差箭头采用同一种视觉 |
| drscotthawley：变换分布，不是点 | — | FM-8 的均值标记与"条件均值"解说 |
| π0 字幕：专家 Q 看 VLM 的 K/V，复用 KV cache | — | FM-3 的 K/V 箭头；FM-8 的"前缀 K/V 相同"断言 |

---

## 7. 交互流程

- 新章节组"动作专家与流匹配"放在"输出头"之后，共 FM-0 到 FM-9 十个 phase。Harness_Law 保持原位。
- 每个 phase 内部，`commentary` 文本分段推进；每段的 `breakAfter` 对应上表"播放条"一列中的 ①②③。相机用 `cameraKeyframes` 在段间平滑移动。
- 可拖动的内部时钟有两个：
  - FM-3 的 activeLayer：0 → 27，用 `phaseClock` 驱动，也可以拖动；
  - FM-6 / FM-8 的 τ：0 → 1。
- **HeadsOverlay 在所有 FM 阶段移除**；knack 和 done 两章是否也改为场景内演示，**请 Frank 确认**。
- 离开 FM 章节时，actionViz 恢复默认：连线为 0，柱场显示最终动作块。

---

## 8. 自测计划

1. 导出器断言全部通过（§2），`k005b-check.json` 中有记录。
2. 箭头自检：两个列表为空，没有使用 hidden 或 opacity=0 端点的箭头（§4）。
3. 每个 FM 阶段的每个 breakAfter 点各截一张图（约 25 张），重点检查：
   - 箭头两端都贴在块面上；
   - 标签没有超出视野；
   - 没有大块 2D 覆盖。
4. `npx tsc --noEmit`，加上在 /tmp/vlabuild 中完成的构建。
5. 性能：帧时间不高于当前水平（swiftshader 下约 330 ms）。

---

## 9. 待 Frank 确认

1. ~~采用方案 A~~ **已确认**：真实单步（无 Euler 迭代可 hook），28 层逐层推进 + 8 噪声粒子；反事实 10 步只作为文字说明。
2. x1 的处理：残差轨（推荐）还是底部块加文字标注。
3. 粒子数 K=8 是否合适。
4. knack 和 done 章节是否也去掉 HeadsOverlay，改为场景内演示。
