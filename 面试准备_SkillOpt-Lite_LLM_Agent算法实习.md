# SkillOpt-Lite 项目深度解读 & LLM/Agent 算法实习面试准备

> 面向 MS 在读 / 找 LLM 算法实习的同学  
> 基于 `SkillOpt-Lite` 项目代码与文档全面梳理，提炼面试可深挖的核心考点与细节

---

## 目录

1. [项目核心概览](#1-项目核心概览)
2. [ReflACT 算法深度拆解](#2-reflact-算法深度拆解)
3. [核心数据流与数据结构](#3-核心数据流与数据结构)
4. [面试重点：概念类比与迁移](#4-面试重点概念类比与迁移)
5. [深度挖掘题（按面试轮次/深度分层）](#5-深度挖掘题按面试轮次深度分层)
6. [实现细节暗礁（容易答错的点）](#6-实现细节暗礁容易答错的点)
7. [设计取舍讨论（展示工程品味）](#7-设计取舍讨论展示工程品味)
8. [可能的改进方向（加分项）](#8-可能的改进方向加分项)
9. [推荐自我介绍话术](#9-推荐自我介绍话术)

---

## 1. 项目核心概览

### 1.1 一句话定义

**SkillOpt-Lite**（又名 **ReflACT**）是一个**纯文本空间的 Agent 行为优化框架**：不微调模型权重，而是通过**迭代编辑一份 Markdown skill 文档**来提升冻结 LLM Agent 的表现，全程用**验证门控（val-gate）**决定是否保留修改。

### 1.2 两种模式

| 模式 | 修改对象 | 使用场景 |
|------|---------|---------|
| `/skillopt-loop` | 仅 `skill.md` | 已有 Agent，只优化 prompt/指令 |
| `/harnessopt-loop` | `skill.md` + Agent Python 代码 | 联合优化 skill 和 harness（如 rollout、react agent） |

### 1.3 为什么叫 "Lite"

- 不用独立 runner，直接在 **VS Code Copilot Chat / Codex CLI / Claude Code** 等编码助手里用 slash-command 驱动
- Coding agent 负责：环境准备、执行 rollout、检查样本、patch skill.md、val-gate、rollback、归档
- 比完整的 SkillOpt（微软）更轻量、更自动化

### 1.4 论文定位

- **ICML 2026 Poster**
- 在 6 个 benchmark 上显著超越 SkillOpt 基线（LiveMath +8.8~25.4 pts，ALFWorld +9.5，SpreadsheetBench +12.6 avg）
- 核心贡献：**minibatch reflection + 慢速纵向更新（Slow Update）+ Meta Skill** 的组合

---

## 2. ReflACT 算法深度拆解

### 2.1 6-Stage Pipeline（每 step）

```
   ┌─────────┐    ┌─────────┐    ┌──────────┐    ┌───────┐    ┌───────┐    ┌─────────┐
   │ Rollout │───▶│ Reflect │───▶│ Aggregate│───▶│Select │───▶│Update │───▶│Evaluate│
   └─────────┘    └─────────┘    └──────────┘    └───────┘    └───────┘    └─────────┘
                                                                                  │
                                                                                  ▼ accept?
                                                                             ┌──────────┐
                                                                             │  Keep    │
                                                                             └──────────┘
```

**① Rollout**  
- 用当前 skill 执行 `batch_size × accumulation` 个 episode
- 每个 episode 得到一个 `RolloutResult`（hard/soft score、turn 数、失败原因、task_type）
- 并行执行：`ThreadPoolExecutor`，支持断点续跑（已有 `results.jsonl` 时跳过）

**② Reflect（Minibatch Reflection）**  
- **核心创新**：不是逐条分析轨迹，而是把失败轨迹分成大小为 M 的 minibatch，**一个 LLM 调用分析一批**
- 类比：SGD（逐样本）vs Minibatch SGD（批次）
  - 好处：**成本更低 + 能发现跨样本的共性失败模式 + 天然并行**
- 分两条线并行运行：
  - **Error Analyst**：分析失败轨迹 → 产生 `failure_patch`
  - **Success Analyst**：分析成功轨迹 → 产生 `success_patch`
- 输出：`RawPatch`（含 `failure_summary` + `edits` + `source_type` + `batch_size`）

**③ Aggregate（层级合并）**  
- N 个 patch → 1 个合并 patch，分三层：
  1. **同层并行**：同一层的多个 patch 用 `ThreadPoolExecutor` 并行合并
  2. **失败优先**：failure patches 先合并，success patches 后合并
  3. **最终合并**：两者再合并，prompt 明确要求 "FAILURE PATCHES TAKE PRIORITY"
- `support_count` 在合并链中传播，高支持度的 edit 更可能被保留

**④ Select（选 Top-L）**  
- 两种模式：
  - **固定预算**：用 LR scheduler（constant / linear / cosine）决定本 step 最多应用 L 个 edit
  - **Autonomous**：让 optimizer LLM 自己判断该应用多少个 edit
- 类比：**Gradient Clipping** — rank edits by importance，保留 top-L

**⑤ Update（应用 Edit）**  
- 4 种 edit 操作：`append` / `insert_after` / `replace` / `delete`
- **保护区域**：`<!-- SLOW_UPDATE_START -->...<!-- SLOW_UPDATE_START -->` 内的内容被 step-level analyst 跳过
- 3 种更新模式：
  - `patch`：传统逐条 edit
  - `rewrite_from_suggestions`：先产出 "revise suggestions"，再整体重写
  - `full_rewrite_minibatch`：每个 minibatch 产出完整 skill candidate

**⑥ Evaluate（验证门控）**  
- 类比：**Early Stopping / Model Selection**
- 在 selection set（`valid_seen`）上评估 candidate skill
- 三种 gate metric：`hard`（exact match）/ `soft`（F1/partial credit）/ `mixed`
- 决策：
  - `accept_new_best`：超过历史最优 → 同时更新 current 和 best
  - `accept`：超过当前但未超最优 → 只更新 current
  - `reject`：≤ 当前 → 回滚，保持 current

### 2.2 Epoch-Level 机制

**Slow Update（慢速纵向更新）** — 类比：**Long-term Memory / Strategic Reflection**
- 在 epoch 边界（epoch 2+）触发
- 用**相同的 20 个样本**，对比上一 epoch 的 skill 和当前 epoch 的 skill 的表现
- 分类：`improved` / `regressed` / `persistent_fail` / `stable_success`
- 让 optimizer LLM 写出**战略性指导**写入 protected 区域
- **强制应用**（不经过 gate），因为这是宏观策略层面的调整

**Meta Skill（优化器侧记忆）**
- 从相邻 epoch 的优化行为中蒸馏出元知识
- 注入到后续 step 的 analyst/merge/ranking prompt 中
- 类比：**Optimizer State（如 Adam 的 momentum）**

**Step Buffer（上下文积累）**
- 记录当前 epoch 内每一步的失败模式和 rejected edits
- 传递给后续 analyst 作为 "## Previous Steps in This Epoch"
- 让 optimizer 知道哪些 edit 已经被试过但被 gate 拒绝了

### 2.3 关键设计：Dual Deployment

```
┌──────────────────────────────────────────────────────────────────┐
│  Optimizer Model (proposes patches)                              │
│  - 通常用更强的模型（如 GPT-5.5）                                 │
│  - 独立 endpoint / API key / auth                                │
├──────────────────────────────────────────────────────────────────┤
│  Target Model (rolls out agent)                                   │
│  - 可以是便宜模型（GPT-5.4-nano）或 exec backend（Codex/Claude）  │
│  - 独立 endpoint                                                │
└──────────────────────────────────────────────────────────────────┘
```

**为什么分离？**
1. **成本优化**：rollout 量大，用便宜模型；patch 分析量小，用强模型
2. **HarnessOpt**：target 可能是 exec backend（Codex CLI），optimizer 必须是 chat model
3. **灵活性**：可以用不同 provider 的模型

---

## 3. 核心数据流与数据结构

### 3.1 主要 Dataclass（`skillopt/types.py`）

```python
@dataclass
class Edit:
    op: str              # "append" | "insert_after" | "replace" | "delete"
    content: str         # markdown 内容
    target: str          # 针对 insert_after/replace/delete 的目标文本
    support_count: int   # 该 edit 被多少个 minibatch 支持
    source_type: str     # "failure" | "success"
    merge_level: int     # 经过了几层合并

@dataclass
class Patch:
    reasoning: str       # 为什么这些 edit 能解决批次问题
    edits: list[Edit]

@dataclass
class RolloutResult:
    id: str
    hard: float          # 0 或 1（是否完全正确）
    soft: float          # 0~1（partial credit / F1）
    n_turns: int         # 对话轮数
    fail_reason: str     # 失败原因分类
    task_type: str       # 任务类型
    extras: dict         # 附加信息

@dataclass
class RawPatch:
    patch: Patch         # 实际 patch 内容
    source_type: str     # "failure" | "success"
    batch_size: int      # 产生该 patch 的 minibatch 大小
    failure_summary: list[FailureSummaryEntry]
```

### 3.2 关键文件与职责映射

| 文件 | 核心职责 | 类比 |
|------|---------|------|
| `trainer.py` | 6-stage loop 编排 + 状态管理 | Training Loop / Epoch Manager |
| `reflect.py` | Minibatch trajectory 分析 | Mini-batch Gradient Computation |
| `aggregate.py` | 层级 patch 合并 | Gradient Aggregation / All-reduce |
| `gate.py` | accept/reject 决策 | Validation-based Early Stopping |
| `slow_update.py` | 纵向对比 + protected region | Long-term Memory / Replay Buffer |
| `meta_skill.py` | 优化器元知识蒸馏 | Optimizer State (momentum) |
| `clip.py` | Edit 排序 / Top-L 选择 | Gradient Clipping |
| `scheduler.py` | Edit budget 调度 | Learning Rate Scheduler |
| `skill.py` | Edit 应用 + 保护区域 | Weight Update / Parameter Assignment |
| `lr_autonomous.py` | 自主决定 edit 预算 | Adaptive Learning Rate |

---

## 4. 面试重点：概念类比与迁移

> 面试官最可能问的问题类型：**"你这个项目里的 XXX 和经典 ML 里的 YYY 有什么区别/联系？"**

### 4.1 SkillOpt 与 Fine-tuning 的本质区别

| 维度 | Fine-tuning | SkillOpt-Lite |
|------|-------------|---------------|
| 优化对象 | 模型权重（连续空间） | 自然语言文本（离散空间） |
| 优化方法 | 梯度下降 + backprop | LLM 作为 "optimizer" 生成 edit |
| 可解释性 | 黑盒（权重难以人工解读） | 白盒（skill.md 是人类可读的） |
| 部署成本 | 需要托管 finetuned 模型 | 仅需一份 skill.md，零推理成本变化 |
| 迭代速度 | 需要重新训练/量化 | 直接编辑文本，秒级生效 |
| 适用于 | 需要深度改变模型行为 | 需要改变 Agent 策略/流程 |

**一句话总结**：SkillOpt-Lite 把 "训练" 从**权重空间**搬到了**文本空间**，optimizer 从一个计算图变成了一个 LLM。

### 4.2 LLM 作为 Optimizer

传统优化器（SGD/Adam）：
- 输入：梯度（连续向量）
- 输出：权重更新

SkillOpt-Lite 的 LLM Optimizer：
- 输入：trajectory 文本 + 当前 skill + 历史 context
- 输出：**自然语言 edit 操作**（离散文本）
- 需要 LLM 自己 "理解" 失败模式并 "设计" 修复方案

**面试深挖点**：
- 为什么不用 LLM 直接输出新的 skill？（vs 输出 edit）→ **Edit 更可控、可合并、可追溯**
- LLM optimizer 的 "梯度" 是什么？→ **Trajectory 分析结果 + failure summary**
- 如何保证 LLM optimizer 的输出质量？→ **Minibatch 聚合 + gate 验证 + 回滚**

### 4.3 Validation Gate 与 Early Stopping 的类比

```
传统 NN 训练：
  train_loss ↓ → val_loss ↓ → keep weights
  train_loss ↓ → val_loss ↑ → rollback (early stopping)

SkillOpt-Lite：
  rollout score ↑ → gate accept → keep new skill
  rollout score ↓ → gate reject → rollback to previous skill
```

**差异**：
- NN 的 early stopping 是基于**连续指标**（loss/accuracy），SkillOpt-Lite 的 gate 也是连续的
- 但 NN 的权重更新是**确定性的**（梯度下降），SkillOpt-Lite 的 edit 是**非确定性的**（LLM 生成），所以 gate 更关键

### 4.4 Minibatch Reflection 与 Minibatch SGD

| | Minibatch SGD | Minibatch Reflection |
|--|---------------|---------------------|
| 输入 | M 个样本的梯度 | M 条 trajectory 文本 |
| 计算 | 平均梯度 → 权重更新 | LLM 分析共性模式 → 生成 edit |
| 好处 | 梯度噪声更小 + 并行 | 发现跨样本模式 + 成本更低 |
| 超参 | batch size M | `gradient.minibatch_size`（默认 8） |

**面试可能问**：M 设太大/太小会怎样？
- 太小：噪声大，可能学到的 pattern 太局部
- 太大：一次分析轨迹太多，LLM 可能遗漏细节，成本也高

### 4.5 Protected Region 与 "Fixed Parameters"

- NN 中：某些 layer 被冻结（`requires_grad=False`）
- SkillOpt-Lite 中：`<!-- SLOW_UPDATE_START -->` 区域被 step-level analyst 跳过
- **类比**：Slow Update 是 "meta-optimizer"，只更新战略层；step-level 是 "base-optimizer"，只更新战术层

### 4.6 Step Buffer 与 Experience Replay

- Step Buffer 记录当前 epoch 内观察到的失败模式和被拒绝的 edit
- 类似于 RL 中的 **Experience Replay Buffer**
- 区别：Step Buffer 只在 epoch 内有效，而且目的是给 LLM 提供上下文，不是用于训练数据采样

---

## 5. 深度挖掘题（按面试轮次/深度分层）

### 5.1 第一层：项目理解（基础轮）

> 目标：确认你真的做过这个项目，不是纸上谈兵

1. **"请用 3 分钟介绍 SkillOpt-Lite 的核心思想和 Pipeline"**
   - 评分点：能否区分 skill 优化 vs 模型微调；能否讲清 6-stage loop

2. **"为什么选择优化 skill 文本而不是微调模型权重？"**
   - 评分点：部署成本、可解释性、迭代速度、适用场景

3. **"什么是 validation gate？它解决什么问题？"**
   - 评分点：防止过拟合到训练 batch、自动回滚、类比 early stopping

4. **"项目中有哪两种模式？有什么区别？"**
   - 评分点：`/skillopt-loop` 只改 skill.md；`/harnessopt-loop` 也改 Python 代码

### 5.2 第二层：技术细节（算法轮）

> 目标：考察你对系统设计和技术细节的理解深度

5. **"Reflect 阶段为什么要用 minibatch 而不是逐条分析？具体怎么实现的？"**
   - 深入点：
     - 查看 `run_minibatch_reflect()` 的实现
     - `minibatch_size` 默认是 8，怎么分 batch 的？
     - 失败和成功轨迹是分开处理还是混合？
     - `ThreadPoolExecutor` 的并行度怎么控制？

6. **"Aggregate 阶段的层级合并具体是怎么做的？为什么要有三层？"**
   - 深入点：
     - 第一层：同批次的 minibatch patches 并行合并
     - 第二层：failure patches 和 success patches 分别合并
     - 第三层：两者最终合并，failure 优先
     - `support_count` 如何在合并中传播？

7. **"Slow Update 和 Meta Skill 的区别和联系是什么？"**
   - 深入点：
     - Slow Update：更新 skill 的 protected 区域（**内容层面**的纵向反思）
     - Meta Skill：更新 optimizer 的 prompt/行为（**过程层面**的元认知）
     - 都在 epoch 边界触发，但作用对象不同

8. **"Dual Deployment 架构是怎么设计的？为什么需要分离 optimizer 和 target？"**
   - 深入点：
     - `backend_config.py` 中的 `optimizer_backend` / `target_backend`
     - 不同的 endpoint、API key、auth mode
     - cost 优化 + harness exec backend 的支持

9. **"Protected Region（SLOW_UPDATE）的机制具体是怎么实现的？"**
   - 深入点：
     - HTML comment markers：`<!-- SLOW_UPDATE_START -->`...`<!-- SLOW_UPDATE_END -->`
     - `apply_patch()` 中怎么跳过这些区域？
     - `replace_slow_update_field()` 怎么安全替换？
     - 如果 markers 不完整（只有 start 没有 end）怎么处理？

10. **"Step Buffer 具体记录什么？怎么传递给后续步骤？"**
    - 深入点：
      - 记录内容：failure patterns + rejected edits + score drop
      - 序列化到 `step_buffer.json`
      - 注入到 analyst/merge/ranking prompt 的 "## Previous Steps" 部分

### 5.3 第三层：系统设计（系统轮）

> 目标：考察你的系统设计能力和工程品味

11. **"整个训练循环是怎么做到可恢复的（resume）的？"**
    - 深入点：
      - `runtime_state.json` 记录最后完成的 step
      - `history.json` 记录完整历史
      - 每个 step 的产物保存在 `step_{NNNN}/` 目录
      - minibatch 的 `.json` 文件持久化，resume 时跳过已完成的

12. **"Selection Cache 是怎么工作的？避免什么问题？"**
    - 深入点：
      - key = `skill_hash(skill_content)`
      - value = `(hard, soft)` 分数
      - 避免对相同的 skill 内容重复跑 selection set（节省成本）

13. **"Config 的 `_base_` 继承系统是怎么实现的？"**
    - 深入点：
      - 查看 `config.py` 中的 `_load_config_with_inheritance()`
      - 递归加载 `_base_` 链
      - deep merge（不是 shallow merge）
      - 与 MMEngine/MMDetection 的异同

14. **"Coding-Agent-Driven Loop 是怎么设计的？和传统脚本驱动有什么优劣？"**
    - 深入点：
      - `.github/prompts/*.prompt.md` 是 slash-command 定义
      - Coding agent 负责所有工具调用（pip install、运行脚本、编辑文件）
      - 优势：用户零配置，agent 自动处理环境
      - 劣势：依赖特定 coding agent 生态，调试困难

### 5.4 第四层：扩展与思考（资深轮）

> 目标：考察你的研究品味和创新能力

15. **"如果把 minibatch reflection 换成 chain-of-thought reflection 会怎样？"**
    - 思考：当前的 analyst prompt 已经要求分类 failure type，但有没有让 LLM 做更深的因果推理？
    - 可能的改进：要求 LLM 输出 "如果修改了 skill 中的 X，预期 Y 会变化"

16. **"如何评价当前 gate 指标的合理性？有没有可能 gate 本身被 LLM 的 scoring bias 影响？"**
    - 思考：
      - hard metric 依赖 exact match，对 partial credit 的任务不友好
      - soft metric 依赖 LLM judge 或 F1，可能有 bias
      - 如果 target model 和 gate 的 scorer 是同一个模型，可能存在自我强化

17. **"Slow Update 的 protected region 是静态的（只有 skill 的一个 section），有没有可能做成动态的？"**
    - 思考：
      - 当前是固定的 `<!-- SLOW_UPDATE_START -->...<!-- SLOW_UPDATE_END -->`
      - 如果根据当前 skill 的章节结构动态决定哪些区域受保护？
      - 如果允许 Slow Update 修改 step-level 之前的内容（有版本控制的前提下）？

18. **"如果让你把这个系统从 skill 优化扩展到 model 微调，你会怎么做？"**
    - 思考：LoRA / prefix tuning 作为中间层，skill 作为 soft prompt，gate 用来决定是否保留微调

---

## 6. 实现细节暗礁（容易答错的点）

> 这些点如果你没有仔细读过代码，很容易在面试中露怯

### 6.1 `source_type` 的传播

- `RawPatch.source_type` 标记是 "failure" 还是 "success"
- 在 `_normalise_patches()` 中，`item.source_type` 默认继承自 patch 的 `source_type`
- 但如果是 `success` 类型的 patch，它的 edit 仍然会被应用（只是优先级低于 failure）

**常见误解**：认为 success patch 的 edit 会被直接丢弃。实际上它们会被合并，但在最终 merge 中 failure 优先。

### 6.2 `support_count` 的计算

```python
support = max(int(p.get("batch_size", 0) or 0), 1)
```

- `support_count` 不是该 edit 被多少个 minibatch 选中，而是**产生该 patch 的 minibatch 的大小**
- 在 `aggregate.py` 中，合并时取 `max(support_a, support_b)`，不是 `sum`
- 这意味着一个大 batch（如 16 条轨迹）产生的 patch 比小 batch（如 2 条）权重更高

### 6.3 `normalize_patches` 中的 `inner = p.get("patch", p)`

- `RawPatch` 的结构是 `{"patch": {...}, "source_type": ..., "batch_size": ...}`
- 但有时 LLM 直接返回了 patch dict 而没有外层包装
- 代码兼容了两种情况：`inner = p.get("patch", p)`

### 6.4 `accumulation > 1` 的含义

- `train.accumulation` 不是 gradient accumulation（因为不是真梯度）
- 它表示：**每个 step 执行多少个 rollout batch 来累积足够的轨迹**
- 比如 `batch_size=40, accumulation=2` → 每个 step 跑 80 条 trajectory

### 6.5 `lr_scheduler` 的三种模式

```python
# scheduler.py
class ConstantScheduler:    # 每步固定 L 个 edit
class LinearScheduler:      # 从 max_lr 线性衰减到 min_lr
class CosineScheduler:      # cosine annealing
class AutonomousScheduler:  # 不限（999），由 LLM 决定
```

- 注意 `CosineScheduler` 是**基于 step 数**的余弦退火，不是 epoch
- `AutonomousScheduler` 返回 999，实际限制由 `decide_autonomous_learning_rate()` 控制

### 6.6 Gate 的 `metric` 参数

- `hard`：只看 exact match，对 partial credit 任务可能太严格
- `soft`：只看 F1/partial credit，对必须完全正确的任务可能太宽松
- `mixed`：加权平均，`mixed_weight` 控制 soft 的权重
- **面试高频**：为什么 gate 不能只看 soft？→ 因为 soft metric 可能被 LLM judge 的 bias 影响，hard metric 提供硬约束

### 6.7 `failure_only` 配置

- `gradient.failure_only: true` 时，跳过 success analyst
- 只分析失败轨迹，不生成 success patch
- 适用于：失败样本足够多，成功样本的 pattern 对 skill 改进帮助不大

### 6.8 `longitudinal_pair_policy`

- `mixed`：分析所有对比对（improved + regressed + persistent_fail + stable_success）
- `changed`：只看结果发生变化的对（improved + regressed）
- `unchanged`：只看结果没变化的对（persistent_fail + stable_success）
- **面试点**：不同 policy 对 slow update 内容的影响

---

## 7. 设计取舍讨论（展示工程品味）

> 这些问题没有标准答案，展示你的思考过程更重要

### 7.1 为什么用 Markdown 作为 skill 格式？

- **优点**：人类可读、可 diff、可 patch、结构化（headings、lists）
- **缺点**：LLM 生成时可能破坏格式、markdown 解析有歧义
- **替代方案**：JSON/YAML 结构、DSL、Python 代码
- **SkillOpt-Lite 的取舍**：Markdown 是因为 Agent 的 prompt/instruction 天然是文本，而且人类要审核

### 7.2 为什么不用强化学习（RL）？

- RL 需要 reward signal，但 Agent 任务的 reward 通常是二元的（成功/失败）
- 稀疏 reward 问题：大部分轨迹是失败的，RL 难以学习
- SkillOpt-Lite 用**验证门控**替代了 reward，用**LLM 分析**替代了 policy gradient

### 7.3 为什么需要两个 analyst（error + success）？

- 只看失败：可能过度修复，引入不必要的限制
- 只看成功：可能学不到什么（成功轨迹的信息量通常低于失败）
- 两者结合：success analyst 可以发现"哪些做法是对的，应该保留"，failure analyst 发现"哪些做法是错的，应该修改"

### 7.4 为什么 Slow Update 是 force-applied 而不是 gated？

- Slow Update 的内容是**战略性指导**（如 "在处理表格时，先确认列名而不是假设 A1 开始"）
- 这种指导不应该被单个 step 的短期表现否决
- 类比：公司的战略调整不应被短期业绩波动否决

### 7.5 为什么不用向量数据库做长期记忆？

- 当前项目用**文本块**（protected region + meta skill context）做记忆
- 向量数据库适合**检索式记忆**，但这里需要的是**可被 LLM 直接读取并遵循的指导**
- 文本块可以直接注入 prompt，而向量检索的结果需要额外的 "synthesis" 步骤

---

## 8. 可能的改进方向（加分项）

> 如果你在面试中被问到 "这个项目有什么可以改进的"，这些方向值得讨论

### 8.1 多轮次 Edit 冲突检测

- 当前 `apply_patch` 是顺序执行的，后一个 edit 可能覆盖前一个
- 改进：在 Select 阶段检测 edit 之间的冲突（如两个 `replace` 操作 target 同一段文本）

### 8.2 动态 Minibatch 大小

- 当前 `minibatch_size` 是固定的
- 改进：根据轨迹的多样性动态调整 — 相似轨迹用更小的 batch，多样轨迹用更大的 batch

### 8.3 Self-Play / 对抗性验证

- 当前 gate 是用 selection set 评估
- 改进：引入 "adversarial eval" — 用不同模型或不同 prompt 生成的 harder 样本做验证

### 8.4 Edit 的 Causal 验证

- 当前 gate 只看整体 score 变化
- 改进：对于每个 edit，单独应用并验证其因果效果（类似 ablation study）

### 8.5 Skill 的结构化版本控制

- 当前 skill 是一份 flat markdown
- 改进：按章节/功能模块拆分，每个模块独立版本化，支持 cherry-pick edit

### 8.6 跨任务迁移

- 当前每个 env 独立训练
- 改进：在一个 env 上学到的 skill edit pattern 迁移到类似 env（如 SpreadsheetBench 的 edit pattern 迁移到其他表格任务）

---

## 9. 推荐自我介绍话术

> 面试时用来介绍你做过的相关项目

### 9.1 30 秒电梯演讲

> "我做过一个叫 SkillOpt-Lite 的项目，核心思想是**不微调模型，而是通过迭代编辑一份 Markdown skill 文档来优化 Agent 行为**。整个流程是：先用当前 skill 跑一批 trajectory，然后用一个更强的 LLM 分析失败原因并生成 edit，经过层级合并和排序后，用验证门控决定是否保留。我们还做了两个 epoch 级的机制：Slow Update 做纵向反思，Meta Skill 做优化器的元记忆。项目在 6 个 benchmark 上跑通了，其中 LiveMath 上比 SkillOpt  baseline 高了 8-25 个点。"

### 9.2 1 分钟深度介绍

> "这个项目最核心的挑战是：**如何让一个 LLM 作为 optimizer，稳定地改进另一个 LLM 的行为**。我们用了几个关键技术：第一是 minibatch reflection，类比 minibatch SGD，把轨迹分组后让 LLM 批量分析，这样既降低成本又能发现跨样本的共性失败模式；第二是 validation gate，类比 early stopping，防止 LLM 的编辑过拟合到训练 batch；第三是 Slow Update，在 epoch 边界用纵向对比写战略性指导，保护区域机制确保战术层（step-level）的编辑不会干扰战略层。整个系统的 dual deployment 架构让 optimizer 和 target 可以用不同的模型，cost 优化很灵活。"

### 9.3 被问到 "你遇到的最大困难是什么" 时的回答

> "最大的困难是**LLM optimizer 输出的稳定性和可控性**。因为 LLM 生成的 edit 是离散的文本，不像梯度是连续的向量，所以很容易出现：生成的 edit 语法正确但语义无效、或者 edit 之间互相冲突、或者 skill 被改得越来越长越来越冗余。我们通过几个机制缓解：一是 minibatch 合并时用 support_count 投票，降低噪声；二是 validation gate 自动回滚不好的修改；三是 Slow Update 定期清理 skill 的结构。但如果要进一步提升，我觉得可以引入 edit 的 causal 验证 — 对每个 edit 单独做 ablation，确认它的确带来了改进。"

---

## 附录：关键代码路径速查

| 你想理解的机制 | 核心文件 | 关键函数/类 |
|--------------|---------|------------|
| 6-stage loop | `skillopt/engine/trainer.py` | `ReflACTTrainer.train()` |
| Minibatch reflection | `skillopt/gradient/reflect.py` | `run_minibatch_reflect()` |
| 层级合并 | `skillopt/gradient/aggregate.py` | `merge_patches()` |
| Validation gate | `skillopt/evaluation/gate.py` | `evaluate_gate()` |
| Edit 应用 | `skillopt/optimizer/skill.py` | `apply_patch()` |
| Edit 排序 | `skillopt/optimizer/clip.py` | `rank_and_select()` |
| Slow Update | `skillopt/optimizer/slow_update.py` | `run_slow_update()` |
| Meta Skill | `skillopt/optimizer/meta_skill.py` | `run_meta_skill()` |
| LR Scheduler | `skillopt/optimizer/scheduler.py` | `build_scheduler()` |
| Protected Region | `skillopt/optimizer/slow_update.py` | `replace_slow_update_field()` |
| Dual Deployment | `skillopt/model/backend_config.py` | `ModelConfig` |
| Config 继承 | `skillopt/config.py` | `_load_config_with_inheritance()` |
| 断点续跑 | `skillopt/engine/trainer.py` | `_save_runtime_state()`, `_resume()` |

---

## 附录：面试高频追问预判

| 你提到的关键词 | 面试官可能的追问 |
|--------------|----------------|
| "minibatch reflection" | M 选多少的依据？失败和成功轨迹分开分析还是混合？ |
| "validation gate" | selection set 和 test set 的区别？gate metric 选 hard/soft/mixed 的依据？ |
| "protected region" | 如果 Slow Update 的内容本身有问题怎么办？ |
| "dual deployment" | optimizer 和 target 用同一个模型可以吗？会有什么问题？ |
| "LLM as optimizer" | 和 prompt optimization（如 OPRO/APE）的区别？ |
| "skill.md" | 如果 skill 很长，LLM 的 context window 不够怎么办？ |
| "resume" | 如果训练中断在 apply_patch 之后、evaluate 之前，resume 会重复计算吗？ |
| "aggregate" | 如果 LLM 在 merge 时输出格式错误，fallback 机制是什么？ |
| "step buffer" | step buffer 信息太多会污染 context 吗？怎么过滤？ |

---

*最后更新：基于 SkillOpt-Lite 项目代码与文档全面分析*  
*面试准备建议：先精读 `trainer.py` 的 `train()` 函数，然后按附录的代码路径速查逐个理解核心模块*
