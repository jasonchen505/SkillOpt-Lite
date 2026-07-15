# SkillOpt-Lite 学习路线图 & 掌握指南

> 基于你的 8× RTX 4090 算力，设计从零到精通的完整学习路径  
> 每阶段包含：学习目标、必读代码、实操任务、验收标准、预计时间

---

## 目录

1. [学习路径总览](#1-学习路径总览)
2. [Stage 0：项目骨架感知（Day 1-2）](#2-stage-0项目骨架感知day-1-2)
3. [Stage 1：理解 ReflACT Pipeline（Day 2-4）](#3-stage-1理解-reflact-pipelineday-2-4)
4. [Stage 2：掌握核心模块（Day 4-7）](#4-stage-2掌握核心模块day-4-7)
5. [Stage 3：理解 EnvAdapter 抽象（Day 7-9）](#5-stage-3理解-envadapter-抽象day-7-9)
6. [Stage 4：模型后端与部署（Day 9-11）](#6-stage-4模型后端与部署day-9-11)
7. [Stage 5：Prompt 工程与 LLM 交互（Day 11-13）](#7-stage-5prompt-工程与-llm-交互day-11-13)
8. [Stage 6：系统韧性与工程细节（Day 13-15）](#8-stage-6系统韧性与工程细节day-13-15)
9. [Stage 7：复现实战（Day 15-22）](#9-stage-7复现实战day-15-22)
10. [Stage 8：面试准备（Day 22-30）](#10-stage-8面试准备day-22-30)
11. [核心文件阅读优先级](#11-核心文件阅读优先级)
12. [常见困惑 & 解答](#12-常见困惑--解答)

---

## 1. 学习路径总览

```
┌─────────────────────────────────────────────────────────────────┐
│  理解层（What）                                                    │
│  ├── Stage 0: 项目骨架感知 — 目录结构、README、数据流              │
│  └── Stage 1: ReflACT Pipeline — 6-stage loop 是什么             │
├─────────────────────────────────────────────────────────────────┤
│  原理层（Why）                                                     │
│  ├── Stage 2: 核心模块 — gradient/optimizer/evaluation           │
│  ├── Stage 3: EnvAdapter — 环境抽象接口                           │
│  └── Stage 4: 模型后端 — dual deployment、vLLM 本地化             │
├─────────────────────────────────────────────────────────────────┤
│  细节层（How）                                                     │
│  ├── Stage 5: Prompt 工程 — LLM 交互的精确设计                    │
│  └── Stage 6: 系统韧性 — resume、gate、cache、retry               │
├─────────────────────────────────────────────────────────────────┤
│  实战层（Do）                                                     │
│  ├── Stage 7: 复现实战 — 从 smoke test 到完整 loop               │
│  └── Stage 8: 面试准备 — 深度追问应对                              │
└─────────────────────────────────────────────────────────────────┘
```

**每阶段产出**：一个可验证的 "我掌握了 XXX" 的 checkpoint。

---

## 2. Stage 0：项目骨架感知（Day 1-2）

### 2.1 学习目标

- 理解项目的目录结构
- 理解数据从下载到训练的完整流程
- 能跑通一个最简单的 eval（smoke test）

### 2.2 必读文件

| 文件 | 阅读目标 | 预计时间 |
|------|---------|---------|
| `README.md` | 项目定位、TL;DR、性能数字 | 15 min |
| `docs/quickstart.md` | 安装步骤、数据下载、smoke test | 10 min |
| `docs/data.md` | 数据来源、split 格式、下载方式 | 10 min |
| `docs/authentication.md` | 三种 auth mode 的区别 | 10 min |
| `copilot_example/README.md` | 6 个 benchmark 的概览 | 10 min |

### 2.3 实操任务

**任务 1：目录结构导航**
```bash
cd /data/home/yizhou/SkillOpt-Lite
ls -la
# 回答以下问题：
# 1. skillopt/ 和 scripts/ 的区别是什么？
# 2. copilot_example/ 和 harness_example/ 的区别是什么？
# 3. skillopt_lite_ckpt/ 和 skillopt_ckpt/ 的区别是什么？
# 4. configs/ 目录的结构是什么样的？_base_/ 的作用是什么？
```

**任务 2：数据下载**
```bash
python data/download.py --config livemath
# 确认 data/ablation_splits/livemathematicianbench/2-2-6_seed42/ 下有 items.json
wc -l data/ablation_splits/livemathematicianbench/2-2-6_seed42/*/items.json
```

**任务 3：Smoke Test**
```bash
# 需要先配置 auth（见 Phase 0）
python scripts/eval_only.py \
    --config configs/livemathematicianbench/default.yaml \
    --skill skillopt/envs/livemathematicianbench/skills/initial.md \
    --split train \
    --eval_limit 5 \
    --target_model gpt-5.4-nano \
    --workers 4 \
    --out_root outputs/stage0_smoke
```

### 2.4 验收标准

- [ ] 能画出项目的完整目录结构图
- [ ] 能解释每个顶层目录的作用
- [ ] Smoke test 产出 `results.jsonl`，包含 5 条结果
- [ ] 理解 `data/download.py` 的数据来源（HuggingFace）

### 2.5 常见困惑

**Q: 为什么有 `skillopt_lite_ckpt/` 和 `skillopt_ckpt/` 两个目录？**

A: `skillopt_ckpt/` 是**原始 SkillOpt（微软）论文**用的 GPT-5.5 skills。`skillopt_lite_ckpt/` 是 **SkillOpt-Lite（本项目）** 产出的、针对不同模型优化的 skills。目录名中的 `gpt5.5`、`gpt5.4-nano` 等是 target 模型名。

**Q: `copilot_example/` 里的 `run.sh` 和 `scripts/train.py` 有什么区别？**

A: `run.sh` 是**面向 coding agent 的 wrapper**，它调用 `scripts/eval_only.py` 做单次 eval，然后调用 `make_samples.py` 导出样本。`scripts/train.py` 是**完整的训练 loop**，包含 6-stage pipeline。Coding agent 的 slash-command（`/skillopt-loop`）是更高层的 orchestrator，它反复调用 `run.sh` 来实现迭代优化。

---

## 3. Stage 1：理解 ReflACT Pipeline（Day 2-4）

### 3.1 学习目标

- 理解 6-stage pipeline 的每个阶段做什么
- 理解整个 loop 的控制流（trainer.py 的 `train()` 方法）
- 能画出 pipeline 的数据流图

### 3.2 必读文件

| 文件 | 阅读目标 | 预计时间 |
|------|---------|---------|
| `skillopt/engine/trainer.py` (lines 1-100, 700-900, 1400-1500) | `ReflACTTrainer` 类、`train()` 主循环 | 60 min |
| `skillopt/types.py` | 所有 dataclass 的定义 | 20 min |
| `skillopt/gradient/reflect.py` (lines 1-100) | Reflect 引擎的入口 | 30 min |
| `skillopt/gradient/aggregate.py` (lines 1-100) | Aggregate 引擎的入口 | 30 min |
| `skillopt/evaluation/gate.py` | Gate 决策逻辑 | 20 min |

### 3.3 核心问题清单

阅读 `trainer.py` 时，回答以下问题：

1. **train() 方法的整体结构是什么？**
   - 初始化阶段做了什么？
   - 主循环的骨架是什么？
   - 每个 epoch 的边界做了什么？

2. **Rollout 阶段的输入输出是什么？**
   - 输入：current_skill、items
   - 输出：list of RolloutResult
   - 并行怎么实现的？

3. **Reflect 阶段的输入输出是什么？**
   - 输入：rollout results、current_skill
   - 输出：list of RawPatch
   - minibatch 怎么分的？

4. **Aggregate 阶段的输入输出是什么？**
   - 输入：list of RawPatch
   - 输出：单个 merged Patch
   - 层级合并怎么实现的？

5. **Select 阶段的输入输出是什么？**
   - 输入：merged Patch、可选的历史上下文
   - 输出：ranked list of Edit
   - Top-L 怎么选的？

6. **Update 阶段的输入输出是什么？**
   - 输入：current_skill、ranked edits
   - 输出：candidate_skill
   - Protected region 怎么处理的？

7. **Evaluate 阶段的输入输出是什么？**
   - 输入：candidate_skill
   - 输出：GateResult（accept/reject）
   - Selection set 怎么选的？

### 3.4 实操任务

**任务 1：画出 Pipeline 数据流**

用 ASCII art 或工具画出以下数据流：
```
items → Rollout → RolloutResult[]
    → Reflect → RawPatch[] (fail + success)
    → Aggregate → Patch
    → Select → Edit[]
    → Update → candidate_skill
    → Evaluate → GateResult
```

并在图上标注：
- 每个阶段的 LLM 调用次数
- 每个阶段的并行方式
- 每个阶段的输出文件（如果有）

**任务 2：追踪一个 Edit 的完整生命周期**

从 `RawPatch` 中的 edit 开始，追踪它经过的所有代码路径：
1. `reflect.py` 中 edit 是怎么被创建的？
2. `aggregate.py` 中 edit 是怎么被合并的？`support_count` 怎么传播？
3. `clip.py` 中 edit 是怎么被排序的？
4. `skill.py` 中 edit 是怎么被应用的？
5. `gate.py` 中 candidate 是怎么被评估的？

### 3.5 验收标准

- [ ] 能手绘 6-stage pipeline 数据流图
- [ ] 能解释每个阶段的输入输出
- [ ] 能追踪一个 edit 从创建到应用的完整生命周期
- [ ] 理解 `RolloutResult`、`RawPatch`、`Patch`、`Edit` 的关系

---

## 4. Stage 2：掌握核心模块（Day 4-7）

### 4.1 学习目标

- 深入理解 Gradient（Reflect + Aggregate）的实现细节
- 深入理解 Optimizer（Select + Update + Slow Update + Meta Skill）的实现细节
- 深入理解 Evaluation（Gate）的实现细节

### 4.2 必读文件

| 文件 | 阅读目标 | 预计时间 |
|------|---------|---------|
| `skillopt/gradient/reflect.py` (全 588 行) | Minibatch reflection 的完整实现 | 90 min |
| `skillopt/gradient/aggregate.py` | 层级 patch 合并 | 60 min |
| `skillopt/optimizer/skill.py` | Edit 应用 + protected region | 45 min |
| `skillopt/optimizer/clip.py` | Edit 排序/选择 | 30 min |
| `skillopt/optimizer/slow_update.py` | Slow update 完整实现 | 60 min |
| `skillopt/optimizer/meta_skill.py` | Meta skill 蒸馏 | 45 min |
| `skillopt/optimizer/scheduler.py` | LR 调度器 | 30 min |
| `skillopt/optimizer/lr_autonomous.py` | 自主 LR 决策 | 30 min |

### 4.3 深度问题清单

**Reflect 模块**：
1. `run_minibatch_reflect()` 的完整流程是什么？
2. 失败轨迹和成功轨迹是分开处理还是混合？
3. `analyst_workers` 控制什么？实际并行度是多少？
4. `fmt_trajectory()` 怎么处理不同格式的轨迹？
5. 如果某个 minibatch 的 LLM 调用失败，怎么 fallback？
6. `_MAX_TRAJ_CHARS = 12_000` 是怎么来的？超长了怎么办？

**Aggregate 模块**：
1. `merge_patches()` 的层级结构是什么？
2. 同层合并和跨层合并有什么区别？
3. `support_count` 在合并时怎么计算？
4. 如果 LLM merge 调用失败，fallback 是什么？
5. Failure-first 策略在代码中怎么体现？

**Select 模块**：
1. `rank_and_select()` 的三种模式（fixed/autonomous/none）怎么切换？
2. Ranking prompt 的评估标准是什么？
3. 如果 ranking LLM 输出格式错误，怎么 fallback？

**Update 模块**：
1. `apply_patch()` 的四种 edit 操作分别怎么实现？
2. Protected region 的检测逻辑是什么？
3. 如果 edit 的 target 是 protected region 内的文本，会怎么样？
4. `replace_slow_update_field()` 怎么安全替换？
5. 3 种 update mode（patch/rewrite/full_rewrite）的区别？

**Slow Update 模块**：
1. `build_comparison_pairs()` 怎么分类 improved/regressed/persistent_fail/stable_success？
2. `longitudinal_pair_policy` 的三种模式怎么影响 pair 选择？
3. Slow update 的 prompt 和 analyst prompt 有什么区别？
4. 为什么 slow update 是 force-applied 而不是 gated？
5. `inject_empty_slow_update_field()` 什么时候被调用？

**Meta Skill 模块**：
1. Meta skill 的内容和 slow update 的内容有什么区别？
2. Meta skill 怎么注入到后续步骤的 prompt 中？
3. Meta skill 的存储位置和生命周期？

**Gate 模块**：
1. `evaluate_gate()` 的三种 action 的精确条件？
2. `select_gate_score()` 的三种 metric 怎么计算？
3. Gate 的 `current_score` 和 `best_score` 什么时候更新？
4. `mixed_weight` 的合理范围？

### 4.4 实操任务

**任务 1：手动触发一个 Minibatch Reflect**

找到一段已保存的 minibatch patch 文件（`outputs/*/steps/step_0001/patches/minibatch_fail_*.json`），读懂它的内容：
- 输入是什么（trajectory 摘要）？
- 输出是什么（edits）？
- 这些 edits 合理吗？

**任务 2：模拟 Edit 应用**

```python
# 写一个脚本，模拟 apply_patch 的过程
from skillopt.optimizer.skill import apply_patch_with_report

skill = open("skillopt/envs/livemathematicianbench/skills/initial.md").read()
edits = [
    {"op": "append", "content": "\n## New Section\nSome content"},
    {"op": "insert_after", "target": "## Existing Section", "content": "\nMore content"},
]
new_skill, report = apply_patch_with_report(skill, edits)
print(report)
```

**任务 3：手动计算 Gate 决策**

```python
from skillopt.evaluation.gate import evaluate_gate

result = evaluate_gate(
    candidate_skill="...",
    cand_hard=0.45,
    current_skill="...",
    current_score=0.40,
    best_skill="...",
    best_score=0.50,
    best_step=3,
    global_step=4,
    metric="hard",
)
print(result.action, result.current_score, result.best_score)
```

### 4.5 验收标准

- [ ] 能手写 minibatch reflection 的伪代码
- [ ] 能手写层级合并的伪代码
- [ ] 能解释 protected region 的精确检测机制
- [ ] 能解释 slow update 和 meta skill 的区别和联系
- [ ] 能手动模拟一个 edit 的 apply 过程

---

## 5. Stage 3：理解 EnvAdapter 抽象（Day 7-9）

### 5.1 学习目标

- 理解 EnvAdapter 接口的设计
- 能读一个具体的 EnvAdapter 实现
- 知道添加新 benchmark 需要改哪些文件

### 5.2 必读文件

| 文件 | 阅读目标 | 预计时间 |
|------|---------|---------|
| `skillopt/envs/base.py` | EnvAdapter 抽象接口 | 30 min |
| `skillopt/envs/livemathematicianbench/adapter.py` | 最简单的 adapter | 45 min |
| `skillopt/envs/livemathematicianbench/rollout.py` | rollout 实现 | 30 min |
| `skillopt/envs/livemathematicianbench/reflect.py` | reflect 实现 | 20 min |
| `skillopt/envs/livemathematicianbench/dataloader.py` | dataloader 实现 | 20 min |
| `skillopt/envs/spreadsheetbench/adapter.py` | 最复杂的 adapter | 60 min |

### 5.3 EnvAdapter 接口清单

```python
class EnvAdapter(ABC):
    @abstractmethod
    def setup(self, cfg: dict) -> None: ...
    
    @abstractmethod
    def get_dataloader(self) -> BaseDataLoader: ...
    
    def build_train_env(self, batch_size: int, seed: int) -> EnvManager: ...
    def build_eval_env(self, env_num: int, split: str, seed: int) -> EnvManager: ...
    
    @abstractmethod
    def rollout(self, env_manager, skill_content: str, out_dir: str) -> list[dict]: ...
    
    @abstractmethod
    def reflect(self, results: list[dict], skill_content: str, out_dir: str) -> list[RawPatch]: ...
    
    @abstractmethod
    def get_task_types(self) -> list[str]: ...
```

**关键问题**：
1. `build_train_env` 和 `build_eval_env` 的区别？
2. `rollout()` 返回的 dict 必须包含哪些字段？
3. `reflect()` 可以直接用 `run_minibatch_reflect()` 吗？什么时候需要自定义？
4. `setup()` 什么时候被调用？一次还是多次？

### 5.4 LiveMath Adapter 深度阅读

重点理解：
- `rollout()` 怎么调用 target model？
- `parse_choice()` 怎么从模型输出中提取答案？
- `compute_score()` 怎么计算 hard/soft？
- `reflect()` 怎么委托给 `run_minibatch_reflect()`？

### 5.5 实操任务

**任务 1：手动实现一个 Minimal EnvAdapter**

创建一个只支持 3 条固定数据的 toy adapter：
```python
class ToyAdapter(EnvAdapter):
    def setup(self, cfg): pass
    
    def get_dataloader(self):
        return SplitDataLoader(split_dir="data/toy_split")
    
    def rollout(self, env_manager, skill_content, out_dir):
        # 直接用 target model 跑 3 条固定问题
        ...
    
    def reflect(self, results, skill_content, out_dir):
        return run_minibatch_reflect(results, skill_content, out_dir, ...)
    
    def get_task_types(self):
        return ["toy"]
```

**任务 2：对比两个 Adapter**

做一个表格，对比 LiveMath 和 SpreadsheetBench 的 adapter：

| 维度 | LiveMath | SpreadsheetBench |
|------|---------|-----------------|
| rollout 复杂度 | 低（单轮 QA） | 高（多轮 ReAct + 代码执行） |
| 需要的工具 | 无 | pandas、openpyxl |
| 评分方式 | exact match | case-level 检查 |
| reflect 定制 | 通用 | 可能定制 |
| exec_timeout | 300s | 600s |

### 5.6 验收标准

- [ ] 能手写一个最小 EnvAdapter
- [ ] 理解 rollout 的输入输出契约
- [ ] 理解 reflect 的默认实现（run_minibatch_reflect）
- [ ] 知道添加新 benchmark 需要创建哪些文件

---

## 6. Stage 4：模型后端与部署（Day 9-11）

### 6.1 学习目标

- 理解 dual deployment 架构
- 能在 8× 4090 上部署 vLLM
- 理解不同 backend 的适用场景

### 6.2 必读文件

| 文件 | 阅读目标 | 预计时间 |
|------|---------|---------|
| `skillopt/model/__init__.py` | 统一 model API | 20 min |
| `skillopt/model/common.py` | TokenTracker、normalize_backend | 30 min |
| `skillopt/model/backend_config.py` | Optimizer/target backend 配置 | 30 min |
| `skillopt/model/azure_openai.py` | Azure OpenAI wrapper | 45 min |
| `skillopt/model/qwen_backend.py` | Qwen/vLLM backend | 30 min |
| `skillopt/model/codex_harness.py` | Exec backend harness | 45 min |

### 6.3 深度问题清单

**后端架构**：
1. `chat_optimizer` 和 `chat_target` 的区别？
2. `set_optimizer_backend()` 和 `set_target_backend()` 什么时候被调用？
3. `backend_config.py` 中的环境变量和 config YAML 的优先级？

**Azure OpenAI Backend**：
1. 三种 auth mode（azure_cli / azure_key / openai）的实现差异？
2. Dual deployment 怎么配置不同的 endpoint？
3. Retry 逻辑是怎么实现的？
4. TokenTracker 怎么统计 token 用量？

**Qwen Backend**：
1. Qwen backend 只用于 target，为什么？
2. `qwen_chat_base_url` 和 `openai_base_url` 的区别？
3. 怎么在 config 中指定 Qwen 模型？

**Exec Backend**：
1. Codex exec 和 Claude Code exec 的区别？
2. `codex_harness.py` 怎么准备 workspace？
3. 怎么把 raw trace 解析成 RolloutResult？

### 6.4 实操任务

**任务 1：部署本地 vLLM**

```bash
# 在 8×4090 上启动双实例
vllm serve Qwen/Qwen2.5-14B-Instruct \
    --tensor-parallel-size 1 \
    --port 8000 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --enable-chunked-prefill \
    --dtype half &

vllm serve Qwen/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --port 8001 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 32768 \
    --enable-chunked-prefill \
    --dtype half &
```

**任务 2：配置本地 Auth**

```bash
export SKILLOPT_AUTH_MODE=openai
export OPENAI_API_KEY=EMPTY
export OPENAI_BASE_URL=http://localhost:8000/v1

# 验证
python -c "
from openai import OpenAI
client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY')
resp = client.chat.completions.create(
    model='Qwen/Qwen2.5-14B-Instruct',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=10,
)
print(resp.choices[0].message.content)
"
```

**任务 3：用本地模型跑 Smoke Test**

```bash
python scripts/eval_only.py \
    --config configs/livemathematicianbench/default.yaml \
    --skill skillopt/envs/livemathematicianbench/skills/initial.md \
    --split train \
    --eval_limit 5 \
    --target_model Qwen/Qwen2.5-7B-Instruct \
    --reasoning_effort medium \
    --workers 4 \
    --out_root outputs/stage4_local_vllm
```

### 6.5 验收标准

- [ ] 能成功启动 vLLM 双实例
- [ ] 能用本地模型跑通 smoke test
- [ ] 理解 dual deployment 的配置方式
- [ ] 知道 optimizer 和 target 可以用不同 backend

---

## 7. Stage 5：Prompt 工程与 LLM 交互（Day 11-13）

### 7.1 学习目标

- 理解所有 prompt 的设计意图
- 能根据 benchmark 特性定制 prompt
- 理解 prompt 在 pipeline 各阶段的作用

### 7.2 必读文件

| 文件 | 阅读目标 | 预计时间 |
|------|---------|---------|
| `skillopt/prompts/analyst_error.md` | Error analyst prompt | 15 min |
| `skillopt/prompts/analyst_success.md` | Success analyst prompt | 15 min |
| `skillopt/prompts/merge_failure.md` | Merge failure prompt | 15 min |
| `skillopt/prompts/merge_success.md` | Merge success prompt | 15 min |
| `skillopt/prompts/merge_final.md` | Final merge prompt | 15 min |
| `skillopt/prompts/ranking.md` | Ranking prompt | 15 min |
| `skillopt/prompts/slow_update.md` | Slow update prompt | 15 min |
| `skillopt/prompts/meta_skill.md` | Meta skill prompt | 15 min |
| `skillopt/prompts/rewrite_skill.md` | Rewrite prompt | 15 min |
| `skillopt/prompts/lr_autonomous.md` | LR autonomous prompt | 10 min |

### 7.3 Prompt 分析框架

对每个 prompt，回答以下问题：

| 问题 | analyst_error | merge_failure | slow_update |
|------|--------------|--------------|------------|
| **System role** 是什么？ | "expert failure-analysis agent" | "coordinator" | "strategic advisor" |
| **输入包含什么**？ | trajectories + skill + budget | patches + skill | comparison pairs + prev guidance |
| **输出格式**？ | JSON with failure_summary + edits | JSON with merged patch | Markdown guidance |
| **约束条件**？ | at most L edits, no protected region edits | no duplicate targets, preserve support_count | must be actionable, write to protected region |
| **失败 fallback**？ | return empty patch | concatenation | return empty guidance |

### 7.4 Prompt 设计的深层模式

**模式 1：Priority-based（失败优先）**
```markdown
# merge_final.md
FAILURE PATCHES TAKE PRIORITY: the primary goal of skill reflection
is to fix failures. When merging, if a success-driven edit conflicts
with a failure-driven edit, keep the failure-driven one.
```

**模式 2：Protected Region Awareness**
```markdown
# analyst_error.md
IMPORTANT: The skill document may contain a section between
<!-- SLOW_UPDATE_START --> and <!-- SLOW_UPDATE_END --> markers.
This is a PROTECTED section. Do NOT propose any edits that target
content within these markers.
```

**模式 3：Generality Constraint**
```markdown
# analyst_error.md
Edits must be generalizable; do not hardcode task-specific values.
Only patch gaps in the skill — do not duplicate existing content.
```

### 7.5 实操任务

**任务 1：Prompt 对比分析**

做一个表格，对比 5 个 analyst/merge/ranking prompt 的关键差异：
- 输入上下文
- 输出格式
- 约束条件
- 失败处理

**任务 2：Prompt 工程实验**

修改 `analyst_error.md`，在 system prompt 中增加 "think step by step" 指令，然后跑一个 minibatch reflect，观察：
- JSON 解析成功率是否提高？
- Edit 质量是否提高？

**任务 3：设计一个 Custom Prompt**

为 LiveMath 设计一个 custom analyst prompt，利用数学任务的特性（如 "注意检查符号错误"、"注意验证单位一致性"）。

### 7.6 验收标准

- [ ] 能解释每个 prompt 的设计意图
- [ ] 能对比不同 prompt 的输入输出差异
- [ ] 知道如何为特定 benchmark 定制 prompt
- [ ] 做过至少一次 prompt 微调实验

---

## 8. Stage 6：系统韧性与工程细节（Day 13-15）

### 8.1 学习目标

- 理解 resume 机制的完整实现
- 理解 retry、timeout、fallback 的处理
- 理解 HITL 机制

### 8.2 必读文件

| 文件 | 阅读目标 | 预计时间 |
|------|---------|---------|
| `skillopt/engine/trainer.py` (lines 250-500) | Resume 逻辑 | 60 min |
| `skillopt/model/azure_openai.py` | Retry + timeout | 45 min |
| `skillopt/engine/hitl.py` | Human-in-the-loop | 30 min |
| `skillopt/utils/scoring.py` | Score 计算 | 20 min |
| `skillopt/utils/json_utils.py` | JSON 提取 | 20 min |

### 8.3 深度问题清单

**Resume 机制**：
1. `_resume()` 怎么恢复 trainer 状态？
2. `runtime_state.json` 和 `history.json` 的关系？
3. Selection cache 怎么从 history 重建？
4. 如果中断发生在 `apply_patch` 之后、`evaluate` 之前，resume 会发生什么？

**Retry 机制**：
1. Azure OpenAI 的 retry 策略是什么？（几次？backoff 策略？）
2. 哪些阶段的 LLM 调用有不同的 retry 次数？
3. Exponential backoff 的最大 sleep 时间？
4. 如果所有 retry 都失败，怎么 fallback？

**HITL 机制**：
1. HITL 的 IPC 机制是什么？（文件系统）
2. 五种决策（ACCEPT/REJECT/EDIT/SELECT/SKIP）分别做什么？
3. HITL 什么时候被触发？

### 8.4 实操任务

**任务 1：模拟 Resume**

```bash
# 1. 开始一个训练，让它跑几步
python scripts/train.py --config ... --num_epochs 4 ...

# 2. 在训练过程中 kill 掉进程
# 3. 检查 runtime_state.json 的内容
cat outputs/*/runtime_state.json | python -m json.tool

# 4. 重新运行相同的命令，观察是否从断点恢复
python scripts/train.py --config ... --num_epochs 4 ...
```

**任务 2：模拟 API 失败**

```bash
# 1. 启动 vLLM 后，临时杀掉服务
# 2. 跑 eval，观察 retry 行为
# 3. 检查 step_record.json 中的错误信息
```

### 8.5 验收标准

- [ ] 能解释 resume 的完整流程
- [ ] 能解释 retry 和 fallback 机制
- [ ] 理解 HITL 的 5 种决策
- [ ] 做过 resume 和 API 失败的模拟实验

---

## 9. Stage 7：复现实战（Day 15-22）

### 9.1 学习目标

- 在 8× 4090 上完整跑通至少 1 个 benchmark
- 产出一个有正向提升的 skill
- 理解复现过程中的实际挑战

### 9.2 实战步骤

按照 `复现计划_8x4090_全流程.md` 执行：

1. **Day 15-16**: Phase 0 + Phase 1（环境搭建 + smoke test）
2. **Day 17-19**: Phase 2（LiveMath 完整 loop）
3. **Day 20-22**: Phase 3（SpreadsheetBench 完整 loop）

### 9.3 每日日志模板

每天记录：
```markdown
## Day X 日志

### 今日目标
- [ ] ...

### 实际完成
- ...

### 遇到的问题
1. 问题描述
   - 原因分析
   - 解决方案
   - 是否解决

### 明日计划
- ...
```

### 9.4 关键检查点

| 检查点 | 时间 | 验收标准 |
|--------|------|---------|
| vLLM 双实例运行 | Day 15 | curl 验证通过 |
| Smoke test 跑通 | Day 16 | results.jsonl 有内容 |
| LiveMath 1 epoch 完成 | Day 17 | history.json 有 1+ 条记录 |
| LiveMath 完整 loop | Day 19 | score 上升，accept 比例合理 |
| SpreadsheetBench smoke | Day 20 | 5 条 eval 通过 |

---

## 10. Stage 8：面试准备（Day 22-30）

### 10.1 学习目标

- 能回答所有深度挖掘题（见第一份 md）
- 能应对五类能力面试（见第二份 md）
- 有完整的项目介绍话术

### 10.2 必做任务

1. **复述练习**：对着录音设备，完整介绍项目（3 分钟）
2. **代码深挖**：每个核心模块，能手写关键代码片段
3. **追问应对**：对每道追问题，写出自己的回答
4. **模拟面试**：找同学或自己模拟面试，录音回听

### 10.3 面试 Checklist

| 类别 | 面试官可能问 | 你能回答 |
|------|------------|---------|
| 底层原理 | 为什么用 skill 优化而不是 finetune？ | ✅ |
| 底层原理 | Minibatch reflection 的动机？ | ✅ |
| 底层原理 | Gate 和 early stopping 的区别？ | ✅ |
| 底层原理 | Protected region 的实现？ | ✅ |
| 实验验证 | 怎么证明有效？ | ✅ |
| 实验验证 | 25pts 的来源？ | ✅ |
| 问题定位 | Score 暴跌怎么排查？ | ✅ |
| 工程落地 | Resume 机制？ | ✅ |
| 业务场景 | 适合什么场景？ | ✅ |

---

## 11. 核心文件阅读优先级

### 11.1 第一优先级（必读，精读）

| 文件 | 行数 | 作用 | 阅读阶段 |
|------|------|------|---------|
| `skillopt/engine/trainer.py` | 1953 | 核心 loop | Stage 1-2 |
| `skillopt/gradient/reflect.py` | 588 | Minibatch reflection | Stage 2 |
| `skillopt/optimizer/slow_update.py` | 393 | Slow update | Stage 2 |
| `skillopt/evaluation/gate.py` | 148 | Gate 决策 | Stage 2 |
| `skillopt/envs/base.py` | ~200 | EnvAdapter 接口 | Stage 3 |

### 11.2 第二优先级（选读，理解架构）

| 文件 | 作用 | 阅读阶段 |
|------|------|---------|
| `skillopt/optimizer/aggregate.py` | 层级合并 | Stage 2 |
| `skillopt/optimizer/skill.py` | Edit 应用 | Stage 2 |
| `skillopt/optimizer/clip.py` | Edit 排序 | Stage 2 |
| `skillopt/model/azure_openai.py` | Azure 后端 | Stage 4 |
| `skillopt/model/codex_harness.py` | Exec backend | Stage 4 |
| `skillopt/config.py` | Config 加载 | Stage 0 |

### 11.3 第三优先级（参考，按需）

| 文件 | 作用 | 阅读阶段 |
|------|------|---------|
| `skillopt/optimizer/meta_skill.py` | Meta skill | Stage 2 |
| `skillopt/optimizer/scheduler.py` | LR 调度 | Stage 2 |
| `skillopt/optimizer/lr_autonomous.py` | 自主 LR | Stage 2 |
| `skillopt/optimizer/rewrite.py` | Full rewrite | Stage 2 |
| `skillopt/engine/hitl.py` | HITL | Stage 6 |
| `skillopt/datasets/base.py` | Data loading | Stage 3 |

---

## 12. 常见困惑 & 解答

### Q1: trainer.py 有 1953 行，怎么看？

**A**: 不需要逐行看。按以下顺序精读关键段落：
1. Lines 1-100：Imports + 辅助函数
2. Lines 100-250：Longitudinal pair 相关函数
3. Lines 250-500：History/persistence 辅助函数
4. Lines 500-700：`__init__`
5. Lines 700-1000：`train()` 的前半部分（初始化 + epoch 循环）
6. Lines 1000-1400：`_run_step()` — **这是核心**
7. Lines 1400-1600：Slow update + meta skill 阶段
8. Lines 1600-1953：Final eval + summary

**最关键的函数**：`_run_step()`（lines 1000-1400），包含了 6-stage pipeline 的完整编排。

### Q2: 那么多 optimizer 文件，它们的关系是什么？

**A**: 一张图说清：

```
trainer.py 调用:
  ├── reflect.py → RawPatch[]
  │     └── model.chat_optimizer()  [LLM call]
  ├── aggregate.py → Patch
  │     └── model.chat_optimizer()  [LLM call]
  ├── clip.py → Edit[]
  │     ├── model.chat_optimizer()  [LLM call, ranking]
  │     └── lr_autonomous.py → decide budget
  │           └── model.chat_optimizer()  [LLM call]
  ├── skill.py → candidate_skill
  │     └── apply_patch() / rewrite_skill()
  ├── gate.py → GateResult
  ├── slow_update.py → protected region content
  │     └── model.chat_optimizer()  [LLM call]
  └── meta_skill.py → optimizer memory
        └── model.chat_optimizer()  [LLM call]
```

### Q3: LLM 调用都在哪里？有几个不同的 LLM 调用点？

**A**: 按阶段统计（每 step）：

| 阶段 | LLM 调用 | 调用者 | 模型 |
|------|---------|--------|------|
| Rollout | batch_size 次 | target backend | target model |
| Reflect | ~2 × ceil(N/M) 次 | chat_optimizer | optimizer model |
| Aggregate | ~2 × ceil(N/merge_batch) 次 | chat_optimizer | optimizer model |
| Select | 1-2 次 | chat_optimizer | optimizer model |
| Slow Update | 2 × slow_update_samples | chat_optimizer | optimizer model |
| Meta Skill | 1 次 | chat_optimizer | optimizer model |

**总计 per step**: rollout 40 + reflect ~10 + aggregate ~5 + select 1 = **~56 次 LLM 调用**（batch_size=40, M=8）

### Q4: 8× 4090 够跑吗？会不会很慢？

**A**: 够跑，但比 cloud API 慢。估算：
- Cloud API (GPT-5.5): 每 step ~30-90s
- Local vLLM (14B + 7B): 每 step ~90-300s

主要瓶颈是 vLLM 的 token 生成速度。可以通过以下方式加速：
1. 增大 `tensor-parallel-size`（用更多 GPU per 模型）
2. 用 4-bit 量化
3. 减小 `max_tokens`（rollout 阶段）
4. 增大 `max-model-len` 的 chunked prefill

### Q5: 如果 Qwen2.5-14B 作为 optimizer 质量不够怎么办？

**A**: 几个降级/升级策略：
1. **降级目标**：把 target 也换成 14B，增大 batch_size，靠数据量弥补
2. **Hybrid 方案**：optimizer 用 cloud API（GPT-5.4-mini），target 用本地
3. **Prompt 补偿**：在 analyst prompt 中加强 "think step by by"、"be thorough" 等指令
4. **换模型**：试试 DeepSeek-V2.5、GLM-4-9B 等其他开源模型

### Q6: 代码中有很多 `try/except ImportError`，是什么意思？

**A**: 这是**可选依赖**的懒加载模式。例如：
```python
try:
    from skillopt.envs.alfworld.adapter import ALFWorldAdapter
    _ENV_REGISTRY["alfworld"] = ALFWorldAdapter
except ImportError:
    pass  # ALFWorld deps not installed — skip
```

如果用户没有安装 `alfworld` 包，这个 adapter 就不会被注册。这样 `pip install -r requirements.txt` 后，即使没有安装所有可选依赖，核心功能仍然可用。

### Q7: 为什么 skill 优化比 prompt engineering 更好？

**A**: 几个关键差异：
1. **结构化 vs Flat**：Skill 是结构化文档（有章节、列表），prompt 通常是 flat 文本
2. **持久化 vs 一次性**：Skill 是文件，可以版本化、对比、回滚；prompt 通常在代码中硬编码
3. **迭代优化 vs 人工调优**：SkillOpt-Lite 是自动化的闭环迭代；prompt engineering 通常是人工试错
4. **Agent 级 vs Query 级**：Skill 控制 Agent 的完整行为；prompt 控制单次查询

---

## 附录：学习节奏建议

| 类型 | 推荐节奏 |
|------|---------|
| 精读代码 | 每天 1-2 小时，不要连续超过 3 小时 |
| 实操实验 | 每天 1-2 小时，最好有实际输出（日志、表格、截图） |
| 写笔记 | 每读完一个模块，写 3-5 行总结 |
| 复现 | 集中时间（如周末）跑长任务，平时调试 |
| 面试准备 | 每天 30 分钟复述，每周 1 次模拟面试 |

---

*最后更新：基于 8× RTX 4090 算力的项目学习路线图*
