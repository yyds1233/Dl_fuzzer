# DL-Fuzz: Deep Learning Library Fuzzing Framework

基于 LLM 驱动的深度学习库模糊测试（Fuzzing）框架，主要用于 PyTorch 等 DL 库的覆盖率导向漏洞挖掘。

## 项目概述

本项目通过 **知识增强的 LLM 生成 + 双通道赌博机筛选** 的方式，自动化完成从 API 参数空间建模、Fuzz Harness 生成到高效 Fuzzing 执行的全流程。核心思路是：

1. 从 PyTorch 源码中提取 API schema，结合官方文档用 LLM 生成结构化的参数空间 YAML 表示
2. 基于 YAML 参数空间，用 LLM 批量生成符合 Atheris 框架的可执行 Fuzz Harness
3. 使用双层 UCB 赌博机算法动态筛选高效 Harness 执行，最大化覆盖率产出

## 架构总览

```
                        API 列表
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
    ┌──────────┐   ┌──────────────┐   ┌──────────┐
    │ 源码提取  │   │  文档 txt    │   │ LLM 增强 │
    │ schema   │   │  (api_txt)   │   │          │
    └────┬─────┘   └──────┬───────┘   └────┬─────┘
         │                │                │
         └────────────────┼────────────────┘
                          │
              ┌───────────▼───────────┐
              │   build_yaml          │
              │   结构化参数空间 YAML   │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   fuzz_api_one        │
              │   LLM 生成 Fuzz       │
              │   Harness (.py)       │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   screen              │
              │   双通道 UCB 筛选执行  │
              │   覆盖率收集 & 审计    │
              └───────────────────────┘
```

## 目录结构

```
.
├── build_yaml/               # 模块一：YAML 参数空间生成
│   ├── pipeline.py           #   批处理主 pipeline（6 阶段）
│   ├── export_schema.py      #   从 PyTorch 源码提取 API schema → JSON
│   ├── doc_rank_extractor.py #   从 API 文档中提取 shape/rank 约束
│   ├── schema2yaml.py        #   Schema JSON → YAML skeleton
│   ├── normalize_yaml_skeleton.py  #   YAML 规范化
│   ├── llm_patch_yaml.py     #   LLM Stage-C：注入多 rank 约束
│   ├── patch_constraints.py  #   LLM Stage-D：注入额外约束
│   ├── check_yaml.py         #   YAML 质量校验
│   └── view_model.py         #   可视化工具
│
├── fuzz_api_one/             # 模块二：Harness 生成
│   ├── pepiline.py           #   批量生成 pipeline（并发调度）
│   ├── llm_gen_harness.py    #   LLM 驱动生成 Atheris Fuzz Harness
│   ├── gen_harness.py        #   模板驱动 Harness 生成
│   ├── llm_harness_prompt_template.md  # LLM Prompt 模板
│   └── api-yaml/             #   各 API 的参数 YAML（示例）
│
├── screen/                   # 模块三：双通道筛选执行
│   ├── cli/main.py           #   命令行入口 & 配置解析
│   ├── bandit_audit_driver_hier.py  # 主 orchestration 逻辑
│   ├── bandit/               #   双通道 UCB 赌博机算法
│   │   ├── bandit_core.py    #     DualChannelUCB 核心实现
│   │   ├── policy.py         #     选择/淘汰策略
│   │   └── rewards.py        #     Reward 计算（快速代理 + 慢速审计）
│   ├── pool/                 #   Profile 池管理
│   │   └── profile_pool.py   #     池刷新/精英保留/变异
│   ├── prior/                #   跨 Harness 先验知识共享
│   │   └── group_prior.py    #     GroupPrior 精英记忆
│   ├── runner/               #   执行引擎
│   │   ├── fuzz_runner.py    #     LibFuzzer/Atheris 进程管理
│   │   └── audit_runner.py   #     覆盖率审计运行器
│   ├── metrics/              #   覆盖率/指标解析
│   │   ├── compute.py        #     增量覆盖率计算
│   │   └── parse_libfuzzer.py#     LibFuzzer 日志解析
│   ├── config/               #   配置 schema
│   └── cov_global_union_audit.py  # LLVM source-based 全局覆盖率审计
│
├── yaml_save_path/           # YAML pipeline 工作目录示例
│   ├── 01_schema_json/       #   Stage 1: 导出的 schema JSON
│   ├── 02_rank_hints/        #   Stage 2: 文档提取的 rank 约束
│   ├── 03_yaml_skeleton/     #   Stage 3: YAML skeleton
│   ├── 04_yaml_normalized/   #   Stage 4: 规范化 YAML
│   ├── 05_stagec_multirank/  #   Stage 5: LLM 注入多 rank 变体
│   └── 06_final_yaml/        #   Stage 6: 最终参数空间 YAML
│
├── api_txt/                  # PyTorch API 文档 txt
├── seed_torch/               # 初始种子语料库
├── fuzz_output_result/       # Fuzzing 输出目录
├── cov-tools/                # LLVM 覆盖率工具链
├── run_screen.sh             # Screen 模块启动脚本
└── replay_titan_fuzz.py      # Crash 复现脚本
```

## 模块一：YAML 参数空间生成 (`build_yaml`)

将 API 列表 + PyTorch 源码 + 官方文档转换为结构化的参数空间 YAML 文件。

### Pipeline 流程

| Stage | 名称 | 输入 | 输出 | 功能 |
|-------|------|------|------|------|
| 0 | 加载 | API 列表 (txt/json) | API 列表 | 加载目标 API 列表 |
| 1 | `export` | PyTorch 运行时 | `01_schema_json/` | 反射导出 API schema（参数名、类型、默认值） |
| 2 | `rank` | API 文档 txt | `02_rank_hints/` | 从文档中提取 rank/shape 约束信息 |
| 3 | `skeleton` | Schema JSON + Rank hints | `03_yaml_skeleton/` | 生成 YAML skeleton（含参数空间骨架） |
| 4 | `normalize` | YAML skeleton | `04_yaml_normalized/` | 规范化 YAML 结构 |
| 5 | `stagec` (LLM) | 规范化 YAML + API 文档 | `05_stagec_multirank/` | LLM 注入多 rank 变体（如矩阵乘法的多维兼容） |
| 6 | `staged` (LLM) | Stage-C YAML + API 文档 | `06_final_yaml/` | LLM 注入约束条件（dtype、device、value range 等） |

### 运行方式

```bash
python build_yaml/pipeline.py \
  --api_list api4titan_fuzz.txt \
  --docs_root api_txt/ \
  --work_dir yaml_output/ \
  --stages all \
  --model gpt-5-codex \
  --overwrite
```

### YAML 参数空间示例

```yaml
api_name: torch.addbmm
aten:
  overload: out
params:
  - name: input
    dtype: float32
    shape_spec:
      - [B, M, N]
  - name: batch1
    dtype: float32
    shape_spec:
      - [B, N, M]  # 转置兼容
  - name: batch2
    dtype: float32
    shape_spec_by_rank:
      2: [M, K]      # 2D batch 场景
      3: [B, N, K]   # 3D batch 场景
constraints:
  - "batch1.shape[-1] == input.shape[-2]"
```

## 模块二：Harness 生成 (`fuzz_api_one`)

基于 LLM，将 YAML 参数空间描述转换为可执行的 Atheris Fuzz Harness。

### 核心流程

```
YAML 参数空间 + API 文档 + Atheris 说明
  → System Prompt（角色设定 + 硬性规则）
  → User Prompt（模板填充具体 API 信息）
  → LLM 生成
  → 提取 Python 代码块
  → AST 静态校验（语法 + 关键标记）
  → 不通过 → Repair Prompt 修复（最多 N 轮）
  → 输出 .py Harness
```

### 运行方式

```bash
# 单个 API 生成
python fuzz_api_one/llm_gen_harness.py \
  --yaml yaml_1/06_final_yaml/torch.addbmm__ov_out__self__MULTIRANK.yaml \
  --api-name torch.addbmm \
  --api-txt api_txt/torch.addbmm.txt \
  --atheris-doc atheris-doc/atheris_readme.txt \
  --out fuzz_output/llm.torch.addbmm.py \
  --model gpt-5.4

# 批量生成（从 JSON 清单并发调度）
python fuzz_api_one/pepiline.py \
  --json screen/supply_titan_fuzz.json \
  --out-dir /root/fuzz_output_experiment/ \
  --workers 2 \
  --skip-existing
```

### Prompt 设计要点

- **YAML 优先但不可盲从**：如果与 PyTorch API 文档语义冲突则降权
- **覆盖率导向**：生成的 Harness 需要最大化参数空间和下游客代码覆盖
- **Atheris 框架规范**：遵循 LibFuzzer 接口，支持 `FuzzedDataProvider`

## 模块三：双通道筛选执行 (`screen`)

对大量已生成的 Harness 进行智能调度和筛选，将 Fuzzing 预算集中到高产出 Harness 上。

### 核心算法：双层 UCB 赌博机

```
                        ┌─────────────────────┐
                        │   Harness 级 UCB     │
                        │   选择本轮执行的      │
                        │   Harness            │
                        └──────────┬──────────┘
                                   │ 选出 Harness h
                        ┌──────────▼──────────┐
                        │   Profile 级 UCB     │
                        │   为 h 选择最优的     │
                        │   变异参数组合        │
                        └──────────┬──────────┘
                                   │ 选出 Profile p
                        ┌──────────▼──────────┐
                        │   Fuzzing 执行       │
                        │   (epoch=60s)        │
                        └──────────┬──────────┘
                                   │ log + new_files
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
        ┌──────────┐      ┌──────────────┐      ┌──────────┐
        │ 快速通道  │      │ 慢速审计通道  │      │ 降权/淘汰 │
        │ 代理奖励  │      │ 全局覆盖率    │      │ UCB比较  │
        │ 每轮更新  │      │ 每N轮触发    │      │ 冷却机制  │
        └──────────┘      └──────────────┘      └──────────┘
```

### 双通道 Reward 机制

| 通道 | 频率 | 指标 | 说明 |
|------|------|------|------|
| **快速通道** | 每轮（~60s） | 代理奖励（Δcov + Δft + 速度因子） | 低成本、高频、有噪声 |
| **慢速审计** | 每 N 轮 | 全局覆盖率增量（BRH/LH） | 高成本、低频、精确 |

**快速代理奖励公式**：
```
r_proxy = ((1-mix) × ln(1+Δcov) + mix × ln(1+Δft)) × √exec_speed × cov_bonus
r_fast  = r_proxy × (1 + 0.1 × ln(1+|new_files|))
```

**慢速审计奖励**：基于 LLVM source-based coverage 的全局 union 覆盖率精确增量。

### 关键机制

- **ε-贪心探索**：harness 级 ε=0.02，profile 级 ε=0.05，避免过早收敛
- **软淘汰 + 冷却**：UCB < best_LCB 连续 N 轮 → 冷却 K 步（可恢复，非永久禁用）
- **事件比例降权**：REDUCE/pulse 占比 ≥ 70% → 触发冷却
- **Profile 池进化**：保留 Top 50% → 变异生成 30% → 从 GroupPrior 注入精英
- **跨 Harness 知识共享**：同一 group 的 Harness 共享精英 profile

### 运行方式

```bash
# 使用提供的启动脚本
bash run_screen.sh 6h

# 或直接调用
python3 -m screen.cli.main \
  --harnesses_json screen/auto_harness_all.json \
  --groups_map screen/groups_map.json \
  --root fuzz_output_result/ \
  --epoch 30 \
  --steps 0 \
  --audit_every 3 \
  --cov_audit_script screen/cov_global_union_audit.py
```

## 环境要求

- **Python**: 3.10+
- **PyTorch**: 含调试符号的覆盖率插桩版本
- **LLVM**: 用于 source-based coverage 收集（`llvm-profdata`、`llvm-cov`）
- **LLM API**: OpenAI 兼容接口（如 GPT-5）
- **LibFuzzer**: 通过 Atheris 框架使用

### 关键依赖

```bash
# PyTorch coverage 版本
pip install torch --index-url ...

# Atheris fuzzing 框架
pip install atheris

# LLVM 工具链
apt install llvm

# Python 依赖
pip install pyyaml openai
```

## 工作流程

### 完整 Fuzzing 流程

```bash
# 1. 准备 API 列表和文档
#    - api_list.txt: 目标 API 列表
#    - api_txt/: API 文档 txt 文件

# 2. 生成 YAML 参数空间
python build_yaml/pipeline.py \
  --api_list api_list.txt \
  --docs_root api_txt/ \
  --work_dir yaml_output/ \
  --stages all \
  --model gpt-5-codex \
  --overwrite

# 3. 按需筛选/合并 API → 生成 supply JSON
python make_json_for_titan_fuzz_supply.py  # 或自定义

# 4. 批量生成 Fuzz Harness
python fuzz_api_one/pepiline.py \
  --json screen/supply_titan_fuzz.json \
  --out-dir fuzz_output/ \
  --workers 2

# 5. 构建 Harness 清单
#    → screen/auto_harness_all.json

# 6. 启动筛选执行
bash run_screen.sh 24h
```

## 关键文件说明

| 文件 | 用途 |
|------|------|
| `yaml2harness.json` | YAML → Harness 映射清单（主实验） |
| `yaml2harness_supplement.json` | YAML → Harness 映射清单（补充） |
| `screen/auto_harness_all.json` | 所有可用的 Harness 及参数 |
| `screen/groups_map.json` | Harness → API group 的映射（用于先验共享） |
| `screen/supply_titan_fuzz.json` | 待生成 Harness 的 API 清单 |
| `api_txt/` | 各 API 的官方文档文本 |
| `seed_torch/` | LibFuzzer 初始种子语料库 |

## 相关论文/设计文档

- `screen/screen_harness_selection_pseudocode.md` — Screen 模块双通道 UCB 算法完整设计
- `fuzz_api_one/llm_harness_prompt_template.md` — LLM Harness 生成 Prompt 工程文档

