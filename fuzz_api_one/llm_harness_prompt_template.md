# LLM 生成 Harness 的 Prompt 模板

## 整体流程

```
输入: YAML(API约束+重载) + API文档(txt) + Atheris说明
  → System Prompt (角色设定 + 硬性规则)
  → User Prompt (模板填充具体API信息)
  → LLM 生成
  → 提取 Python 代码块 (```python ... ```)
  → 静态校验 (AST语法 + 关键标记检查)
  → 不通过? → Repair Prompt 修复 (最多 N 轮)
  → 输出 .py harness
```

---

## 1. System Prompt — 角色与硬约束

### 角色设定
你是 PyTorch + Atheris 覆盖率导向 fuzzing 专家。

### 优先级 (降序)
1. harness 必须可执行、语法正确
2. 必须实际调用目标 API
3. 最大化参数空间覆盖 + 下游代码覆盖
4. YAML 优先使用，但不可盲从（与 API 文档 / PyTorch 语义矛盾时降权）

### YAML 解读规则
- 多个 YAML 可能对应不同重载，需要协调兼容
- YAML 可信但不完美 → 矛盾时优先保证可执行性

### 覆盖率目标
- 探索 dtypes / ranks / shapes / 可选参数 / 重载 / 边界值的广泛组合
- 不坍缩为单一"最合法"模式
- 偏向合法输入但保留多样性
- 默认小 tensor，但允许变异到不同 rank / shape

### 硬性要求 (17条)
1. 仅输出一个 Python 代码块
2. 完整独立的 `.py` 文件
3. 首先 `import atheris`
4. 禁用 `with atheris.instrument_imports():`
5. 禁用 `atheris.instrument_all()`
6. `TestOneInput` 前加 `@atheris.instrument_func`
7. 定义 `TestOneInput(data: bytes)`
8. `atheris.Setup(sys.argv, TestOneInput)` → `atheris.Fuzz()` 标准流程
9. 使用 `FuzzedDataProvider(data)` 解码
10. 必须调用目标 API
11. 生成合理有效的 PyTorch 值同时保持多样性
12. 异常吞噬策略: `RuntimeError, TypeError, ValueError, IndexError, AssertionError, NotImplementedError` → `return`
13. 无占位符 / TODO / 伪代码
14. 不依赖本地项目模块
15. 辅助函数内联到同文件
16. 直接可运行

---

## 2. User Prompt 模板

### 模板占位符

| 占位符 | 来源 | 说明 |
|--------|------|------|
| `{api_name}` | YAML 或 CLI 参数 | 目标 API 名，如 `torch.select_scatter` |
| `{yaml_summary}` | `build_yaml_summary()` | YAML 结构化摘要：params、dtype_choices、shape_spec、constraints、overload 等 |
| `{all_yaml_text}` | 原始 YAML 文件 | 所有匹配 YAML 的全文 |
| `{api_txt_text}` | API 文档 txt | PyTorch API 官方文档原文 |
| `{atheris_doc_text}` | Atheris README | Atheris fuzz 框架使用说明 |

### User Prompt 核心逻辑

- **目标**: 生成覆盖率导向的可执行 harness
- **信息来源优先级**: YAML 为主结构提示，但矛盾时 API txt > YAML
- **多 YAML 协调**: 同一 API 多个 YAML 可能对应不同重载，需协调兼容
- **设计目标**: 覆盖全参数空间，暴露不同 dtype / rank / shape / 可选参数 / 重载分支 / 边界值
- **有效性策略**: "偏合法但非仅合法" — 经常足够合法以执行深层代码，同时保持多样性让 fuzzer 能变异
- **冲突推理**: YAML 与 API 文本一致时遵从；YAML 缺失细节时推断；YAML 明显错误时降权
- **实现提示**: 通过 fuzz 输入驱动分支选择重载模式；复用解码选择协调关联参数

### 完整 User Prompt 结构

```
Generate one executable Python Atheris harness for the PyTorch API described below.

Primary goal: 覆盖率导向的可执行 harness

Reliability and source-priority rules:
  - YAML 为主结构提示，但不盲从
  - 矛盾时: 可执行性 > 盲从 YAML

Output contract:
  - 仅输出一个 fenced Python 代码块
  - 无前后说明文字

Target API: {api_name}

Multi-YAML guidance: 多 YAML 可能对应不同重载，需协调

Harness design objectives:
  - 覆盖全参数空间
  - 暴露不同 dtype / rank / shape / 可选参数 / 重载 / 边界值
  - 通过 FuzzedDataProvider 结构化解码
  - 小 tensor 为主，但保留变异空间
  - 辅助函数内联、无日志打印、自包含

Validity strategy: 偏合法但非仅合法

Reasoning policy for conflicting information:
  - YAML + API txt 一致 → 遵从
  - YAML 缺失 → 推断
  - YAML 明显错误 → 降权

Implementation hints: ...

==== YAML SUMMARY BEGIN ====
{yaml_summary}
==== YAML SUMMARY END ====

==== ALL API YAMLs BEGIN ====
{all_yaml_text}
==== ALL API YAMLs END ====

==== API TXT BEGIN ====
{api_txt_text}
==== API TXT END ====

==== ATHERIS DOC BEGIN ====
{atheris_doc_text}
==== ATHERIS DOC END ====
```

---

## 3. Repair Prompt — 修复失败生成

```
校验失败时触发，最多 repair_attempts 轮 (默认 1):

Prompt = {
    "你的上一轮输出未通过校验",
    "校验错误: {validation_error}",
    "上一轮原始响应: {bad_response}",
    "提取出的代码: {bad_code}",
    "请修复，重新输出一个 fenced Python 代码块",
    "硬性要求复述: import atheris 为首, @instrument_func, TestOneInput, Setup→Fuzz, 自包含...",
    "原始任务: {original_prompt}"
}
```

---

## 4. 静态校验项

| 检查项 | 规则 |
|--------|------|
| AST 解析 | `ast.parse(code)` 必须通过 |
| 首个 import | 必须是 `import atheris` |
| 必需标记 | `@atheris.instrument_func`, `TestOneInput`, `FuzzedDataProvider`, `atheris.Setup`, `atheris.Fuzz`, `torch` |
| 禁用项 | `instrument_imports()`, `instrument_all()` |
| API 引用 | 代码中必须出现目标 API 名或其尾部名称 |

---

## 5. YAML 摘要结构 (`build_yaml_summary`)

对每个 YAML 提取以下关键信息注入 prompt:

```
Target API: torch.xxx
Number of YAML files: N

[i] source_yaml: path/to/xxx.yaml
  category: ...
  rank_hints: status=... candidates=... rank_any=...
  aten: aten_name=... overload=... schema_str=...
  shape_vars_keys: [...]
  params: [...]
    - param_name: kind=... dtype_choices=... shape_spec=... range=...
  constraints_count: N
    - constraint_description
  generator: stage=... version=...
```

---

## 6. 拼装后的完整 Prompt 报文结构

```
┌──────────────────────────────────────────┐
│ System Prompt                            │
│  角色 + 17条硬约束 + YAML解读规则         │
│  + 覆盖率目标 + 质量门槛                  │
├──────────────────────────────────────────┤
│ User Prompt                              │
│  ┌ 目标: 生成可执行 harness               │
│  ├ 信息来源优先级: YAML为主, API txt兜底   │
│  ├ 输出合同: 仅一个代码块                  │
│  ├ Target API: {api_name}                │
│  ├ 多YAML指引                            │
│  ├ Harness设计目标 (参数空间覆盖)          │
│  ├ 有效性策略 (偏合法非仅合法)             │
│  ├ 冲突推理策略                           │
│  ├ 实现提示                              │
│  ├ ==== YAML SUMMARY (结构化摘要) ====   │
│  ├ ==== ALL YAML TEXT (原文) ====        │
│  ├ ==== API TXT (文档原文) ====          │
│  └ ==== ATHERIS DOC (框架说明) ====      │
└──────────────────────────────────────────┘
```

> **核心设计理念**: 多源信息融合 (YAML 结构约束 + API 文档语义 + 框架硬规范)，以可执行性为底线，以覆盖率最大化为目标，YAML 优先但不盲从。
