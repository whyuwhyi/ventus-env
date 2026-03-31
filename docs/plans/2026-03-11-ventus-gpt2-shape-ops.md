# Ventus GPT-2 Shape Ops Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 Ventus 后端补齐 `view`、`reshape`、`contiguous`，打通 GPT 类模型第一批 shape/layout 基础算子。

**Architecture:** 在 Ventus 专属注册层补齐 `view` / `reshape`，优先复用现有 `as_strided` 与 alias 逻辑；对 `contiguous` 则先复用现有 Ventus kernel，再最小化修补 `AutogradVentus` 分发缺口。整个实现遵循 TDD，先用现有 testcase 做红绿，再加一条最小 Transformer 路径验证。

**Tech Stack:** PyTorch C++ dispatcher, ATen Ventus backend, CMake/Ninja, Python testcase

---

### Task 1: 固定失败样例并确认红态

**Files:**
- Test: `ventus-pytorch/testcases/view.py`
- Test: `ventus-pytorch/testcases/reshape.py`
- Test: `ventus-pytorch/testcases/contiguous.py`

**Step 1: 运行 `view` testcase 记录失败**

Run: `source env.sh && python ventus-pytorch/testcases/view.py`
Expected: FAIL with `Could not run 'aten::view' with arguments from the 'Ventus' backend`

**Step 2: 运行 `reshape` testcase 记录失败**

Run: `source env.sh && python ventus-pytorch/testcases/reshape.py`
Expected: FAIL with `aten::view` 缺失

**Step 3: 运行 `contiguous` testcase 记录失败**

Run: `source env.sh && python ventus-pytorch/testcases/contiguous.py`
Expected: FAIL with `Could not run 'aten::contiguous' with arguments from the 'AutogradVentus' backend`

### Task 2: 为 Ventus 补齐 `view`

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/Factory.cpp`

**Step 1: 写最小 failing 行为锚点**

Use existing `ventus-pytorch/testcases/view.py` as the failing test.

**Step 2: 先实现与原生语义一致的 `view`**

- 在 Ventus 注册层增加 `aten::view`
- 复用 `infer_size` 和 stride compatibility 逻辑
- stride 兼容时返回 metadata-only view
- stride 不兼容时保留原生报错语义

**Step 3: 重新运行 `view` testcase**

Run: `source env.sh && python ventus-pytorch/testcases/view.py`
Expected: PASS

### Task 3: 为 Ventus 补齐 `reshape`

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/Factory.cpp`

**Step 1: 使用现有 `reshape.py` 作为 failing test**

Run: `source env.sh && python ventus-pytorch/testcases/reshape.py`
Expected: FAIL because internal path still reaches missing `aten::view`

**Step 2: 增加 Ventus `reshape` 注册**

- 优先走与原生一致的 `reshape -> view when possible` 路径
- 不额外引入 Ventus 专用 copy 语义

**Step 3: 重新运行 `reshape` testcase**

Run: `source env.sh && python ventus-pytorch/testcases/reshape.py`
Expected: PASS

### Task 4: 修复 `contiguous` 的 `AutogradVentus` 路径

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/Factory.cpp`
- Modify if needed: `ventus-pytorch/c10/core/DispatchKeySet.h`
- Modify if needed: `ventus-pytorch/c10/core/TensorImpl.h`
- Modify if needed: other minimal `ventus-pytorch/c10/core/*` glue files directly proven necessary by reproduction

**Step 1: 保持 testcase 处于失败状态**

Run: `source env.sh && python ventus-pytorch/testcases/contiguous.py`
Expected: FAIL on `AutogradVentus`

**Step 2: 做最小实现**

- 优先验证是否只需调整现有 Ventus `contiguous` 路径即可
- 若仍掉到 `AutogradVentus`，只补最小 dispatch glue
- 不顺手改 unrelated backend key plumbing

**Step 3: 重新运行 `contiguous` testcase**

Run: `source env.sh && python ventus-pytorch/testcases/contiguous.py`
Expected: PASS

### Task 5: 增加最小 Transformer 路径验证

**Files:**
- Create or Modify: `ventus-pytorch/testcases/` 下一个最小 attention / transformer shape 路径脚本

**Step 1: 写最小 failing script**

覆盖以下链路：

- `qkv.view(...)`
- `transpose(1, 2)`
- `.contiguous().view(...)`

**Step 2: 先运行确认失败或触发旧问题**

Run: `source env.sh && python <new-test>.py`
Expected: 在修复前失败，且失败点与本轮目标一致

**Step 3: 用修复后的实现重新运行**

Run: `source env.sh && python <new-test>.py`
Expected: PASS

### Task 6: 汇总验证

**Files:**
- Modify: none

**Step 1: 运行全部第一阶段验证**

Run: `source env.sh && python ventus-pytorch/testcases/view.py`
Expected: PASS

Run: `source env.sh && python ventus-pytorch/testcases/reshape.py`
Expected: PASS

Run: `source env.sh && python ventus-pytorch/testcases/contiguous.py`
Expected: PASS

Run: `source env.sh && python <new-test>.py`
Expected: PASS

**Step 2: 记录仍未覆盖的 GPT-2 缺口**

把下一批缺失算子整理出来，作为后续阶段输入，而不是在这一轮无限扩 scope。

