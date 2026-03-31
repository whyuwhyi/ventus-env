# Ventus Shared GEMM Launcher Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 Ventus 新增 shared GEMM launcher，把 `aten::mm.out`、`aten::addmm.out`、`aten::bmm.out` 统一挂到同一个 runtime primitive 上，并移除这三条路径上的 CPU fallback 依赖。

**Architecture:** 入口层在 `BinaryOps.cpp` 里只负责参数归一化和 `TORCH_CHECK`；真正的 GEMM 判定、artifact 选择、metadata 打包和 launch 全部收敛到 `VentusRuntime.cpp`。当前已有的 plain `addmm` 和 `addmm_mma` 资产继续保留，但由 shared launcher 统一调度。

**Tech Stack:** PyTorch C++ dispatcher, ATen Ventus runtime, Ventus kernel artifacts, Python testcase, Ninja

---

### Task 1: 写 `mm.out` 和 `linear` 的 failing test

**Files:**
- Create: `ventus-pytorch/testcases/mm.py`
- Create: `ventus-pytorch/testcases/linear_mm.py`
- Test: `ventus-pytorch/testcases/addmm.py`
- Test: `ventus-pytorch/testcases/bmm.py`

**Step 1: 写 `mm.py`**

脚本要求：

- 构造最小 `torch.mm(a, b, out=...)` 或等价 `out` 路径
- 输入放在 Ventus
- 与 CPU 基准比较
- 运行时如果落到 CPU fallback，应明确失败

**Step 2: 运行确认红态**

Run: `source env.sh && python ventus-pytorch/testcases/mm.py`  
Expected: FAIL with `aten::mm.out` missing on Ventus or fallback warning on the current tree

**Step 3: 写 `linear_mm.py`**

脚本要求：

- 构造一个无 bias 的 `torch.nn.Linear`
- 强制走 `linear -> mm.out`
- 比较 Ventus 与 CPU 输出

**Step 4: 运行确认红态**

Run: `source env.sh && python ventus-pytorch/testcases/linear_mm.py`  
Expected: FAIL because `linear` still reaches `aten::mm.out` fallback on the current tree

### Task 2: 定义 shared GEMM runtime 接口

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.h`
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`

**Step 1: 在头文件里新增统一 request 类型**

至少包含：

- `GemmKind`
- batch
- `M/N/K`
- bias 是否存在
- `beta / alpha`
- `mat1 / mat2 / out`

**Step 2: 声明统一入口**

新增类似接口：

- `bool canUseGemm(...)`
- `void launchGemm(...)`
- 必要时保留 `canUseAddmmMmaKernel(...)` 作为 shared launcher 内部子判断

**Step 3: 保留现有对外 API，准备后续迁移**

短期可以保留：

- `launchAddmm(...)`
- `launchAddmmMma(...)`

但让它们逐步变成 shared launcher 的薄包装。

### Task 3: 抽取 plain GEMM 公共实现

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`

**Step 1: 从现有 `launchAddmm(...)` 中抽出公共逻辑**

公共部分包括：

- plain artifact 准备
- metadata/data 读取
- arg buffer 分配
- metadata payload 改写
- kernel launch

**Step 2: 把 bias 变成可选输入**

约定：

- `addmm` 使用真实 bias
- `mm` 使用 `beta=0` 且无 bias
- `bmm` 先不支持 bias

**Step 3: 把 `M/N/K` 和输出元素数改为 request 驱动**

不要再让 plain GEMM 代码只理解“1D bias + 2D mat1 + 2D mat2”。

### Task 4: 让 `mm.out` 接到 shared GEMM

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/BinaryOps.cpp`

**Step 1: 新增 `mm_out_tensor(...)`**

要求：

- 只接受 Ventus tensor
- 校验 2D、contiguous、dense、dtype 一致性
- 若当前不满足 shared GEMM 约束，直接 `TORCH_CHECK`

**Step 2: 在 Ventus dispatcher 中注册**

新增：

- `m.impl(TORCH_SELECTIVE_NAME("aten::mm.out"), TORCH_FN(mm_out_tensor));`

**Step 3: 不允许 CPU bridge**

当前主线要求是 no fallback，因此 `mm.out` 不要写 CPU bridge。

### Task 5: 把 `addmm.out` 改挂 shared GEMM

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/BinaryOps.cpp`
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`

**Step 1: 保留当前 MMA 判定策略**

继续支持：

- plain `addmm`
- `addmm_mma`

**Step 2: 把入口改成统一 request**

`addmm_out_tensor(...)` 不再直接拼 launch 细节，只负责：

- `out` reshape / allocate
- request 构造
- 调用 shared launcher

**Step 3: 删除 `addmm.out` 的 CPU bridge**

如果 plain 和 MMA 都不满足当前约束，直接报错。

### Task 6: 把 `bmm.out` 从 fallback 挪到 shared GEMM

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/BinaryOps.cpp`
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusFallback.cpp`

**Step 1: 在 `BinaryOps.cpp` 中新增 `bmm_out_tensor(...)`**

要求：

- 输入是 3D Ventus tensor
- 批维相等
- `K` 匹配
- 输出为 3D Ventus tensor

**Step 2: 在 runtime 中实现 batched plain GEMM**

第一版可以接受：

- 逐 batch 循环 launch plain GEMM kernel

但不能接受：

- CPU fallback

**Step 3: 从 fallback 中移除 `try_handle_bmm_out(...)` 主路径**

`VentusFallback.cpp` 不应再拦截 `aten::bmm.out`。

### Task 7: 统一 artifact / request 选择与错误信息

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`

**Step 1: 统一 GEMM 家族的错误信息**

错误必须能区分：

- plain gemm requirement 不满足
- mma requirement 不满足
- profile / hardware cap 不匹配
- artifact 缺失

**Step 2: 保留现有环境开关语义**

例如：

- `VENTUS_ADDMM_REQUIRE_MMA=1`

shared launcher 需要继续理解这个 gate。

**Step 3: 确保 spike / rtlsim 的 artifact 选择不被破坏**

不要重新引入之前那类路径拼接错误。

### Task 8: 跑本地验证并整理编译交接

**Files:**
- Modify: none

**Step 1: 运行无需重编译即可跑的脚本检查**

Run:

- `source env.sh && python ventus-pytorch/testcases/mm.py`
- `source env.sh && python ventus-pytorch/testcases/linear_mm.py`
- `source env.sh && python ventus-pytorch/testcases/addmm.py`
- `source env.sh && python ventus-pytorch/testcases/bmm.py`

Expected:

- 在实现前红态
- 在重编译并同步库后转绿

**Step 2: 整理需要用户执行的编译命令**

由用户执行，不由助手执行。至少包括：

- `source env.sh && env CCACHE_DISABLE=1 ninja -C ventus-pytorch/build torch_python -j16`
- 同步 `ventus-pytorch/build/lib/*.so` 到 `ventus-pytorch/torch/lib/`

**Step 3: 重编译后回归**

Run:

- `source env.sh && python ventus-pytorch/testcases/mm.py`
- `source env.sh && python ventus-pytorch/testcases/linear_mm.py`
- `source env.sh && python ventus-pytorch/testcases/addmm.py`
- `source env.sh && python ventus-pytorch/testcases/addmm_mma.py`
- `source env.sh && python ventus-pytorch/testcases/addmm_mma_tail.py`
- `source env.sh && python ventus-pytorch/testcases/bmm.py`
- `source env.sh && VENTUS_BACKEND=spike python ventus-pytorch/testcases/gpt2_transformers_cache_smoke.py`
- `source env.sh && VENTUS_BACKEND=spike python ventus-pytorch/testcases/gpt2_transformers_decode_smoke.py`

Expected:

- `mm.out` / `addmm.out` / `bmm.out` 不再因未实现而 fallback
- `linear -> mm.out` 留在 Ventus 路径
- GPT-2 smoke 继续向 attention 主路径推进
