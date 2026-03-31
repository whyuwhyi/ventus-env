# Ventus GPT-2 Addmm MMA Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把 Ventus `aten::addmm.out` 从当前的普通 FP32 GEMM fast path 升级为基于 `m16n16k16` 的 MMA fast path，采用 FP16 输入和 FP32 累加，同时保持不回退到 CPU。

**Architecture:** 在 `addmm.out` 中新增 MMA 路径判定，优先命中 `m16n16k16_fp16_fp32`。输入在 host 侧按 benchmark 的 tile 方式 pack 为 FP16，内部使用 `__builtin_riscv_ventus_mma_m16n16k16_row_col_f32_f16_f16_f32`，输出保持 FP32。非 tile 整倍数的情况通过内部 padding/cropping 处理，而不是回退到 CPU。

**Tech Stack:** PyTorch C++ dispatcher, ATen Ventus runtime, Ventus MMA builtins, OpenCL kernels, benchmark packing utilities, Ninja, Python testcase

---

### Task 1: 写 MMA 路径的 failing test

**Files:**
- Create: `ventus-pytorch/testcases/addmm_mma.py`
- Modify: `ventus-pytorch/testcases/addmm.py`

**Step 1: 写 `addmm_mma.py`**

脚本要求：

- 构造 `M=N=K=16` 的最小 `torch.addmm`
- 输入仍然使用 FP32 张量，外部语义不变
- 通过环境变量强制要求 `addmm` 命中 MMA 路径，例如 `VENTUS_ADDMM_REQUIRE_MMA=1`
- 结果与 CPU 基准对齐

**Step 2: 跑脚本确认红态**

Run: `source env.sh && VENTUS_ADDMM_REQUIRE_MMA=1 python ventus-pytorch/testcases/addmm_mma.py`
Expected: FAIL because runtime 当前还没有 MMA fast path / env gate

**Step 3: 扩展 `addmm.py`**

保留现有 `addmm.py` 作为普通数值回归，不把它改成 MMA 专用。

### Task 2: 写非整倍数尾块验证

**Files:**
- Create: `ventus-pytorch/testcases/addmm_mma_tail.py`

**Step 1: 写 17x18x19 一类的 testcase**

脚本要求：

- 输入 shape 故意不是 16 的整倍数
- 通过 `VENTUS_ADDMM_REQUIRE_MMA=1` 要求仍留在 Ventus 路径
- 数值与 CPU 基准对齐

**Step 2: 运行确认红态**

Run: `source env.sh && VENTUS_ADDMM_REQUIRE_MMA=1 python ventus-pytorch/testcases/addmm_mma_tail.py`
Expected: FAIL because当前没有内部 padding/cropping 的 MMA 路线

### Task 3: 抽取 benchmark 的 packing 参考

**Files:**
- Read/Reference: `benchmarks/mma/m16n16k16_fp16_fp32/kernel.cl`
- Read/Reference: `benchmarks/workloads/gelu_mlp_fp16/kernel.cl`
- Read/Reference: `benchmarks/workloads/workload_case.hpp`
- Create: `ventus-pytorch/aten/src/ATen/ventus/MmaPacking.h`
- Create: `ventus-pytorch/aten/src/ATen/ventus/MmaPacking.cpp`

**Step 1: 对齐现有 tile 数据格式**

明确：

- A tile 的 row-major FP16 packing
- B tile 的 col-major FP16 packing
- C/out tile 的 FP32 unpack 方式

**Step 2: 提炼最小 host-side packing helper**

只先支持：

- `m16n16k16`
- `float32 -> fp16 pack`
- `fp32 out` unpack

**Step 3: 只复制最小必要逻辑**

不要把整个 benchmark helper 全搬进 ATen；只提炼 `addmm` 需要的 packing/unpacking。

### Task 4: 新增 MMA kernel 资产

**Files:**
- Create: `ventus-pytorch/aten/src/ATen/ventus/kernels/src/addmm_mma.cl`
- Create: `ventus-pytorch/aten/src/ATen/ventus/kernels/build-addmm-mma-artifact.sh`
- Create: `ventus-pytorch/aten/src/ATen/ventus/kernels/generate-addmm-mma-artifact.py`

**Step 1: 参考 benchmark kernel**

以 `benchmarks/mma/m16n16k16_fp16_fp32/kernel.cl` 为基线，改成适合 ATen runtime 资产格式的 kernel。

**Step 2: kernel 约束**

要求：

- 输入是 packed FP16 tile
- 调用 `__builtin_riscv_ventus_mma_m16n16k16_row_col_f32_f16_f16_f32`
- 输出是 FP32 tile

**Step 3: 接入 artifact 生成链**

要求：

- 生成 `object.riscv`
- 生成 `addmm_mma_0.metadata`
- 生成 `addmm_mma_0.data`
- 安装到 `torch/share/ventus/kernels/addmm_mma/<backend>/<profile>/`

### Task 5: 在 runtime 中接入 MMA 资产与分流

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.h`
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`

**Step 1: 新增 MMA 入口**

新增：

- `canUseAddmmMmaKernel(...)`
- `launchAddmmMma(...)`

**Step 2: 为 `DriverState` 增加 `addmm_mma` 状态**

包括：

- mutex
- uploaded 标志
- configured_artifacts
- path 缓存

**Step 3: 接入 artifact 解析**

要求：

- 支持 installed artifacts
- 支持 `VENTUS_SOURCE_KERNEL_FALLBACK=1`

**Step 4: 增加调试/验证 gate**

新增环境开关，例如：

- `VENTUS_ADDMM_REQUIRE_MMA=1`

作用：

- 测试时强制要求命中 MMA 路径
- 若没命中则直接报错，而不是悄悄走普通 fast path 或 CPU bridge

### Task 6: 在 runtime 中实现 padding/cropping

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`
- Modify: `ventus-pytorch/aten/src/ATen/ventus/MmaPacking.cpp`

**Step 1: 计算 padded `M/N/K`**

规则：

- 向上对齐到 16

**Step 2: 分配临时 packed buffers**

要求：

- A/B 走 FP16 packed tile buffer
- C/out 走 FP32 tile buffer

**Step 3: 对原输入做 pad + pack**

要求：

- 超出原矩阵边界的元素补零

**Step 4: kernel 执行后 crop 回原尺寸**

要求：

- 最终 `out` 仍为原始 `M x N`
- 不向用户暴露 padded shape

### Task 7: 修改 `addmm.out` 的三层分流

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/BinaryOps.cpp`

**Step 1: 保留现有普通 Ventus fast path**

不要删除当前普通 `launchAddmm(...)` 路径。

**Step 2: 新增优先级**

顺序改成：

1. `canUseAddmmMmaKernel(...)` -> `launchAddmmMma(...)`
2. `canUseAddmmKernel(...)` -> `launchAddmm(...)`
3. CPU bridge

**Step 3: 在 `VENTUS_ADDMM_REQUIRE_MMA=1` 时禁止降级**

要求：

- 若环境变量要求 MMA，但只命中普通 `addmm` 或 CPU bridge，则直接报错

### Task 8: 构建 MMA 资产并重编译

**Files:**
- Modify: none

**Step 1: 生成 MMA kernel 资产**

Run: `source env.sh && bash ventus-pytorch/aten/src/ATen/ventus/kernels/build-addmm-mma-artifact.sh`
Expected: `torch/share/ventus/kernels/addmm_mma/<backend>/<profile>/` 下生成三件套

**Step 2: 重编译**

Run: `env CCACHE_DISABLE=1 ninja -C ventus-pytorch/build torch_python -j16`
Expected: build succeeds

**Step 3: 同步运行时库**

Run:

- `cp -f ventus-pytorch/build/lib/libtorch_cpu.so ventus-pytorch/torch/lib/libtorch_cpu.so`
- `cp -f ventus-pytorch/build/lib/libtorch.so ventus-pytorch/torch/lib/libtorch.so`
- `cp -f ventus-pytorch/build/lib/libtorch_python.so ventus-pytorch/torch/lib/libtorch_python.so`
- `cp -f ventus-pytorch/build/lib/libshm.so ventus-pytorch/torch/lib/libshm.so`

### Task 9: 验证 `addmm` MMA 路径

**Files:**
- Modify: none

**Step 1: 跑基础数值回归**

Run: `source env.sh && python ventus-pytorch/testcases/addmm.py`
Expected: PASS

**Step 2: 跑 MMA 强制命中用例**

Run: `source env.sh && VENTUS_ADDMM_REQUIRE_MMA=1 python ventus-pytorch/testcases/addmm_mma.py`
Expected: PASS

**Step 3: 跑尾块用例**

Run: `source env.sh && VENTUS_ADDMM_REQUIRE_MMA=1 python ventus-pytorch/testcases/addmm_mma_tail.py`
Expected: PASS

### Task 10: 验证 GPT-2 线性层代理与真实路径

**Files:**
- Modify: none

**Step 1: 跑 `Conv1D` 代理**

Run: `source env.sh && python ventus-pytorch/testcases/conv1d_addmm.py`
Expected: PASS

**Step 2: 跑 GPT-2 smoke**

Run:

- `source env.sh && python ventus-pytorch/testcases/gpt2_transformers_smoke.py`
- `source env.sh && python ventus-pytorch/testcases/gpt2_transformers_cache_smoke.py`
- `source env.sh && python ventus-pytorch/testcases/gpt2_transformers_decode_smoke.py`

Expected: all PASS

### Task 11: 跑基础 shape/layout 回归

**Files:**
- Modify: none

**Step 1: 跑第一阶段回归**

Run:

- `source env.sh && python ventus-pytorch/testcases/view.py`
- `source env.sh && python ventus-pytorch/testcases/reshape.py`
- `source env.sh && python ventus-pytorch/testcases/contiguous.py`
- `source env.sh && python ventus-pytorch/testcases/gpt_shape_path.py`

Expected: all PASS

### Task 12: 记录下一阶段输入

**Files:**
- Modify: none

**Step 1: 记录仍未覆盖的矩阵乘路径**

明确：

- 哪些 `addmm` shape 还没命中 MMA
- attention 的 `mm/bmm` 还差什么

**Step 2: 把下一阶段收敛成 `mm/bmm` MMA 计划输入**

不要在本轮继续扩到 `mm/bmm` 实现。
