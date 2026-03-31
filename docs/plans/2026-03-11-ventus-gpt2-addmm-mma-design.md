# Ventus GPT-2 Addmm MMA Design

**Date:** 2026-03-11

**Goal:** 把当前 Ventus 上的 `aten::addmm.out` 从“普通 FP32 GEMM fast path”升级为“基于 MMA 指令的 FP16 输入、FP32 累加 fast path”，作为后续把 GPT-2 真正拉回 Ventus 计算路径的第一步。

## Background

当前 Ventus 已经能在功能上跑通 GPT-2：

- `use_cache=False` 的最小前向
- `use_cache=True` 的 prefill
- 第二 token decode

并且上一阶段已经把 `transformers.Conv1D -> torch.addmm(...)` 的路径从 CPU bridge 切回了 Ventus 原生 kernel。

但这个原生 `addmm` 仍然只是普通 GEMM 内核：

- 使用 `float32`
- 在 kernel 里做朴素乘加循环
- 没有使用自定义 `mma` 指令

这意味着它虽然已经不是 CPU fallback，但仍然不是最终想要的 tensor core / MMA 路径。

## New Evidence From `benchmarks/`

用户指出新增指令测试在顶层 `benchmarks/` 中。已确认：

### 1. MMA 微基准已存在

`benchmarks/mma/` 下已经有四个 MMA 形状的可运行例子：

- `m8n8k16_fp16_fp32`
- `m16n8k16_fp16_fp32`
- `m8n16k16_fp16_fp32`
- `m16n16k16_fp16_fp32`

这些 kernel 直接调用：

- `__builtin_riscv_ventus_mma_*`

对应反汇编里也确实出现了：

- `mma.m8n8k16...`
- `mma.m16n8k16...`
- `mma.m16n16k16...`

因此，MMA 不是推测中的未来能力，而是当前工具链已经暴露的真实路径。

### 2. 更高层 FP16 workload 已存在

`benchmarks/workloads/` 里已经有更接近模型算子的例子：

- `bmm_fp16`
- `gelu_mlp_fp16`
- `swiglu_mlp_fp16`

这些 workload 已经包含：

- FP16 tile packing
- `vcvt.fp16.fp32`
- `__builtin_riscv_ventus_mma_m16n16k16_row_col_f32_f16_f16_f32(...)`

这说明“FP16 输入 + FP32 累加 + MMA tile 化”的模式在仓库里已经有成体系的参考实现。

## Scope

本阶段只做一件事：

- 为 Ventus `aten::addmm.out` 增加第一条 MMA fast path

并且严格限制为：

- 输入 `mat1` / `mat2` 走 FP16 tile packing
- 内部调用 `m16n16k16_fp16_fp32`
- 累加保持 FP32
- 优先覆盖 GPT-2 `Conv1D` 的常见线性层形态

明确不做：

- 不在这一轮补 `mm`
- 不在这一轮补 `bmm`
- 不在这一轮补 BF16 MMA
- 不在这一轮重做所有 layout
- 不在这一轮保证所有 `addmm` 形态都命中 MMA

## Approaches Considered

### 1. 保留当前 FP32 普通 GEMM fast path

优点：

- 已经能跑
- 风险最小

缺点：

- 无法利用 MMA
- 与真实目标偏离

### 2. 直接复用 `benchmarks/mma/m16n16k16_fp16_fp32` 作为第一版 `addmm` MMA 基线

优点：

- 有现成 builtin 和反汇编验证
- 与 workload 中的 tile 形状一致
- 最容易先拿到第一条可工作的 MMA `addmm`

缺点：

- 只适合尺寸可 tile 化的子集
- 需要加入 FP16 packing 和尾块处理逻辑

### 3. 一次做多种 MMA tile 形状自动选择

优点：

- 长期更优

缺点：

- 范围过大
- 会拖慢闭环

## Recommendation

推荐方案 2：

- 先以 `m16n16k16_fp16_fp32` 作为唯一目标 tile
- 先让 `addmm` 拿到第一条 MMA 快路径
- 其余情况继续回退到当前 FP32 普通 GEMM fast path 或 CPU bridge

## Design

### 1. 三层分流

`aten::addmm.out` 最终采用三层分流：

1. MMA fast path
   - 命中 `m16n16k16_fp16_fp32`
   - 使用 FP16 tile packing + FP32 accumulate

2. 现有 FP32 普通 GEMM fast path
   - 对还不能走 MMA 但能在 Ventus 上直接算的场景保留

3. 现有 CPU bridge
   - 作为最后兜底

这样可以保证：

- MMA 引入后不破坏当前已可用路径
- GPT-2 能逐步迁移，而不是一次性赌全量切换

### 2. 对齐 `benchmarks/mma` 与 `benchmarks/workloads`

MMA 版 `addmm` 不应重新发明一套格式，而应对齐仓库已有参考：

- tile 形状：`16x16x16`
- 数据格式：FP16 packed input
- 累加：FP32 vector accumulator
- 输出：FP32

优先复用的参考：

- `benchmarks/mma/m16n16k16_fp16_fp32/kernel.cl`
- `benchmarks/workloads/gelu_mlp_fp16/kernel.cl`
- `benchmarks/workloads/swiglu_mlp_fp16/kernel.cl`

### 3. Runtime 侧新增 MMA 资产，而不是覆盖普通 `addmm`

建议不要直接把当前 `addmm` kernel 资产整体替换为 MMA 版，而是：

- 保留当前 `addmm` 普通 GEMM 资产
- 新增 `addmm_mma` 或等价的 runtime 分支与资产
- 在 `canUseAddmmMmaKernel(...)` 命中时，走 MMA 资产

这样更利于调试，也方便做 A/B 验证。

### 4. 最小形态限制

第一版 MMA `addmm` 只支持：

- `float32` bias / output
- `float32` 输入张量在进入 kernel 前由 host 侧转换并 pack 为 FP16 tile
- `M/N/K` 为 16 的整数倍，或尾块先暂时回退
- contiguous dense layout

对 GPT-2 的含义是：

- 只有一部分线性层 shape 会先命中
- 但可以先证明 MMA 路线可行

### 5. 数值与精度策略

外部语义：

- 用户仍然看到 `addmm` 的 FP32 输出

内部策略：

- 输入 tile 转 FP16
- MMA 累加使用 FP32 accumulator
- 输出保持 FP32

这与“GPT-2 不要求全图 FP32，但关键累加尽量保持 FP32”这一部署目标一致。

## Files Likely Involved

- `ventus-pytorch/aten/src/ATen/ventus/BinaryOps.cpp`
- `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.h`
- `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`
- `ventus-pytorch/aten/src/ATen/ventus/kernels/src/*`
- `ventus-pytorch/aten/src/ATen/ventus/kernels/build-*.sh`
- `ventus-pytorch/aten/src/ATen/ventus/kernels/generate-*.py`
- `ventus-pytorch/testcases/`

## Validation Strategy

分三层验证：

1. MMA 微基准对齐验证
   - 与 `benchmarks/mma/m16n16k16_fp16_fp32` 的结果一致

2. `addmm` 算子级验证
   - shape 命中时确认走 MMA
   - 数值与 CPU 参考对齐

3. GPT-2 路径验证
   - `conv1d_addmm.py`
   - `gpt2_transformers_smoke.py`
   - `gpt2_transformers_cache_smoke.py`
   - `gpt2_transformers_decode_smoke.py`

## Risks

- host 侧 FP16 packing 与 benchmark 格式不完全一致时，数值可能错
- `M/N/K` 非 tile 整倍数的尾块策略如果没处理好，会导致大量场景回退
- LLVM builtin 降低到 MMA 指令的路径，和 benchmark 可以工作，不代表直接搬到 PyTorch kernel 资产后立即稳定
