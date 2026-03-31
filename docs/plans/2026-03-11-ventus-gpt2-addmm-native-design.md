# Ventus GPT-2 Addmm Native Fast Path Design

**Date:** 2026-03-11

**Goal:** 为 Ventus 后端补齐 `aten::addmm.out` 的第一条原生 fast path，优先覆盖 GPT-2 `Conv1D` 的线性层调用，减少对 CPU bridge 的依赖。

## Background

当前 GPT-2 在 Ventus 上已经可以“功能跑通”，包括：

- `use_cache=False` 的最小前向
- `use_cache=True` 的 prefill
- 第二 token decode 复用 `past_key_values`

但这条链路并不是“原生 Ventus 计算”：

- `transformers` 里的 GPT-2 `Conv1D.forward()` 直接调用 `torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)`
- Ventus 当前的 `aten::addmm.out` 实现会把 `self / mat1 / mat2` 全部搬到 CPU，调用 `at::addmm(...)`，再把结果拷回 Ventus
- 这保证了功能闭环，但不是真正的 Ventus GEMM 路径，更谈不上 MMA

因此，下一阶段最小且高价值的目标，不是一次补全 `mm / bmm / matmul`，而是先把 GPT-2 一定会命中的 `addmm` 拉回 Ventus。

## Current Constraints

现有 Ventus runtime 已经有较完整的 kernel artifact 工作流：

- `aten/src/ATen/ventus/kernels/src/*.cl` 作为 source of truth
- `build-*.sh` 负责编译 `object.riscv`
- `generate-*.py` 负责生成 `.metadata` / `.data`
- runtime 优先从 `torch/share/ventus/kernels/<op>/<backend>/<profile>/` 解析产物

已确认的约束：

- `torch/share/ventus/kernels/` 下没有现成的 `addmm` / `gemm` 产物
- runtime 的 source fallback 也不是直接执行 `.cl`，仍然要求 `object.riscv + .metadata + .data`
- 仓库里虽然有 `testcases/_get_case/AIops/GEMM/gemm.cl` 和 `gemm_fused.cl`，但它们当前没有接进 ATen Ventus kernel artifact 流程

这意味着本阶段不能只写 C++ dispatch，还必须把 `addmm` 接进现有 kernel artifact 生成链路。

## Approaches Considered

### 1. 保持现有 CPU bridge

优点：

- 零风险
- 不需要新增内核资产

缺点：

- 对“真正在 Ventus 上跑 GPT-2”没有推进
- 线性层仍全部绕回 CPU

### 2. 只为 `aten::addmm.out` 增加 Ventus 原生 fast path

优点：

- 范围最小
- 直接命中 GPT-2 `Conv1D`
- 可以复用现有 Ventus kernel artifact 工作流

缺点：

- attention 真正高成本的 `bmm/mm` 仍未解决

### 3. 一次抽象 `addmm + mm + bmm` 通用 GEMM 栈

优点：

- 长期结构更合理
- 更接近后续 MMA 路线

缺点：

- 本轮范围明显过大
- 验证面会急剧扩大

## Recommendation

推荐采用方案 2：

- 只做 `aten::addmm.out`
- 先拿到 GPT-2 第一条真正的 Ventus 线性层路径
- 其他形态继续保留 CPU bridge，避免本轮失控

## Scope

本阶段只覆盖 GPT-2 `Conv1D` 的常见 `addmm` 形态：

- `mat1` 是 2D dense strided tensor
- `mat2` 是 2D dense strided tensor
- `self` 是 1D bias，长度等于输出列数，或已是输出同形张量
- `out` 在 Ventus
- dtype 先只支持 `float32`
- `alpha` / `beta` 先只支持可以安全落到 kernel 标量参数的常见标量路径

明确不做：

- 不补 `mm`
- 不补 `bmm`
- 不补一般 batched matmul
- 不补量化 / 稀疏 / 混合 layout
- 不做 MMA 最终抽象

## Design

### 1. 双路径 `addmm.out`

`aten::addmm.out` 改成双路径：

- Fast path：命中本阶段支持形态时，直接走 Ventus 原生 kernel
- Slow path：不满足条件时，继续沿用当前 CPU bridge

这样可以保证：

- GPT-2 命中的主要线性层路径先被原生化
- 其他复杂 `addmm` 形态先不炸
- 当前已打通的 GPT-2 功能路径不回退

### 2. Runtime 层新增 `addmm` kernel 接口

在 `VentusRuntime.h/.cpp` 中新增：

- `canUseAddmmKernel(...)`
- `launchAddmm(...)`

理由：

- kernel 产物解析、launch descriptor、buffer staging 本来就集中在 runtime 层
- 不应该把硬件 launch 细节塞进 `BinaryOps.cpp`
- 也方便后续 `mm/bmm` 继续沿同一套模式扩展

### 3. 内核资产接入现有 artifact 流程

新增 `addmm` 的内核资产布局，复用现有模式：

- `aten/src/ATen/ventus/kernels/src/addmm.cl`
- `aten/src/ATen/ventus/kernels/build-addmm-artifact.sh`
- `aten/src/ATen/ventus/kernels/generate-addmm-artifact.py`
- 安装目标：
  `torch/share/ventus/kernels/addmm/<backend>/<profile>/`

kernel 语义以 `C = beta * self + alpha * (mat1 @ mat2)` 为目标，但实现只需优先保证 GPT-2 `Conv1D` 会命中的常见形态正确。

### 4. Kernel 语义优先于通用性

第一版不追求完美的 BLAS 兼容，只追求：

- GPT-2 `Conv1D` 输入形态正确
- 输出 shape 正确
- 数值与 CPU `addmm` 足够接近

如果某些 stride / broadcast / dtype 不满足 fast path 假设，则由 `canUseAddmmKernel(...)` 直接拒绝，回到 CPU bridge。

## Files Likely Involved

- `ventus-pytorch/aten/src/ATen/ventus/BinaryOps.cpp`
- `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.h`
- `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`
- `ventus-pytorch/aten/src/ATen/ventus/kernels/src/addmm.cl`
- `ventus-pytorch/aten/src/ATen/ventus/kernels/build-addmm-artifact.sh`
- `ventus-pytorch/aten/src/ATen/ventus/kernels/generate-addmm-artifact.py`
- `ventus-pytorch/testcases/` 下新增 `addmm` 和 `Conv1D` 最小 testcase

## Validation Strategy

分三层验证：

1. `addmm` 最小算子级验证
   - 最小 2D `mat1 @ mat2 + bias`
   - 对齐 CPU 数值结果

2. GPT-2 线性层验证
   - 复用 `transformers.Conv1D`
   - 覆盖 `x.view(-1, hidden) -> addmm -> view(size_out)`

3. GPT-2 真实路径回归
   - `gpt2_transformers_smoke.py`
   - `gpt2_transformers_cache_smoke.py`
   - `gpt2_transformers_decode_smoke.py`

## Risks

- 第一版 `addmm` kernel 很可能只覆盖 `float32` 和最简单 layout
- `spike/rtlsim/cyclesim` 三种 backend/profile 的 artifact 生成可能暴露额外工具链问题
- 即使 `addmm` 原生化成功，GPT-2 attention 仍会卡在 `mm/bmm` 路径，这属于下一阶段问题
