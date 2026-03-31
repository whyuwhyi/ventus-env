# Ventus Shared GEMM Launcher Design

**Date:** 2026-03-19

**Goal:** 把 Ventus 上分散的 `mm.out`、`addmm.out`、`bmm.out` 收敛到同一个 runtime GEMM primitive，先消灭 attention 主路径里 `linear -> mm.out` 的 CPU fallback，同时不给后续 MMA 路线继续制造重复代码。

## Background

当前 Ventus 在 attention / GPT-2 bring-up 里已经补了一批关键算子，但 GEMM 家族仍然是分裂状态：

- `aten::addmm.out` 已经有 Ventus fast path
- `aten::addmm.out` 还保留一套独立的 MMA fast path
- `aten::bmm.out` 仍主要走 `VentusFallback.cpp` 里的 CPU fallback
- `aten::mm.out` 还没有真正挂上 Ventus runtime

这会直接导致：

- `torch.nn.functional.linear` 走到 `aten::mm.out` 时仍可能 fallback
- `addmm / bmm / mm` 三条路径分别维护参数校验、artifact 选择、metadata 打包和 launch 逻辑
- 后续再补 attention、matmul、batched linear 时，代码会继续散开

## Scope

这一轮只做 GEMM family 的 runtime 收敛，不额外扩大战场：

- 新增 shared GEMM launcher
- 接上 `aten::mm.out`
- 让 `aten::addmm.out` 改为走 shared launcher
- 让 `aten::bmm.out` 退出 CPU fallback，改为走 shared launcher

明确不做：

- 不改 Chisel / driver / benchmark
- 不新增新的 ISA 能力
- 不做通用 `matmul`
- 不引入任何 CPU fallback 作为兜底

## Constraints

- 只能修改 `ventus-pytorch/` 下的文件
- 需要尽可能复用现有 `addmm` / `addmm_mma` 资产
- 不允许 required path 悄悄退回 CPU
- 需要保留后续把 MMA 继续做强的空间
- 编译由用户执行，助手只给出需要执行的命令

## Approaches Considered

### 1. 先只补 `mm.out`

优点：

- 改动最小
- 最快解除 `linear` 阻塞

缺点：

- `addmm` / `bmm` 继续各自为政
- 很快又要做第二轮重构

### 2. 先做 shared runtime primitive，再把 `mm/addmm/bmm` 三个入口一起收敛

优点：

- 一次解决结构散乱问题
- `mm`、`addmm`、`bmm` 后续都能复用同一套 artifact / launch / validation
- 最符合后续 attention 路径的扩展方向

缺点：

- 这一轮改动会比只补 `mm` 更大

### 3. 彻底做成“一个 kernel 覆盖所有 GEMM 变体”

优点：

- 长期最整洁

缺点：

- 范围过大
- 当前没有必要

## Recommendation

选择方案 2。

也就是：

- 统一 runtime primitive
- 保留必要的 kernel 资产分支
- 先解决入口层和 launch 层的重复问题

这样既能立即解除 `mm.out` 阻塞，又不会把 kernel 目录抽象工作提前做过头。

## Design

### 1. 入口层只做归一化和强校验

`BinaryOps.cpp` 中新增或改造：

- `mm.out`
- `addmm.out`
- `bmm.out`

三者只负责：

- 输入输出必须在 Ventus
- shape / dtype / contiguous / storage offset 校验
- 归一化成统一 GEMM request
- 调用 `VentusRuntime` 的 shared GEMM launcher

如果 request 当前不满足 Ventus kernel 约束，直接 `TORCH_CHECK`，而不是退回 CPU。

### 2. Runtime 引入统一 GEMM request

在 `VentusRuntime.{h,cpp}` 中新增统一抽象，例如：

- `enum class GemmKind { Mm, Addmm, Bmm }`
- `struct GemmRequest`
- `bool canUseGemm(...)`
- `void launchGemm(...)`

`GemmRequest` 至少要包含：

- kind
- self / bias 是否存在
- mat1 / mat2
- out
- `beta / alpha`
- batch、`M/N/K`
- 是否允许走 MMA fast path

这样 `mm/addmm/bmm` 的共性逻辑就能合到一个地方。

### 3. 共享的非 MMA GEMM 路径

当前普通 `addmm` kernel 已经可用，第一步不推翻它，而是把它扩成 shared GEMM 基线：

- `mm` 视为 `beta=0`、无 bias 的特例
- `addmm` 视为带 bias 的 2D GEMM
- `bmm` 视为 batched 版本

如果现有 plain GEMM kernel 还不能直接吃 batch 维，则先由 runtime 在 batch 维上循环 launch，同样不允许 CPU fallback。

### 4. MMA 仍作为 shared launcher 内部的一个分支

shared launcher 内部保留 MMA 分流，但只在满足当前约束时触发：

- dtype 命中 MMA 约束
- layout / shape 命中当前 kernel 假设
- 需要时命中 `VENTUS_ADDMM_REQUIRE_MMA=1`

也就是说，shared launcher 统一的是“入口和调度”，不是强行把所有 kernel 资产物理合并成一个文件。

### 5. Artifact 解析与缓存也要统一

当前 runtime 对 `addmm`、`addmm_mma` 有各自的 artifact state。重构后需要把 GEMM 家族统一成一套更清楚的组织方式：

- plain gemm artifacts
- mma gemm artifacts

shared launcher 依据 request 和 profile 选择相应 artifacts，并复用同一套：

- profile 解析
- hardware cap 校验
- metadata buffer 准备
- kernel launch

### 6. `bmm.out` 退出 fallback

`VentusFallback.cpp` 里的 `try_handle_bmm_out` 需要移除出主路径。  
`bmm.out` 应该像 `addmm.out` 一样在 `TORCH_LIBRARY_IMPL(aten, Ventus, m)` 中显式注册。

目标是：

- GEMM family 在 Ventus 下有明确的原生入口
- fallback 只留给暂时完全没实现的非关键算子

## File Map

主要涉及：

- `ventus-pytorch/aten/src/ATen/ventus/BinaryOps.cpp`
- `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.h`
- `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`
- `ventus-pytorch/aten/src/ATen/ventus/VentusFallback.cpp`
- `ventus-pytorch/testcases/`

大概率新增：

- `ventus-pytorch/testcases/mm.py`
- `ventus-pytorch/testcases/linear_mm.py`

## Validation Strategy

按从低到高三层验证：

### 1. 单算子

- `addmm.py`
- `addmm_mma.py`
- `addmm_mma_tail.py`
- `bmm.py`
- 新增 `mm.py`

### 2. 模块级

- 新增 `linear_mm.py`

确认 `linear -> mm.out` 已经留在 Ventus 路径。

### 3. 主线 smoke

- `gpt2_transformers_cache_smoke.py`
- `gpt2_transformers_decode_smoke.py`

这一步的目标不是“一次跑完 GPT-2”，而是继续缩小 attention 主路径上的剩余缺口。

## Expected Outcome

完成后应达到：

- `mm.out` 不再因为“未注册 Ventus kernel”而 fallback
- `addmm.out`、`mm.out`、`bmm.out` 共用同一个 runtime GEMM primitive
- `bmm.out` 不再依赖 `VentusFallback.cpp`
- 后续继续补 attention / linear / matmul 时，有统一的 GEMM 扩展落点

## Compile Handoff

本设计默认：

- 代码修改由助手完成
- 编译与产物同步由用户执行

需要重新编译时，助手只通知，不自行运行编译命令。
