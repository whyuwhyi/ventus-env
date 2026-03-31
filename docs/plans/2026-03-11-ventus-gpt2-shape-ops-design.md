# Ventus GPT-2 Shape Ops Design

**Date:** 2026-03-11

**Goal:** 为 Ventus 后端补齐 `view`、`reshape`、`contiguous` 的第一阶段能力，作为后续跑通 GPT-2 inference 的基础。

## Background

当前 Ventus 后端已经有最小张量分配与部分数据搬运能力，但在 GPT 类模型常见的 shape/layout 链路上仍存在缺口。

已确认的失败现象：

- `view.py` 在 Ventus 上报 `aten::view` 缺失。
- `reshape.py` 内部同样会走到 `aten::view`，因此一起失败。
- `contiguous.py` 在 `transpose(...).contiguous()` 场景下卡在 `AutogradVentus` 分发路径。

这些算子都属于 GPT-2 / LLaMA / Mistral 一类 decoder-only Transformer 的高频基础算子，尤其是：

- QKV 投影后的 `view`
- 注意力前后的 `transpose`
- 输出投影前的 `contiguous().view(...)`

## Scope

本阶段只处理 dense strided Ventus tensor，范围严格限制为：

- `aten::view`
- `aten::reshape`
- `aten::contiguous`

明确不做：

- 不补一整套 shape-op 兼容层
- 不扩展到 `flatten`、`permute`、`transpose` 的新实现
- 不为了通过 testcase 改写 `view` 语义，使其在 stride 不兼容时偷偷 copy

## Design

### 1. `view`

在 Ventus 后端显式注册 `aten::view`。

行为目标与 PyTorch 原生一致：

- 若输入 tensor 的 size/stride 与目标形状兼容，则返回 metadata-only view。
- 若不兼容，则报出与原生接近的错误，而不是退化为 copy。

实现上优先复用现有 Ventus `as_strided` / `alias_with_sizes_and_strides` 逻辑，而不是新造一套 view 表示。

### 2. `reshape`

在 Ventus 后端显式注册 `aten::reshape`。

优先策略：

- 先走与原生 `reshape` 一致的“可 view 则 view”路径。
- 对本阶段最小目标来说，不主动扩展复杂 fallback。

这样可以确保 `reshape` 修复主要来自分发补齐，而不是额外引入 Ventus 专用语义。

### 3. `contiguous`

`Ventus` 已经有 `aten::contiguous` 注册，但 `transpose(...).contiguous()` 仍然会掉到 `AutogradVentus` 缺口，说明问题不只是 kernel 是否存在，还包括 dispatch / autograd glue。

本阶段目标：

- 保证转置后的 Ventus tensor 调用 `.contiguous()` 时可以正确命中可执行路径。
- 返回真正 contiguous 的 Ventus tensor。

修复顺序：

1. 先确认是否只需要补齐后端显式注册与 view 链路。
2. 若仍然卡在 `AutogradVentus`，则只修改最小必要的 dispatch glue，不做大范围 c10 重构。

## Files Likely Involved

- `ventus-pytorch/aten/src/ATen/ventus/Factory.cpp`
- `ventus-pytorch/aten/src/ATen/ventus/VentusAten.cpp`
- `ventus-pytorch/c10/core/*` 中与 Ventus dispatch / autograd key 相关的最小必要文件
- `ventus-pytorch/testcases/view.py`
- `ventus-pytorch/testcases/reshape.py`
- `ventus-pytorch/testcases/contiguous.py`

## Validation Strategy

分三层验证：

1. 算子级验证
   - `view.py`
   - `reshape.py`
   - `contiguous.py`

2. 最小 Transformer 路径验证
   - 覆盖 `view`
   - 覆盖 `transpose`
   - 覆盖 `contiguous`
   - 覆盖末尾 `view(..., dim)`

3. 继续推进到 GPT-2 inference 的最小前向
   - 本阶段不承诺一次性跑通完整生成
   - 目标是补齐第一批高频基础算子，为下一轮缺口收敛铺路

## Risks

- `contiguous` 的问题可能暴露更底层的 `AutogradVentus` key 路由缺陷。
- `reshape` 在部分 layout 下可能需要更多 fallback 才能覆盖真实模型路径。
- 即使这 3 个算子补齐，GPT-2 后续仍可能在别的基础算子上继续失败，但这不影响本阶段作为最小闭环。
