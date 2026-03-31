# Ventus GPT-2 Addmm Native Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 Ventus 后端补齐 `aten::addmm.out` 的原生 fast path，先覆盖 GPT-2 `Conv1D` 线性层。

**Architecture:** 保留当前 `addmm.out` CPU bridge 作为慢路径，同时在 `BinaryOps.cpp` 中新增 Ventus fast path 分流，具体 kernel 选择和 launch 逻辑下沉到 `VentusRuntime.cpp`。内核资产完全复用现有 `kernels/src/*.cl + build-*.sh + generate-*.py + torch/share/ventus/kernels/<op>/...` 工作流，只新增 `addmm` 这一套。

**Tech Stack:** PyTorch C++ dispatcher, ATen Ventus backend, Ventus runtime artifact pipeline, OpenCL kernel source, Ninja, Python testcase

---

### Task 1: 补最小 `addmm` testcase

**Files:**
- Create: `ventus-pytorch/testcases/addmm.py`

**Step 1: 写最小算子级脚本**

脚本要求：

- 在 CPU 上构造 `bias`, `mat1`, `mat2`
- 拷到 Ventus
- 调用 `torch.addmm(bias, mat1, mat2)`
- 回 CPU 与 `torch.addmm` CPU 基准比对

**Step 2: 运行确认当前行为**

Run: `source env.sh && python ventus-pytorch/testcases/addmm.py`
Expected: 目前大概率已经 PASS，但它只证明功能正确，不证明走了原生路径

**Step 3: 保留脚本作为后续数值回归**

该脚本不是本轮唯一红测，但必须作为原生化后的基础数值回归。

### Task 2: 补最小 GPT `Conv1D` 线性层 testcase

**Files:**
- Create: `ventus-pytorch/testcases/conv1d_addmm.py`

**Step 1: 写最小 `transformers.Conv1D` 路径脚本**

脚本要求：

- 构造一个极小 `Conv1D(nf, nx)`
- 输入 `x` shape 为 `[batch, seq, hidden]`
- 走 `Conv1D.forward()`，实际命中 `torch.addmm(...)`
- 输出转回 CPU 对齐 CPU 基准

**Step 2: 运行确认基线行为**

Run: `source env.sh && python ventus-pytorch/testcases/conv1d_addmm.py`
Expected: 当前 PASS，但本质仍走 CPU bridge

**Step 3: 用它作为 GPT-2 线性层代理回归**

这个 testcase 是比整条 GPT-2 smoke 更便宜的 addmm 代理。

### Task 3: 为 `addmm` 新增 kernel 资产骨架

**Files:**
- Create: `ventus-pytorch/aten/src/ATen/ventus/kernels/src/addmm.cl`
- Create: `ventus-pytorch/aten/src/ATen/ventus/kernels/build-addmm-artifact.sh`
- Create: `ventus-pytorch/aten/src/ATen/ventus/kernels/generate-addmm-artifact.py`

**Step 1: 参考已有 kernel 资产模式**

读取参考：

- `ventus-pytorch/aten/src/ATen/ventus/kernels/build-vecadd-artifact.sh`
- `ventus-pytorch/aten/src/ATen/ventus/kernels/generate-vecadd-artifact.py`
- `ventus-pytorch/aten/src/ATen/ventus/kernels/src/vecadd.cl`

**Step 2: 写 `addmm.cl` 最小 kernel**

要求：

- 只支持 2D `mat1 @ mat2`
- 输出 `out[m, n] = beta * self[n] + alpha * sum_k(mat1[m, k] * mat2[k, n])`
- 先按 `float32` 写

**Step 3: 写 `generate-addmm-artifact.py`**

要求：

- 生成 `object.riscv`, `addmm_0.metadata`, `addmm_0.data`
- metadata/data 布局遵循现有 runtime 读取方式
- 参数缓冲区要能表达 `self`, `mat1`, `mat2`, `out`, `M`, `K`, `N`, `alpha`, `beta`

**Step 4: 写 `build-addmm-artifact.sh`**

要求：

- 复用 `build-vecadd-artifact.sh` 的环境约定
- 安装到 `torch/share/ventus/kernels/addmm/<backend>/<profile>/`

### Task 4: 在 runtime 层接入 `addmm`

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.h`
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`

**Step 1: 新增 runtime 接口声明**

新增：

- `canUseAddmmKernel(...)`
- `launchAddmm(...)`

**Step 2: 复用 artifact 解析/上传模式**

参考现有：

- `resolveVecAddArtifacts(...)`
- `prepareVecAddArtifactsReadyLocked()`
- `launchAdd(...)`

**Step 3: 加入 `addmm` 的 artifact 解析分支**

要求：

- 支持 installed artifacts
- 支持 `VENTUS_SOURCE_KERNEL_FALLBACK=1` 时的 source-tree fallback

**Step 4: 写最小 `canUseAddmmKernel(...)`**

先只允许：

- `float32`
- `self` 是 1D bias 或输出同形
- `mat1`/`mat2` 是 2D dense strided
- 输出 shape 合法

### Task 5: 在 `BinaryOps.cpp` 中切换 `addmm.out` 双路径

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/BinaryOps.cpp`

**Step 1: 保留当前 CPU bridge 代码**

不要删除当前 CPU bridge 逻辑，只把它收成慢路径。

**Step 2: 在 `addmm_out_tensor(...)` 中先判定 fast path**

逻辑：

- 输入 device/type/shape 校验
- 若 `canUseAddmmKernel(...)` 为真，则直接 `launchAddmm(...)`
- 否则继续走现有 CPU bridge

**Step 3: 确保 `out` 直接在 Ventus 上写入**

避免 fast path 再多做一次 CPU staging。

### Task 6: 构建 `addmm` kernel 产物

**Files:**
- Modify: none

**Step 1: 运行 artifact build**

Run: `source env.sh && bash ventus-pytorch/aten/src/ATen/ventus/kernels/build-addmm-artifact.sh`
Expected: 在 `ventus-pytorch/torch/share/ventus/kernels/addmm/<backend>/<profile>/` 下生成三件套

**Step 2: 检查产物是否存在**

Run: `find ventus-pytorch/torch/share/ventus/kernels/addmm -maxdepth 4 -type f`
Expected: 至少包含 `object.riscv`, `addmm_0.metadata`, `addmm_0.data`

### Task 7: 重建 PyTorch 运行时

**Files:**
- Modify: none

**Step 1: 重新编译**

Run: `env CCACHE_DISABLE=1 ninja -C ventus-pytorch/build torch_python`
Expected: build succeeds

**Step 2: 同步运行时库**

Run:

- `cp -f ventus-pytorch/build/lib/libtorch_cpu.so ventus-pytorch/torch/lib/libtorch_cpu.so`
- `cp -f ventus-pytorch/build/lib/libtorch.so ventus-pytorch/torch/lib/libtorch.so`
- `cp -f ventus-pytorch/build/lib/libtorch_python.so ventus-pytorch/torch/lib/libtorch_python.so`
- `cp -f ventus-pytorch/build/lib/libshm.so ventus-pytorch/torch/lib/libshm.so`

Expected: runtime libs updated

### Task 8: 验证 `addmm` 和 `Conv1D`

**Files:**
- Modify: none

**Step 1: 跑最小 `addmm` testcase**

Run: `source env.sh && python ventus-pytorch/testcases/addmm.py`
Expected: PASS

**Step 2: 跑最小 `Conv1D` testcase**

Run: `source env.sh && python ventus-pytorch/testcases/conv1d_addmm.py`
Expected: PASS

**Step 3: 如有需要，观察日志确认不再只依赖 CPU bridge**

重点检查：

- 不应再只有 `self.cpu()/mat1.cpu()/mat2.cpu()` 这一路
- `addmm` artifact 应能被解析和 launch

### Task 9: 跑 GPT-2 真实路径回归

**Files:**
- Modify: none

**Step 1: 跑无 cache 前向**

Run: `source env.sh && python ventus-pytorch/testcases/gpt2_transformers_smoke.py`
Expected: PASS

**Step 2: 跑 cache prefill**

Run: `source env.sh && python ventus-pytorch/testcases/gpt2_transformers_cache_smoke.py`
Expected: PASS

**Step 3: 跑 decode**

Run: `source env.sh && python ventus-pytorch/testcases/gpt2_transformers_decode_smoke.py`
Expected: PASS

### Task 10: 跑既有基础回归

**Files:**
- Modify: none

**Step 1: 跑 shape/layout 基础回归**

Run:

- `source env.sh && python ventus-pytorch/testcases/view.py`
- `source env.sh && python ventus-pytorch/testcases/reshape.py`
- `source env.sh && python ventus-pytorch/testcases/contiguous.py`
- `source env.sh && python ventus-pytorch/testcases/gpt_shape_path.py`

Expected: all PASS

### Task 11: 记录下一阶段输入

**Files:**
- Modify: none

**Step 1: 明确剩余矩阵乘缺口**

记录：

- GPT-2 attention 里还剩哪些 `mm/bmm` 路径
- 当前 `addmm` fast path 的 dtype/layout 限制

**Step 2: 把下一阶段收敛成 `mm/bmm` 计划输入**

不要在本轮继续扩到 `bmm/mm` 实现。
