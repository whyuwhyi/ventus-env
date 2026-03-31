# GPT-2 Small Generate Script Design

**Date:** 2026-03-19

**Goal:** 在本地 `ventus-pytorch/checkpoints/gpt2/` 目录下放一个最小可运行脚本，直接加载真实预训练 `GPT-2 small` 权重，在 Ventus 上执行一次标准 `generate()` 路径，并打印输入/输出 token 与解码文本。

## Scope

这一步只做最小直跑脚本，不引入新的测试框架抽象：

- 脚本路径固定为 `ventus-pytorch/checkpoints/gpt2/run_generate.py`
- 默认从同目录 checkpoint 文件加载
- 使用 `transformers` 标准 `AutoTokenizer` 和 `GPT2LMHeadModel`
- 单 batch
- greedy decode
- 固定 `max_new_tokens`

明确不做：

- 不做采样策略扩展
- 不做 batch>1
- 不做性能优化
- 不做 dtype 切换

## Design

### 1. 本地 checkpoint 自包含

脚本默认使用自身所在目录作为 checkpoint 根目录，这样用户直接 `cd ventus-pytorch/checkpoints/gpt2 && python run_generate.py` 就能运行。

### 2. 标准生成接口

脚本使用 `model.generate(...)`，而不是手写 decode loop。这样更接近真实用户调用方式，也能直接覆盖 `transformers` 生成链路上的实际问题。

### 3. 最小参数面

脚本支持：

- 默认 prompt
- 命令行覆盖 prompt
- 默认 `max_new_tokens=8`

其余参数保持固定，避免把脚本写成一个 CLI 工具。

### 4. 输出内容

脚本打印：

- checkpoint 目录
- prompt 文本
- prompt token ids
- output token ids
- 解码文本

### 5. 失败处理

如果生成失败，直接抛出真实异常，便于继续沿主线补算子/运行时问题，不在脚本里吞错误或做复杂恢复。
