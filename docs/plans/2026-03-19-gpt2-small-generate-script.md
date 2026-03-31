# GPT-2 Small Generate Script Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 `ventus-pytorch/checkpoints/gpt2/` 下新增一个最小 `run_generate.py` 脚本，直接加载本地真实预训练 `GPT-2 small` checkpoint，并在 Ventus 上执行一次单 batch、greedy、固定 `max_new_tokens` 的标准生成。

**Architecture:** 脚本只依赖本地 checkpoint 目录中的 `config/tokenizer/model.safetensors` 文件，使用 `transformers` 标准加载与 `model.generate()`，并把输入 prompt、token ids、输出文本直接打印出来。失败时不吞异常，保持主线诊断透明。

**Tech Stack:** Python, Transformers, local checkpoint loading, Ventus device backend

---

### Task 1: 写最小生成脚本

**Files:**
- Create: `ventus-pytorch/checkpoints/gpt2/run_generate.py`

**Step 1: 写脚本主体**

脚本要求：

- 默认 checkpoint 根目录为脚本自身所在目录
- 默认 prompt 为一行短文本
- 支持命令行覆盖 prompt
- `local_files_only=True`
- `device='ventus'`
- `do_sample=False`
- `max_new_tokens=8`

**Step 2: 打印关键信息**

打印：

- checkpoint 路径
- prompt
- prompt ids
- output ids
- decoded text

### Task 2: 做最小本地语法验证

**Files:**
- Test: `ventus-pytorch/checkpoints/gpt2/run_generate.py`

**Step 1: 运行语法检查**

Run: `python -m py_compile ventus-pytorch/checkpoints/gpt2/run_generate.py`

Expected: PASS

### Task 3: 首次运行交给用户

**Files:**
- Modify: none

**Step 1: 给出运行命令**

Run:

- `source env.sh && python ventus-pytorch/checkpoints/gpt2/run_generate.py`
- 或 `source env.sh && python ventus-pytorch/checkpoints/gpt2/run_generate.py "Hello, my name is"`

Expected:

- 成功打印输入/输出 token ids 与生成文本
- 若失败，抛出第一处真实异常
