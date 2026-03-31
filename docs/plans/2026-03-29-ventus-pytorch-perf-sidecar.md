# Ventus PyTorch Perf Sidecar Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the `ventus-pytorch` backend dump one perf JSON file after each `rtlsim` kernel launch, with a per-kernel launch index in the filename.

**Architecture:** Extend the Ventus PyTorch runtime's driver API to load `vt_dump_perf`, then add a runtime-side perf sidecar helper that runs after each synchronous `wait()`. The helper keeps a per-kernel launch counter in `DriverState`, writes `<kernel_name>_<launch_idx>.perf.json` into `VENTUS_BENCH_PERF_DIR` when `VENTUS_BENCH_ENABLE_PERF` is enabled, and is explicitly gated to `rtlsim`.

**Tech Stack:** C++17, ATen Ventus runtime, existing Ventus driver ABI, standard library filesystem/stdio.

---

### Task 1: Extend runtime state for perf sidecars

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/ventus_runtime_internal.h`
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`

**Step 1: Write the failing test**

No existing isolated unit-test target covers Ventus runtime internals. Use a focused build verification instead.

**Step 2: Run test to verify it fails**

Run: `rg -n "vt_dump_perf" ventus-pytorch/aten/src/ATen/ventus/ventus_runtime_internal.h`
Expected: no `DriverApi::vt_dump_perf` entry exists yet.

**Step 3: Write minimal implementation**

Add `vt_dump_perf` to `DriverApi`. Add per-kernel launch counters and a mutex to `DriverState`.

**Step 4: Run test to verify it passes**

Run: `rg -n "vt_dump_perf|perf_launch_counts" ventus-pytorch/aten/src/ATen/ventus/ventus_runtime_internal.h`
Expected: the new API slot and counter state are present.

**Step 5: Commit**

```bash
git add ventus-pytorch/aten/src/ATen/ventus/ventus_runtime_internal.h ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp
git commit -m "feat: add ventus pytorch perf sidecar state"
```

### Task 2: Dump per-launch perf JSON for rtlsim

**Files:**
- Modify: `ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`

**Step 1: Write the failing test**

No existing runtime harness exists. Use source-level verification plus a targeted runtime build if available.

**Step 2: Run test to verify it fails**

Run: `rg -n "dumpPerf|VENTUS_BENCH_ENABLE_PERF|perf.json" ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`
Expected: no PyTorch-side perf sidecar helper exists yet.

**Step 3: Write minimal implementation**

Load `vt_dump_perf`, add helpers for:
- env gating via `VENTUS_BENCH_ENABLE_PERF`
- backend gating to `rtlsim`
- perf directory resolution via `VENTUS_BENCH_PERF_DIR`
- filename format `<kernel_name>_<launch_idx>.perf.json`
- writing after each `wait()` in the common launch helpers

**Step 4: Run test to verify it passes**

Run: `rg -n "dumpPerfSidecarAfterLaunch|VENTUS_BENCH_ENABLE_PERF|perf.json" ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp`
Expected: helper and post-`wait()` hooks are present.

**Step 5: Commit**

```bash
git add ventus-pytorch/aten/src/ATen/ventus/VentusRuntime.cpp
git commit -m "feat: dump per-launch rtlsim perf sidecars for ventus pytorch"
```
