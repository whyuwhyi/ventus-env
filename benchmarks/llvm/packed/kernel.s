    .text
    .globl bench_packed
bench_packed:
    vadd.f16x2 v8, v4, v12
    vmul.f16x2 v9, v5, v13
    vfma.f16x2 v10, v6, v14
    vadd.bf16x2 v11, v7, v15
    vmul.bf16x2 v12, v8, v16
    vfma.bf16x2 v13, v9, v17
    ret
