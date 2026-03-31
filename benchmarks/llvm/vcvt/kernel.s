    .text
    .globl bench_vcvt
bench_vcvt:
    vcvt.fp32.fp16 v8, v4
    vcvt.fp16.fp32 v9, v5
    vcvt.fp32.bf16 v10, v6
    vcvt.bf16.fp32 v11, v7
    ret
