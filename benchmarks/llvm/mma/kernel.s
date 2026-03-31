    .text
    .globl bench_mma
bench_mma:
    mma.m8n8k16.row.row.f32.f16.f16.f32 v8, v4, v12
    mma.m16n8k16.row.col.f16.f16.f16.f16 v9, v5, v13
    mma.m16n16k16.col.col.f32.bf16.bf16.f32 v10, v6, v14
    ret
