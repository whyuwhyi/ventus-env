    .text
    .globl bench_sfu
bench_sfu:
    vex2.approx.f32 v8, v4
    vlg2.approx.f32 v9, v5
    vrcp.approx.f32 v10, v6
    vsqrt.approx.f32 v11, v7
    vrsqrt.approx.f32 v12, v8
    vsin.approx.f32 v13, v9
    vcos.approx.f32 v14, v10
    vtanh.approx.f32 v15, v11
    vgelu.approx.f32 v16, v12
    vsilu.approx.f32 v17, v13
    vex2.approx.f16x2 v18, v14
    vrcp.approx.f16x2 v19, v15
    vsqrt.approx.f16x2 v20, v16
    vrsqrt.approx.f16x2 v21, v17
    vtanh.approx.f16x2 v22, v18
    vgelu.approx.f16x2 v23, v19
    vsilu.approx.f16x2 v24, v20
    vex2.approx.bf16x2 v25, v21
    vrcp.approx.bf16x2 v26, v22
    vsqrt.approx.bf16x2 v27, v23
    vrsqrt.approx.bf16x2 v28, v24
    vtanh.approx.bf16x2 v29, v25
    vgelu.approx.bf16x2 v30, v26
    vsilu.approx.bf16x2 v31, v27
    ret
