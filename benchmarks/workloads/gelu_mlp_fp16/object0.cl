#define MLP_D 96
#define MLP_H 384
#define K_TILES (MLP_D / 16)
#define H_TILES (MLP_H / 16)
#define O_TILES (MLP_D / 16)

static inline uint4 load_u4(__global const uint *buf, int tileIdx, int tid) {
  int base = (tileIdx * 32 + tid) * 4;
  return (uint4)(buf[base + 0], buf[base + 1], buf[base + 2], buf[base + 3]);
}

static inline void store_f8(__global float *buf, int tileIdx, int tid, float8 v) {
  ((__global float8 *)buf)[tileIdx * 32 + tid] = v;
}

static inline float8 gelu8(float8 x) {
  return (float8)(
      __builtin_riscv_ventus_vgelu_approx_f32(x.s0),
      __builtin_riscv_ventus_vgelu_approx_f32(x.s1),
      __builtin_riscv_ventus_vgelu_approx_f32(x.s2),
      __builtin_riscv_ventus_vgelu_approx_f32(x.s3),
      __builtin_riscv_ventus_vgelu_approx_f32(x.s4),
      __builtin_riscv_ventus_vgelu_approx_f32(x.s5),
      __builtin_riscv_ventus_vgelu_approx_f32(x.s6),
      __builtin_riscv_ventus_vgelu_approx_f32(x.s7));
}

static inline uint4 fp32x8_to_fp16x8(float8 x) {
  uint h0 = __builtin_riscv_ventus_vcvt_fp16_fp32(x.s0) & 0xffffu;
  uint h1 = __builtin_riscv_ventus_vcvt_fp16_fp32(x.s1) & 0xffffu;
  uint h2 = __builtin_riscv_ventus_vcvt_fp16_fp32(x.s2) & 0xffffu;
  uint h3 = __builtin_riscv_ventus_vcvt_fp16_fp32(x.s3) & 0xffffu;
  uint h4 = __builtin_riscv_ventus_vcvt_fp16_fp32(x.s4) & 0xffffu;
  uint h5 = __builtin_riscv_ventus_vcvt_fp16_fp32(x.s5) & 0xffffu;
  uint h6 = __builtin_riscv_ventus_vcvt_fp16_fp32(x.s6) & 0xffffu;
  uint h7 = __builtin_riscv_ventus_vcvt_fp16_fp32(x.s7) & 0xffffu;
  return (uint4)(h0 | (h1 << 16),
                 h2 | (h3 << 16),
                 h4 | (h5 << 16),
                 h6 | (h7 << 16));
}

__kernel void gelu_mlp_fp16(__global const uint *x_tiles,
                            __global const uint *w1_tiles,
                            __global const uint *w2_tiles,
                            __global float *out_tiles) {
  int tid = get_global_id(0);
  float8 outputs[O_TILES];
  for (int o = 0; o < O_TILES; ++o)
    outputs[o] = (float8)(0.0f);

  for (int h = 0; h < H_TILES; ++h) {
    float8 hidden = (float8)(0.0f);
    for (int k = 0; k < K_TILES; ++k) {
      uint4 a = load_u4(x_tiles, k, tid);
      uint4 b = load_u4(w1_tiles, h * K_TILES + k, tid);
      hidden = __builtin_riscv_ventus_mma_m16n16k16_row_col_f32_f16_f16_f32(a, b, hidden);
    }
    float8 act = gelu8(hidden);
    uint4 act_fp16 = fp32x8_to_fp16x8(act);
    for (int o = 0; o < O_TILES; ++o) {
      uint4 b2 = load_u4(w2_tiles, h * O_TILES + o, tid);
      outputs[o] = __builtin_riscv_ventus_mma_m16n16k16_row_col_f32_f16_f16_f32(act_fp16, b2, outputs[o]);
    }
  }

  for (int o = 0; o < O_TILES; ++o)
    store_f8(out_tiles, o, tid, outputs[o]);
}
