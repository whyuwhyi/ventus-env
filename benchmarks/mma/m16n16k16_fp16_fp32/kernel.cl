__kernel void mma_m16n16k16_fp16_fp32(
    __global const uint *a_window,
    __global const uint *b_window,
    __global const float *c_window,
    __global float *d_window) {
  int tid = get_global_id(0);
  int a_base = tid * 4;
  int b_base = tid * 4;
  int c_base = tid * 8;

  uint a0 = a_window[a_base + 0];
  uint a1 = a_window[a_base + 1];
  uint a2 = a_window[a_base + 2];
  uint a3 = a_window[a_base + 3];
  uint b0 = b_window[b_base + 0];
  uint b1 = b_window[b_base + 1];
  uint b2 = b_window[b_base + 2];
  uint b3 = b_window[b_base + 3];
  float c0 = c_window[c_base + 0];
  float c1 = c_window[c_base + 1];
  float c2 = c_window[c_base + 2];
  float c3 = c_window[c_base + 3];
  float c4 = c_window[c_base + 4];
  float c5 = c_window[c_base + 5];
  float c6 = c_window[c_base + 6];
  float c7 = c_window[c_base + 7];

  uint4 a = (uint4)(a0, a1, a2, a3);
  uint4 b = (uint4)(b0, b1, b2, b3);
  float8 c = (float8)(c0, c1, c2, c3, c4, c5, c6, c7);

  float8 d = __builtin_riscv_ventus_mma_m16n16k16_row_col_f32_f16_f16_f32(a, b, c);
  ((__global float8 *)d_window)[tid] = d;
}
