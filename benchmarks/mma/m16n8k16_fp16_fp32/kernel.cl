__kernel void mma_m16n8k16_fp16_fp32(
    __global const uint *a_window,
    __global const uint *b_window,
    __global const float *c_window,
    __global float *d_window) {
  int tid = get_global_id(0);
  int a_base = tid * 4;
  int b_base = tid * 2;

  uint a0 = a_window[a_base + 0];
  uint a1 = a_window[a_base + 1];
  uint a2 = a_window[a_base + 2];
  uint a3 = a_window[a_base + 3];
  uint b0 = b_window[b_base + 0];
  uint b1 = b_window[b_base + 1];
  float c0 = c_window[a_base + 0];
  float c1 = c_window[a_base + 1];
  float c2 = c_window[a_base + 2];
  float c3 = c_window[a_base + 3];

  uint4 a = (uint4)(a0, a1, a2, a3);
  uint2 b = (uint2)(b0, b1);
  float4 c = (float4)(c0, c1, c2, c3);

  float4 d = __builtin_riscv_ventus_mma_m16n8k16_row_col_f32_f16_f16_f32(a, b, c);
  ((__global float4 *)d_window)[tid] = d;
}
