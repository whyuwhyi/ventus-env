__kernel void mma_m16n16k16_fp16_fp32_random(
    __global const uint *a_window,
    __global const uint *b_window,
    __global const float *c_window,
    __global float *d_window) {
  int tid = get_global_id(0);
  int a_base = tid * 4;
  int b_base = tid * 4;
  int c_base = tid * 8;

  uint4 a = (uint4)(a_window[a_base + 0], a_window[a_base + 1],
                    a_window[a_base + 2], a_window[a_base + 3]);
  uint4 b = (uint4)(b_window[b_base + 0], b_window[b_base + 1],
                    b_window[b_base + 2], b_window[b_base + 3]);
  float8 c = (float8)(c_window[c_base + 0], c_window[c_base + 1],
                      c_window[c_base + 2], c_window[c_base + 3],
                      c_window[c_base + 4], c_window[c_base + 5],
                      c_window[c_base + 6], c_window[c_base + 7]);

  float8 d = __builtin_riscv_ventus_mma_m16n16k16_row_col_f32_f16_f16_f32(a, b, c);
  ((__global float8 *)d_window)[tid] = d;
}
