__kernel void vadd_f16x2(__global const uint *a, __global const uint *b, __global uint *output) {
  int tid = get_global_id(0);
  output[tid] = __builtin_riscv_ventus_vadd_f16x2(a[tid], b[tid]);
}
