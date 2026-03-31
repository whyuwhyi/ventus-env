__kernel void vfma_bf16x2(__global const uint *a, __global const uint *b,
                     __global const uint *c, __global uint *output) {
  int tid = get_global_id(0);
  output[tid] = __builtin_riscv_ventus_vfma_bf16x2(a[tid], b[tid], c[tid]);
}
