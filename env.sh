DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export VENTUS_INSTALL_PREFIX=${VENTUS_INSTALL_PREFIX:-${DIR}/install}
export VENTUS_PYTORCH_DIR=${VENTUS_PYTORCH_DIR:-${DIR}/ventus-pytorch}
export VENTUS_PYTORCH_VENV=${VENTUS_PYTORCH_VENV:-${VENTUS_PYTORCH_DIR}/.venv}
export PATH=${VENTUS_INSTALL_PREFIX}/bin:$PATH
export LD_LIBRARY_PATH=${VENTUS_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH:-}
export POCL_DEVICES="ventus"
export OCL_ICD_VENDORS=${VENTUS_INSTALL_PREFIX}/lib/libpocl.so
export POCL_CACHE_DIR=${POCL_CACHE_DIR:-${DIR}/.pocl-cache}

# see https://pcn2po10nqam.feishu.cn/wiki/XHNXwIdRkiFtZDkCW6Mc58Uon3b
export POCL_ENABLE_UNINIT=1

# remove extra colons
export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | sed -e 's/^:*//' -e 's/:*$//')

export NUM_WARP=2
export NUM_THREAD=32
export VENTUS_BACKEND=${VENTUS_BACKEND:-rtlsim}
export LUT_PATH=${LUT_PATH:-${DIR}/spike/dependencies/unfu/lut}
export VENTUS_KERNEL_PROFILE=${VENTUS_KERNEL_PROFILE:-${NUM_WARP}w${NUM_THREAD}t}
export VENTUS_BUILD_KERNELS=${VENTUS_BUILD_KERNELS:-1}
export VENTUS_KERNEL_BUILD_STRICT=${VENTUS_KERNEL_BUILD_STRICT:-0}
export VENTUS_SOURCE_KERNEL_FALLBACK=${VENTUS_SOURCE_KERNEL_FALLBACK:-0}
export BUILD_TYPE=${BUILD_TYPE:-RelWithDebInfo}
export LLVM_ENABLE_ASSERTIONS=${LLVM_ENABLE_ASSERTIONS:-ON}
export LLVM_ENABLE_EXPENSIVE_CHECKS=${LLVM_ENABLE_EXPENSIVE_CHECKS:-ON}
export CLANG_TOOLING_BUILD_AST_INTROSPECTION=${CLANG_TOOLING_BUILD_AST_INTROSPECTION:-OFF}
export USE_CUDA=${USE_CUDA:-0}
export USE_ROCM=${USE_ROCM:-0}
export USE_XPU=${USE_XPU:-0}
export USE_MKLDNN=${USE_MKLDNN:-0}
export USE_FBGEMM=${USE_FBGEMM:-0}
export USE_NNPACK=${USE_NNPACK:-0}
export USE_PYTORCH_QNNPACK=${USE_PYTORCH_QNNPACK:-0}
export USE_XNNPACK=${USE_XNNPACK:-0}
export USE_DISTRIBUTED=${USE_DISTRIBUTED:-0}
export USE_TENSORPIPE=${USE_TENSORPIPE:-0}
export USE_GLOO=${USE_GLOO:-0}

if [ -f "${VENTUS_PYTORCH_VENV}/bin/activate" ]; then
  # shellcheck disable=SC1090
  source "${VENTUS_PYTORCH_VENV}/bin/activate"
else
  echo "[ventus-env] ventus-pytorch virtualenv not found at ${VENTUS_PYTORCH_VENV}" >&2
  echo "[ventus-env] Run: bash ${DIR}/build-ventus.sh --build pytorch" >&2
fi
