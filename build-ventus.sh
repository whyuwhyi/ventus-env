#!/usr/bin/env bash

set -euo pipefail

DIR=$(cd "$(dirname "${0}")" &>/dev/null && (pwd -W 2>/dev/null || pwd))
VENTUS_INSTALL_PREFIX=${VENTUS_INSTALL_PREFIX:-${DIR}/install}
PROGRAMS_TOBUILD_DEFAULT=(systemc llvm ocl-icd libclc spike gvm driver pocl rodinia cts test-pocl)
PROGRAMS_TOBUILD_DEFAULT_FULL=(systemc llvm ocl-icd libclc spike rtlsim cyclesim gvm driver pocl rodinia cts test-pocl)
PROGRAMS_TOBUILD=(${PROGRAMS_TOBUILD_DEFAULT_FULL[@]})

BUILD_PARALLEL=$(nproc)

# Helper function
help() {
  cat <<END

Build [systemc llvm, pocl, ocl-icd, libclc, driver, spike, rtlsim|gpgpu, cyclesim|simulator, gvm, pytorch, ventus-kernels] programs.
Run the rodinia and test-pocl test suites.
Read ${DIR}/llvm/README.md to get started.

Usage: ${DIR}/$(basename ${0})
                          [--build <build programs>]
                          [--help | -h]

Options:
  --build <build programs>
    Chosen programs to build : [${PROGRAMS_TOBUILD}]
    Option format : "llvm;pocl" or "driver;pytorch", strings are separated by semicolon.
    ( Note that quotation marks are necessary, or bash will parse the semicolon as command ending )
    Default : "llvm;ocl-icd;libclc;spike;rtlsim;cyclesim;driver;pocl;rodinia;test-pocl"
    Extra target : "pytorch" builds ventus-pytorch into ${VENTUS_PYTORCH_DIR}/.venv
    Extra target : "ventus-kernels" stages flat kernel artifacts under torch/share/ventus/kernels/<op>/
    'BUILD_TYPE' defaults to 'Release' and may be overridden by environment variables
    'LLVM_ENABLE_ASSERTIONS' defaults to 'OFF' and is forwarded to LLVM CMake
    'LLVM_ENABLE_EXPENSIVE_CHECKS' defaults to 'OFF' and is forwarded to LLVM CMake
    'CLANG_TOOLING_BUILD_AST_INTROSPECTION' defaults to 'OFF' and skips ASTNodeAPI tooling generation
    'VENTUS_BUILD_KERNELS=0' opts out of the default kernel artifact stage in the pytorch build path
    'VENTUS_KERNEL_OPS' may be set to a comma/semicolon separated subset when staging flat kernel artifacts
    'VENTUS_KERNEL_FORCE_REBUILD=1' forces rebuilding kernel artifacts even when flat artifacts already exist

  --help | -h
    Print this help message and exit.
END
}

# Check the to be built program exits in file system or not
check_if_program_exits() {
  if [ ! -d "$1" ]; then
    echo "WARNING:*************************************************************"
    echo
    echo "$2 folder not found, please set or check!"
    echo "Default folder is set to be $(realpath $1)"
    echo
    echo "WARNING:*************************************************************"
    exit 1
  fi
}

# Parse command line options
while [ $# -gt 0 ]; do
  case $1 in
  -h | --help)
    help
    exit 0
    ;;

  --build)
    shift
    if [ ! -z "${1}" ]; then
      PROGRAMS_TOBUILD=(${1//;/ })
    fi
    ;;

  # --build-full)
  #   PROGRAMS_TOBUILD=(${PROGRAMS_TOBUILD_DEFAULT_FULL[@]})
  #   ;;

  ?*)
    echo "Invalid options: \"$1\" , try ${DIR}/$(basename ${0}) --help for help"
    exit -1
    ;;
  esac
  # Process next command-line option
  shift
done

# Get build type from env, otherwise use default value 'Release'
BUILD_TYPE=${BUILD_TYPE:-Release}
LLVM_ENABLE_ASSERTIONS=${LLVM_ENABLE_ASSERTIONS:-OFF}
LLVM_ENABLE_EXPENSIVE_CHECKS=${LLVM_ENABLE_EXPENSIVE_CHECKS:-OFF}
CLANG_TOOLING_BUILD_AST_INTROSPECTION=${CLANG_TOOLING_BUILD_AST_INTROSPECTION:-OFF}

# Need to get the systemc folder from enviroment variables
SYSTEMC_DIR=${SYSTEMC_DIR:-${DIR}/systemc}
SYSTEMC_INSTALL_DIR=${SYSTEMC_INSTALL_DIR:-${VENTUS_INSTALL_PREFIX}/systemc}
check_if_program_exits $SYSTEMC_DIR "lib systemc"

# Need to get the ventus-llvm folder from enviroment variables
LLVM_DIR=${LLVM_DIR:-${DIR}/llvm}
check_if_program_exits $LLVM_DIR "ventus-llvm"
LIBCLC_DIR=${LLVM_DIR}/libclc
LLVM_BUILD_DIR=${LLVM_DIR}/build
LIBCLC_BUILD_DIR=${LLVM_DIR}/build-libclc

# Need to get the cpp-cycle-level-simulator folder from enviroment variables
CYCLESIM_DIR=${CYCLESIM_DIR:-${DIR}/cyclesim}
check_if_program_exits $CYCLESIM_DIR "ventus-gpgpu cpp cycle-level simulator"
CYCLESIM_BUILD_DIR=${CYCLESIM_DIR}/build

# Need to get the ventus-gpgpu (Chisel RTL) folder from enviroment variables
GPGPU_DIR=${GPGPU_DIR:-${DIR}/gpgpu}
check_if_program_exits $GPGPU_DIR "ventus-gpgpu chisel RTL"

# Need to get the pocl folder from enviroment variables
POCL_DIR=${POCL_DIR:-${DIR}/pocl}
check_if_program_exits $POCL_DIR "pocl"
POCL_BUILD_DIR=${POCL_DIR}/build

# Need to get the ventus-driver folder from enviroment variables
DRIVER_DIR=${DRIVER_DIR:-${DIR}/driver}
check_if_program_exits ${DRIVER_DIR} "ventus-driver"
DRIVER_BUILD_DIR=${DRIVER_DIR}/build

# Need to get the ventus-spike folder from enviroment variables
SPIKE_DIR=${SPIKE_DIR:-${DIR}/spike}
check_if_program_exits ${SPIKE_DIR} "spike"
SPIKE_BUILD_DIR=${SPIKE_DIR}/build

# Need to get the icd_loader folder from enviroment variables
OCL_ICD_DIR=${OCL_ICD_DIR:-${DIR}/ocl-icd}
check_if_program_exits ${OCL_ICD_DIR} "ocl-icd"
OCL_ICD_BUILD_DIR=${OCL_ICD_DIR}/build

# Need to get the OpenCL-CTS folder from enviroment variables
OPENCL_CTS_DIR=${OPENCL_CTS_DIR:-${DIR}/OpenCL-CTS}
check_if_program_exits ${OPENCL_CTS_DIR} "OpenCL Conformance Test Suite (CTS)"
OPENCL_CTS_BUILD_DIR=${OPENCL_CTS_DIR}/build

# Need to get the gpu-rodinia folder from enviroment variables
RODINIA_DIR=${RODINIA_DIR:-${DIR}/rodinia}
check_if_program_exits ${RODINIA_DIR} "gpu-rodinia"

# Need to get the ventus-pytorch folder from environment variables
VENTUS_PYTORCH_DIR=${VENTUS_PYTORCH_DIR:-${DIR}/ventus-pytorch}
check_if_program_exits ${VENTUS_PYTORCH_DIR} "ventus-pytorch"
VENTUS_PYTORCH_VENV=${VENTUS_PYTORCH_VENV:-${VENTUS_PYTORCH_DIR}/.venv}
USE_CUDA=${USE_CUDA:-0}
USE_ROCM=${USE_ROCM:-0}
USE_XPU=${USE_XPU:-0}
VENTUS_BACKEND=${VENTUS_BACKEND:-rtlsim}
VENTUS_BUILD_KERNELS=${VENTUS_BUILD_KERNELS:-1}
VENTUS_KERNEL_BUILD_STRICT=${VENTUS_KERNEL_BUILD_STRICT:-0}
VENTUS_KERNEL_FORCE_REBUILD=${VENTUS_KERNEL_FORCE_REBUILD:-0}

# Build library systemc: depended by cyclesim
build_systemc() {
  cd ${SYSTEMC_DIR}
  ./config/bootstrap
  mkdir -p ${SYSTEMC_DIR}/build
  cd ${SYSTEMC_DIR}/build
  ../configure 'CXXFLAGS=-std=c++20' --prefix=${SYSTEMC_INSTALL_DIR} --enable-debug
  make -j${BUILD_PARALLEL}
  make -j${BUILD_PARALLEL} check
  make install
}

# Build llvm
build_llvm() {
  if [ -e "${LLVM_DIR}/prebuilt" ]; then
    echo "Using prebuilt llvm-ventus, skip building"
    cp --reflink=auto -a ${LLVM_DIR}/install ${VENTUS_INSTALL_PREFIX}
    return 0
  fi
  mkdir -p ${LLVM_BUILD_DIR}
  cd ${LLVM_BUILD_DIR}
  cmake -G Ninja -B ${LLVM_BUILD_DIR} -S ${LLVM_DIR}/llvm \
    -DLLVM_CCACHE_BUILD=ON \
    -DLLVM_OPTIMIZED_TABLEGEN=ON \
    -DLLVM_PARALLEL_LINK_JOBS=12 \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DLLVM_ENABLE_ASSERTIONS=${LLVM_ENABLE_ASSERTIONS} \
    -DLLVM_ENABLE_EXPENSIVE_CHECKS=${LLVM_ENABLE_EXPENSIVE_CHECKS} \
    -DCLANG_TOOLING_BUILD_AST_INTROSPECTION=${CLANG_TOOLING_BUILD_AST_INTROSPECTION} \
    -DLLVM_ENABLE_PROJECTS="clang;lld;libclc" \
    -DLLVM_TARGETS_TO_BUILD="AMDGPU;X86;RISCV" \
    -DLLVM_TARGET_ARCH=riscv32 \
    -DBUILD_SHARED_LIBS=ON \
    -DLLVM_BUILD_LLVM_DYLIB=ON \
    -DCMAKE_INSTALL_PREFIX=${VENTUS_INSTALL_PREFIX}
  ninja
  ninja install
}

# Build ventus driver
build_driver() {
  mkdir -p ${DRIVER_BUILD_DIR}
  cd ${DRIVER_DIR}
  cmake -G Ninja -B ${DRIVER_BUILD_DIR} -S ${DRIVER_DIR} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${VENTUS_INSTALL_PREFIX} \
    -DVENTUS_INSTALL_PREFIX=${VENTUS_INSTALL_PREFIX} \
    -DSPIKE_SRC_DIR=${SPIKE_DIR} \
    -DDRIVER_ENABLE_AUTOSELECT=ON \
    -DDRIVER_ENABLE_RTLSIM=ON \
    -DDRIVER_ENABLE_CYCLESIM=ON \
    -DDRIVER_ENABLE_GVM=ON

  ninja -C ${DRIVER_BUILD_DIR}

  ninja -C ${DRIVER_BUILD_DIR} install # -DCMAKE_C_COMPILER=clang
  # -DCMAKE_CXX_COMPILER=clang++
}

# Build spike simulator
build_spike() {
  # rm -rf ${SPIKE_BUILD_DIR} || true
  mkdir -p ${SPIKE_BUILD_DIR}
  cd ${SPIKE_BUILD_DIR}
  ../configure --prefix=${VENTUS_INSTALL_PREFIX} --enable-commitlog
  make -j${BUILD_PARALLEL}
  make install
}

# Build ventus cpp cycle-level simulator
build_gpgpu_cyclesim() {
  cd ${CYCLESIM_DIR}
  cmake -G Ninja -B ${CYCLESIM_BUILD_DIR} -S ${CYCLESIM_DIR} \
    -DSYSTEMC_HOME=${SYSTEMC_INSTALL_DIR} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_INSTALL_PREFIX=${VENTUS_INSTALL_PREFIX}
  ninja -C ${CYCLESIM_BUILD_DIR}
  ninja -C ${CYCLESIM_BUILD_DIR} install
}

# Build ventus cpp cycle-level simulator
build_gpgpu_rtlsim() {
  cd ${GPGPU_DIR}/sim-verilator
  make -j${BUILD_PARALLEL} RELEASE=1
  make install RELEASE=1 PREFIX=${VENTUS_INSTALL_PREFIX}
}

build_gvm() {
  cd ${GPGPU_DIR}/sim-verilator
  make -f gvm.mk -j${BUILD_PARALLEL} RELEASE=1 GVM_TRACE=0
  make -f gvm.mk install RELEASE=1 PREFIX=${VENTUS_INSTALL_PREFIX}
}

# Build pocl from THU
build_pocl() {
  local pocl_c_flags=""
  local pocl_cxx_flags=""

  if [[ "${LLVM_ENABLE_ASSERTIONS}" == "ON" || "${LLVM_ENABLE_EXPENSIVE_CHECKS}" == "ON" ]]; then
    pocl_c_flags="${pocl_c_flags} -D_DEBUG -UNDEBUG"
    pocl_cxx_flags="${pocl_cxx_flags} -D_DEBUG -D_GLIBCXX_DEBUG -UNDEBUG"
  fi

  mkdir -p ${POCL_BUILD_DIR}
  cd ${POCL_DIR}
  cmake -G Ninja -B ${POCL_BUILD_DIR} -S ${POCL_DIR} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DENABLE_HOST_CPU_DEVICES=OFF \
    -DENABLE_VENTUS=ON \
    -DENABLE_ICD=ON \
    -DDEFAULT_ENABLE_ICD=ON \
    -DENABLE_TESTS=OFF \
    -DSTATIC_LLVM=OFF \
    -DVENTUS_INSTALL_PREFIX=${VENTUS_INSTALL_PREFIX} \
    -DCMAKE_INSTALL_PREFIX=${VENTUS_INSTALL_PREFIX} \
    -DCMAKE_C_FLAGS="${pocl_c_flags}" \
    -DCMAKE_CXX_FLAGS="${pocl_cxx_flags}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
  # -DCMAKE_C_COMPILER=clang \
  # -DCMAKE_CXX_COMPILER=clang++ \
  ninja -C ${POCL_BUILD_DIR}
  ninja -C ${POCL_BUILD_DIR} install
}

build_ventus_kernel_artifacts() {
  check_if_ventus_llvm_built
  local kernels_root="${VENTUS_PYTORCH_DIR}/aten/src/ATen/ventus/kernels"
  local builders_root="${kernels_root}/builders"
  local flat_builder="${builders_root}/build-kernel-flat.sh"
  local install_root="${VENTUS_KERNEL_INSTALL_DIR:-${VENTUS_PYTORCH_DIR}/torch/share/ventus/kernels}"
  local -a kernel_ops=()
  local normalized_kernel_ops="${VENTUS_KERNEL_OPS:-}"
  normalized_kernel_ops="${normalized_kernel_ops//,/ }"
  normalized_kernel_ops="${normalized_kernel_ops//;/ }"
  if [[ -n "${normalized_kernel_ops}" ]]; then
    # shellcheck disable=SC2206
    kernel_ops=(${normalized_kernel_ops})
  else
    kernel_ops=(
      vecadd mul sub div relu tanh neg
      eq_scalar ne_scalar ne_tensor lt_scalar lt_scalar_f32 rsub_scalar
      bitwise_and_bool bitwise_and_i64_bool bitwise_not_bool bitwise_or_bool
      masked_fill_i64_bool pow_scalar_f32 addcmul_f32 tril_f32 where_f32
      arange_f32 arange_i64 isin_i64 isneginf_f32 rsqrt_f32
      add_i64 sub_i64 mul_i64 all_bool any_bool
      gelu_f32 gelu_f16 gelu_bf16 silu_f32 silu_f16 silu_bf16
      convert_f32_f16 convert_f16_f32 convert_f32_bf16 convert_bf16_f32
      vecadd_f16 vecadd_bf16 vecmul_f16 vecmul_bf16
      vecsub_f16 vecsub_bf16 vecneg_f16 vecneg_bf16
      vecdiv_f16 vecdiv_bf16
      sum mean cumsum max_i64 argmax_f32
      softmax_f32 softmax_f16 softmax_bf16
      layer_norm_f32 layer_norm_f16
      index_select_f32 addmm addmm_mma addmm_mma_bf16 gemm_tf32
    )
  fi

  if [[ ! -x "${flat_builder}" ]]; then
    chmod +x "${flat_builder}" 2>/dev/null || true
  fi
  if [[ ! -x "${flat_builder}" ]]; then
    echo "Flat kernel builder script is missing: ${flat_builder}"
    exit 1
  fi

  local op
  for op in "${kernel_ops[@]}"; do
    local op_install_dir="${install_root}/${op}"
    if [[ "${VENTUS_KERNEL_FORCE_REBUILD}" != "1" ]] \
      && [[ -f "${op_install_dir}/object.riscv" ]] \
      && compgen -G "${op_install_dir}/*.metadata" > /dev/null \
      && compgen -G "${op_install_dir}/*.data" > /dev/null; then
      echo "[ventus-kernels] skip op=${op} install=${op_install_dir}"
      continue
    fi
    env \
      VENTUS_INSTALL_PREFIX="${VENTUS_INSTALL_PREFIX}" \
      VENTUS_PYTORCH_DIR="${VENTUS_PYTORCH_DIR}" \
      LLVM_DIR="${LLVM_DIR}" \
      VENTUS_KERNEL_BUILD_STRICT="${VENTUS_KERNEL_BUILD_STRICT}" \
      VENTUS_KERNEL_COMPILE_OBJECT="${VENTUS_KERNEL_COMPILE_OBJECT:-1}" \
      VENTUS_KERNEL_INSTALL_DIR="${VENTUS_KERNEL_INSTALL_DIR:-${VENTUS_PYTORCH_DIR}/torch/share/ventus/kernels}" \
      VENTUS_KERNEL_BUILD_DIR="${VENTUS_KERNEL_BUILD_DIR:-${VENTUS_PYTORCH_DIR}/build/ventus-kernels}" \
      "${flat_builder}" "${op}"
  done
}

# Build ventus-pytorch in a dedicated virtual environment
build_ventus_pytorch() {
  check_if_spike_built
  check_if_cyclesim_built
  check_if_rtlsim_built
  check_if_gvm_built
  check_if_gvmref_built
  check_if_pocl_built
  if [[ "${VENTUS_BUILD_KERNELS}" == "1" ]]; then
    build_ventus_kernel_artifacts
  fi
  cd ${VENTUS_PYTORCH_DIR}
  if [ ! -d "${VENTUS_PYTORCH_VENV}" ]; then
    python3 -m venv --system-site-packages "${VENTUS_PYTORCH_VENV}"
  else
    python3 -m venv --upgrade --system-site-packages "${VENTUS_PYTORCH_VENV}"
  fi
  # shellcheck disable=SC1090
  source "${VENTUS_PYTORCH_VENV}/bin/activate"
  python - <<'PYCODE'
import setuptools.build_meta
print('setuptools.build_meta available')
PYCODE
  pip install -r requirements-build.txt
  USE_CUDA=${USE_CUDA:-0} \
    USE_ROCM=${USE_ROCM:-0} \
    USE_XPU=${USE_XPU:-0} \
    MAX_JOBS=${MAX_JOBS:-${BUILD_PARALLEL}} \
    pip install -e . -v --no-build-isolation
}

# Build libclc for pocl
build_libclc() {
  if [ -e "${LLVM_DIR}/prebuilt" ]; then
    echo "Using prebuilt llvm libclc, skip building"
    cp --reflink=auto -a ${LLVM_DIR}/install ${VENTUS_INSTALL_PREFIX}
    return 0
  fi
  if [ ! -d "${LIBCLC_BUILD_DIR}" ]; then
    mkdir ${LIBCLC_BUILD_DIR}
  fi
  cd ${LIBCLC_BUILD_DIR}
  cmake -G Ninja -B ${LIBCLC_BUILD_DIR} -S ${LLVM_DIR}/libclc \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DCMAKE_CLC_COMPILER=clang \
    -DCMAKE_LLAsm_COMPILER_WORKS=ON \
    -DCMAKE_CLC_COMPILER_WORKS=ON \
    -DCMAKE_CLC_COMPILER_FORCED=ON \
    -DCMAKE_LLAsm_FLAGS="-target riscv32 -mcpu=ventus-gpgpu -cl-std=CL2.0 -Dcl_khr_fp64 -ffunction-sections -fdata-sections" \
    -DCMAKE_CLC_FLAGS="-target riscv32 -mcpu=ventus-gpgpu -cl-std=CL2.0 -I${LLVM_DIR}/libclc/generic/include -Dcl_khr_fp64  -ffunction-sections -fdata-sections" \
    -DLIBCLC_TARGETS_TO_BUILD="riscv32--" \
    -DCMAKE_CXX_FLAGS="-I ${LLVM_DIR}/llvm/include/ -std=c++17 -Dcl_khr_fp64 -ffunction-sections -fdata-sections" \
    -DCMAKE_INSTALL_PREFIX=${VENTUS_INSTALL_PREFIX} \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE}
  # -DCMAKE_C_COMPILER=clang \
  # -DCMAKE_CXX_COMPILER=clang++ \
  ninja
  ninja install
  # TODO: There are bugs in linking all libclc object files now
  echo "************* Building riscv32 libclc object file ************"
  bash ${LLVM_DIR}/libclc/build_riscv32clc.sh ${LLVM_DIR}/libclc ${LIBCLC_BUILD_DIR} ${VENTUS_INSTALL_PREFIX} || true

  DstDir=${VENTUS_INSTALL_PREFIX}/share/pocl
  if [ ! -d "${DstDir}" ]; then
    mkdir -p ${DstDir}
  fi
}

# Build icd_loader
build_icd_loader() {
  cd ${OCL_ICD_DIR}
  ./bootstrap
  mkdir -p ${OCL_ICD_DIR}/build
  cd ${OCL_ICD_DIR}/build
  ../configure --prefix=${VENTUS_INSTALL_PREFIX}
  make -j${BUILD_PARALLEL} && make install
}

build_opencl_cts() {
  cd ${OPENCL_CTS_DIR}
  cmake -S ${OPENCL_CTS_DIR} -B ${OPENCL_CTS_BUILD_DIR} -G Ninja \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCL_INCLUDE_DIR=${VENTUS_INSTALL_PREFIX}/include \
    -DCL_LIB_DIR=${VENTUS_INSTALL_PREFIX}/lib \
    -DOPENCL_LIBRARIES=OpenCL
  cmake --build ${OPENCL_CTS_BUILD_DIR} -j ${BUILD_PARALLEL}
}

# Test the rodinia test suit
test_rodinia() {
  cd ${RODINIA_DIR}
  make OCL_clean
  make OPENCL
}

# TODO : More test cases of the pocl will be added
test_pocl() {
  cd ${POCL_BUILD_DIR}/examples
  ./vecadd/vecadd
  ./matadd/matadd
  VENTUS_BACKEND=cyclesim ./matadd/matadd
  VENTUS_BACKEND=rtlsim ./matadd/matadd
}

# Export needed path and enviroment variables
export_elements() {
  export PATH=${VENTUS_INSTALL_PREFIX}/bin:$PATH
  export LD_LIBRARY_PATH=${VENTUS_INSTALL_PREFIX}/lib:${LD_LIBRARY_PATH:-}
  export SPIKE_SRC_DIR=${SPIKE_DIR}
  export SPIKE_TARGET_DIR=${VENTUS_INSTALL_PREFIX}
  export VENTUS_INSTALL_PREFIX=${VENTUS_INSTALL_PREFIX}
  export POCL_DEVICES="ventus"
  export OCL_ICD_VENDORS=${VENTUS_INSTALL_PREFIX}/lib/libpocl.so
}

# When no need to build llvm, export needed elements
if [[ ! "${PROGRAMS_TOBUILD[*]}" =~ "llvm" ]]; then
  export_elements
fi

# Check dep-library systemc is built or not
check_if_systemc_built() {
  if [ ! -f "${SYSTEMC_INSTALL_DIR}/lib-linux64/libsystemc.so" ]; then
    echo "Please build library systemc first!"
    exit 1
  fi
}

# Check llvm is built or not
check_if_ventus_llvm_built() {
  if [ ! -d "${VENTUS_INSTALL_PREFIX}" ]; then
    echo "Please build llvm first!"
    exit 1
  fi
}

# Check isa simulator is built or not
check_if_spike_built() {
  if [ ! -f "${VENTUS_INSTALL_PREFIX}/lib/libspike_main.so" ]; then
    if [ ! -f "${SPIKE_BUILD_DIR}/lib/libspike_main.so" ]; then
      echo "Please build spike isa-simulator first!"
      exit 1
    else
      cp ${SPIKE_BUILD_DIR}/lib/libspike_main.so ${VENTUS_INSTALL_PREFIX}/lib
    fi
  fi
}

check_if_gvmref_built() {
  if [ -f "${SPIKE_BUILD_DIR}/libgvmref.so" ]; then
    cp ${SPIKE_BUILD_DIR}/libgvmref.so ${VENTUS_INSTALL_PREFIX}/lib
    return 0
  fi
  if [ -f "${SPIKE_BUILD_DIR}/lib/libgvmref.so" ]; then
    cp ${SPIKE_BUILD_DIR}/lib/libgvmref.so ${VENTUS_INSTALL_PREFIX}/lib
    return 0
  fi
  echo "Please build spike gvm reference library (libgvmref.so) for GVM!"
  exit 1
}

# Check gpgpu rtlsim sim-verilator is built or not
check_if_rtlsim_built() {
  if [ ! -f "${VENTUS_INSTALL_PREFIX}/lib/libVentusRTL.so" ]; then
    echo "Please build Ventus Chisel RTL sim-verilator (rtlsim) first!"
    exit 1
  fi
}

check_if_gvm_built() {
  if [ ! -f "${VENTUS_INSTALL_PREFIX}/lib/libVentusGVM.so" ]; then
    echo "Please build Ventus GVM backend first (use --build gvm)!"
    exit 1
  fi
}

# Check gpgpu cpp cycle-level simulator is built or not
check_if_cyclesim_built() {
  if [ ! -f "${VENTUS_INSTALL_PREFIX}/lib/libVentusCycleSim.so" ]; then
    echo "Please build Ventus Chisel C++ cycle-level simulator (cyclesim) first!"
    exit 1
  fi
}

# Check ocl-icd loader is built or not
# since pocl need ocl-icd and llvm built first
check_if_ocl_icd_built() {
  if [ ! -f "${VENTUS_INSTALL_PREFIX}/lib/libOpenCL.so" ]; then
    echo "Please build ocl-icd first!"
    exit 1
  fi
}

check_if_pocl_built() {
  if [ ! -f "${VENTUS_INSTALL_PREFIX}/lib/pocl/libpocl-devices-ventus.so" ]; then
    echo "Please build POCL first!"
    exit 1
  fi
}

# Process build options
for program in "${PROGRAMS_TOBUILD[@]}"; do
  if [ "${program}" == "systemc" ]; then
    build_systemc
  elif [ "${program}" == "llvm" ]; then
    build_llvm
    export_elements
  elif [ "${program}" == "ocl-icd" ]; then
    build_icd_loader
  elif [ "${program}" == "libclc" ]; then
    check_if_ventus_llvm_built
    build_libclc
  elif [ "${program}" == "spike" ]; then
    build_spike
  elif [ "${program}" == "rtlsim" ] || [ "${program}" == "rtl" ] || [ "${program}" == "gpgpu" ]; then
    build_gpgpu_rtlsim
  elif [ "${program}" == "cyclesim" ] || [ "${program}" == "simulator" ]; then
    check_if_systemc_built
    build_gpgpu_cyclesim
  elif [ "${program}" == "gvm" ]; then
    build_gvm
  elif [ "${program}" == "driver" ]; then
    check_if_spike_built
    check_if_cyclesim_built
    check_if_rtlsim_built
    check_if_gvm_built
    check_if_gvmref_built
    build_driver
  elif [ "${program}" == "pocl" ]; then
    check_if_ventus_llvm_built
    check_if_ocl_icd_built
    build_pocl
  elif [ "${program}" == "rodinia" ]; then
    check_if_ventus_llvm_built
    check_if_ocl_icd_built
    check_if_spike_built
    test_rodinia
  elif [ "${program}" == "cts" ] || [ "${program}" == "opencl-cts" ] || [ "${program}" == "OpenCL-CTS" ]; then
    check_if_pocl_built
    check_if_spike_built
    build_opencl_cts
  elif [ "${program}" == "pytorch" ]; then
    build_ventus_pytorch
  elif [ "${program}" == "ventus-kernels" ]; then
    build_ventus_kernel_artifacts
  elif [ "${program}" == "test-pocl" ]; then
    check_if_ventus_llvm_built
    check_if_ocl_icd_built
    check_if_spike_built
    test_pocl
  else
    echo "Invalid build options: \"${program}\" , try $0 --help for help"
    exit 1
  fi
done
