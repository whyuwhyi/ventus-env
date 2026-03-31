CASES := \
	vcvt_fp32_fp16 \
	vcvt_fp16_fp32 \
	vcvt_fp16_fp32_workloadlike \
	vcvt_fp32_bf16 \
	vcvt_bf16_fp32 \
	vadd_f16x2 \
	vmul_f16x2 \
	vfma_f16x2 \
	vadd_bf16x2 \
	vmul_bf16x2 \
	vfma_bf16x2

.PHONY: all clean run $(CASES)

all: $(CASES)

$(CASES):
	$(MAKE) -C $@

run: all
	for d in $(CASES); do $(MAKE) -C $$d run; done

clean:
	for d in $(CASES); do $(MAKE) -C $$d clean; done
