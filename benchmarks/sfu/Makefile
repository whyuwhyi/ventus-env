CASES := exp2_fp32 gelu_fp32 silu_fp32 rcp_fp16x2 rsqrt_bf16x2

.PHONY: all clean run $(CASES)

all: $(CASES)

$(CASES):
	$(MAKE) -C $@

run: all
	for d in $(CASES); do $(MAKE) -C $$d run; done

clean:
	for d in $(CASES); do $(MAKE) -C $$d clean; done
