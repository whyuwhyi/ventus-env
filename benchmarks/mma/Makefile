CASES := \
	m8n8k16_fp16_fp32 \
	m16n8k16_fp16_fp32 \
	m8n16k16_fp16_fp32 \
	m16n16k16_fp16_fp32 \
	m16n16k16_fp16_fp32_random \
	m16n16k8_tf32_fp32_random

.PHONY: all clean run $(CASES)

all: $(CASES)

$(CASES):
	$(MAKE) -C $@

run: all
	for d in $(CASES); do $(MAKE) -C $$d run; done

clean:
	for d in $(CASES); do $(MAKE) -C $$d clean; done
