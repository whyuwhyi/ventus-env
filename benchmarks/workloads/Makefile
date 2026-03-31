CASES := \
	bmm_fp16 \
	gelu_mlp_fp16 \
	swiglu_mlp_fp16

.PHONY: all clean run $(CASES)

all: $(CASES)

$(CASES):
	$(MAKE) -C $@

run: all
	for d in $(CASES); do $(MAKE) -C $$d run; done

clean:
	for d in $(CASES); do $(MAKE) -C $$d clean; done
