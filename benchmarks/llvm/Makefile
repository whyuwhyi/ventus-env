CASES := mma shuffle vcvt packed sfu

.PHONY: all clean $(CASES)

all: $(CASES)

$(CASES):
	$(MAKE) -C $@

clean:
	for d in $(CASES); do $(MAKE) -C $$d clean; done
