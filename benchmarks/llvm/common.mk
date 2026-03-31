ROOT     := $(abspath $(CURDIR)/../../..)
LLVM_BIN := $(ROOT)/llvm/build/bin
CLANG    := $(LLVM_BIN)/clang
OBJDUMP  := $(LLVM_BIN)/llvm-objdump
TARGET   := $(notdir $(CURDIR))
ASM      := kernel.s
OBJ      := $(TARGET).o
DUMP     := $(TARGET).dump

.PHONY: all clean

all: $(OBJ) $(DUMP)

$(OBJ): $(ASM)
	$(CLANG) -c -target riscv32 -mcpu=ventus-gpgpu $< -o $@

$(DUMP): $(OBJ)
	$(OBJDUMP) -d $< > $@

clean:
	rm -f $(OBJ) $(DUMP)
