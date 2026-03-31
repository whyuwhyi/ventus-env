ROOT       := $(abspath $(CURDIR)/../../..)
OPENCL_INC := $(ROOT)/install/include
OPENCL_LIB := $(ROOT)/install/lib
CXX        := clang++
CXXFLAGS   := -g -O2 -std=c++17 -I. -I..
LDFLAGS    := -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result
BUILD_TMPDIR ?= $(CURDIR)/.tmp-build

TARGET := $(notdir $(CURDIR)).out
EXTRA_HOST_DEPS ?=
HOST_DEPS := main.cpp ../common.hpp ../workload_case.hpp $(EXTRA_HOST_DEPS)

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(HOST_DEPS)
	mkdir -p $(BUILD_TMPDIR)
	TMPDIR=$(BUILD_TMPDIR) $(CXX) $(CXXFLAGS) main.cpp -o $@ -I$(OPENCL_INC) $(LDFLAGS)

clean:
	rm -rf $(BUILD_TMPDIR)
	rm -f *.out *.o *~
