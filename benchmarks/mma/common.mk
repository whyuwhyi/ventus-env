ROOT       := $(abspath $(CURDIR)/../../..)
OPENCL_INC := $(ROOT)/install/include
OPENCL_LIB := $(ROOT)/install/lib
CXX        := clang++
CXXFLAGS   := -g -O2 -std=c++17 -I. -I..
LDFLAGS    := -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

TARGET := $(notdir $(CURDIR)).out
EXTRA_HOST_DEPS ?=
HOST_DEPS := main.cpp ../common.hpp ../mma_case.hpp $(EXTRA_HOST_DEPS)

.PHONY: all clean
all: $(TARGET)

$(TARGET): $(HOST_DEPS)
	$(CXX) $(CXXFLAGS) main.cpp -o $@ -I$(OPENCL_INC) $(LDFLAGS)

clean:
	rm -f *.out *.o *~
