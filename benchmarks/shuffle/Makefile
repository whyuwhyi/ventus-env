ROOT       := $(abspath $(CURDIR)/../..)
OPENCL_INC := $(ROOT)/install/include
OPENCL_LIB := $(ROOT)/install/lib
CXX        := clang++
CXXFLAGS   := -g -O2 -std=c++17 -I. -I.. -I../fpu
LDFLAGS    := -L$(OPENCL_LIB) -lOpenCL -Wno-unused-result

TARGET := $(notdir $(CURDIR)).out
HOST_DEPS := main.cpp ../fpu/common.hpp ../fpu/fpu_case.hpp

.PHONY: all clean run

all: $(TARGET)

$(TARGET): $(HOST_DEPS)
	$(CXX) $(CXXFLAGS) main.cpp -o $@ -I$(OPENCL_INC) $(LDFLAGS)

run: all
	bash ./run

clean:
	rm -f *.out *.o *~
