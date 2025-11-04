# Makefile for CUDA softmax kernels
# Compiler
NVCC = nvcc

# Compiler flags
#NVCCFLAGS = -O3

# NVML library path
NVML_LIB = /usr/lib/x86_64-linux-gnu/libnvidia-ml.so

# Find all .cu files in current directory
CU_SOURCES = $(wildcard *.cu)

# Generate executable names (remove .cu extension)
EXECUTABLES = $(CU_SOURCES:.cu=)

# Energy measurement programs that need NVML
ENERGY_PROGRAMS = softmax_sharedmem_energy softmax_sharedmemwarp_energy sharedmemwarpvec4_energy

# Regular programs (all except energy measurement programs)
REGULAR_PROGRAMS = $(filter-out $(ENERGY_PROGRAMS), $(EXECUTABLES))

# Default target: build all executables
.PHONY: all
all: $(EXECUTABLES)

# Rule for energy measurement programs (with NVML)
$(ENERGY_PROGRAMS): %: %.cu
	$(NVCC) -o $@ $< $(NVML_LIB)

# Rule for regular CUDA programs
$(REGULAR_PROGRAMS): %: %.cu
	$(NVCC) -o $@ $<

# Clean target: remove all executables
.PHONY: clean
clean:
	rm -f $(EXECUTABLES)

# Help target
.PHONY: help
help:
	@echo "Available targets:"
	@echo "  all     - Build all executables (default)"
	@echo "  clean   - Remove all executables"
	@echo "  help    - Show this help message"
	@echo ""
	@echo "Programs to be built:"
	@echo "  Regular programs: $(REGULAR_PROGRAMS)"
	@echo "  Energy programs:  $(ENERGY_PROGRAMS)"

# List all targets
.PHONY: list
list:
	@echo "Executables that will be built:"
	@for prog in $(EXECUTABLES); do echo "  $$prog"; done
