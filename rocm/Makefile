# Makefile for libcompressed_kernel.so
#
# Detects AMD GPU architecture automatically via rocminfo.
# Override with:  make ARCH=gfx1100
#
# Targets:
#   make          — build shared library
#   make test     — build and run the sanity-check binary
#   make clean    — remove build artifacts

HIPCC  := hipcc
TARGET := libcompressed_kernel.so
TEST   := test_kernel

# Detect GPU architecture (first gfxXXXX found in rocminfo output)
ARCH := $(shell rocminfo 2>/dev/null | grep -o 'gfx[0-9a-z]*' | head -1)
ifeq ($(ARCH),)
  $(warning rocminfo did not return a GPU architecture. Defaulting to gfx906.)
  $(warning Override with: make ARCH=gfxXXXX)
  ARCH := gfx906
endif

# Optional: override HIP code-object version for compatibility with a specific
# HIP runtime (e.g. PyTorch may bundle ROCm 6.0 which needs code object v5
# while system hipcc defaults to v6).
# Usage: make HIP_COV=5
ifneq ($(HIP_COV),)
  HIP_COV_FLAG := -mcode-object-version=$(HIP_COV)
else
  HIP_COV_FLAG :=
endif

CFLAGS := -O3 --offload-arch=$(ARCH) -fPIC -fno-gpu-rdc $(HIP_COV_FLAG)

# Optional: link against a specific HIP library directory instead of system ROCm.
# When set (e.g. to PyTorch's lib dir), the resulting .so will depend on
# libamdhip64.so from that path, avoiding version conflicts at runtime.
# Usage: make HIP_LIB_DIR=/home/user/.local/lib/python3.12/site-packages/torch/lib
ifneq ($(HIP_LIB_DIR),)
  HIP_RPATH := -L$(HIP_LIB_DIR) -Wl,-rpath,$(HIP_LIB_DIR)
else
  HIP_RPATH :=
endif

.PHONY: all test clean

all: $(TARGET)
	@echo "Built $(TARGET) for $(ARCH)"

$(TARGET): compressed_kernel.hip compressed_kernel.h
	$(HIPCC) $(CFLAGS) $(HIP_RPATH) -shared -o $@ compressed_kernel.hip

# Minimal correctness test — no external dependencies
test: $(TARGET) test_kernel.cpp
	$(HIPCC) --offload-arch=$(ARCH) -O2 -o $(TEST) test_kernel.cpp \
	         -L. -lcompressed_kernel -Wl,-rpath,.
	./$(TEST)

clean:
	rm -f $(TARGET) $(TEST)
