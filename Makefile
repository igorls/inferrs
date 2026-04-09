.DEFAULT_GOAL := build

# Detect OS and architecture for conditional package inclusion.
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# inferrs-backend-cuda is only built on Linux or Windows x86_64 (not macOS,
# not Windows ARM64).
ifeq ($(UNAME_S),Darwin)
  CUDA_PKG :=
else
  CUDA_PKG := -p inferrs-backend-cuda
endif

# inferrs-backend-hexagon compiles cleanly on all host platforms (it fast-fails
# at runtime on non-Snapdragon hardware).  Include it unconditionally.
HEXAGON_PKG := -p inferrs-backend-hexagon

# Packages that can be built/tested without GPU toolchains (CUDA, ROCm).
# The Hexagon backend is included because it has no exotic toolchain requirement.
NO_GPU_PKGS := -p inferrs -p inferrs-benchmark -p inferrs-backend-vulkan $(HEXAGON_PKG) $(CUDA_PKG)

.PHONY: all build release fmt clippy test

all: fmt clippy test build

build:
	cargo build $(NO_GPU_PKGS)

release:
	cargo build --release $(NO_GPU_PKGS)

fmt:
	cargo fmt --check $(NO_GPU_PKGS)

clippy:
	cargo clippy $(NO_GPU_PKGS) -- -D warnings

test:
	cargo test $(NO_GPU_PKGS)
