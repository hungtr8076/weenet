# Build the FusedHaarStem / FusedHaarStemU8 TensorRT plugin on the Jetson Nano (JetPack 4.6.1: CUDA 10.2, TensorRT 8.2, sm_53).
#   make -C plugin          -> build/libfused_haar_stem.so (relative to the repository root)
NVCC ?= /usr/local/cuda/bin/nvcc
TRT_INC ?= /usr/include/aarch64-linux-gnu
OUT ?= ../build/libfused_haar_stem.so

$(OUT): fused_haar_stem_plugin.cpp fused_haar_stem.cu fused_haar_stem.h
	mkdir -p $(dir $(OUT))
	$(NVCC) -O3 -arch=sm_53 -std=c++14 -Xcompiler -fPIC -shared -o $@ fused_haar_stem_plugin.cpp fused_haar_stem.cu -I$(TRT_INC) -lnvinfer

clean:
	rm -f $(OUT)

.PHONY: clean
