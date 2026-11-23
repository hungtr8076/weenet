// fused_haar_stem.h -- interface of the fused Haar-scattering stem kernel of WeeNet (fused_haar_stem.cu).
#pragma once
#include <cuda_runtime.h>

const int kFusedHaarStemFeatures = 21;  // per color: mean, |H|,|V|,|D| at 4 px, mean |h|,|v|,|d| at 2 px

// x: device FP32 linear NCHW, N = 1, 3 x H x W (H, W multiples of 4).
// y: device FP16 kCHW2, cout x H/4 x W/4 (cout even, <= 64).
// wb: device floats, cout*21 weights (row = output channel) followed by cout biases.
// Computes y = ReLU(W * haar_scattering(x) + b) in one launch. Returns 0 on success.
int fusedHaarStemLaunch(const void* x, void* y, const void* wb, int H, int W, int cout, cudaStream_t st);

// Same stem from the raw frame. x: device bytes, H x W x 3 interleaved, channel order BGR (OpenCV).
// wbn: wb as above followed by 3 scales and 3 offsets in RGB order (normalized pixel = byte * scale + offset).
int fusedHaarStemLaunchU8(const void* x, void* y, const void* wbn, int H, int W, int cout, cudaStream_t st);
