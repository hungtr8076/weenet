// fused_haar_stem.cu -- one-launch stem for WeeNet on sm_53 (Jetson Nano): order-1 Haar scattering of every 4x4 cell
// (add / subtract / abs), then the 1x1 conv (BN folded) + ReLU kept in registers; only the 32-channel FP16 output is
// written to memory, in the kCHW2 layout. The body of the network runs on stock TensorRT layers.
#include <cuda_fp16.h>

#include "fused_haar_stem.h"

namespace {
const int NF = kFusedHaarStemFeatures;

__global__ void haar_stem(const float* __restrict__ x, const float* __restrict__ wb, half2* __restrict__ y, int H, int W,
                          int Ho, int Wo, int C2o) {
  extern __shared__ float sw[];
  const int cout = 2 * C2o, nw = cout * NF + cout;
  for (int i = threadIdx.x; i < nw; i += blockDim.x) sw[i] = wb[i];
  __syncthreads();
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Ho * Wo) return;
  const int oy = idx / Wo, ox = idx - oy * Wo;
  float f[NF];
  for (int c = 0; c < 3; ++c) {
    const float* p = x + (size_t)c * H * W + (size_t)(4 * oy) * W + 4 * ox;
    float a[4], sh = 0.f, sv = 0.f, sd = 0.f;
    for (int b = 0; b < 4; ++b) {  // the four 2x2 blocks of the cell: TL, TR, BL, BR
      const float* q = p + (b >> 1) * 2 * W + (b & 1) * 2;
      const float p00 = q[0], p01 = q[1], p10 = q[W], p11 = q[W + 1];
      a[b] = (p00 + p01 + p10 + p11) * 0.25f;
      sh += fabsf((p00 - p01 + p10 - p11) * 0.25f);
      sv += fabsf((p00 + p01 - p10 - p11) * 0.25f);
      sd += fabsf((p00 - p01 - p10 + p11) * 0.25f);
    }
    f[c] = (a[0] + a[1] + a[2] + a[3]) * 0.25f;
    f[3 + c] = fabsf((a[0] - a[1] + a[2] - a[3]) * 0.25f);
    f[6 + c] = fabsf((a[0] + a[1] - a[2] - a[3]) * 0.25f);
    f[9 + c] = fabsf((a[0] - a[1] - a[2] + a[3]) * 0.25f);
    f[12 + c] = sh * 0.25f;
    f[15 + c] = sv * 0.25f;
    f[18 + c] = sd * 0.25f;
  }
  const float* bias = sw + cout * NF;
  for (int o2 = 0; o2 < C2o; ++o2) {
    const float* w0 = sw + (2 * o2) * NF;
    const float* w1 = w0 + NF;
    float s0 = bias[2 * o2], s1 = bias[2 * o2 + 1];
    for (int j = 0; j < NF; ++j) {
      s0 = fmaf(w0[j], f[j], s0);
      s1 = fmaf(w1[j], f[j], s1);
    }
    y[(size_t)o2 * Ho * Wo + idx] = __floats2half2_rn(fmaxf(s0, 0.f), fmaxf(s1, 0.f));
  }
}

// The same stem from the raw camera layout: uint8, interleaved (HWC), channel order BGR as OpenCV decodes it.
// nrm (after the biases in wb) = 3 scales then 3 offsets in RGB order: byte * scale[c] + offset[c] is the normalized
// pixel the network was trained on. The Haar details are differences, so the offset cancels and the positive scale
// leaves the modulus; only the cell mean takes the offset. The CPU no longer converts, normalizes or transposes.
__global__ void haar_stem_u8(const unsigned char* __restrict__ x, const float* __restrict__ wb, half2* __restrict__ y,
                             int W, int Ho, int Wo, int C2o) {
  extern __shared__ float sw[];
  const int cout = 2 * C2o, nw = cout * NF + cout + 6;
  for (int i = threadIdx.x; i < nw; i += blockDim.x) sw[i] = wb[i];
  __syncthreads();
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= Ho * Wo) return;
  const int oy = idx / Wo, ox = idx - oy * Wo;
  const float* nrm = sw + cout * NF + cout;
  const unsigned char* cell = x + ((size_t)(4 * oy) * W + 4 * ox) * 3;
  float f[NF];
  for (int c = 0; c < 3; ++c) {
    const unsigned char* p = cell + (2 - c);  // BGR bytes -> color c of RGB
    const float sc = nrm[c];
    float a[4], sh = 0.f, sv = 0.f, sd = 0.f;
    for (int b = 0; b < 4; ++b) {  // the four 2x2 blocks of the cell: TL, TR, BL, BR
      const unsigned char* q = p + ((b >> 1) * 2 * W + (b & 1) * 2) * 3;
      const float p00 = q[0], p01 = q[3], p10 = q[3 * W], p11 = q[3 * W + 3];
      a[b] = (p00 + p01 + p10 + p11) * 0.25f;
      sh += fabsf((p00 - p01 + p10 - p11) * 0.25f);
      sv += fabsf((p00 + p01 - p10 - p11) * 0.25f);
      sd += fabsf((p00 - p01 - p10 + p11) * 0.25f);
    }
    f[c] = fmaf((a[0] + a[1] + a[2] + a[3]) * 0.25f, sc, nrm[3 + c]);
    f[3 + c] = fabsf((a[0] - a[1] + a[2] - a[3]) * 0.25f) * sc;
    f[6 + c] = fabsf((a[0] + a[1] - a[2] - a[3]) * 0.25f) * sc;
    f[9 + c] = fabsf((a[0] - a[1] - a[2] + a[3]) * 0.25f) * sc;
    f[12 + c] = sh * 0.25f * sc;
    f[15 + c] = sv * 0.25f * sc;
    f[18 + c] = sd * 0.25f * sc;
  }
  const float* bias = sw + cout * NF;
  for (int o2 = 0; o2 < C2o; ++o2) {
    const float* w0 = sw + (2 * o2) * NF;
    const float* w1 = w0 + NF;
    float s0 = bias[2 * o2], s1 = bias[2 * o2 + 1];
    for (int j = 0; j < NF; ++j) {
      s0 = fmaf(w0[j], f[j], s0);
      s1 = fmaf(w1[j], f[j], s1);
    }
    y[(size_t)o2 * Ho * Wo + idx] = __floats2half2_rn(fmaxf(s0, 0.f), fmaxf(s1, 0.f));
  }
}
}  // namespace

int fusedHaarStemLaunchU8(const void* x, void* y, const void* wbn, int H, int W, int cout, cudaStream_t st) {
  if (H % 4 || W % 4 || cout % 2 || cout > 64) return 1;
  const int Ho = H / 4, Wo = W / 4, TB = 128;
  const size_t smem = (size_t)(cout * NF + cout + 6) * sizeof(float);
  haar_stem_u8<<<(Ho * Wo + TB - 1) / TB, TB, smem, st>>>((const unsigned char*)x, (const float*)wbn, (half2*)y, W, Ho, Wo, cout / 2);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}

int fusedHaarStemLaunch(const void* x, void* y, const void* wb, int H, int W, int cout, cudaStream_t st) {
  if (H % 4 || W % 4 || cout % 2 || cout > 64) return 1;
  const int Ho = H / 4, Wo = W / 4, TB = 128;
  const size_t smem = (size_t)(cout * NF + cout) * sizeof(float);
  haar_stem<<<(Ho * Wo + TB - 1) / TB, TB, smem, st>>>((const float*)x, (const float*)wb, (half2*)y, H, W, Ho, Wo, cout / 2);
  return cudaGetLastError() == cudaSuccess ? 0 : 1;
}
