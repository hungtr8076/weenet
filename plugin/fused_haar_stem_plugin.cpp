// fused_haar_stem_plugin.cpp -- TensorRT 8.2 plugin "FusedHaarStem" v1: FP32 linear image in, FP16 kCHW2 features out
// (kernel in fused_haar_stem.cu). ONNX node op_type "FusedHaarStem" with attributes cout, weight (cout*21), bias (cout).
#include <NvInfer.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "fused_haar_stem.h"

using namespace nvinfer1;

namespace {
const char* kName = "FusedHaarStem";
const char* kVersion = "1";

class FusedHaarStem : public IPluginV2DynamicExt {
 public:
  FusedHaarStem(int cout, std::vector<float> wb) : cout_(cout), wb_(std::move(wb)) {}
  FusedHaarStem(const void* data, size_t) {
    const char* p = static_cast<const char*>(data);
    std::memcpy(&cout_, p, sizeof(int)); p += sizeof(int);
    size_t n; std::memcpy(&n, p, sizeof(size_t)); p += sizeof(size_t);
    wb_.resize(n); std::memcpy(wb_.data(), p, n * sizeof(float));
  }
  IPluginV2DynamicExt* clone() const noexcept override {
    auto* c = new FusedHaarStem(cout_, wb_);
    c->setPluginNamespace(ns_.c_str());
    return c;
  }
  DimsExprs getOutputDimensions(int32_t, const DimsExprs* in, int32_t, IExprBuilder& eb) noexcept override {
    DimsExprs o;
    o.nbDims = 4;
    o.d[0] = in[0].d[0];
    o.d[1] = eb.constant(cout_);
    for (int i = 2; i < 4; ++i) o.d[i] = eb.operation(DimensionOperation::kFLOOR_DIV, *in[0].d[i], *eb.constant(4));
    return o;
  }
  bool supportsFormatCombination(int32_t pos, const PluginTensorDesc* io, int32_t, int32_t) noexcept override {
    const bool ok = pos == 0 ? (io[0].type == DataType::kFLOAT && io[0].format == TensorFormat::kLINEAR)
                             : (io[1].type == DataType::kHALF && io[1].format == TensorFormat::kCHW2);
    if (std::getenv("FUSED_HAAR_STEM_DEBUG"))
      std::fprintf(stderr, "[fused_haar_stem] supportsFormatCombination pos=%d type=%d format=%d -> %d\n", pos,
                   (int)io[pos].type, (int)io[pos].format, (int)ok);
    return ok;
  }
  void configurePlugin(const DynamicPluginTensorDesc*, int32_t, const DynamicPluginTensorDesc*, int32_t) noexcept override {}
  size_t getWorkspaceSize(const PluginTensorDesc*, int32_t, const PluginTensorDesc*, int32_t) const noexcept override {
    return 0;
  }
  int32_t enqueue(const PluginTensorDesc* in, const PluginTensorDesc*, const void* const* inputs, void* const* outputs,
                  void*, cudaStream_t st) noexcept override {
    if (!dev_ && initialize() != 0) return 1;  // execution-context clones may never see initialize()
    if (in[0].dims.d[0] != 1) return 1;        // batch 1 only
    return fusedHaarStemLaunch(inputs[0], outputs[0], dev_, in[0].dims.d[2], in[0].dims.d[3], cout_, st);
  }
  DataType getOutputDataType(int32_t, const DataType*, int32_t) const noexcept override { return DataType::kHALF; }
  const char* getPluginType() const noexcept override { return kName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  int32_t getNbOutputs() const noexcept override { return 1; }
  int32_t initialize() noexcept override {
    if (dev_) return 0;
    if (cudaMalloc(&dev_, wb_.size() * sizeof(float)) != cudaSuccess) return 1;
    if (cudaMemcpy(dev_, wb_.data(), wb_.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
      cudaFree(dev_);
      dev_ = nullptr;
      return 1;
    }
    return 0;
  }
  void terminate() noexcept override {
    if (dev_) cudaFree(dev_);
    dev_ = nullptr;
  }
  size_t getSerializationSize() const noexcept override { return sizeof(int) + sizeof(size_t) + wb_.size() * sizeof(float); }
  void serialize(void* buffer) const noexcept override {
    char* p = static_cast<char*>(buffer);
    std::memcpy(p, &cout_, sizeof(int)); p += sizeof(int);
    const size_t n = wb_.size();
    std::memcpy(p, &n, sizeof(size_t)); p += sizeof(size_t);
    std::memcpy(p, wb_.data(), n * sizeof(float));
  }
  void destroy() noexcept override {
    terminate();
    delete this;
  }
  void setPluginNamespace(const char* ns) noexcept override { ns_ = ns ? ns : ""; }
  const char* getPluginNamespace() const noexcept override { return ns_.c_str(); }

 protected:
  int cout_{0};
  std::vector<float> wb_;  // cout*21 weights then cout biases (FusedHaarStemU8: then 3 scales and 3 offsets)
  void* dev_{nullptr};
  std::string ns_;
};

// "FusedHaarStemU8" v1: the image arrives as raw bytes (H x W x 3, BGR, as OpenCV decodes it). TensorRT 8.2 has no uint8
// tensor, so the binding is declared FP32 linear 1 x 3 x H/2 x W/2 -- the same number of bytes -- and this plugin,
// which is the only consumer of the network input, reads the buffer as bytes. Normalization happens in the kernel.
class FusedHaarStemU8 : public FusedHaarStem {
 public:
  using FusedHaarStem::FusedHaarStem;
  IPluginV2DynamicExt* clone() const noexcept override {
    auto* c = new FusedHaarStemU8(cout_, wb_);
    c->setPluginNamespace(ns_.c_str());
    return c;
  }
  DimsExprs getOutputDimensions(int32_t, const DimsExprs* in, int32_t, IExprBuilder& eb) noexcept override {
    DimsExprs o;
    o.nbDims = 4;
    o.d[0] = in[0].d[0];
    o.d[1] = eb.constant(cout_);
    for (int i = 2; i < 4; ++i) o.d[i] = eb.operation(DimensionOperation::kFLOOR_DIV, *in[0].d[i], *eb.constant(2));
    return o;
  }
  int32_t enqueue(const PluginTensorDesc* in, const PluginTensorDesc*, const void* const* inputs, void* const* outputs,
                  void*, cudaStream_t st) noexcept override {
    if (!dev_ && initialize() != 0) return 1;
    if (in[0].dims.d[0] != 1 || in[0].dims.d[1] != 3) return 1;  // batch 1, 3 float planes = H x W x 3 bytes
    return fusedHaarStemLaunchU8(inputs[0], outputs[0], dev_, 2 * in[0].dims.d[2], 2 * in[0].dims.d[3], cout_, st);
  }
  const char* getPluginType() const noexcept override { return "FusedHaarStemU8"; }
};

class FusedHaarStemCreator : public IPluginCreator {
 public:
  FusedHaarStemCreator() {
    fields_.emplace_back("cout", nullptr, PluginFieldType::kINT32, 1);
    fields_.emplace_back("weight", nullptr, PluginFieldType::kFLOAT32, 0);
    fields_.emplace_back("bias", nullptr, PluginFieldType::kFLOAT32, 0);
    fc_.nbFields = static_cast<int32_t>(fields_.size());
    fc_.fields = fields_.data();
  }
  const char* getPluginName() const noexcept override { return kName; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  const PluginFieldCollection* getFieldNames() noexcept override { return &fc_; }
  IPluginV2* createPlugin(const char*, const PluginFieldCollection* fc) noexcept override {
    int cout = -1;
    std::vector<float> w, b;
    for (int i = 0; i < fc->nbFields; ++i) {
      const PluginField& f = fc->fields[i];
      if (!std::strcmp(f.name, "cout")) cout = *static_cast<const int*>(f.data);
      if (!std::strcmp(f.name, "weight")) w.assign(static_cast<const float*>(f.data), static_cast<const float*>(f.data) + f.length);
      if (!std::strcmp(f.name, "bias")) b.assign(static_cast<const float*>(f.data), static_cast<const float*>(f.data) + f.length);
    }
    // refuse rather than guess when a field is missing or has the wrong size
    if (cout <= 0 || cout % 2 || cout > 64 || w.size() != (size_t)cout * kFusedHaarStemFeatures || b.size() != (size_t)cout)
      return nullptr;
    w.insert(w.end(), b.begin(), b.end());
    auto* p = new FusedHaarStem(cout, std::move(w));
    p->setPluginNamespace(ns_.c_str());
    return p;
  }
  IPluginV2* deserializePlugin(const char*, const void* data, size_t len) noexcept override {
    auto* p = new FusedHaarStem(data, len);
    p->setPluginNamespace(ns_.c_str());
    return p;
  }
  void setPluginNamespace(const char* ns) noexcept override { ns_ = ns ? ns : ""; }
  const char* getPluginNamespace() const noexcept override { return ns_.c_str(); }

 private:
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
  std::string ns_;
};
class FusedHaarStemU8Creator : public IPluginCreator {
 public:
  FusedHaarStemU8Creator() {
    fields_.emplace_back("cout", nullptr, PluginFieldType::kINT32, 1);
    fields_.emplace_back("weight", nullptr, PluginFieldType::kFLOAT32, 0);
    fields_.emplace_back("bias", nullptr, PluginFieldType::kFLOAT32, 0);
    fields_.emplace_back("norm", nullptr, PluginFieldType::kFLOAT32, 6);
    fc_.nbFields = static_cast<int32_t>(fields_.size());
    fc_.fields = fields_.data();
  }
  const char* getPluginName() const noexcept override { return "FusedHaarStemU8"; }
  const char* getPluginVersion() const noexcept override { return kVersion; }
  const PluginFieldCollection* getFieldNames() noexcept override { return &fc_; }
  IPluginV2* createPlugin(const char*, const PluginFieldCollection* fc) noexcept override {
    int cout = -1;
    std::vector<float> w, b, n;
    for (int i = 0; i < fc->nbFields; ++i) {
      const PluginField& f = fc->fields[i];
      const float* d = static_cast<const float*>(f.data);
      if (!std::strcmp(f.name, "cout")) cout = *static_cast<const int*>(f.data);
      if (!std::strcmp(f.name, "weight")) w.assign(d, d + f.length);
      if (!std::strcmp(f.name, "bias")) b.assign(d, d + f.length);
      if (!std::strcmp(f.name, "norm")) n.assign(d, d + f.length);
    }
    // refuse rather than guess when a field is missing or has the wrong size; the scales must be positive
    if (cout <= 0 || cout % 2 || cout > 64 || w.size() != (size_t)cout * kFusedHaarStemFeatures || b.size() != (size_t)cout ||
        n.size() != 6 || !(n[0] > 0.f && n[1] > 0.f && n[2] > 0.f))
      return nullptr;
    w.insert(w.end(), b.begin(), b.end());
    w.insert(w.end(), n.begin(), n.end());
    auto* p = new FusedHaarStemU8(cout, std::move(w));
    p->setPluginNamespace(ns_.c_str());
    return p;
  }
  IPluginV2* deserializePlugin(const char*, const void* data, size_t len) noexcept override {
    auto* p = new FusedHaarStemU8(data, len);
    p->setPluginNamespace(ns_.c_str());
    return p;
  }
  void setPluginNamespace(const char* ns) noexcept override { ns_ = ns ? ns : ""; }
  const char* getPluginNamespace() const noexcept override { return ns_.c_str(); }

 private:
  std::vector<PluginField> fields_;
  PluginFieldCollection fc_{};
  std::string ns_;
};
}  // namespace

REGISTER_TENSORRT_PLUGIN(FusedHaarStemCreator);
REGISTER_TENSORRT_PLUGIN(FusedHaarStemU8Creator);
