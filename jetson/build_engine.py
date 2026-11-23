#!/usr/bin/env python3
"""Build a TensorRT 8.2 FP16 engine from ONNX through the Python API (on the board, Python 3.6, no f-strings).

Needed for graphs with the FusedHaarStem / FusedHaarStemU8 plugin node: trtexec --fp16 cannot build them, because the
plugin writes FP16 in the kCHW2 layout while the ONNX tensors are FP32. Here the plugin layer's output type is fixed to
FP16 under OBEY_PRECISION_CONSTRAINTS (its input stays FP32: the image or the byte buffer); every other layer is left to
the builder as trtexec --fp16 would. Without --plugin it builds a stock FP16 engine with the same settings (equivalent
to trtexec --fp16). Prints the number of network layers.
  LD_PRELOAD=build/libfused_haar_stem.so python3 jetson/build_engine.py --onnx weenet_do_u8.onnx --out weenet_do_u8.plan
      --plugin build/libfused_haar_stem.so
  python3 jetson/build_engine.py --onnx weenet_do.onnx --out weenet_do.plan
"""
from __future__ import print_function
import argparse
import ctypes
import sys

import tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--plugin", default="", help="comma-separated plugin libraries (build/libfused_haar_stem.so)")
ap.add_argument("--workspace_mb", type=int, default=256)
a = ap.parse_args()

logger = trt.Logger(trt.Logger.WARNING)
for lib in [p for p in a.plugin.split(",") if p]:
    ctypes.CDLL(lib)
trt.init_libnvinfer_plugins(logger, "")
builder = trt.Builder(logger)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(net, logger)
if not parser.parse(open(a.onnx, "rb").read()):
    for i in range(parser.num_errors):
        print("ONNX parse error:", parser.get_error(i), file=sys.stderr)
    sys.exit(2)
config = builder.create_builder_config()
config.max_workspace_size = a.workspace_mb << 20
config.set_flag(trt.BuilderFlag.FP16)
n_plugin = 0
for i in range(net.num_layers):
    layer = net.get_layer(i)
    if layer.type == trt.LayerType.PLUGIN_V2:
        if not layer.name.endswith("_fused_haar_stem"):  # stem plugin reads the FP32 image: constrain its output only
            layer.precision = trt.float16
        layer.set_output_type(0, trt.float16)
        n_plugin += 1
if a.plugin and n_plugin == 0:
    print("expected a plugin layer in %s but found none" % a.onnx, file=sys.stderr)
    sys.exit(3)
if n_plugin:
    config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
plan = builder.build_serialized_network(net, config)
if plan is None:
    print("engine build failed", file=sys.stderr)
    sys.exit(4)
with open(a.out, "wb") as f:
    f.write(plan)
print("built %s: %d layers, %d plugin layers, input dtype %s" % (a.out, net.num_layers, n_plugin, net.get_input(0).dtype))
