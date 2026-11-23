#!/usr/bin/env python3
"""Build a fixed-batch TensorRT 8.2 FP16 engine on the board from a batch-1 ONNX (Python 3.6, no f-strings).

Used for the batched GPU policy of jetson/streams.py. The ONNX graphs have a fixed batch dimension of 1 and the onnx
package is not part of JetPack, so the batch is changed in the parsed TensorRT network instead; the batched engines then
come from the same ONNX file as the batch-1 engine:
  - the network input gets batch B;
  - every shuffle (reshape) layer with static reshape dims whose input carries the batch and whose target dim 0 is 1
    gets target dim 0 = B (the exporter writes batch 1 into Flatten / Reshape / the Gemm reshapes);
  - every slice layer with a static output shape whose input carries the batch gets output dim 0 = B;
  - YOLO11's C2f Split becomes slices whose start/size the ONNX parser computes from constants holding the static
    data shape (1, C, H, W); those constants get dim 0 = B.
"Carries the batch" is data flow from the network input, shape tensors excluded; constants reshaped for
broadcasting are left unchanged. A batch-carrying tensor without batch B after patching is an error (build stops).
jetson/verify_batch.py then checks every batched engine against the batch-1 engine on real frames. Builder settings as
trtexec --fp16 --workspace=1024.
  python3 jetson/build_batch.py --onnx onnx/weenet_do.onnx --batch 4 --out engines/weenet_do_b4.plan
"""
from __future__ import print_function
import argparse
import sys

import numpy as np
import tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument("--onnx", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--batch", type=int, required=True)
ap.add_argument("--workspace_mb", type=int, default=1024)
ap.add_argument("--dry", action="store_true", help="parse and patch only, no build")
a = ap.parse_args()
B = a.batch
if B < 2:
    raise SystemExit("--batch must be >= 2 (build the batch-1 engine with trtexec)")

logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
builder = trt.Builder(logger)
net = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
parser = trt.OnnxParser(net, logger)
if not parser.parse(open(a.onnx, "rb").read()):
    for i in range(parser.num_errors):
        print("ONNX parse error:", parser.get_error(i), file=sys.stderr)
    sys.exit(2)
if net.num_inputs != 1 or net.num_outputs != 1:
    raise SystemExit("expected 1 input and 1 output")
inp = net.get_input(0)
shape = list(inp.shape)
if shape[0] != 1:
    raise SystemExit("input batch is %s, expected 1" % shape[0])
# Pass 1, on the unmodified network (querying a shape after the input change triggers shape analysis of the whole
# graph, which fails until every reshape is patched): which tensors carry the batch (data flow from the input,
# shape tensors excluded), and which static reshapes / slices fix batch 1 on such a tensor.
carry = set([inp.name])
patch_shuffle, patch_slice, dynamic = [], [], []
producer = {}
for i in range(net.num_layers):
    for k in range(net.get_layer(i).num_outputs):
        producer[net.get_layer(i).get_output(k).name] = i
patch_const = {}                             # constant layer index -> original int32 values


def baked_shape_constants(layer, data_shape):
    """Constants in the shape-tensor subgraph of a dynamic slice that hold the parse-time shape of its data input
    (the ONNX parser bakes the static input dims, batch 1 included, into the slice size computation)."""
    stack = [layer.get_input(j) for j in range(1, layer.num_inputs) if layer.get_input(j) is not None]
    seen, found = set(), []
    while stack:
        t = stack.pop()
        if t.name in seen or t.name not in producer:
            continue
        seen.add(t.name)
        li = producer[t.name]
        src = net.get_layer(li)
        if src.type == trt.LayerType.CONSTANT:
            src.__class__ = trt.IConstantLayer
            v = np.array(src.weights).reshape(-1)
            if v.size == len(data_shape) and [int(x) for x in v] == list(data_shape):
                found.append((li, v))
        else:
            stack.extend(src.get_input(j) for j in range(src.num_inputs) if src.get_input(j) is not None)
    return found


for i in range(net.num_layers):
    layer = net.get_layer(i)
    ins = [layer.get_input(j) for j in range(layer.num_inputs)]
    c = any(t is not None and t.name in carry and not t.is_shape_tensor for t in ins)
    if not c:
        continue
    for k in range(layer.num_outputs):
        o = layer.get_output(k)
        if not o.is_shape_tensor:
            carry.add(o.name)
    if layer.type == trt.LayerType.SHUFFLE:
        layer.__class__ = trt.IShuffleLayer
        if layer.num_inputs > 1:
            dynamic.append(layer.name)       # reshape dims from a shape tensor: follows the real input shape
            continue
        try:
            rd = list(layer.reshape_dims)
        except Exception:
            rd = None                        # pure transpose, no reshape
        if rd and rd[0] == 1:
            patch_shuffle.append((i, rd))
    elif layer.type == trt.LayerType.SLICE:
        layer.__class__ = trt.ISliceLayer
        if layer.num_inputs > 1:
            # start/size computed at run time from shape tensors, but from parse-time constants that hold the
            # static data shape (batch 1): patch those constants
            dynamic.append(layer.name)
            ds = tuple(layer.get_input(0).shape)
            if ds[0] == 1:
                for li, v in baked_shape_constants(layer, ds):
                    patch_const[li] = v
            continue
        sh = list(layer.shape)
        if sh[0] == 1:
            patch_slice.append((i, sh))
# Pass 2: change the batch and apply the patches.
inp.shape = [B] + shape[1:]
for i, rd in patch_shuffle:
    layer = net.get_layer(i)
    layer.__class__ = trt.IShuffleLayer
    layer.reshape_dims = [B] + rd[1:]
for i, sh in patch_slice:
    layer = net.get_layer(i)
    layer.__class__ = trt.ISliceLayer
    layer.shape = [B] + sh[1:]
keep = []                                    # the network keeps pointers to the weight arrays
for li, v in sorted(patch_const.items()):
    layer = net.get_layer(li)
    layer.__class__ = trt.IConstantLayer
    nv = np.array(v, dtype=v.dtype)
    nv[0] = B
    keep.append(nv)
    layer.weights = trt.Weights(nv)
n_shuffle, n_slice = len(patch_shuffle), len(patch_slice)
# Pass 3: every batch-carrying tensor must now have batch B.
bad = []
for i in range(net.num_layers):
    layer = net.get_layer(i)
    for k in range(layer.num_outputs):
        o = layer.get_output(k)
        if o.name in carry and tuple(o.shape)[:1] != (B,):
            bad.append("%d %s -> %s" % (i, layer.name, tuple(o.shape)))
if bad:
    raise SystemExit("batch-carrying tensors without batch %d: %s" % (B, "; ".join(bad[:5])))
out = tuple(net.get_output(0).shape)
if out[0] != B:
    raise SystemExit("network output %s does not carry batch %d" % (out, B))
print("dynamic slices/reshapes: %d, baked shape constants patched: %d" % (len(dynamic), len(patch_const)))
if a.dry:
    print("dry %s: batch %d, input %s, output %s, patched %d shuffles + %d slices" % (
        a.onnx, B, tuple(inp.shape), out, n_shuffle, n_slice))
    sys.exit(0)
config = builder.create_builder_config()
config.max_workspace_size = a.workspace_mb << 20
config.set_flag(trt.BuilderFlag.FP16)
plan = builder.build_serialized_network(net, config)
if plan is None:
    print("engine build failed", file=sys.stderr)
    sys.exit(4)
with open(a.out, "wb") as f:
    f.write(plan)
print("built %s: batch %d, input %s, output %s, %d network layers, patched %d shuffles + %d slices" % (
    a.out, B, tuple(inp.shape), out, net.num_layers, n_shuffle, n_slice))
