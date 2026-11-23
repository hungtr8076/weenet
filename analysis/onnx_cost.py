#!/usr/bin/env python3
"""FLOPs and bytes of every node of an ONNX graph at batch 1, and its roofline bound on the Jetson Nano B01 (paper,
Sect. 6, Eq. (7)).

Per node: FLOPs (a multiply-add counts as two: Conv 2 * output elements * Cin/g * kh * kw, MatMul/Gemm 2 * output
elements * inner dimension, any other node one per output element) and bytes of its FP16 activation inputs, weights and
outputs. Bound of a node: max(FLOPs / P, bytes / Q) with
  P = 128 CUDA cores x 2 FLOPs per fused multiply-add x 2 (FP16) x 0.9216 GHz = 471.9 GFLOP/s
  Q = 25.6 GB/s (64-bit LPDDR4)
both datasheet peaks. Conv nodes also record group, kernel size and input channels, from which analysis/roofline.py
assigns the layer class. Tensor shapes come from one ONNX Runtime run with every node output exposed (symbolic
dimensions left by ONNX shape inference would otherwise be undercounted).
The summary is the engine-level bound: the sum over the nodes that TensorRT runs as separate kernels, approximated by
the "heavy" nodes (Conv, ConvTranspose, MatMul, Gemm, pooling, Resize); element-wise nodes are assumed fused into them.
  python analysis/onnx_cost.py onnx/weenet_do.onnx --json results/cost/weenet_do.json
"""
import argparse
import json
import os

import numpy as np
import onnx
from onnx import shape_inference

P_PEAK = 128 * 2 * 2 * 0.9216e9   # FLOP/s, FP16
Q_PEAK = 25.6e9                   # byte/s
BYTES = 2                         # FP16
HEAVY = {"Conv", "ConvTranspose", "MatMul", "Gemm", "MaxPool", "AveragePool", "GlobalAveragePool", "Resize"}


def shapes_of(model):
    """Tensor shapes at batch 1 from one ONNX Runtime run on zeros, every node output exposed as a graph output."""
    import onnxruntime as ort
    m = shape_inference.infer_shapes(model)
    known = {o.name for o in m.graph.output}
    for n in m.graph.node:
        for o in n.output:
            if o and o not in known:
                m.graph.output.append(onnx.ValueInfoProto(name=o))
                known.add(o)
    sess = ort.InferenceSession(m.SerializeToString(), providers=["CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        if i.type != "tensor(float)":
            raise SystemExit("unexpected input type %s for %s" % (i.type, i.name))
        feeds[i.name] = np.zeros([d if isinstance(d, int) else 1 for d in i.shape], np.float32)
    names = [o.name for o in sess.get_outputs()]
    outs = sess.run(names, feeds)
    s = {k: list(v.shape) for k, v in zip(names, outs) if hasattr(v, "shape")}
    for i in m.graph.input:
        s[i.name] = list(feeds[i.name].shape) if i.name in feeds else s.get(i.name, [])
    for t in m.graph.initializer:
        s[t.name] = list(t.dims)
    return m, s


def numel(d):
    return int(np.prod(d)) if d else 1


def conv_attrs(n, inits):
    """group, kernel size (largest of kernel_shape) and input channels of a Conv node with an initializer weight."""
    at = {a.name: a for a in n.attribute}
    if "kernel_shape" not in at or n.input[1] not in inits:
        raise SystemExit("conv %s has no kernel_shape attribute or no initializer weight" % n.name)
    g = at["group"].i if "group" in at else 1
    return {"group": g, "kernel": max(at["kernel_shape"].ints), "cin": inits[n.input[1]].dims[1] * g}


def node_cost(n, s, inits):
    ins = [x for x in n.input if x]
    act_in = [x for x in ins if x not in inits]
    w_in = [x for x in ins if x in inits]
    out = n.output[0]
    if out not in s:
        raise SystemExit("no shape for %s (%s)" % (out, n.op_type))
    o = numel(s[out])
    if n.op_type in ("Conv", "ConvTranspose"):
        w = s[n.input[1]]                       # Cout, Cin/g, kh, kw
        flops = 2 * o * numel(w[1:])
    elif n.op_type in ("MatMul", "Gemm"):
        flops = 2 * o * s[n.input[0]][-1]
    else:
        flops = o
    by = BYTES * (sum(numel(s[x]) for x in act_in if x in s) + sum(numel(s[x]) for x in w_in) + o)
    row = {"name": n.name, "op": n.op_type, "flops": flops, "bytes": by,
           "t_compute_ms": 1e3 * flops / P_PEAK, "t_mem_ms": 1e3 * by / Q_PEAK,
           "inputs": list(n.input), "outputs": list(n.output)}
    if n.op_type == "Conv":
        row.update(conv_attrs(n, inits))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--json", required=True, help="output: summary and per-node costs")
    a = ap.parse_args()
    m, s = shapes_of(onnx.load(a.onnx))
    inits = {t.name: t for t in m.graph.initializer}
    rows = [node_cost(n, s, inits) for n in m.graph.node]
    heavy = [r for r in rows if r["op"] in HEAVY]
    summ = {"onnx": os.path.basename(a.onnx), "nodes": len(rows), "heavy": len(heavy),
            "gflops": sum(r["flops"] for r in heavy) / 1e9, "mbytes": sum(r["bytes"] for r in heavy) / 1e6,
            "roofline_ms": sum(max(r["t_compute_ms"], r["t_mem_ms"]) for r in heavy),
            "mem_bound_frac": sum(r["t_mem_ms"] > r["t_compute_ms"] for r in heavy) / float(max(1, len(heavy)))}
    print(json.dumps(summ))
    out_dir = os.path.dirname(os.path.abspath(a.json))
    os.makedirs(out_dir, exist_ok=True)
    with open(a.json, "w") as f:
        json.dump({"summary": summ, "nodes": rows}, f)
        f.write("\n")


if __name__ == "__main__":
    main()
