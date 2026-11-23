#!/usr/bin/env python3
"""ONNX graphs for the TensorRT 8.2 engines on the B01 (opset 13, batch 1, input "image", output "logits", FP32).

WeeNet-DO is first folded into a plain WeeNet (fold_do). --graph selects one of four forms of the Haar stem:
  two-conv   (default) the stem written as conv 4x4/4 + ReLU + conv 1x1 + ReLU (export_graph), so the whole network
             is Conv / Relu / Add / GlobalAveragePool / Gemm. Input 1 x 3 x 224 x 224, normalized. Every WeeNet latency
             in the comparison tables uses this graph.
  haar-ops   the Haar stem exported as written in PyTorch (pixel unshuffle, add, subtract, abs, average pool).
  plugin     the stem's BN folded into the 1x1 conv, then the Haar transform, the 1x1 conv and its ReLU replaced by one
             FusedHaarStem node (plugin/make_stem_onnx.py). Input 1 x 3 x 224 x 224, normalized.
  plugin-u8  the same with a FusedHaarStemU8 node. Input = raw BGR bytes of a 224 x 224 x 3 frame, declared as FP32
             1 x 3 x 112 x 112 (the same byte count); the normalization runs in the kernel.
The plugin graphs need build/libfused_haar_stem.so (plugin/Makefile) to build and run.

Weights: --ckpt <state_dict.pt> (a checkpoint of train/merged3.py), or --random for a latency-only graph (TensorRT
timing does not depend on the weight values; random BN statistics keep the folded biases distinct).

Every graph ONNX Runtime can run is checked against PyTorch on random inputs: rel-RMS <= 1e-5 and equal argmax, else
exit 1. For the plugin graphs, the part of the graph after the plugin node is checked the same way.
  python weenet/export_onnx.py --model weenet_z_ds4_do --ckpt checkpoints/weenet_do_merged3_seed0.pt --out onnx/weenet_do.onnx
  python weenet/export_onnx.py --model weenet_z_ds4_do --ckpt checkpoints/weenet_do_merged3_seed0.pt --graph plugin-u8 \
      --out onnx/weenet_do_u8.onnx
"""
import argparse
import os
import sys
import tempfile

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from weenet import model as wm  # noqa: E402

GRAPHS = ("two-conv", "haar-ops", "plugin", "plugin-u8")


class Tail(nn.Module):
    """WeeNet after the fixed Haar transform (input = the 21 features): the stem BN folded into the 1x1 conv, then the
    1x1 conv's BN + ReLU, the body, pool and head. Its first ONNX node is Conv(image) followed by Relu."""

    def __init__(self, m):
        super().__init__()
        bn, cba = m.stem[1], m.stem[2]
        s = bn.weight / (bn.running_var + bn.eps).sqrt()
        t = bn.bias - bn.running_mean * s
        conv = nn.Conv2d(21, cba.conv.out_channels, 1, bias=True)
        with torch.no_grad():
            conv.weight.copy_(cba.conv.weight * s.view(1, -1, 1, 1))
            conv.bias.copy_((cba.conv.weight[:, :, 0, 0] * t.view(1, -1)).sum(1))
        self.conv, self.bn, self.act, self.body, self.pool, self.head = conv, cba.bn, cba.act, m.body, m.pool, m.head

    def forward(self, f):
        x = self.body(self.act(self.bn(self.conv(f))))
        return self.head(self.pool(x).flatten(1))


def rand_bn(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for mod in m.modules():
            if isinstance(mod, nn.BatchNorm2d):
                n = mod.num_features
                mod.weight.copy_(0.5 + torch.rand(n, generator=g))
                mod.bias.copy_(0.2 * torch.randn(n, generator=g))
                mod.running_mean.copy_(0.2 * torch.randn(n, generator=g))
                mod.running_var.copy_(0.5 + torch.rand(n, generator=g))
    return m


def rel_rms(path, fn, inputs):
    import onnxruntime as ort
    sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
    worst, same = 0.0, True
    for x in inputs:
        with torch.no_grad():
            ref = fn(x).numpy()
        y = sess.run(None, {"image": x.numpy()})[0]
        worst = max(worst, float(np.sqrt(((y - ref) ** 2).mean()) / (np.sqrt((ref ** 2).mean()) + 1e-12)))
        same &= bool((y.argmax(1) == ref.argmax(1)).all())
    return worst, same


def onnx_export(m, path, shape):
    torch.onnx.export(m, torch.zeros(*shape), path, input_names=["image"], output_names=["logits"], opset_version=13,
                      dynamo=False)


def check(path, fn, shape, what, n=4, tol=1e-5):
    torch.manual_seed(0)
    worst, same = rel_rms(path, fn, [torch.randn(*shape) for _ in range(n)])
    if worst > tol or not same:
        raise SystemExit("%s: ORT vs PyTorch rel-RMS %.2e, argmax equal %s" % (what, worst, same))
    return worst


def load_weenet(name, ckpt, random):
    m = wm.build(name, 6)
    if ckpt:
        m.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    elif random:
        rand_bn(m)
    else:
        raise SystemExit("WeeNet needs --ckpt or --random")
    m.eval()
    return m


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True, choices=wm.MODELS)
    ap.add_argument("--out", required=True)
    w = ap.add_mutually_exclusive_group(required=True)
    w.add_argument("--ckpt", default="")
    w.add_argument("--random", action="store_true")
    ap.add_argument("--graph", choices=GRAPHS, default="two-conv", help="how the Haar stem is exported")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    torch.manual_seed(0)
    shape = (1, 3, 224, 224)
    m = load_weenet(a.model, a.ckpt, a.random)
    plain = wm.fold_do(m) if a.model == "weenet_z_ds4_do" else m
    if a.graph in ("two-conv", "haar-ops"):
        onnx_export(wm.export_graph(plain) if a.graph == "two-conv" else plain, a.out, shape)
        r = check(a.out, m, shape, a.out)
        print("wrote %s (%s graph), rel-RMS vs the PyTorch model %.1e" % (a.out, a.graph, r))
        return
    if not isinstance(plain.stem[0], wm.HaarScat):
        raise SystemExit("--graph %s needs a model with the Haar stem" % a.graph)
    from plugin.make_stem_onnx import rewrite
    tail = Tail(plain).eval()
    haar = plain.stem[0]
    with tempfile.TemporaryDirectory() as tmp:
        p_tail = os.path.join(tmp, "tail.onnx")
        onnx_export(tail, p_tail, (1, 21, 56, 56))
        torch.manual_seed(0)
        imgs = [torch.randn(*shape) for _ in range(4)]
        with torch.no_grad():
            full = max(float((tail(haar(x)) - m(x)).abs().max()) for x in imgs)
            r, same = rel_rms(p_tail, tail, [haar(x) for x in imgs])
        if full > 1e-4 or r > 1e-5 or not same:
            raise SystemExit("tail check failed: max|tail - model| %.2e, ORT rel-RMS %.2e, argmax %s" % (full, r, same))
        n = rewrite(p_tail, a.out, u8=a.graph == "plugin-u8")
    print("wrote %s (%s graph, %d nodes); tail rel-RMS %.1e, tail vs model max|diff| %.1e"
          % (a.out, a.graph, n, r, full))


if __name__ == "__main__":
    main()
