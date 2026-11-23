#!/usr/bin/env python3
"""Per-layer GPU time of the TensorRT engines against the roofline bound (paper, Sect. 6: Table 6, Fig. 3, the input
stage of WeeNet and the layer-class latency model).

Inputs per engine (keys and ONNX stems in analysis/engines.py):
  <profiles>/prof_<key>.json   trtexec per-layer profile (--dumpProfile --separateProfileRun --exportProfile)
  <costs>/<stem>.json          per-node FLOPs, bytes and bound of the engine's ONNX graph (analysis/onnx_cost.py)
TensorRT names a fused layer after its ONNX nodes ("A + B + C"), so each profiled layer is matched to its nodes by name.
Class of a layer, by its heaviest node (most FLOPs among Conv, ConvTranspose, MatMul, Gemm, pooling, Resize):
  dense      Conv with groups = 1 and kernel > 1x1
  pointwise  Conv with groups = 1 and kernel 1x1
  depthwise  Conv with groups = input channels > 1
  grouped    other grouped Conv
  matmul     MatMul or Gemm
  pool       MaxPool, AveragePool, GlobalAveragePool, Resize
  myelin     TensorRT "ForeignNode" blocks (attention blocks fused by TensorRT; not classified)
  other      no heavy node: element-wise, activation, format conversion (reformatting), shuffle and plugin layers
Bound of a layer: sum over its heavy nodes of max(FLOPs / P, bytes / Q) (Eq. (7)); none for myelin and other layers.
The layer is memory-bound when the summed memory term exceeds the summed compute term.
  python analysis/roofline.py --profiles results/profiles --costs results/cost
"""
import argparse
import collections
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engines as E  # noqa: E402

HEAVY = ("Conv", "ConvTranspose", "MatMul", "Gemm", "MaxPool", "AveragePool", "GlobalAveragePool", "Resize")
CLASSES = ("dense", "pointwise", "depthwise", "matmul", "pool")
LABEL = {"dense": "Dense conv (kernel > 1x1)", "pointwise": "Pointwise conv (1x1)", "depthwise": "Depthwise conv",
         "matmul": "MatMul / Gemm", "pool": "Pooling", "grouped": "Grouped conv", "myelin": "Attention (fused)",
         "other": "Non-convolution"}
# Layer-class latency model, fixed before it was tested: the pooled measured-over-bound ratio of each class over the
# 13 profiled engines (twelve models at 224 px and YOLO11n-cls at 160 px), rounded, plus the mean time of one
# non-convolution layer over the same engines.
MODEL_RATIO = {"dense": 1.87, "pointwise": 3.44, "depthwise": 15.3, "matmul": 14.3, "pool": 6.8}
MODEL_OTHER_MS = 0.045


def conv_class(node):
    g, k, cin = node["group"], node["kernel"], node["cin"]
    if g > 1 and g == cin:
        return "depthwise"
    if g > 1:
        return "grouped"
    return "pointwise" if k == 1 else "dense"


def classify(layer_name, nodes):
    """(class, heavy nodes of the layer) for one profiled TensorRT layer."""
    if layer_name.startswith("{ForeignNode"):
        return "myelin", []
    hit = [nodes[x.strip()] for x in layer_name.split(" + ") if x.strip() in nodes]
    heavy = [h for h in hit if h["op"] in HEAVY]
    if not heavy:
        return "other", []
    top = max(heavy, key=lambda h: h["flops"])
    if top["op"] in ("MatMul", "Gemm"):
        return "matmul", heavy
    if top["op"] in ("MaxPool", "AveragePool", "GlobalAveragePool", "Resize"):
        return "pool", heavy
    if top["op"] != "Conv":
        raise SystemExit("no class rule for %s (%s)" % (top["name"], top["op"]))
    return conv_class(top), heavy


def load_json(path):
    if not os.path.isfile(path):
        raise SystemExit("missing %s" % path)
    with open(path) as f:
        return json.load(f)


def load_profile(path):
    prof = load_json(path)
    if "count" not in prof[0]:
        raise SystemExit("%s is not a trtexec --exportProfile file" % path)
    return prof[1:]


def layers_of(key, a):
    """Per profiled layer: dict(name, cls, ms, bound, t_compute, t_mem)."""
    cost = load_json(os.path.join(a.costs, E.STEM[key] + ".json"))
    nodes = {r["name"]: r for r in cost["nodes"]}
    out = []
    for p in load_profile(os.path.join(a.profiles, "prof_%s.json" % key)):
        c, heavy = classify(p["name"], nodes)
        out.append({"name": p["name"], "cls": c, "ms": p["averageMs"],
                    "bound": sum(max(h["t_compute_ms"], h["t_mem_ms"]) for h in heavy),
                    "t_compute": sum(h["t_compute_ms"] for h in heavy), "t_mem": sum(h["t_mem_ms"] for h in heavy)})
    return out, cost["summary"]


def class_totals(lay, keys):
    t = collections.defaultdict(lambda: [0.0, 0.0, 0])          # class -> measured ms, bound ms, layers
    for k in keys:
        for r in lay[k]:
            t[r["cls"]][0] += r["ms"]
            t[r["cls"]][1] += r["bound"]
            t[r["cls"]][2] += 1
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", required=True, help="folder with prof_<key>.json")
    ap.add_argument("--costs", required=True, help="folder with <onnx stem>.json from analysis/onnx_cost.py")
    a = ap.parse_args()

    twelve = [k for k, _, _, _ in E.TWELVE]
    abl = [k for k, _, _, _ in E.ABLATION]
    lay, summ = {}, {}
    for k in twelve + ["yolo160"] + abl:
        lay[k], summ[k] = layers_of(k, a)

    print("== Engine-level bound (sum over heavy ONNX nodes) against engine latency, twelve models at 224 px")
    for k in twelve:
        print("%-22s bound %6.2f ms  latency %6.2f ms  bound/latency %5.1f %%  ONNX GFLOP %.3f"
              % (E.NAME[k], summ[k]["roofline_ms"], E.LATENCY_MS[k], 100 * summ[k]["roofline_ms"] / E.LATENCY_MS[k],
                 summ[k]["gflops"]))
    lat = [E.LATENCY_MS[k] for k in twelve]
    frac = [summ[k]["roofline_ms"] / E.LATENCY_MS[k] for k in twelve]
    rho_b, p_b = spearmanr([summ[k]["roofline_ms"] for k in twelve], lat)
    rho_f, p_f = spearmanr([summ[k]["gflops"] for k in twelve], lat)
    print("bound / latency: %.0f to %.0f %%; Spearman with latency: bound %.3f (p = %.3f), ONNX FLOPs %.3f (p = %.3f)"
          % (100 * min(frac), 100 * max(frac), rho_b, p_b, rho_f, p_f))

    print("\n== Profiled engines (Fig. 3): layers, summed profile against engine latency, time share by class (%)")
    shares = {}
    for k in twelve + ["yolo160"]:
        s = sum(r["ms"] for r in lay[k])
        sh = collections.defaultdict(float)
        for r in lay[k]:
            sh[r["cls"]] += 100 * r["ms"] / s
        shares[k] = (len(lay[k]), s, sh)
        print("%-22s layers %3d  profiled %6.2f ms  (%+5.1f %% vs %.2f ms) | %s"
              % (E.NAME[k], len(lay[k]), s, 100 * (s / E.LATENCY_MS[k] - 1), E.LATENCY_MS[k],
                 ", ".join("%s %.0f" % (c, v) for c, v in sorted(sh.items(), key=lambda kv: -kv[1]))))
    anal = [k for k in twelve if k != "yolo224"]       # YOLO11n-cls: profile sum exceeds the engine latency
    dev = max(abs(shares[k][1] / E.LATENCY_MS[k] - 1) for k in anal)
    big = [shares[k][2].get("other", 0.0) for k in anal if shares[k][0] > 100]
    print("largest |profile - latency| over the %d engines of Table 6: %.1f %%; non-convolution share of the engines "
          "above 100 layers: %.0f to %.0f %%" % (len(anal), 100 * dev, min(big), max(big)))

    print("\n== Table 6: measured time and bound by layer class, %d engines (twelve models without YOLO11n-cls)"
          % len(anal))
    non_conv = [r["ms"] for k in anal for r in lay[k] if r["cls"] == "other"]
    if not non_conv:
        raise SystemExit("no non-convolution layer in the profiles")
    b0 = float(np.mean(non_conv))
    print("mean non-convolution layer: %.3f ms over %d layers (per-layer overhead subtracted in 'adjusted')"
          % (b0, len(non_conv)))
    print("%-27s %6s %9s %9s %6s %8s %11s %7s" % ("class", "layers", "meas ms", "bound ms", "ratio", "adjusted",
                                                 "per engine", "mem %"))
    for c in CLASSES:
        rows = [r for k in anal for r in lay[k] if r["cls"] == c]
        m, b = sum(r["ms"] for r in rows), sum(r["bound"] for r in rows)
        adj = (m - len(rows) * b0) / b
        per = []
        for k in anal:
            rk = [r for r in lay[k] if r["cls"] == c]
            if len(rk) >= 3:
                per.append(sum(r["ms"] for r in rk) / sum(r["bound"] for r in rk))
        mem = 100.0 * sum(r["t_mem"] > r["t_compute"] for r in rows) / len(rows)
        print("%-27s %6d %9.2f %9.2f %6.1f %8s %11s %7.0f"
              % (LABEL[c], len(rows), m, b, m / b, ("%.1f" % adj) if adj > 0 else "-",
                 ("%.1f-%.1f" % (min(per), max(per))) if len(per) >= 2 else "-", mem))
    print("%-27s %6d %9.2f" % (LABEL["other"], len(non_conv), sum(non_conv)))
    rest = sorted({r["cls"] for k in anal for r in lay[k]} - set(CLASSES) - {"other"})
    for c in rest:
        print("%-27s %6d (not classified)" % (LABEL[c], sum(r["cls"] == c for k in anal for r in lay[k])))
    mm = [r["ms"] for k in anal for r in lay[k] if r["cls"] == "matmul"]
    print("mean MatMul / Gemm layer: %.3f ms" % np.mean(mm))

    print("\n== Input stage of WeeNet (stock engine) and the plugin engine")
    cost = {r["name"]: r for r in load_json(os.path.join(a.costs, E.STEM["wee"] + ".json"))["nodes"]}
    stem = [r for r in lay["wee"] if r["name"].startswith("/stem/") or "to /stem/" in r["name"]]
    for r in stem:
        top = r["name"].split(" + ")[0]
        b = max(cost[top]["t_compute_ms"], cost[top]["t_mem_ms"]) if top in cost else 0.0
        print("  %-80s %.3f ms%s" % (r["name"][:80], r["ms"], ("  %.1f x its bound" % (r["ms"] / b)) if b else ""))
    plug = load_profile(os.path.join(a.profiles, "prof_wee_u8.json"))
    p_stem = [p for p in plug if p["name"].startswith("stem_")]   # plugin node name (plugin/make_stem_onnx.py)
    if len(p_stem) != 1:
        raise SystemExit("expected one plugin layer in prof_wee_u8.json, found %s" % [p["name"] for p in p_stem])
    s_ms = sum(r["ms"] for r in stem)
    s_bound = sum(r["bound"] for r in stem)
    rest_w = sum(r["ms"] for r in lay["wee"]) - s_ms
    rest_u = sum(p["averageMs"] for p in plug) - p_stem[0]["averageMs"]
    print("stock: %d layers, %.2f ms, bound %.2f ms | plugin layer %s: %.2f ms | saving %.2f ms | rest of the two "
          "engines differs by %.2f ms" % (len(stem), s_ms, s_bound, p_stem[0]["name"], p_stem[0]["averageMs"],
                                          s_ms - p_stem[0]["averageMs"], abs(rest_w - rest_u)))

    print("\n== Layer-class latency model: bound x fixed class ratio + %.3f ms per non-convolution layer"
          % MODEL_OTHER_MS)
    t13 = class_totals(lay, twelve + ["yolo160"])
    print("pooled over the 13 profiled engines: %s; non-convolution %.4f ms per layer"
          % (", ".join("%s %.2f" % (c, t13[c][0] / t13[c][1]) for c in CLASSES), t13["other"][0] / t13["other"][2]))
    pred = {}
    for k in abl + ["yolo160"]:
        pred[k] = sum(MODEL_OTHER_MS if r["cls"] == "other" else
                      (0.0 if r["cls"] == "myelin" else r["bound"] * MODEL_RATIO[r["cls"]]) for r in lay[k])
        err = 100 * (pred[k] - E.LATENCY_MS[k]) / E.LATENCY_MS[k]
        print("%-22s predicted %.2f ms  measured %.2f ms  error %+.1f %%  (ONNX GFLOP %.2f)"
              % (E.NAME[k], pred[k], E.LATENCY_MS[k], err, summ[k]["gflops"]))
    within = sum(abs(pred[k] - E.LATENCY_MS[k]) <= 0.15 * E.LATENCY_MS[k] for k in pred)
    pairs = list(itertools.combinations(abl, 2))
    wrong = [(x, y) for x, y in pairs if (pred[x] < pred[y]) != (E.LATENCY_MS[x] < E.LATENCY_MS[y])]
    print("within 15 %%: %d of %d engines; ablation pairs in the measured order: %d of %d; misordered: %s"
          % (within, len(pred), len(pairs) - len(wrong), len(pairs),
             ", ".join("%s / %s (measured %.2f ms apart)"
                       % (E.NAME[x], E.NAME[y], abs(E.LATENCY_MS[x] - E.LATENCY_MS[y])) for x, y in wrong)))


if __name__ == "__main__":
    main()
