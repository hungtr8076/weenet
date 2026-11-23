#!/usr/bin/env python3
"""Layers of built TensorRT 8.2 engines: the number of layers and the engine inspector's layer list, a cross-check of
the layers that a per-layer profile (trtexec --exportProfile) reports. Python 3.6 (no f-strings).

For each label=plan argument: deserialize the plan, print engine.num_layers and the bindings, and write the inspector's
layer list to <outdir>/<label>.layers.json (layer names only, unless the engine was built with detailed profiling
verbosity). Plugin libraries are loaded with --plugins (comma-separated). Writes <outdir>/engine_layers.json.
  python3 jetson/engine_layers.py --outdir out/layers wee=engines/weenet_do.plan
  python3 jetson/engine_layers.py --plugins build/libfused_haar_stem.so --outdir out/layers wee_u8=engines/weenet_do_u8.plan
"""
from __future__ import print_function
import argparse
import ctypes
import hashlib
import json
import os

import tensorrt as trt

ap = argparse.ArgumentParser()
ap.add_argument("--plugins", default="", help="comma-separated plugin libraries")
ap.add_argument("--outdir", required=True)
ap.add_argument("items", nargs="+", help="label=plan")
a = ap.parse_args()
for lib in [x for x in a.plugins.split(",") if x]:
    ctypes.CDLL(lib)
if not os.path.isdir(a.outdir):
    os.makedirs(a.outdir)
logger = trt.Logger(trt.Logger.WARNING)
trt.init_libnvinfer_plugins(logger, "")
rt = trt.Runtime(logger)
summary = []
for item in a.items:
    if item.count("=") != 1:
        raise SystemExit("expected label=plan, got %s" % item)
    label, plan = item.split("=")
    data = open(plan, "rb").read()
    eng = rt.deserialize_cuda_engine(data)
    if eng is None:
        raise SystemExit("cannot deserialize %s" % plan)
    info = eng.create_engine_inspector().get_engine_information(trt.LayerInformationFormat.JSON)
    with open(os.path.join(a.outdir, label + ".layers.json"), "w") as f:
        f.write(info)
    row = {"label": label, "plan": os.path.basename(plan), "md5": hashlib.md5(data).hexdigest(),
           "num_layers": eng.num_layers, "inspector_layers": len(json.loads(info).get("Layers", [])),
           "bindings": [[eng.get_binding_name(i), list(eng.get_binding_shape(i)), str(eng.get_binding_dtype(i)),
                         bool(eng.binding_is_input(i))] for i in range(eng.num_bindings)]}
    summary.append(row)
    print("%-14s engine.num_layers %3d  inspector %3d  md5 %s" % (label, row["num_layers"], row["inspector_layers"],
                                                                   row["md5"]))
json.dump(summary, open(os.path.join(a.outdir, "engine_layers.json"), "w"), indent=1)
