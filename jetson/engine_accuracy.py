#!/usr/bin/env python3
"""Top-1 of a TensorRT engine on the uint8 224x224 crops written by train/make_eval_crops.py, the same pixels the host
used for PyTorch, so the only difference left is FP16 against FP32. Engines with a normalized float input (the stock
and the FusedHaarStem plugin graphs) get the normalized crop; with --u8 the FusedHaarStemU8 engine gets the raw BGR
bytes. Python 3.6 (no f-strings).
  python3 jetson/engine_accuracy.py --plan weenet_do.plan --crops crops_u8.npy --labels labels.npy --out acc.json
  LD_PRELOAD=build/libfused_haar_stem.so python3 jetson/engine_accuracy.py --plan weenet_do_u8.plan --u8 \
      --plugin build/libfused_haar_stem.so --crops crops_u8.npy --labels labels.npy --out acc_u8.json
"""
from __future__ import print_function
import argparse
import json
import os

import numpy as np
import tensorrt as trt

from trt_runner import Runner, cuda_init, normalize_hwc_uint8

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--crops", required=True)
ap.add_argument("--labels", required=True)
ap.add_argument("--label", default="")
ap.add_argument("--out", required=True)
ap.add_argument("--plugin", default="", help="plugin library, loaded before the engine is deserialized")
ap.add_argument("--u8", action="store_true", help="FusedHaarStemU8 engine: feed the crop as raw BGR bytes")
a = ap.parse_args()
if a.plugin:
    import ctypes
    ctypes.CDLL(a.plugin)

cuda_init()
X = np.load(a.crops)
Y = np.load(a.labels).astype(np.int64)
if X.shape[0] != Y.shape[0] or X.shape[1:] != (224, 224, 3):
    raise SystemExit("bad test set shapes %s %s" % (X.shape, Y.shape))
r = Runner(open(a.plan, "rb").read(), trt.Logger(trt.Logger.WARNING))
preds, logits, nonfinite = [], [], 0
for k in range(X.shape[0]):
    if a.u8:      # crops are RGB; the kernel expects OpenCV's BGR bytes in a float32-typed buffer of the same size
        l = r(np.ascontiguousarray(X[k][:, :, ::-1]).reshape(-1).view(np.float32)).astype(np.float64)
    else:
        l = r(normalize_hwc_uint8(X[k])).astype(np.float64)
    nonfinite += int(not np.all(np.isfinite(l)))
    preds.append(int(l.argmax()))
    logits.append([round(float(v), 4) for v in l])
preds = np.array(preds)
res = {"label": a.label, "plan": os.path.basename(a.plan), "n": int(len(Y)), "top1": float((preds == Y).mean()),
       "nonfinite": nonfinite, "preds": preds.tolist(), "logits": logits}
json.dump(res, open(a.out, "w"))
print("%s top1 %.4f nonfinite %d" % (a.label, res["top1"], nonfinite))
