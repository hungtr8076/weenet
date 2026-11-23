#!/usr/bin/env python3
"""Isolated inputs of the stream-capacity estimate (paper, Sect. 4.4 and Eq. (8)), measured for one batch-1 engine
without the arrival process of jetson/streams.py. Python 3.6 (no f-strings).

  pre_1   median time of the deployed preprocessing of one frame on one thread (jetson/frames.py: square center crop,
          INTER_AREA to 240 px, bilinear to --res, then normalization (--pre float) or the bytes as they are (--pre u8))
  rate_w  preprocessed frames per second with --workers threads running back to back for --dur seconds
  gpu     median and P99 of the GPU-thread time per frame, launches back to back for --dur seconds: copy into the
          page-locked input, H2D copy, execution and D2H copy on one CUDA stream, argmax (the per-frame work of the GPU
          thread of jetson/streams.py)
  mhz     mean GPU clock sampled during the GPU loop
Frames: jetson/frames.py (64 test images at 640 x 480 BGR in memory). Writes one JSON with these fields and
k_gpu = floor(T / gpu median), k_cpu = floor(rate_w * T), k_pred = min(k_gpu, k_cpu), T = 1/30 s; analysis/capacity.py
computes both estimators from the file.
  python3 jetson/prebench.py --plan engines/weenet_do.plan --pre float --list data/split_seed0/test.csv \
      --root $DATA_ROOT --label wee --out out/capacity/pre_wee.json
"""
from __future__ import print_function
import argparse
import ctypes
import glob
import json
import os
import threading
import time

import numpy as np
import tensorrt as trt

from frames import load_frames, pre_float, pre_u8
from trt_runner import AsyncRunner, cuda_init


def one(pattern):
    hits = glob.glob(pattern)
    if len(hits) != 1:
        raise SystemExit("expected one match for %s, found %s" % (pattern, hits))
    return hits[0]


GPU_FREQ = one("/sys/devices/57000000.gpu/devfreq/*/cur_freq")

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True, help="batch-1 engine")
ap.add_argument("--pre", choices=["float", "u8"], required=True, help="u8 for the FusedHaarStemU8 engine")
ap.add_argument("--res", type=int, default=224, help="engine input size")
ap.add_argument("--plugin", default="", help="plugin library for engines with a plugin layer")
ap.add_argument("--std-plugins", action="store_true", help="register TensorRT's standard plugins")
ap.add_argument("--list", required=True, help="split CSV (path,label,...), e.g. data/split_seed0/test.csv")
ap.add_argument("--root", required=True, help="image root ($DATA_ROOT)")
ap.add_argument("--workers", type=int, default=3, help="preprocessing threads")
ap.add_argument("--dur", type=float, default=15.0, help="seconds of each back-to-back loop")
ap.add_argument("--label", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
if a.plugin:
    ctypes.CDLL(a.plugin)
frames = load_frames(a.list, a.root)
pre_fn = pre_float if a.pre == "float" else pre_u8
n = len(frames)

# 1) one thread
for f in frames[:16]:
    pre_fn(f, a.res)
t1 = []
for i in range(320):
    g0 = time.perf_counter()
    pre_fn(frames[i % n], a.res)
    t1.append((time.perf_counter() - g0) * 1000.0)

# 2) --workers threads back to back
count = [0] * a.workers
stop = [False]


def work(k):
    i = k
    while not stop[0]:
        pre_fn(frames[i % n], a.res)
        count[k] += 1
        i += a.workers


ths = [threading.Thread(target=work, args=(k,)) for k in range(a.workers)]
w0 = time.perf_counter()
for th in ths:
    th.start()
time.sleep(a.dur)
stop[0] = True
for th in ths:
    th.join()
rate_w = sum(count) / (time.perf_counter() - w0)

# 3) GPU-thread work per frame, launches back to back
cuda_init()
lg = trt.Logger(trt.Logger.WARNING)
if a.std_plugins:
    trt.init_libnvinfer_plugins(lg, "")
r = AsyncRunner(open(a.plan, "rb").read(), lg)
xs = [pre_fn(f, a.res) for f in frames]
if xs[0].nbytes != r.meta[r.i_in][2]:
    raise SystemExit("preprocessed frame has %d bytes, engine input wants %d (wrong --pre or --res?)"
                     % (xs[0].nbytes, r.meta[r.i_in][2]))
hin = r.host_in().view(np.uint8).reshape(-1)
for i in range(50):
    np.copyto(hin, xs[i % n].view(np.uint8).reshape(-1))
    r.enqueue()
    r.wait()
g, mhz = [], []
e0 = time.perf_counter()
i = 0
while time.perf_counter() - e0 < a.dur:
    g0 = time.perf_counter()
    np.copyto(hin, xs[i % n].view(np.uint8).reshape(-1))
    r.enqueue()
    int(np.argmax(r.wait()))
    g.append((time.perf_counter() - g0) * 1000.0)
    if i % 50 == 0:
        mhz.append(int(open(GPU_FREQ).read()) / 1e6)
    i += 1

T_MS = 1000.0 / 30.0
row = {"label": a.label, "plan": os.path.basename(a.plan), "pre": a.pre, "res": a.res, "workers": a.workers,
       "dur_s": a.dur, "pre_1_ms_median": float(np.median(t1)), "pre_1_ms_p99": float(np.percentile(t1, 99)),
       "rate_w_fps": rate_w, "gpu_ms_median": float(np.median(g)), "gpu_ms_p99": float(np.percentile(g, 99)),
       "gpu_launches": len(g), "gpu_mhz_mean": float(np.mean(mhz)),
       "k_gpu": int(np.floor(T_MS / np.median(g))), "k_cpu": int(np.floor(rate_w * T_MS / 1000.0))}
row["k_pred"] = min(row["k_gpu"], row["k_cpu"])
out_dir = os.path.dirname(os.path.abspath(a.out))
if not os.path.isdir(out_dir):
    os.makedirs(out_dir)
json.dump(row, open(a.out, "w"), indent=1)
print(json.dumps(row))
