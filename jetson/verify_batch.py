#!/usr/bin/env python3
"""Check fixed-batch engines (jetson/build_batch.py) against the batch-1 engine of the same ONNX.

The 64 frames of jetson/frames.py go through the deployed float path and then through the batch-1 engine one at a time
and through each batch-B engine B at a time (row order shuffled within a launch so that a row mix-up would show).
Reports per B: max |logit difference|, argmax agreement with batch 1, and for each disagreeing frame the batch-1 top-2
logit margin. FP16 engines built separately pick different tactics, so bit equality is not expected. A batched engine
passes if max |diff| < 0.1 (a row mix-up or a wrong batch path moves logits by whole units) and every argmax
disagreement is a near-tie (batch-1 top-2 margin below 2 x max |diff|); exit 1 if any engine fails.
Python 3.6, no f-strings.
  python3 jetson/verify_batch.py --plans 1:engines/weenet_do.plan,2:engines/weenet_do_b2.plan --res 224 \
      --list data/split_seed0/test.csv --root $DATA_ROOT --label weenet_do --out out/verify_weenet_do.json
"""
from __future__ import print_function
import argparse
import json
import sys

import numpy as np
import tensorrt as trt

from frames import load_frames, pre_float
from trt_runner import AsyncRunner, cuda_init

ap = argparse.ArgumentParser()
ap.add_argument("--label", required=True)
ap.add_argument("--plans", required=True, help="comma-separated B:plan, must include 1:")
ap.add_argument("--res", type=int, required=True, help="engine input size")
ap.add_argument("--list", required=True, help="split CSV (path,label,...)")
ap.add_argument("--root", required=True, help="image root ($DATA_ROOT)")
ap.add_argument("--out", required=True)
a = ap.parse_args()

xs = [pre_float(f, a.res) for f in load_frames(a.list, a.root)]
cuda_init()
lg = trt.Logger(trt.Logger.WARNING)
plans = dict((int(kv.split(":", 1)[0]), kv.split(":", 1)[1]) for kv in a.plans.split(","))
if 1 not in plans or len(plans) < 2:
    raise SystemExit("--plans needs 1:<batch-1 plan> and at least one batched plan")


def run(r, b, idx):
    hin = r.host_in().view(np.uint8).reshape(b, -1)
    for k, i in enumerate(idx):
        np.copyto(hin[k], xs[i].view(np.uint8).reshape(-1))
    r.enqueue()
    return r.wait().reshape(b, -1)[:len(idx)].copy()


r1 = AsyncRunner(open(plans[1], "rb").read(), lg)
ref = np.stack([run(r1, 1, [i])[0] for i in range(len(xs))])
rs = np.random.RandomState(0)
res = {"label": a.label, "res": a.res, "n": len(xs), "plans": dict((str(b), p) for b, p in plans.items()), "by_batch": {}}
ok = True
for b in sorted(plans):
    if b == 1:
        continue
    r = AsyncRunner(open(plans[b], "rb").read(), lg)
    perm = list(rs.permutation(len(xs)))
    got = np.zeros_like(ref)
    for j in range(0, len(perm), b):
        idx = perm[j:j + b]
        got[idx] = run(r, b, idx)
    d = float(np.max(np.abs(got - ref)))
    agree = int(np.sum(np.argmax(got, 1) == np.argmax(ref, 1)))
    srt = np.sort(ref, 1)
    margins = [float(srt[i, -1] - srt[i, -2]) for i in range(len(xs)) if np.argmax(got[i]) != np.argmax(ref[i])]
    passed = d < 0.1 and all(m < 2 * d for m in margins)
    ok = ok and passed
    res["by_batch"][str(b)] = {"max_abs_logit_diff": d, "argmax_agree": agree, "disagree_b1_margins": margins,
                               "pass": passed}
    print("[verify] %s B=%d: max |logit diff| %.4f, argmax agree %d/%d (b1 top-2 margin of disagreements %s) -> %s" % (
        a.label, b, d, agree, len(xs), ["%.4f" % m for m in margins], "PASS" if passed else "FAIL"))
res["pass"] = ok
json.dump(res, open(a.out, "w"), indent=1)
sys.exit(0 if ok else 1)
