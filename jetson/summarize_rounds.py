#!/usr/bin/env python3
"""Engine latency and energy over timing rounds, from the <label>.r<round>.json files of jetson/time_engine.py.

Per label: latency = median of the per-round medians of trtexec's GPU compute time; energy per image = mean over rounds
of the board energy (POM_5V_IN power / throughput); also the per-round medians and the lowest GPU clock seen. A label
whose rounds are not exactly --rounds, or a round whose median GPU clock was below 921.6 MHz, is reported as an error
(exit 1). Runs on the board or the host (Python 3.6+).
  python3 jetson/summarize_rounds.py out/latency
"""
from __future__ import print_function
import argparse
import glob
import json
import os
import re


def med(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


ap = argparse.ArgumentParser()
ap.add_argument("outdir")
ap.add_argument("--rounds", type=int, default=3)
a = ap.parse_args()
by = {}
for f in sorted(glob.glob(os.path.join(a.outdir, "*.r*.json"))):
    if not re.search(r"\.r[0-9]+\.json$", f):
        continue
    d = json.load(open(f))
    by.setdefault(d["label"], []).append(d)
if not by:
    raise SystemExit("no <label>.r<round>.json in %s" % a.outdir)
bad = []
print("%-24s %9s %9s  %-26s %s" % ("label", "GPU ms", "mJ/image", "per-round GPU ms", "min GPU MHz"))
for lab in sorted(by):
    rs = sorted(by[lab], key=lambda d: d["round"])
    rounds = [d["round"] for d in rs]
    if rounds != list(range(1, a.rounds + 1)):
        bad.append("%s: rounds %s, expected 1..%d" % (lab, rounds, a.rounds))
    slow = [d["round"] for d in rs if d["clocks"]["gpu_below_921"]]
    if slow:
        bad.append("%s: GPU clock median below 921.6 MHz in round(s) %s" % (lab, slow))
    ms = med([d["gpu_ms"]["median"] for d in rs])
    mj = sum(d["energy_mj"]["board_in"] for d in rs) / float(len(rs))
    print("%-24s %9.3f %9.1f  %-26s %.1f" % (lab, ms, mj, ", ".join("%.3f" % d["gpu_ms"]["median"] for d in rs),
                                            min(d["clocks"]["gpu_mhz_min"] for d in rs)))
if bad:
    raise SystemExit("\n".join(bad))
