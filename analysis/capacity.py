#!/usr/bin/env python3
"""Stream capacity of one board: the estimate of Eq. (8) from isolated measurements against the measured capacity
(paper, Sect. 7: Tables 7 and 8, the synchronized and the batched runs).

Inputs in one folder, per engine key (analysis/engines.py):
  pre_<key>.json         isolated inputs (jetson/prebench.py): rate r_c of three preprocessing threads, median and
                         P99 of the GPU-thread time per frame t_g, each over 15 s without the arrival process
  even_<key>_<n>.json    staggered streams (jetson/streams.py --phase even, policy b1), one file per repeat n; a repeat
                         split over two files is written <n> and <n>b
  aligned_<key>.json     optional: synchronized streams (--phase aligned)
  batch_<key>.json       optional: batched GPU policy (--policy batch)
Estimate: K_hat = min(floor(T / t_g), floor(r_c T)), T = 1/30 s, with t_g the median ("median" estimator) or the P99
("P99" estimator). Measured capacity K* (Eq. (4)): the largest K at which every run, in every repeat, has a miss
rate below 1 %. The board power and the P99 frame latency at K* are read from the last repeat at K*.
  python analysis/capacity.py results/capacity
"""
import argparse
import glob
import json
import math
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import engines as E  # noqa: E402

MISS_CAP = 0.01
# Engines whose capacity was measured after both estimators had been fixed (the second block of Table 8)
MEASURED_AFTER = ("shufflenet", "repvgg", "resnet18", "effformer")
LABEL = dict(E.NAME, yolo224="YOLO11n-cls 224 px")


def load(path):
    with open(path) as f:
        return json.load(f)


def cap(runs):
    ok = [r["K"] for r in runs if r["miss_rate"] < MISS_CAP]
    return max(ok) if ok else 0


def measured(folder, key):
    files = []
    for fn in sorted(glob.glob(os.path.join(folder, "even_*.json"))):
        m = re.fullmatch(r"even_(.+)_([0-9]+)b?\.json", os.path.basename(fn))
        if not m:
            raise SystemExit("unexpected file name %s" % fn)
        if m.group(1) == key:
            files.append((fn, int(m.group(2))))
    if not files:
        raise SystemExit("no even_%s_<n>.json in %s" % (key, folder))
    reps = {}
    for fn, n in files:
        d = load(fn)
        if d["phase"] != "even" or d["policy"] != "b1" or d["workers"] != 3:
            raise SystemExit("%s: expected phase even, policy b1, 3 workers" % fn)
        rep = reps.setdefault(n, {"runs": [], "dur": set()})
        rep["runs"] += d["runs"]
        rep["dur"].add(d["dur_s"])
    durs = set.union(*(r["dur"] for r in reps.values()))
    if len(durs) != 1:
        raise SystemExit("%s: repeats of different lengths %s" % (key, sorted(durs)))
    allruns = [r for rep in reps.values() for r in rep["runs"]]
    k_star = max([k for k in {r["K"] for r in allruns} if all(r["miss_rate"] < MISS_CAP for r in allruns if r["K"] == k)],
                 default=0)
    last = reps[max(reps)]["runs"]
    at = [r for r in last if r["K"] == k_star]
    if len(at) != 1:
        raise SystemExit("%s: last repeat has %d runs at K* = %d" % (key, len(at), k_star))
    near = [r for rep in reps.values() for r in rep["runs"] if r["K"] in (k_star, k_star + 1)]
    return {"k": k_star, "repeats": len(reps), "dur": durs.pop(), "w": at[0]["board_w_mean"],
            "miss": at[0]["miss_rate"], "mhz": [r["gpu_mhz_mean"] for r in near],
            "tg_run": [r["infer_ms_median"] for r in near if r.get("infer_ms_median") is not None]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", help="folder with pre_<key>.json and even_<key>_<n>.json")
    a = ap.parse_args()
    keys = sorted((re.fullmatch(r"pre_(.+)\.json", os.path.basename(f)).group(1)
                   for f in glob.glob(os.path.join(a.folder, "pre_*.json"))),
                  key=lambda k: E.LATENCY_MS[k])
    if not keys:
        raise SystemExit("no pre_<key>.json in %s" % a.folder)
    T = E.T_MS
    rows = {}
    for k in keys:
        p = load(os.path.join(a.folder, "pre_%s.json" % k))
        if p["workers"] != 3 or p["dur_s"] != 15.0:
            raise SystemExit("pre_%s.json: expected 3 workers and 15 s" % k)
        kc = math.floor(p["rate_w_fps"] * T / 1000.0)
        rows[k] = dict(p=p, m=measured(a.folder, k), kc=kc, med=min(math.floor(T / p["gpu_ms_median"]), kc),
                       p99=min(math.floor(T / p["gpu_ms_p99"]), kc))

    print("== Table 7: stream capacity, estimate against measurement")
    print("%-22s %8s %7s %8s %6s %8s %6s %3s %7s %5s %6s" % ("engine", "engine ms", "t_g ms", "P99/med", "T/t_g",
                                                           "K median", "K P99", "K*", "runs", "W", "mJ"))
    for k in keys:
        r, p, m = rows[k], rows[k]["p"], rows[k]["m"]
        print("%-22s %8.2f %7.2f %8.2f %6.2f %8d %6d %3d %7s %5.2f %6.1f"
              % (LABEL[k], E.LATENCY_MS[k], p["gpu_ms_median"], p["gpu_ms_p99"] / p["gpu_ms_median"],
                 T / p["gpu_ms_median"], r["med"], r["p99"], m["k"], "%dx%.0f" % (m["repeats"], m["dur"]), m["w"],
                 1000.0 * m["w"] / (30.0 * m["k"])))
    over = [rows[k]["p"]["gpu_ms_median"] - E.LATENCY_MS[k] for k in keys]
    print("CPU term: three threads preprocess %.0f to %.0f frames/s (K_c >= %d); t_g - engine latency: %.2f to %.2f ms"
          % (min(rows[k]["p"]["rate_w_fps"] for k in keys), max(rows[k]["p"]["rate_w_fps"] for k in keys),
             min(rows[k]["kc"] for k in keys), min(over), max(over)))
    print("one-thread preprocessing (median ms): " +
          ", ".join("%s %.2f" % (k, rows[k]["p"]["pre_1_ms_median"]) for k in keys))
    print("miss rate at K* in the last repeat (%): " +
          ", ".join("%s %.2f" % (k, 100 * rows[k]["m"]["miss"]) for k in keys))
    print("in the staggered runs at K* and K*+1, GPU clock (MHz) and GPU-thread time per frame (ms):")
    for k in keys:
        m = rows[k]["m"]
        tg = ("%.2f-%.2f" % (min(m["tg_run"]), max(m["tg_run"]))) if m["tg_run"] else "-"
        p = rows[k]["p"]
        print("  %-22s %4.0f-%4.0f MHz  t_g %s ms (isolated %.2f ms at %.0f MHz)"
              % (LABEL[k], min(m["mhz"]), max(m["mhz"]), tg, p["gpu_ms_median"], p["gpu_mhz_mean"]))

    print("\n== Table 8: the estimate as a model of K*")
    print("%-38s %5s %4s %4s %5s %6s %5s %4s %6s" % ("", "exact", "<=1", "over", "under", "bias", "MAE", "max",
                                                     "rel %"))
    after = [k for k in keys if k in MEASURED_AFTER]
    for title, ks in (("all %d engines" % len(keys), keys), ("%d engines measured afterward" % len(after), after)):
        if not ks:
            continue
        for which in ("med", "p99"):
            e = [rows[k][which] - rows[k]["m"]["k"] for k in ks]
            rel = 100.0 * sum(abs(v) / float(rows[k]["m"]["k"]) for v, k in zip(e, ks)) / len(ks)
            print("%-38s %5d %4d %4d %5d %+6.2f %5.2f %4d %6.0f"
                  % ("%s, %s" % (title, "median" if which == "med" else "P99"), sum(v == 0 for v in e),
                     sum(abs(v) <= 1 for v in e), sum(v > 0 for v in e), sum(v < 0 for v in e),
                     sum(e) / float(len(e)), sum(abs(v) for v in e) / float(len(e)), max(abs(v) for v in e), rel))

    extra = sorted(glob.glob(os.path.join(a.folder, "aligned_*.json"))) + sorted(
        glob.glob(os.path.join(a.folder, "batch_*.json")))
    if extra:
        print("\n== Synchronized streams at the staggered K*, and the batched GPU policy")
    for fn in extra:
        d = load(fn)
        kind, k = re.fullmatch(r"(aligned|batch)_(.+)\.json", os.path.basename(fn)).groups()
        if kind == "aligned":
            if d["phase"] != "aligned":
                raise SystemExit("%s: phase is %s" % (fn, d["phase"]))
            r = [r for r in d["runs"] if r["K"] == rows[k]["m"]["k"]][0]
            print("%-22s synchronized, K = %2d: miss %.2f %%" % (LABEL[k], r["K"], 100 * r["miss_rate"]))
        else:
            if d["policy"] != "batch":
                raise SystemExit("%s: policy is %s" % (fn, d["policy"]))
            kb = cap(d["runs"])
            r = [r for r in d["runs"] if r["K"] == kb][0]
            print("%-22s batched, K_B = %2d (largest K run: %d): median frame latency %.1f ms, %.2f W"
                  % (LABEL[k], kb, max(x["K"] for x in d["runs"]), r["lat_ms"]["median"], r["board_w_mean"]))


if __name__ == "__main__":
    main()
