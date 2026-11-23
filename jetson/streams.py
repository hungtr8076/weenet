#!/usr/bin/env python3
"""Frame latency and stream capacity of one Jetson Nano B01 for K camera streams at a fixed frame rate (paper:
frame latency with one 30 fps camera, and the spare capacity of one board). Python 3.6 (no f-strings).

Load: K streams; stream s delivers frame n at a_n = t0 + s*T/K + n*T (T = 1/fps, phases spread evenly; with
--phase aligned every stream delivers at t0 + n*T, all K frames at the same instant). Frames come
from jetson/frames.py (64 test images at 640 x 480 BGR in memory). Per frame, the deployed path: preprocessing
(jetson/frames.py; --pre float or u8) on one of --workers CPU threads, then on one GPU thread H2D copy, TensorRT
execution (one CUDA stream, TensorRT Python API, no CUDA graph), D2H copy and argmax.
Latest frame wins: when a newer frame of a stream arrives while the previous one still waits (not yet taken by a
worker, or preprocessed but not yet taken by the GPU thread), the older frame is dropped; each queue holds at most one
frame per stream.
GPU thread policy:
  b1     one frame per launch, oldest arrival first (--plan, a batch-1 engine)
  batch  every preprocessed frame in the queue (oldest first, at most the largest batch) in one launch, on the smallest
         fixed-batch engine that holds them (--batch-plans, built with jetson/build_batch.py); unused rows are padding
Frame latency = time the argmax is ready - arrival. A frame misses its deadline if it is dropped or its latency is above
T. Frames whose index n lies in the timed window (after --warm seconds, for --dur seconds) are counted: due =
K * round(dur * fps), and every due frame is either completed or dropped (checked, exit 1 otherwise).
Reported per K: miss rate, latency median / P95 / P99 / max, median GPU-thread time per launch (copies, execution,
argmax), GPU busy fraction, CPU utilization (/proc/stat), mean GPU clock and mean board power (INA3221 POM_5V_IN), both
sampled every 50 ms. Clocks are left to nvpmodel and the default DVFS governor (no root needed). At the end: capacity =
the largest K run with a miss rate below 1 %.
K is stepped through --streams and the sweep stops after the first K whose miss rate is above --stop-miss.
  python3 jetson/streams.py --plan engines/weenet_do.plan --pre float --list data/split_seed0/test.csv \
      --root $DATA_ROOT --streams 1 --label weenet_do --out out/streams/k1_weenet_do.json
"""
from __future__ import print_function
import argparse
import collections
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


INA = one("/sys/bus/i2c/drivers/ina3221x/*/iio:device0")
RAIL = [i for i in range(3) if open(os.path.join(INA, "rail_name_%d" % i)).read().strip() == "POM_5V_IN"][0]
GPU_FREQ = one("/sys/devices/57000000.gpu/devfreq/*/cur_freq")
MISS_CAP = 0.01


def copy_in(runner, x):
    """Byte copy into the page-locked input (a u8 engine declares a float32 binding of the same byte size)."""
    np.copyto(runner.host_in().view(np.uint8).reshape(-1), x.view(np.uint8).reshape(-1))


def cpu_times():
    v = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(v), v[3] + v[4]          # total, idle + iowait


class Sampler(threading.Thread):
    """Board power (W) and GPU clock (MHz) every 50 ms while the run is on."""

    def __init__(self):
        threading.Thread.__init__(self)
        self.daemon, self.stop, self.p, self.f = True, False, [], []

    def run(self):
        path = os.path.join(INA, "in_power%d_input" % RAIL)
        while not self.stop:
            self.p.append(int(open(path).read()) / 1000.0)
            self.f.append(int(open(GPU_FREQ).read()) / 1e6)
            time.sleep(0.05)


def run_k(runners, frames, pre, K, fps, dur, warm, workers, policy, phase):
    """runners: {batch: AsyncRunner}; policy b1 uses runners[1] only."""
    T = 1.0 / fps
    n_warm, n_dur = int(round(warm * fps)), int(round(dur * fps))
    in_win = lambda n: n_warm <= n < n_warm + n_dur  # noqa: E731
    lock = threading.Condition()
    pending = collections.OrderedDict()      # stream -> (n, t_arr, frame index), latest frame only
    ready = collections.deque()              # (stream, n, t_arr, x) preprocessed
    dropped = [0]
    comp = []                                # (n, t_arr, latency ms) of every completed frame
    busy_ms = [0.0]
    infer_ms = []                            # GPU-thread time of each launch with a frame in the window
    batch_hist = collections.Counter()       # frames per launch, launches with a frame in the window
    done_flag = [False]
    stop_gpu = [False]
    off = [s * T / K for s in range(K)] if phase == "even" else [0.0] * K
    t0 = time.perf_counter() + 0.2
    t_end = t0 + warm + dur
    sizes = sorted(runners)
    max_b = sizes[-1] if policy == "batch" else 1

    def drop(old_n):
        if in_win(old_n):
            dropped[0] += 1

    def producer():
        n_frames = len(frames)
        events = []
        n = 0
        while True:
            for s in range(K):
                events.append((t0 + off[s] + n * T, s, n))
            if t0 + n * T > t_end:
                break
            n += 1
        events.sort()
        for ta, s, n in events:
            d = ta - time.perf_counter()
            if d > 0:
                time.sleep(d)
            with lock:
                if s in pending:                 # previous frame of this stream never started: drop it
                    drop(pending.pop(s)[0])
                pending[s] = (n, ta, (s * 7 + n) % n_frames)
                lock.notify()
        with lock:
            done_flag[0] = True
            lock.notify_all()

    def worker():
        while True:
            with lock:
                while not pending and not done_flag[0]:
                    lock.wait()
                if not pending and done_flag[0]:
                    return
                s, (n, ta, fi) = min(pending.items(), key=lambda kv: kv[1][1])
                del pending[s]
            x = pre(frames[fi])
            with lock:
                # latest frame wins here too: an older preprocessed frame of this stream still queued is dropped
                for j in [j for j in range(len(ready)) if ready[j][0] == s and ready[j][2] < ta]:
                    drop(ready[j][1])
                    del ready[j]
                    break
                ready.append((s, n, ta, x))
                lock.notify_all()

    def gpu():
        while True:
            with lock:
                while not ready and not stop_gpu[0]:
                    lock.wait(0.01)
                if not ready:
                    return
                order = sorted(range(len(ready)), key=lambda j: ready[j][2])[:max_b]   # oldest arrival first
                take = [ready[j] for j in order]
                for j in sorted(order, reverse=True):
                    del ready[j]
            nb = len(take)
            bsel = [b for b in sizes if b >= nb][0] if policy == "batch" else 1
            r = runners[bsel]
            g0 = time.perf_counter()
            if bsel == 1:
                copy_in(r, take[0][3])
            else:
                hin = r.host_in().view(np.uint8).reshape(bsel, -1)
                for k, item in enumerate(take):  # rows nb..bsel-1 keep stale data (padding, results ignored)
                    np.copyto(hin[k], item[3].view(np.uint8).reshape(-1))
            r.enqueue()
            out = r.wait().reshape(bsel, -1)
            [int(np.argmax(out[k])) for k in range(nb)]
            g1 = time.perf_counter()
            any_in = False
            for s, n, ta, x in take:
                comp.append((n, ta, (g1 - ta) * 1000.0))
                any_in = any_in or in_win(n)
            if any_in:
                busy_ms[0] += (g1 - g0) * 1000.0
                infer_ms.append((g1 - g0) * 1000.0)
                batch_hist[nb] += 1

    ws = [threading.Thread(target=worker) for _ in range(workers)]
    pt = threading.Thread(target=producer)
    gt = threading.Thread(target=gpu)
    smp = Sampler()
    for w in ws:
        w.daemon = True
        w.start()
    gt.daemon = True
    gt.start()
    smp.start()
    pt.start()
    time.sleep(max(0.0, t0 + warm - time.perf_counter()))
    c1, n_p0 = cpu_times(), len(smp.p)
    pt.join()
    for w in ws:
        w.join()
    with lock:
        stop_gpu[0] = True
        lock.notify_all()
    gt.join()
    c2 = cpu_times()
    smp.stop = True
    smp.join()
    due = K * n_dur
    lat = [v for n, ta, v in comp if in_win(n)]
    late = int(sum(1 for v in lat if v > T * 1000.0))
    unacc = due - len(lat) - dropped[0]
    a = np.array(lat) if lat else np.array([np.nan])
    nl = sum(batch_hist.values())
    if len(smp.f) <= n_p0:
        raise SystemExit("no clock / power sample in the timed window")
    return {"K": K, "fps_per_stream": fps, "phase": phase, "offsets_ms": [1000.0 * v for v in off], "policy": policy,
            "due": due, "done": len(lat), "dropped": dropped[0], "late": late, "unaccounted": unacc,
            "miss_rate": (dropped[0] + late) / float(due),
            "lat_ms": {"median": float(np.median(a)), "p95": float(np.percentile(a, 95)),
                       "p99": float(np.percentile(a, 99)), "max": float(np.max(a)), "mean": float(np.mean(a))},
            "infer_ms_median": float(np.median(infer_ms)) if infer_ms else None,
            "launches": nl, "frames_per_launch_mean": (sum(k * v for k, v in batch_hist.items()) / float(nl)) if nl else None,
            "frames_per_launch_hist": dict((str(k), v) for k, v in sorted(batch_hist.items())),
            "gpu_busy": busy_ms[0] / (dur * 1000.0),
            "cpu_util": 1.0 - (c2[1] - c1[1]) / float(max(1, c2[0] - c1[0])),
            "gpu_mhz_mean": float(np.mean(smp.f[n_p0:])), "board_w_mean": float(np.mean(smp.p[n_p0:]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True, help="batch-1 engine")
    ap.add_argument("--batch-plans", default="", help="policy batch: comma-separated B:plan of fixed-batch engines "
                                                      "(--plan is used as B=1)")
    ap.add_argument("--policy", choices=["b1", "batch"], default="b1")
    ap.add_argument("--phase", choices=["even", "aligned"], default="even",
                    help="stream offsets: spread evenly over the frame period, or all streams at the same instant")
    ap.add_argument("--plugin", default="", help="plugin library for engines with a plugin layer")
    ap.add_argument("--std-plugins", action="store_true",
                    help="register TensorRT's standard plugins (engines whose build used them, e.g. MobileViTv2)")
    ap.add_argument("--pre", choices=["float", "u8"], required=True, help="u8 for the FusedHaarStemU8 engine")
    ap.add_argument("--res", type=int, default=224, help="engine input size")
    ap.add_argument("--list", required=True, help="split CSV (path,label,...), e.g. data/split_seed0/test.csv")
    ap.add_argument("--root", required=True, help="image root ($DATA_ROOT)")
    ap.add_argument("--streams", default="1,2,4,6,8,10,12,16", help="K values, run in this order")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--dur", type=float, default=60.0, help="timed seconds per K")
    ap.add_argument("--warm", type=float, default=5.0, help="warm-up seconds per K")
    ap.add_argument("--workers", type=int, default=3, help="preprocessing threads")
    ap.add_argument("--stop-miss", type=float, default=0.05, help="stop the sweep after a K above this miss rate")
    ap.add_argument("--label", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.plugin:
        ctypes.CDLL(a.plugin)
    frames = load_frames(a.list, a.root)
    cuda_init()
    lg = trt.Logger(trt.Logger.WARNING)
    if a.std_plugins:
        trt.init_libnvinfer_plugins(lg, "")
    plans = {1: a.plan}
    if a.policy == "batch":
        if not a.batch_plans:
            raise SystemExit("--policy batch needs --batch-plans")
        for kv in a.batch_plans.split(","):
            b, pth = kv.split(":", 1)
            plans[int(b)] = pth
    elif a.batch_plans:
        raise SystemExit("--batch-plans needs --policy batch")
    runners = {}
    for b, pth in sorted(plans.items()):
        runners[b] = AsyncRunner(open(pth, "rb").read(), lg)
        shp = runners[b].meta[runners[b].i_in][0]
        if shp[0] != b:
            raise SystemExit("engine %s has input %s, expected batch %d" % (pth, shp, b))
    pre_fn = pre_float if a.pre == "float" else pre_u8
    pre = lambda bgr: pre_fn(bgr, a.res)  # noqa: E731
    x = pre(frames[0])
    for b, r in sorted(runners.items()):
        if r.meta[r.i_in][2] != b * x.nbytes:
            raise SystemExit("batch-%d engine input has %d bytes, a preprocessed frame %d (wrong --pre or --res?)"
                             % (b, r.meta[r.i_in][2], x.nbytes))
        hin = r.host_in().view(np.uint8).reshape(b, -1)
        for k in range(b):
            np.copyto(hin[k], x.view(np.uint8).reshape(-1))
        for _ in range(30):                      # warm the engine and the clocks
            r.enqueue()
            r.wait()
    out_dir = os.path.dirname(os.path.abspath(a.out))
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    res = {"label": a.label, "plans": dict((str(b), p) for b, p in plans.items()), "policy": a.policy, "phase": a.phase,
           "res": a.res, "pre": a.pre, "workers": a.workers, "fps": a.fps, "dur_s": a.dur, "warm_s": a.warm,
           "frames": "%d images of %s at 640x480 BGR in memory" % (len(frames), os.path.basename(a.list)),
           "nvpmodel": os.popen("nvpmodel -q 2>/dev/null").read().strip(), "runs": [], "ts": int(time.time())}
    for K in [int(k) for k in a.streams.split(",")]:
        r = run_k(runners, frames, pre, K, a.fps, a.dur, a.warm, a.workers, a.policy, a.phase)
        res["runs"].append(r)
        print("%s K=%2d %s: miss %.4f (drop %d late %d of %d) latency median %.1f p99 %.1f max %.1f ms | gpu %.0f%% "
              "cpu %.0f%% | %.0f MHz %.2f W | frames/launch %s"
              % (a.label, K, a.policy, r["miss_rate"], r["dropped"], r["late"], r["due"], r["lat_ms"]["median"],
                 r["lat_ms"]["p99"], r["lat_ms"]["max"], 100 * r["gpu_busy"], 100 * r["cpu_util"], r["gpu_mhz_mean"],
                 r["board_w_mean"], "%.2f" % r["frames_per_launch_mean"] if r["frames_per_launch_mean"] else "-"))
        json.dump(res, open(a.out, "w"), indent=1)
        if r["unaccounted"] != 0:
            raise SystemExit("accounting error: %d frames neither completed nor dropped" % r["unaccounted"])
        if r["miss_rate"] > a.stop_miss:
            break
    ok = [r["K"] for r in res["runs"] if r["miss_rate"] < MISS_CAP]
    res["capacity"] = max(ok) if ok else 0
    json.dump(res, open(a.out, "w"), indent=1)
    print("%s capacity (largest K run with < %.0f %% missed frames): %d" % (a.label, 100 * MISS_CAP, res["capacity"]))


if __name__ == "__main__":
    main()
