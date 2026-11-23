#!/usr/bin/env python3
"""One trtexec timing run of a TensorRT 8.2 engine with board power sampled alongside.

trtexec --loadEngine, warm-up 3 s, timed 15 s. Latency = trtexec's median "GPU Compute Time". tegrastats runs at 10 Hz
and every line is time-stamped on arrival, so the mean power covers exactly the timed window: from "Starting inference"
+ warm-up + 0.3 s to the first trtexec line printed after the run - 0.3 s. Energy per image = mean POM_5V_IN power /
throughput (same for the GPU rail). GPU and CPU0 clocks are read from sysfs at 10 Hz over the same window (the clocks are
left to the default governor), and the result is flagged when the GPU median is below 921.6 MHz.
Writes <outdir>/<label>.r<round>.json plus the raw logs (.log trtexec, .tegra.log, .clk.log).
Python 3.6 (no f-strings).
  python3 jetson/time_engine.py --plan weenet.plan --label weenet --round 1 --outdir out
  python3 jetson/time_engine.py --plan weenet_u8.plan --plugin build/libfused_haar_stem.so --label weenet_u8 --round 1 --outdir out
"""
from __future__ import print_function
import argparse
import json
import os
import re
import subprocess
import threading
import time

TRTEXEC = os.environ.get("TRTEXEC", "/usr/src/tensorrt/bin/trtexec")
GPUF = "/sys/devices/gpu.0/devfreq/57000000.gpu/cur_freq"
CPUF = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
RAILS = ("POM_5V_IN", "POM_5V_GPU", "POM_5V_CPU")

ap = argparse.ArgumentParser()
ap.add_argument("--plan", required=True)
ap.add_argument("--label", required=True)
ap.add_argument("--round", type=int, required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--warmup_ms", type=int, default=3000)
ap.add_argument("--duration", type=int, default=15)
ap.add_argument("--plugin", default="", help="plugin library for engines with a plugin layer")
a = ap.parse_args()
if not os.path.isdir(a.outdir):
    os.makedirs(a.outdir)
base = os.path.join(a.outdir, "%s.r%d" % (a.label, a.round))

tlines = []
ts = subprocess.Popen(["stdbuf", "-oL", "tegrastats", "--interval", "100"], stdout=subprocess.PIPE)


def read_ts():
    for raw in iter(ts.stdout.readline, b""):
        tlines.append((time.time(), raw.decode("utf-8", "replace").rstrip("\n")))


clk = []  # (time, GPU Hz, CPU0 kHz) at 10 Hz from sysfs
halt = threading.Event()


def read_clk():
    while not halt.is_set():
        clk.append((time.time(), int(open(GPUF).read()), int(open(CPUF).read())))
        halt.wait(0.1)


th = threading.Thread(target=read_ts)
th.daemon = True
th.start()
th2 = threading.Thread(target=read_clk)
th2.daemon = True
th2.start()
time.sleep(2.0)  # a few samples before the run, kept in the log only

t0 = time.time()
freq0 = int(open(GPUF).read())
cmd = [TRTEXEC, "--loadEngine=" + a.plan, "--warmUp=%d" % a.warmup_ms, "--duration=%d" % a.duration]
if a.plugin:
    cmd.append("--plugins=" + a.plugin)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
lines = []
for raw in iter(p.stdout.readline, b""):
    lines.append((time.time(), raw.decode("utf-8", "replace").rstrip("\n")))
rc = p.wait()
freq1 = int(open(GPUF).read())
halt.set()
ts.terminate()
ts.wait()
th.join(2.0)

with open(base + ".log", "w") as f:  # raw trtexec log
    f.write("\n".join(l for _, l in lines) + "\n")
with open(base + ".tegra.log", "w") as f:  # seconds since trtexec start, then the tegrastats line
    for t, l in tlines:
        f.write("%.3f %s\n" % (t - t0, l))
if rc != 0:
    raise SystemExit("trtexec rc=%d for %s" % (rc, a.label))

text = "\n".join(l for _, l in lines)
qps = float(re.search(r"Throughput: ([0-9.]+) qps", text).group(1))
g = re.search(r"GPU Compute Time: min = ([0-9.]+) ms, max = ([0-9.]+) ms, mean = ([0-9.]+) ms, "
              r"median = ([0-9.]+) ms, percentile\(99%\) = ([0-9.]+) ms", text)
h = re.search(r"\] Latency: min = .*?median = ([0-9.]+) ms", text)
t_inf = [t for t, l in lines if "Starting inference" in l][0]
after = [t for t, l in lines if t > t_inf + a.warmup_ms / 1000.0 + 1.0][0]
w0, w1 = t_inf + a.warmup_ms / 1000.0 + 0.3, after - 0.3
if w1 - w0 < a.duration - 2:
    raise SystemExit("power window only %.1f s for %s" % (w1 - w0, a.label))
rows = [l for t, l in tlines if w0 <= t <= w1]
if len(rows) < 20:
    raise SystemExit("only %d tegrastats samples in the window for %s" % (len(rows), a.label))
pw = {}
for r in RAILS:
    v = [int(re.search(r + r" ([0-9]+)/", l).group(1)) for l in rows]
    pw[r] = sum(v) / float(len(v)) / 1000.0


def med(v):
    v = sorted(v)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


cw = [(gf, cf) for t, gf, cf in clk if w0 <= t <= w1]
gpu_mhz = [gf / 1e6 for gf, cf in cw]
cpu_mhz = [cf / 1e3 for gf, cf in cw]
clocks = {"gpu_mhz_median": med(gpu_mhz), "gpu_mhz_min": min(gpu_mhz), "cpu_mhz_median": med(cpu_mhz),
          "cpu_mhz_min": min(cpu_mhz), "n": len(cw), "gpu_below_921": med(gpu_mhz) < 921.6}
with open(base + ".clk.log", "w") as f:  # seconds since trtexec start, GPU Hz, CPU0 kHz
    for t, gf, cf in clk:
        f.write("%.3f %d %d\n" % (t - t0, gf, cf))
temps = [float(re.search(r"GPU@([0-9.]+)C", l).group(1)) for l in rows]
res = {"label": a.label, "round": a.round, "plan": os.path.basename(a.plan), "throughput_qps": qps,
       "gpu_ms": {"min": float(g.group(1)), "max": float(g.group(2)), "mean": float(g.group(3)),
                  "median": float(g.group(4)), "p99": float(g.group(5))},
       "latency_median_ms": float(h.group(1)),
       "window_s": w1 - w0, "n_samples": len(rows), "power_w": pw,
       "energy_mj": {"board_in": pw["POM_5V_IN"] / qps * 1000.0, "gpu_rail": pw["POM_5V_GPU"] / qps * 1000.0},
       "gpu_temp_c_mean": sum(temps) / len(temps), "clocks": clocks, "gpu_freq_hz": [freq0, freq1],
       "time": time.strftime("%H:%M:%S")}
json.dump(res, open(base + ".json", "w"), indent=1)
print("%s r%d: gpu median %.3f ms | %.1f qps | IN %.2f W GPU %.2f W | %.2f / %.2f mJ | n=%d %.1fs | "
      "GPU %.1f MHz (min %.1f) CPU %.0f MHz%s" % (
          a.label, a.round, res["gpu_ms"]["median"], qps, pw["POM_5V_IN"], pw["POM_5V_GPU"],
          res["energy_mj"]["board_in"], res["energy_mj"]["gpu_rail"], len(rows), w1 - w0, clocks["gpu_mhz_median"],
          clocks["gpu_mhz_min"], clocks["cpu_mhz_median"], " FLAG: GPU median below 921.6" if clocks["gpu_below_921"] else ""))
