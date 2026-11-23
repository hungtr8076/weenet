#!/usr/bin/env python3
"""Top-1, macro-F1 and paired tests from the per-image test predictions written by train/merged3.py (paper, Table 4
accuracy columns, Sect. 8.1, Table 9).

Each input file holds, per source (trashnet, realwaste, garbage12), the test images (files_<source>, relative to the data
root), their labels and the model's predictions. Printed per file: model, seed, number of test images, correct images,
top-1 and macro-F1 (mean over the six classes of 2PR / (P + R)) in percent. With --ref, each other file is compared with
the reference on the same test images by the exact two-sided McNemar test, with Holm's step-down correction over the
compared files.
  python analysis/accuracy.py --ref results/predictions/twelve_models/weenet_z_ds4_do.json \
      results/predictions/twelve_models/*.json
"""
import argparse
import json
import os
from math import comb

SOURCES = ("trashnet", "realwaste", "garbage12")


def load(path):
    with open(path) as f:
        d = json.load(f)
    rows = {}
    for s in SOURCES:
        for fn, p, y in zip(d["files_" + s], d["preds_" + s], d["labels_" + s]):
            rows[fn] = (p, y)
    if len(rows) != sum(len(d["files_" + s]) for s in SOURCES):
        raise SystemExit("%s: repeated test images" % path)
    return d, rows


def macro_f1(rows, n_classes):
    f1 = []
    for c in range(n_classes):
        tp = sum(1 for p, y in rows.values() if p == c and y == c)
        fp = sum(1 for p, y in rows.values() if p == c and y != c)
        fn = sum(1 for p, y in rows.values() if p != c and y == c)
        f1.append(2.0 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0)
    return sum(f1) / n_classes


def mcnemar_exact(a, b):
    """Two-sided exact McNemar p-value: a, b = images only the first / only the second model classifies correctly."""
    n = a + b
    return min(1.0, 2.0 * sum(comb(n, i) for i in range(min(a, b) + 1)) / 2.0 ** n) if n else 1.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", help="prediction JSON files of train/merged3.py")
    ap.add_argument("--ref", default=None, help="reference file for the McNemar tests (omit: no tests)")
    a = ap.parse_args()
    ref = load(a.ref)[1] if a.ref else None
    pv = {}
    print("%-34s %4s %6s %7s %7s %8s" % ("model", "seed", "images", "correct", "top-1 %", "macro-F1"))
    for path in a.files:
        d, rows = load(path)
        correct = sum(p == y for p, y in rows.values())
        print("%-34s %4d %6d %7d %7.2f %8.2f" % (d["model"], d["seed"], len(rows), correct, 100.0 * correct / len(rows),
                                                 100.0 * macro_f1(rows, len(d["classes"]))))
        if ref is not None and os.path.abspath(path) != os.path.abspath(a.ref):
            if set(rows) != set(ref):
                raise SystemExit("%s and %s are not on the same test images" % (path, a.ref))
            only_ref = sum(1 for k in ref if ref[k][0] == ref[k][1] and rows[k][0] != rows[k][1])
            only_this = sum(1 for k in ref if ref[k][0] != ref[k][1] and rows[k][0] == rows[k][1])
            pv[d["model"]] = (mcnemar_exact(only_ref, only_this), only_ref, only_this)
    if ref is None:
        return
    print("\nExact McNemar test against %s, Holm's correction over %d comparisons" % (os.path.basename(a.ref), len(pv)))
    print("%-34s %9s %10s %8s %8s" % ("model", "only ref", "only this", "p", "p Holm"))
    order = sorted(pv, key=lambda m: pv[m][0])
    run = 0.0
    for i, m in enumerate(order):
        run = max(run, min(1.0, (len(order) - i) * pv[m][0]))
        print("%-34s %9d %10d %8.3f %8.3f" % (m, pv[m][1], pv[m][2], pv[m][0], run))


if __name__ == "__main__":
    main()
