#!/usr/bin/env python3
"""Rebuild the merged split and compare it with the shipped lists (9,726 images, 6,801 / 1,453 / 1,472).

Two ways to list the images:
  --manifest            the 9,726 paths of data/checksums.md5 (no images needed); the split functions are applied to
                        these paths exactly as train/data.py applies them to the folders on disk
  --data-root DIR       the folders on disk, through train/data.py (the code path of fine-tuning)
For seed 0 the result must equal data/split_seed0/{train,val,test}.csv row by row (exit 1 otherwise). --check-md5 also
verifies every image under --data-root against data/checksums.md5. --write DIR writes the CSVs of another --seed (the
ablation trains each seed on its own split).
  python data/make_split.py --manifest
  python data/make_split.py --data-root $DATA_ROOT --check-md5
  python data/make_split.py --data-root $DATA_ROOT --seed 1 --write splits/seed1
"""
import argparse
import csv
import hashlib
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from train.data import CLASSES, G12_MAP, RW_DIR, RW_MAP, SOURCES  # noqa: E402

PARTS = ("train", "val", "test")


def _split(files, rng):
    idx = np.arange(len(files))
    rng.shuffle(idx)
    a, b = int(0.70 * len(files)), int(0.15 * len(files))
    return idx[:a], idx[a:a + b], idx[a + b:]


def split_from_paths(paths, seed):
    """The split of train/data.py applied to relative paths (TrashNet: ImageFolder order; RealWaste: sorted per folder;
    Garbage-12: the de-duplicated list grouped by mapped class, sorted)."""
    out = {}
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for y, c in enumerate(CLASSES):
        fs = sorted(p for p in paths if p.startswith("trashnet/%s/" % c))
        for part, sel in zip(parts, _split(fs, rng)):
            part.extend((fs[i], y) for i in sel)
    out["trashnet"] = parts
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for folder, cls in sorted(RW_MAP.items()):
        fs = sorted(p for p in paths if p.startswith("%s/%s/" % (RW_DIR.replace(os.sep, "/"), folder)))
        for part, sel in zip(parts, _split(fs, rng)):
            part.extend((fs[i], CLASSES.index(cls)) for i in sel)
    out["realwaste"] = parts
    g12 = [p.strip() for p in open(os.path.join(HERE, "garbage12_clean_dedup.txt")) if p.strip()]
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for cls in sorted(set(G12_MAP.values())):
        fs = sorted(p for p in g12 if G12_MAP[p.split("/")[-2]] == cls)
        for part, sel in zip(parts, _split(fs, rng)):
            part.extend((fs[i], CLASSES.index(cls)) for i in sel)
    out["garbage12"] = parts
    return out


def split_from_disk(root, seed):
    from train.data import merged_split
    sp = merged_split(root, os.path.join(HERE, "garbage12_clean_dedup.txt"), seed)
    return {s: tuple([(os.path.relpath(p, root).replace(os.sep, "/"), y) for p, y in part] for part in sp[s])
            for s in SOURCES}


def rows_of(sp):
    return {part: [(p, y, CLASSES[y], s) for s in SOURCES for p, y in sp[s][i]] for i, part in enumerate(PARTS)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--manifest", action="store_true")
    g.add_argument("--data-root")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check-md5", action="store_true")
    ap.add_argument("--write", default="")
    a = ap.parse_args()
    manifest = [l.split(None, 1) for l in open(os.path.join(HERE, "checksums.md5")).read().splitlines() if l.strip()]
    if a.manifest:
        sp = split_from_paths([p for _, p in manifest], a.seed)
    else:
        sp = split_from_disk(a.data_root, a.seed)
    rows = rows_of(sp)
    for s in SOURCES:
        print("%-10s train, val, test %s" % (s, [len(p) for p in sp[s]]))
    print("%-10s train, val, test %s" % ("total", [len(rows[p]) for p in PARTS]))
    if a.check_md5:
        if not a.data_root:
            raise SystemExit("--check-md5 needs --data-root")
        bad = [p for h, p in manifest if hashlib.md5(open(os.path.join(a.data_root, p), "rb").read()).hexdigest() != h]
        print("md5: %d of %d images differ%s" % (len(bad), len(manifest), (": %s" % bad[:5]) if bad else ""))
        if bad:
            sys.exit(1)
    if a.write:
        os.makedirs(a.write, exist_ok=True)
        for part in PARTS:
            with open(os.path.join(a.write, "%s.csv" % part), "w", newline="") as f:
                w = csv.writer(f, lineterminator="\n")
                w.writerow(["path", "label", "class", "source"])
                w.writerows(rows[part])
        print("wrote", a.write)
    if a.seed == 0:
        ok = True
        for part in PARTS:
            with open(os.path.join(HERE, "split_seed0", "%s.csv" % part)) as f:
                ref = [(r["path"], int(r["label"]), r["class"], r["source"]) for r in csv.DictReader(f)]
            same = ref == rows[part]
            ok &= same
            print("%-5s %5d rows, identical to data/split_seed0/%s.csv: %s" % (part, len(ref), part, same))
        if not ok:
            sys.exit(1)


if __name__ == "__main__":
    main()
