#!/usr/bin/env python3
"""Decode ImageNet-1k once into square images for train/pretrain_imagenet.py.

Input: the Hugging Face ILSVRC/imagenet-1k parquet shards (columns `image` with `bytes`, and `label`), or a folder with
one sub-folder per class. Every image: short side resized to floor(size / crop_ratio) (PIL bilinear), center crop
size x size.
Output in --out:
  --format u8   <split>_<size>.u8      N x size x size x 3 uint8 RGB (memory-mapped by the trainer)
  --format jpg  <split>_<size>.jpg     the same squares as JPEG rows of --quality, with <split>_<size>.jpg.off.npy
  <split>_labels.npy (int16) and <split>_meta.json {"n", "size", "crop_ratio", "classes", "format", "quality"}
The reported backbone used:
  python train/imagenet_prep.py --parquet $HF_IMAGENET/data --pattern 'train-*.parquet' --split train --size 176 \
      --format jpg --quality 95 --out $IMAGENET_PREP
  python train/imagenet_prep.py --parquet $HF_IMAGENET/data --pattern 'validation-*.parquet' --split val224 \
      --size 224 --crop-ratio 0.95 --out $IMAGENET_PREP
"""
import argparse
import glob
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

PQ_BUFFER = 8 << 20          # read parquet column chunks in 8 MB pieces


def to_square(img, size, crop_ratio=1.0):
    short = size if crop_ratio == 1.0 else int(size / crop_ratio)       # timm eval: floor(img_size / crop_pct)
    img.draft("RGB", (short, short))               # JPEG: decode at 1/2, 1/4, 1/8 scale when that still covers `short`
    img = img.convert("RGB")
    w, h = img.size
    s = short / min(w, h)
    nw, nh = max(short, round(w * s)), max(short, round(h * s))
    img = img.resize((nw, nh), Image.Resampling.BILINEAR, reducing_gap=None)
    l, t = (nw - size) // 2, (nh - size) // 2
    return np.asarray(img.crop((l, t, l + size, t + size)), dtype=np.uint8)


def rows_of(source):
    """Yield (PIL image, label) of one parquet shard or one list of (file, label)."""
    if isinstance(source, str):
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(source, buffer_size=PQ_BUFFER)
        for batch in pf.iter_batches(batch_size=256, columns=["image", "label"]):
            for im, lab in zip(batch.column("image").to_pylist(), batch.column("label").to_pylist()):
                yield Image.open(io.BytesIO(im["bytes"])), lab
    else:
        for f, lab in source:
            yield Image.open(f), lab


def job(args):
    """Decode one shard into a part file: raw uint8 rows (u8) or concatenated JPEG rows (jpg)."""
    source, part, size, crop, fmt, quality = args
    labels, lengths = [], []
    with open(part + ".tmp", "wb") as f:
        for img, lab in rows_of(source):
            a = to_square(img, size, crop)
            if fmt == "u8":
                f.write(a.tobytes())
            else:
                b = io.BytesIO()
                Image.fromarray(a).save(b, "JPEG", quality=quality)
                f.write(b.getbuffer())
                lengths.append(b.tell())
            labels.append(lab)
    os.replace(part + ".tmp", part)
    return labels, lengths


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--parquet")
    src.add_argument("--folder")
    ap.add_argument("--pattern", default="*.parquet", help="with --parquet: which shards, e.g. 'train-*.parquet'")
    ap.add_argument("--split", required=True, help="name used in the output files: train, val224, ...")
    ap.add_argument("--size", type=int, required=True)
    ap.add_argument("--crop-ratio", type=float, default=1.0, help="center crop ratio (0.95 for the 224 px test crop)")
    ap.add_argument("--format", choices=["u8", "jpg"], default="u8")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality for --format jpg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--procs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.parquet:
        sources = sorted(glob.glob(os.path.join(a.parquet, a.pattern)))
        if not sources:
            raise SystemExit("no shard matches %s in %s" % (a.pattern, a.parquet))
    else:
        classes = sorted(d for d in os.listdir(a.folder) if os.path.isdir(os.path.join(a.folder, d)))
        if not classes:
            raise SystemExit("no class folder in %s" % a.folder)
        items = [(os.path.join(a.folder, c, f), ci) for ci, c in enumerate(classes)
                 for f in sorted(os.listdir(os.path.join(a.folder, c))) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        sources = [items[i:i + 2000] for i in range(0, len(items), 2000)]
    tmp = os.path.join(a.out, "%s_%d_parts" % (a.split, a.size))
    os.makedirs(tmp, exist_ok=True)
    parts = [os.path.join(tmp, "%05d.bin" % i) for i in range(len(sources))]
    labels, lengths = [], []
    with ProcessPoolExecutor(a.procs) as ex:
        for k, (lab, ln) in enumerate(ex.map(job, [(s, p, a.size, a.crop_ratio, a.format, a.quality)
                                                    for s, p in zip(sources, parts)])):
            labels += lab
            lengths += ln
            print("  part %d/%d, %d images, %.1f min" % (k + 1, len(parts), len(labels), (time.time() - t0) / 60),
                  flush=True)
    data = os.path.join(a.out, "%s_%d.%s" % (a.split, a.size, a.format))
    with open(data, "wb") as f:                      # join the parts in source order
        for p in parts:
            with open(p, "rb") as g:
                while True:
                    buf = g.read(64 << 20)
                    if not buf:
                        break
                    f.write(buf)
            os.remove(p)
    os.rmdir(tmp)
    n = len(labels)
    if a.format == "u8" and os.path.getsize(data) != n * a.size * a.size * 3:
        raise SystemExit("%s: size does not match %d images" % (data, n))
    if a.format == "jpg":
        off = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
        if off[-1] != os.path.getsize(data):
            raise SystemExit("%s: size does not match the row lengths" % data)
        np.save(data + ".off.npy", off)
    labels = np.asarray(labels, dtype=np.int64)
    if n == 0 or labels.min() < 0:
        raise SystemExit("no image, or labels missing")
    np.save(os.path.join(a.out, "%s_labels.npy" % a.split), labels.astype(np.int16))
    json.dump({"n": int(n), "size": a.size, "crop_ratio": a.crop_ratio, "classes": int(labels.max()) + 1,
               "format": a.format, "quality": a.quality if a.format == "jpg" else None,
               "minutes": round((time.time() - t0) / 60, 1)},
              open(os.path.join(a.out, "%s_meta.json" % a.split), "w"), indent=1)
    print("done: %d images, %d classes, %.1f min" % (n, labels.max() + 1, (time.time() - t0) / 60), flush=True)


if __name__ == "__main__":
    main()
