#!/usr/bin/env python3
"""ImageNet-1k pretraining of WeeNet: 100 epochs at 160 px, tested at 224 px.

Built from library components: a torch DataLoader over the files written by train/imagenet_prep.py with the timm transform (random
resized crop, scale 0.25-1, ratio 3/4-4/3, flip, bilinear), timm SGD with Nesterov momentum 0.9 (no weight decay on
biases and 1-d parameters), timm CosineLRScheduler stepped per update with a linear warm-up from 1e-5, timm
LabelSmoothingCrossEntropy, torch autocast. Defaults: batch 512, peak LR 0.5 * batch / 1024 = 0.25, weight decay 4e-5.
The reported run (one RTX 3090, 25 loader workers, 339 min):
  python train/pretrain_imagenet.py --data $IMAGENET_PREP --epochs 100 --warmup 1 --batch 512 --res 160 \
      --val-split val224 --test-res 224 --label-smoothing 0.1 --out outputs/in1k_weenet
-> 55.40 % top-1 on the 50,000 validation images at 224 px.
Writes <out>/log.jsonl (one line per epoch), <out>/last.pt after every epoch (a rerun resumes from it; not bit-exact,
the loader workers draw new seeds), and at the end <out>/backbone.pt (the state_dict that train/merged3.py --backbone
loads) and <out>/pretrain.json.
"""
import argparse
import io
import json
import os
import random
import sys
import time

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image
from timm.data import create_transform
from timm.loss import LabelSmoothingCrossEntropy
from timm.optim import create_optimizer_v2
from timm.scheduler import CosineLRScheduler
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from weenet import model as wm  # noqa: E402

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class U8(Dataset):
    """One split stored as <split>_<size>.u8 (N x size x size x 3 uint8), <split>_labels.npy, <split>_meta.json; the
    memmap is opened lazily so each DataLoader worker has its own."""

    def __init__(self, root, split, tf):
        meta = json.load(open(os.path.join(root, "%s_meta.json" % split)))
        self.path = os.path.join(root, "%s_%d.u8" % (split, meta["size"]))
        self.shape = (meta["n"], meta["size"], meta["size"], 3)
        self.y = np.load(os.path.join(root, "%s_labels.npy" % split)).astype(np.int64)
        if len(self.y) != meta["n"]:
            raise SystemExit("%s: %d labels for %d images" % (split, len(self.y), meta["n"]))
        if os.path.getsize(self.path) != int(np.prod(self.shape)):
            raise SystemExit("%s: file size does not match %s" % (self.path, self.shape))
        self.meta, self.tf, self.x = meta, tf, None

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        if self.x is None:
            self.x = np.memmap(self.path, dtype=np.uint8, mode="r", shape=self.shape)
        return self.tf(Image.fromarray(np.array(self.x[i]))), int(self.y[i])


class JPG(U8):
    """One split stored as JPEG rows: <split>_<size>.jpg (row i = bytes off[i]:off[i + 1]) and <split>_<size>.jpg.off.npy;
    a row decodes to the same size x size RGB square a U8 row holds. The reported training split used this layout
    (quality 95), which fits in the page cache of a machine with ~40 GB of RAM."""

    def __init__(self, root, split, tf):
        meta = json.load(open(os.path.join(root, "%s_meta.json" % split)))
        self.path = os.path.join(root, "%s_%d.jpg" % (split, meta["size"]))
        self.off = np.load(self.path + ".off.npy")
        self.y = np.load(os.path.join(root, "%s_labels.npy" % split)).astype(np.int64)
        if not (len(self.y) == meta["n"] == len(self.off) - 1):
            raise SystemExit("%s: %d labels, %d offsets for %d images" % (split, len(self.y), len(self.off), meta["n"]))
        if os.path.getsize(self.path) != self.off[-1]:
            raise SystemExit("%s: file size does not match its offsets" % self.path)
        self.meta, self.tf, self.x = meta, tf, None

    def __getitem__(self, i):
        if self.x is None:
            self.x = os.open(self.path, os.O_RDONLY)
            if hasattr(os, "posix_fadvise"):          # random reads: no read-ahead
                os.posix_fadvise(self.x, 0, 0, os.POSIX_FADV_RANDOM)
        a, b = int(self.off[i]), int(self.off[i + 1])
        raw = os.pread(self.x, b - a, a)
        if len(raw) != b - a:
            raise OSError("%s: row %d read %d of %d bytes" % (self.path, i, len(raw), b - a))
        return self.tf(Image.open(io.BytesIO(raw)).convert("RGB")), int(self.y[i])


def dataset(root, split, tf):
    fmt = json.load(open(os.path.join(root, "%s_meta.json" % split))).get("format", "u8")
    return {"u8": U8, "jpg": JPG}[fmt](root, split, tf)


@torch.no_grad()
def evaluate(model, dl, dev, amp):
    model.eval()
    right = n = 0
    for x, y in dl:
        with torch.autocast(dev, enabled=amp):
            out = model(x.to(dev, non_blocking=True))
        right += int((out.argmax(1).cpu() == y).sum())
        n += len(y)
    return right / n


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", required=True, help="directory written by train/imagenet_prep.py")
    ap.add_argument("--model", default="weenet_z_ds4")
    ap.add_argument("--res", type=int, required=True)
    ap.add_argument("--epochs", type=int, required=True)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=None, help="default 0.5 * batch / 1024")
    ap.add_argument("--wd", type=float, default=4e-5)
    ap.add_argument("--warmup", type=float, default=2.0, help="epochs of warm-up")
    ap.add_argument("--scale-min", type=float, default=0.25)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--val-split", default="val224")
    ap.add_argument("--test-res", type=int, default=224)
    ap.add_argument("--workers", type=int, default=None, help="default: CPUs - 2")
    ap.add_argument("--prefetch", type=int, default=4, help="batches in flight per worker")
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (smoke test); writes no result")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-cpu", action="store_true", help="CPU smoke test only")
    a = ap.parse_args()
    if torch.cuda.is_available():
        dev = "cuda"
    elif a.allow_cpu:
        dev = "cpu"
    else:
        raise SystemExit("CUDA is not available (pass --allow-cpu only for a CPU smoke test)")
    amp = dev == "cuda"
    lr = a.lr if a.lr is not None else 0.5 * a.batch / 1024
    workers = a.workers if a.workers is not None else max(1, os.cpu_count() - 2)
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    np.random.seed(a.seed)
    tf_tr = create_transform(a.res, is_training=True, scale=(a.scale_min, 1.0), ratio=(3 / 4, 4 / 3), hflip=0.5,
                             color_jitter=None, auto_augment=None, interpolation="bilinear", re_prob=0.0, mean=MEAN,
                             std=STD)
    va_size = json.load(open(os.path.join(a.data, "%s_meta.json" % a.val_split)))["size"]
    tf_va = transforms.Compose(([] if va_size == a.test_res else [transforms.Resize(a.test_res, antialias=True)])
                               + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    tr, va = dataset(a.data, "train", tf_tr), dataset(a.data, a.val_split, tf_va)
    if a.res > tr.meta["size"]:
        raise SystemExit("--res %d is larger than the stored images (%d)" % (a.res, tr.meta["size"]))
    nc = tr.meta["classes"]
    dl = DataLoader(tr, a.batch, shuffle=True, drop_last=True, num_workers=workers, pin_memory=amp,
                    persistent_workers=workers > 0, prefetch_factor=a.prefetch if workers > 0 else None)
    dv = DataLoader(va, a.batch, shuffle=False, num_workers=workers, pin_memory=amp)
    steps_ep = len(dl)
    total, warm = a.epochs * steps_ep, int(a.warmup * steps_ep)
    model = wm.build(a.model, nc).to(dev)
    opt = create_optimizer_v2(model, "sgd", lr=lr, weight_decay=a.wd, momentum=0.9)       # timm "sgd" = Nesterov
    sched = CosineLRScheduler(opt, t_initial=total - warm, lr_min=0.0, warmup_t=warm, warmup_lr_init=1e-5,
                              warmup_prefix=True, t_in_epochs=False)
    crit = LabelSmoothingCrossEntropy(smoothing=a.label_smoothing) if a.label_smoothing > 0 else nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(dev, enabled=amp)
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "pretrain.json")) and not a.max_steps:
        raise SystemExit("%s already holds a finished run" % a.out)
    print("%s, timm %s, %d classes, %d train / %d %s images, train crop %d px, test %d px, batch %d, %d steps/epoch, "
          "peak lr %.4g, wd %g, label smoothing %g, %d workers, %s"
          % (a.model, timm.__version__, nc, len(tr), len(va), a.val_split, a.res, a.test_res, a.batch, steps_ep, lr,
             a.wd, a.label_smoothing, workers, dev), flush=True)
    t0, hist, step, start = time.time(), [], 0, 0
    run_args = {k: getattr(a, k) for k in ("model", "res", "epochs", "batch", "wd", "warmup", "scale_min",
                                           "label_smoothing", "val_split", "test_res", "seed")}
    run_args.update(lr=lr, steps_per_epoch=steps_ep, train_format=tr.meta.get("format", "u8"))
    ck_path = os.path.join(a.out, "last.pt")
    if os.path.exists(ck_path) and not a.max_steps:
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        if ck["args"] != run_args:
            raise SystemExit("%s was written with other settings:\n  %s\n  now %s" % (ck_path, ck["args"], run_args))
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        start, step, hist = ck["epoch"], ck["step"], ck["hist"]
        torch.set_rng_state(ck["rng"]["torch"])
        np.random.set_state(ck["rng"]["numpy"])
        random.setstate(ck["rng"]["python"])
        sched.step_update(num_updates=step)
        t0 -= 60 * hist[-1]["minutes"]
        print("resumed from %s: epoch %d done, step %d, lr %.4g" % (ck_path, start, step, opt.param_groups[0]["lr"]),
              flush=True)
    for ep in range(start, a.epochs):
        model.train()
        te, seen, run, nrun = time.time(), 0, 0.0, 0
        for i, (x, y) in enumerate(dl):
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(dev, enabled=amp):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            step += 1
            sched.step_update(num_updates=step)
            seen += len(y)
            if i % 50 == 0:
                if not torch.isfinite(loss):
                    raise SystemExit("loss is not finite at epoch %d step %d" % (ep + 1, i))
                run += loss.item()
                nrun += 1
            if i % 200 == 199:
                print("  ep %d step %d/%d  loss %.3f  lr %.4g  %.0f img/s" % (ep + 1, i + 1, steps_ep, loss.item(),
                      opt.param_groups[0]["lr"], seen / (time.time() - te)), flush=True)
            if a.max_steps and step == a.max_steps:
                print("max-steps %d reached: %.0f img/s, loss %.3f, lr %.4g" % (step, seen / (time.time() - te),
                      loss.item(), opt.param_groups[0]["lr"]), flush=True)
                return
        val = evaluate(model, dv, dev, amp)
        rec = {"epoch": ep + 1, "val_top1": round(val, 4), "loss": round(run / max(1, nrun), 4),
               "img_per_s": round(seen / (time.time() - te)), "minutes": round((time.time() - t0) / 60, 1)}
        hist.append(rec)
        with open(os.path.join(a.out, "log.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        print("ep %3d/%d  val %.4f  loss %.3f  %d img/s  (%.1f min)" % (ep + 1, a.epochs, val, rec["loss"],
              rec["img_per_s"], rec["minutes"]), flush=True)
        ck = {"args": run_args, "epoch": ep + 1, "step": step, "hist": hist, "model": model.state_dict(),
              "opt": opt.state_dict(), "scaler": scaler.state_dict(),
              "rng": {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}}
        torch.save(ck, ck_path + ".tmp")
        os.replace(ck_path + ".tmp", ck_path)
    if a.max_steps:
        raise SystemExit("--max-steps %d is beyond the end of the run" % a.max_steps)
    torch.save(model.state_dict(), os.path.join(a.out, "backbone.pt"))
    best = max(hist, key=lambda r: r["val_top1"])
    json.dump({"model": a.model, "timm": timm.__version__, "torch": torch.__version__,
               "gpu": torch.cuda.get_device_name() if dev == "cuda" else "cpu", "batch": a.batch, "lr": lr, "wd": a.wd,
               "warmup": a.warmup, "scale_min": a.scale_min, "label_smoothing": a.label_smoothing, "epochs": a.epochs,
               "res": a.res, "test_res": a.test_res, "val_split": a.val_split, "workers": workers,
               "val_top1_final": hist[-1]["val_top1"], "val_top1_best": best["val_top1"], "best_epoch": best["epoch"],
               "classes": nc, "train_images": len(tr), "train_format": run_args["train_format"],
               "train_jpeg_quality": tr.meta.get("quality"), "history": hist},
              open(os.path.join(a.out, "pretrain.json"), "w"), indent=1)
    print("done: val %.4f (best %.4f at epoch %d)" % (hist[-1]["val_top1"], best["val_top1"], best["epoch"]), flush=True)


if __name__ == "__main__":
    main()
