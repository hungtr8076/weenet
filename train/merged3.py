#!/usr/bin/env python3
"""Fine-tune WeeNet / WeeNet-DO on the merged set and record per-image test predictions.

Recipe: RandomResizedCrop 0.35-1, flip, TrivialAugmentWide, RandomErasing 0.25, MixUp 0.2 or
CutMix 1.0 per batch, label smoothing 0.1, AdamW lr 6e-4 wd 0.05, OneCycle (pct_start 0.25), batch 32, AMP, run for
ceil(steps / steps_per_epoch) epochs. With 6,801 training images (212 steps per epoch), the default --steps 63750
gives 301 epochs, 63,812 steps, the protocol of the paper.
The checkpoint with the best pooled validation accuracy is evaluated on the three test sets; the last checkpoint is
saved too and its validation accuracy is recorded (val_last).
WeeNet models start from --backbone (WeeNet's own ImageNet weights, train/pretrain_imagenet.py); without it they start
from random weights.

Seeds: --seeds drives initialization, augmentation, sampling and MixUp/CutMix. The data split uses --split-seed, which
defaults to the run seed (each seed on its own split, as in the ablation); --split-seed 0 gives the split of
data/split_seed0/.

Output: <outdir>/<regime>/seed_<s>/<model>.json with test_pooled, test_<source>, preds_/labels_/files_<source> (files
relative to --data-root), val_best, val_last, steps, split sizes; checkpoints in <outdir>/<regime>/seed_<s>/ckpt/.
An existing JSON is skipped, so a job can be resumed.

  python train/merged3.py --data-root $DATA_ROOT --models weenet_z_ds4_do --backbone $BACKBONE \
      --seeds 0 --split-seed 0 --regime-name finetune --outdir outputs
  python train/merged3.py --smoke --models weenet_z_ds4_do     (CPU, fake images, one short epoch, code path only)
"""
import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repository root
from train import data as D  # noqa: E402
from train.models import make_model  # noqa: E402

HERE = Path(__file__).resolve().parent
G12_LIST = HERE.parent / "data" / "garbage12_clean_dedup.txt"


def train_strong(name, train_dl, val_dl, a, device):
    from torchvision.transforms import v2
    nc = len(D.CLASSES)
    model = make_model(name, nc, backbone=a.backbone).to(device)
    epochs = max(1, math.ceil(a.steps / len(train_dl)))
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=epochs * len(train_dl), pct_start=0.25)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    mix = v2.RandomChoice([v2.MixUp(alpha=0.2, num_classes=nc), v2.CutMix(alpha=1.0, num_classes=nc)])
    amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, t0 = -1.0, None, time.time()
    print("=== %s, %d epochs x %d steps (target %d steps), lr %g" % (name, epochs, len(train_dl), a.steps, a.lr),
          flush=True)
    for ep in range(epochs):
        model.train()
        for x, y in train_dl:
            x, y = mix(x, y)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
        acc = D.evaluate(model, val_dl, device)
        if acc > best:
            best, best_state = acc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if (ep + 1) % 5 == 0 or ep == 0:
            print("   ep %3d/%d  val %.4f  best %.4f  (%.1f min)" % (ep + 1, epochs, acc, best, (time.time() - t0) / 60),
                  flush=True)
    os.makedirs(a.ckpt, exist_ok=True)
    torch.save(best_state, os.path.join(a.ckpt, "%s.pt" % name))
    torch.save({k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
               os.path.join(a.ckpt, "%s__last.pt" % name))
    return {"model": name, "val_best": best, "val_last": acc, "epochs": epochs, "steps": epochs * len(train_dl),
            "lr": a.lr, "res": a.res}


def make_fake(root):
    """Random images in the DATA_ROOT layout and a Garbage-12 list (smoke test only)."""
    rng = np.random.default_rng(0)
    lists = []
    for base, names, k in ((os.path.join(root, "trashnet"), D.CLASSES, 20),
                           (os.path.join(root, D.RW_DIR), sorted(D.RW_MAP), 10),
                           (os.path.join(root, "garbage12", "garbage_classification"), sorted(D.G12_MAP), 8)):
        for n in names:
            os.makedirs(os.path.join(base, n), exist_ok=True)
            for i in range(k):
                p = os.path.join(base, n, "%d.jpg" % i)
                Image.fromarray((rng.random((80, 100, 3)) * 255).astype(np.uint8)).save(p)
                if "garbage12" in base:
                    lists.append(os.path.relpath(p, root))
    g12 = os.path.join(root, "g12.txt")
    open(g12, "w").write("\n".join(lists) + "\n")
    return g12


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data-root", default=os.environ.get("DATA_ROOT", ""), help="default: $DATA_ROOT")
    ap.add_argument("--g12-list", default=str(G12_LIST))
    ap.add_argument("--models", default="weenet_z_ds4_do")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--split-seed", type=int, default=None, help="default: the run seed")
    ap.add_argument("--backbone", default="", help="WeeNet's ImageNet backbone.pt (WeeNet models only)")
    ap.add_argument("--regime-name", default="")
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--res", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--steps", type=int, default=63750, help="optimizer steps (rounded up to whole epochs)")
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--smoke", action="store_true", help="fake images, CPU, one short epoch, random init")
    a = ap.parse_args()
    if a.smoke:
        tmp = tempfile.mkdtemp(prefix="merged3_smoke_")
        a.g12_list = make_fake(tmp)
        a.data_root, a.outdir, a.seeds, a.steps, a.batch, a.workers, a.res = tmp, tmp, "0", 3, 8, 0, 96
        a.backbone, a.regime_name = "", "smoke"
    else:
        if not a.data_root:
            raise SystemExit("--data-root (or $DATA_ROOT) is required")
        if not a.regime_name:
            raise SystemExit("--regime-name is required")
    device = "cuda" if torch.cuda.is_available() and not a.smoke else "cpu"
    if device == "cpu" and not a.smoke:
        raise SystemExit("CUDA not available (CPU is for --smoke only)")
    for seed in [int(s) for s in a.seeds.split(",")]:
        split_seed = seed if a.split_seed is None else a.split_seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        parts = D.merged_split(a.data_root, a.g12_list, split_seed)
        split_n = {k: [len(p) for p in parts[k]] for k in D.SOURCES}
        if not a.smoke and split_n["trashnet"][2] != 384:
            raise SystemExit("unexpected TrashNet split sizes %s" % split_n["trashnet"])
        tf_ev = D.eval_tf(a.res)
        mk = lambda items, tf, shuf: DataLoader(D.Files(items, tf), batch_size=a.batch, shuffle=shuf, drop_last=shuf,  # noqa: E731
                                                num_workers=a.workers, pin_memory=device == "cuda")
        train_dl = mk([x for k in D.SOURCES for x in parts[k][0]], D.train_tf(a.res), True)
        val_dl = mk([x for k in D.SOURCES for x in parts[k][1]], tf_ev, False)
        tests = {k: (mk(parts[k][2], tf_ev, False), parts[k][2]) for k in D.SOURCES}
        odir = Path(a.outdir) / a.regime_name / ("seed_%d" % seed)
        odir.mkdir(parents=True, exist_ok=True)
        for name in a.models.split(","):
            out = odir / ("%s.json" % name)
            if out.exists():
                print("skip (exists) %s" % out, flush=True)
                continue
            print("=== merged3 %s seed %d (split seed %d) %s: train, val, test %s"
                  % (name, seed, split_seed, a.regime_name, split_n), flush=True)
            t0 = time.time()
            torch.manual_seed(seed)
            a.ckpt = str(odir / "ckpt")
            row = train_strong(name, train_dl, val_dl, a, device)
            model = make_model(name, len(D.CLASSES)).to(device)
            model.load_state_dict(torch.load(odir / "ckpt" / ("%s.pt" % name), map_location="cpu"))
            right, total = 0, 0
            for k, (dl, items) in tests.items():
                logits, labels = D.predict(model, dl, device)
                pred = logits.argmax(1)
                row["test_" + k] = float((pred == labels).mean())
                row["preds_" + k], row["labels_" + k] = pred.tolist(), labels.tolist()
                row["files_" + k] = [os.path.relpath(p, a.data_root) for p, _ in items]
                right += int((pred == labels).sum())
                total += len(labels)
            row.update({"test_pooled": right / total, "seed": seed, "split_seed": split_seed, "regime": a.regime_name,
                        "classes": D.CLASSES, "splits": split_n, "backbone": os.path.basename(a.backbone),
                        "wall_min": round((time.time() - t0) / 60, 2)})
            out.write_text(json.dumps(row))
            print("   -> pooled %.4f  TrashNet %.4f  RealWaste %.4f  Garbage-12 %.4f (%.1f min)" % (
                row["test_pooled"], row["test_trashnet"], row["test_realwaste"], row["test_garbage12"],
                row["wall_min"]), flush=True)
    print("done: %s" % a.regime_name, flush=True)


if __name__ == "__main__":
    main()
