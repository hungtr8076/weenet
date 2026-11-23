#!/usr/bin/env python3
"""Host side: the test images as uint8 crops with the evaluation transform of fine-tuning (PIL RGB, Resize(int(1.14 *
224)) + CenterCrop(224)), so PyTorch and the FP16 engines on the board (jetson/engine_accuracy.py) see one pixel set.
With --ckpt it also writes the PyTorch FP32 predictions of that checkpoint on the crops.
  python train/make_eval_crops.py --list data/split_seed0/test.csv --root $DATA_ROOT --out crops \
      [--ckpt checkpoints/weenet_do_merged3_seed0.pt --model weenet_z_ds4_do]
Writes <out>/crops_u8.npy (N x 224 x 224 x 3, RGB) and <out>/labels.npy; with --ckpt also <out>/ref_fp32.json.
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.data import MEAN, STD  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--list", required=True)
ap.add_argument("--root", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ckpt", default="")
ap.add_argument("--model", default="weenet_z_ds4_do")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
rows = list(csv.DictReader(open(a.list)))
tf = transforms.Compose([transforms.Resize(int(224 * 1.14)), transforms.CenterCrop(224)])
crops = np.zeros((len(rows), 224, 224, 3), dtype=np.uint8)
for k, r in enumerate(rows):
    crops[k] = np.asarray(tf(Image.open(os.path.join(a.root, r["path"])).convert("RGB")))
labels = np.array([int(r["label"]) for r in rows], dtype=np.int64)
np.save(os.path.join(a.out, "crops_u8.npy"), crops)
np.save(os.path.join(a.out, "labels.npy"), labels)
print("crops %s, class counts %s" % (crops.shape, np.bincount(labels).tolist()))
if a.ckpt:
    from train.models import make_model
    m = make_model(a.model, 6)
    m.load_state_dict(torch.load(a.ckpt, map_location="cpu"), strict=True)
    m.eval()
    mean, std = torch.tensor(MEAN).view(1, 3, 1, 1), torch.tensor(STD).view(1, 3, 1, 1)
    preds = []
    with torch.no_grad():
        for i in range(0, len(rows), 64):
            x = torch.from_numpy(crops[i:i + 64]).permute(0, 3, 1, 2).float() / 255.0
            preds.append(m((x - mean) / std).argmax(1).numpy())
    preds = np.concatenate(preds)
    correct = int((preds == labels).sum())
    json.dump({"correct": correct, "n": len(rows), "preds": preds.tolist()},
              open(os.path.join(a.out, "ref_fp32.json"), "w"))
    print("PyTorch FP32: %d / %d correct" % (correct, len(rows)))
