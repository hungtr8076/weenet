"""The merged data set: TrashNet + RealWaste + de-duplicated Garbage-12, six classes.

Expected layout under DATA_ROOT (the folders the public downloads unpack to):
  trashnet/<class>/*.jpg                                   six folders: cardboard glass metal paper plastic trash
  realwaste/realwaste-main/RealWaste/<Folder>/*.jpg        the RealWaste GitHub archive
  garbage12/garbage_classification/<folder>/*.jpg          the Kaggle "Garbage Classification" (12 classes) archive
Garbage-12 is read through data/garbage12_clean_dedup.txt (paths relative to DATA_ROOT), the 3,612 images left after
de-duplication against TrashNet, RealWaste and itself.

Each source is split 70 / 15 / 15 per class with numpy.random.default_rng(seed); the split with seed 0 is the reported one
(6,801 / 1,453 / 1,472 images) and is also shipped as data/split_seed0/*.csv. Train = the three train parts, val = the
three val parts (checkpoint selection), test per source and pooled.
"""
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SOURCES = ("trashnet", "realwaste", "garbage12")
RW_DIR = os.path.join("realwaste", "realwaste-main", "RealWaste")
RW_MAP = {"Cardboard": "cardboard", "Glass": "glass", "Metal": "metal", "Paper": "paper", "Plastic": "plastic",
          "Miscellaneous Trash": "trash"}
G12_MAP = {"brown-glass": "glass", "green-glass": "glass", "white-glass": "glass", "cardboard": "cardboard",
           "metal": "metal", "paper": "paper", "plastic": "plastic", "trash": "trash"}
IMG = (".jpg", ".jpeg", ".png")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _cut(n):
    return int(0.70 * n), int(0.15 * n)


def trashnet_split(root, seed):
    """Stratified 70/15/15 over torchvision's ImageFolder order (classes sorted, files sorted within a class)."""
    from torchvision import datasets
    full = datasets.ImageFolder(root)
    if full.classes != CLASSES:
        raise SystemExit("TrashNet classes %s != %s" % (full.classes, CLASSES))
    targets = np.array(full.targets)
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for c in np.unique(targets):
        idx = np.where(targets == c)[0]
        rng.shuffle(idx)
        n_tr, n_va = _cut(len(idx))
        for part, sel in zip(parts, (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])):
            part.extend(full.samples[i] for i in sel)
    return parts


def realwaste_split(root, seed):
    """The six RealWaste folders that map onto the classes (Miscellaneous Trash -> trash), stratified 70/15/15."""
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for folder, cls in sorted(RW_MAP.items()):
        files = sorted(str(p) for p in Path(root, folder).iterdir() if p.suffix.lower() in IMG)
        if not files:
            raise SystemExit("%s/%s: no images" % (root, folder))
        idx = np.arange(len(files))
        rng.shuffle(idx)
        n_tr, n_va = _cut(len(files))
        y = CLASSES.index(cls)
        for part, sel in zip(parts, (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])):
            part.extend((files[i], y) for i in sel)
    return parts


def garbage12_split(data_root, list_path, seed):
    """The de-duplicated Garbage-12 list; brown / green / white glass -> glass; stratified 70/15/15."""
    if not os.path.exists(list_path):
        raise SystemExit("%s missing (expected data/garbage12_clean_dedup.txt)" % list_path)
    by = {}
    for rel in open(list_path).read().split("\n"):
        if rel:
            folder = rel.split("/")[-2]
            if folder not in G12_MAP:
                raise SystemExit("%s: folder %s is not a mapped Garbage-12 class" % (rel, folder))
            by.setdefault(G12_MAP[folder], []).append(os.path.join(data_root, rel))
    if sorted(by) != sorted(CLASSES):
        raise SystemExit("Garbage-12 classes %s != %s" % (sorted(by), CLASSES))
    rng = np.random.default_rng(seed)
    parts = ([], [], [])
    for cls in sorted(by):
        files = sorted(by[cls])
        idx = np.arange(len(files))
        rng.shuffle(idx)
        n_tr, n_va = _cut(len(files))
        y = CLASSES.index(cls)
        for part, sel in zip(parts, (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])):
            part.extend((files[i], y) for i in sel)
    return parts


def merged_split(data_root, g12_list, seed):
    """{source: (train, val, test)} with items (absolute path, label)."""
    return {"trashnet": trashnet_split(os.path.join(data_root, "trashnet"), seed),
            "realwaste": realwaste_split(os.path.join(data_root, RW_DIR), seed),
            "garbage12": garbage12_split(data_root, g12_list, seed)}


class Files(Dataset):
    def __init__(self, items, tf):
        self.items, self.tf = items, tf

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tf(Image.open(p).convert("RGB")), y


def eval_tf(res):
    """Resize the short side to int(1.14 r), center crop r (antialiased PIL bilinear), ImageNet normalization."""
    from torchvision import transforms
    return transforms.Compose([transforms.Resize(int(res * 1.14)), transforms.CenterCrop(res), transforms.ToTensor(),
                               transforms.Normalize(MEAN, STD)])


def train_tf(res):
    """RandomResizedCrop 0.35-1, flip, TrivialAugmentWide, RandomErasing 0.25 (MixUp / CutMix are applied per batch)."""
    from torchvision import transforms
    return transforms.Compose([transforms.RandomResizedCrop(res, scale=(0.35, 1.0)), transforms.RandomHorizontalFlip(),
                               transforms.TrivialAugmentWide(), transforms.ToTensor(), transforms.Normalize(MEAN, STD),
                               transforms.RandomErasing(p=0.25)])


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, labels = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        labels.append(y)
    return torch.cat(logits).numpy(), torch.cat(labels).numpy()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    right = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        right += int((model(x).argmax(1) == y).sum())
        total += y.numel()
    return right / max(total, 1)
