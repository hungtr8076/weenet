"""Frames and the deployed preprocessing path for the frame-latency measurement (jetson/streams.py) and the batched
engine check (jetson/verify_batch.py). Python 3.6 (no f-strings).

Frames: the first 64 images of a split CSV (columns path,label,...; paths relative to --root), decoded with OpenCV and
resized once to 640 x 480 BGR (INTER_AREA), held in memory; capture and JPEG decoding are not part of the timing.
Deployed path per frame: square center crop of the short side, INTER_AREA to 240 px, bilinear (INTER_LINEAR) to the
engine input size r, then
  pre_float  BGR -> RGB, ImageNet normalization, CHW float32 (stock engines and the FusedHaarStem engine)
  pre_u8     the r x r x 3 BGR bytes as they are (FusedHaarStemU8 engine; normalization runs in the plugin)
"""
from __future__ import print_function
import csv
import os

import cv2
import numpy as np

from trt_runner import MEAN, STD

N_FRAMES = 64
SCALE_C = (1.0 / (255.0 * STD)).astype(np.float32)[:, None, None]
BIAS_C = (-MEAN / STD).astype(np.float32)[:, None, None]


def load_frames(list_csv, root):
    with open(list_csv) as f:
        rows = list(csv.DictReader(f))[:N_FRAMES]
    if len(rows) != N_FRAMES:
        raise SystemExit("%s has %d rows, need %d" % (list_csv, len(rows), N_FRAMES))
    frames = []
    for r in rows:
        p = os.path.join(root, r["path"])
        im = cv2.imread(p)
        if im is None:
            raise SystemExit("cannot read %s" % p)
        frames.append(cv2.resize(im, (640, 480), interpolation=cv2.INTER_AREA))
    return frames


def crop_resize(bgr, res):
    h, w = bgr.shape[:2]
    s = min(h, w)
    t, l = (h - s) // 2, (w - s) // 2
    c = cv2.resize(bgr[t:t + s, l:l + s], (240, 240), interpolation=cv2.INTER_AREA)
    return cv2.resize(c, (res, res), interpolation=cv2.INTER_LINEAR)


def pre_float(bgr, res):
    x = np.ascontiguousarray(crop_resize(bgr, res).transpose(2, 0, 1)[::-1]).astype(np.float32)
    x *= SCALE_C
    x += BIAS_C
    return x


def pre_u8(bgr, res):
    return np.ascontiguousarray(crop_resize(bgr, res))
