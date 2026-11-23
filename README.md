# WeeNet on a Jetson Nano B01: per-layer roofline analysis, fused input stage and stream capacity

Code and measurement results for the paper

> Hung Tran. *Per-layer efficiency, input-stage fusion and stream capacity of CNN inference on an embedded GPU: a
> Jetson Nano case study.* Manuscript submitted to the Journal of Systems Architecture.

The paper studies where TensorRT FP16 inference time goes on a Jetson Nano B01 and how many 30 fps camera streams one
board serves, with waste classification (six classes: cardboard, glass, metal, paper, plastic, other trash) as the
workload. WeeNet is a 2.03 M-parameter CNN whose stem is a fixed Haar transform over 4x4 pixel cells followed by a
learned 1x1 convolution; the body has four stages of dense ResNet basic blocks that downsample with 4x4 stride-2
convolutions. During fine-tuning the 3x3 convolutions are over-parameterized with DO-Conv (WeeNet-DO) and folded back
into plain convolutions before export, so the deployed engine is WeeNet's. A TensorRT 8.2 plugin computes the stem (and,
for 8-bit input, the normalization) in one CUDA kernel.

This repository contains the model, the training and evaluation code, the ONNX export, the TensorRT plugin, the programs
that measured the engines on the Jetson Nano B01 (engine latency and energy, per-layer profiles, frame latency, the
isolated inputs of the capacity estimate and the measured stream capacity), the analysis programs (per-layer roofline
analysis, capacity estimate, accuracy and paired tests), and the result files from which the paper's tables are
computed, for WeeNet and for the baselines. It also contains the seed-0 data split and the seed-0 WeeNet-DO checkpoint.

## Repository layout

```
weenet/model.py              WeeNet (weenet_z_ds4), WeeNet-DO (weenet_z_ds4_do), ablation variants, DO-Conv folding,
                             Haar stem as two convolutions, MAC count
weenet/export_onnx.py        ONNX export of the four stem graphs (two-conv, haar-ops, plugin, plugin-u8), checked
                             against PyTorch with ONNX Runtime
train/data.py                the three sources, the stratified 70/15/15 split, transforms, evaluation
train/merged3.py             fine-tuning on the merged set; per-image test predictions
train/models.py              model factory, loading of WeeNet's ImageNet-1k backbone
train/imagenet_prep.py       decodes ImageNet-1k into square images for pretraining
train/pretrain_imagenet.py   ImageNet-1k pretraining of WeeNet (writes backbone.pt)
train/make_eval_crops.py     test images as 224x224 uint8 crops, and the PyTorch FP32 accuracy of a checkpoint on them
data/make_split.py           rebuilds the split and compares it with data/split_seed0/
data/split_seed0/            seed-0 split: train.csv, val.csv, test.csv (6,801 / 1,453 / 1,472 images)
data/garbage12_clean_dedup.txt   the 3,612 Garbage-12 images kept after de-duplication
data/checksums.md5           MD5 of all 9,726 images
plugin/                      TensorRT 8.2 plugins FusedHaarStem and FusedHaarStemU8 (CUDA kernel, plugin, Makefile)
plugin/make_stem_onnx.py     the ONNX rewrite to the plugin node, used by weenet/export_onnx.py
jetson/build_engine.py       FP16 engine through the TensorRT Python API (required for the plugin graphs)
jetson/build_batch.py        fixed-batch FP16 engine from a batch-1 ONNX (batched GPU policy)
jetson/verify_batch.py       checks the batched engines against the batch-1 engine
jetson/time_engine.py        one trtexec timing run with board power from tegrastats
jetson/summarize_rounds.py   latency (median of round medians) and energy over the timing rounds
jetson/engine_accuracy.py    top-1 of an engine on the crops of train/make_eval_crops.py
jetson/engine_layers.py      number of layers and layer list of built engines
jetson/prebench.py           isolated inputs of the capacity estimate: preprocessing rate, GPU-thread time per frame
jetson/streams.py            frame latency and stream capacity for K replayed 30 fps streams
jetson/frames.py             frames and deployed preprocessing path shared by streams.py, prebench.py, verify_batch.py
jetson/trt_runner.py         TensorRT runners without pycuda (CUDA runtime through ctypes)
analysis/engines.py          the engines of the paper: keys, names, ONNX stems, engine latencies
analysis/onnx_cost.py        FLOPs, bytes and roofline bound of every ONNX node (Eq. (7))
analysis/roofline.py         per-layer profiles matched to ONNX nodes: layer classes, ratios to the bound (Sect. 6)
analysis/capacity.py         capacity estimate (Eq. (8)) against the measured capacity (Tables 7 and 8)
analysis/accuracy.py         top-1, macro-F1 and exact McNemar tests with Holm's correction from per-image predictions
checkpoints/weenet_do_merged3_seed0.pt   WeeNet-DO, seed 0, 63,812 steps (the checkpoint of Table 4)
results/                     measurement results of the paper (see "Result files")
```

All commands run from the repository root. They create `outputs/`, `onnx/`, `engines/`, `crops/`, `out/` and
`build/`.

## Requirements

**Host** (data split, evaluation, ONNX export and the analysis of `results/` run on a CPU; fine-tuning and pretraining
need a CUDA GPU). The paper's runs used one NVIDIA RTX 3090 24 GB with PyTorch 2.8.0 and timm 1.0.29.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export DATA_ROOT=/path/to/data      # see Data
```

The analysis programs in `analysis/` need only numpy and scipy, and `analysis/onnx_cost.py` also onnx and onnxruntime.

**Jetson Nano B01 4 GB** with JetPack 4.6.1 (L4T R32.7.1, CUDA 10.2, cuDNN 8.2.1, TensorRT 8.2.1, OpenCV 4.1.1,
Python 3.6), MAXN power mode (`nvpmodel -q` must report MAXN), default DVFS governor. The scripts in `jetson/` need
nothing beyond JetPack (no pycuda) and no root access. TensorRT engines are tied to the GPU, the TensorRT version and
the plugin build, so they must be built on the board that runs them.

## Data

No images are included. Download the three public data sets and arrange them under `$DATA_ROOT`:

```
$DATA_ROOT/trashnet/{cardboard,glass,metal,paper,plastic,trash}/*.jpg     TrashNet, "dataset-resized"
$DATA_ROOT/realwaste/realwaste-main/RealWaste/<Folder>/*.jpg              RealWaste, GitHub archive of branch main
$DATA_ROOT/garbage12/garbage_classification/<folder>/*.jpg                Garbage Classification, 12 classes (Kaggle)
```

- TrashNet (<https://github.com/garythung/trashnet>):

  ```bash
  mkdir -p $DATA_ROOT && curl -L -o $DATA_ROOT/trashnet.zip \
    https://raw.githubusercontent.com/garythung/trashnet/master/data/dataset-resized.zip
  python3 -c "import zipfile, os; zipfile.ZipFile(os.environ['DATA_ROOT'] + '/trashnet.zip').extractall(os.environ['DATA_ROOT'])"
  mv $DATA_ROOT/dataset-resized $DATA_ROOT/trashnet && rm -rf $DATA_ROOT/__MACOSX
  ```

- RealWaste (<https://github.com/sam-single/realwaste>): download
  <https://github.com/sam-single/realwaste/archive/refs/heads/main.zip> and unpack it into `$DATA_ROOT/realwaste/`,
  which gives `realwaste-main/`.
- Garbage-12 (<https://www.kaggle.com/datasets/mostafaabla/garbage-classification>, needs a Kaggle account): unpack
  into `$DATA_ROOT/garbage12/`, which gives `garbage_classification/`.

The merged set has six classes. RealWaste keeps the six folders that map onto them (Miscellaneous Trash becomes other
trash); Garbage-12 keeps eight folders (brown-glass, green-glass and white-glass become glass; battery, biological,
clothes and shoes are dropped) and only the 3,612 images listed in `data/garbage12_clean_dedup.txt`. In total 9,726
images (paper, Sect. 4.2 and Table 3).

## Reproducing the paper

Steps 1 to 6 run on the host, steps 7 to 12 on the Jetson Nano. Steps 13 to 16 recompute the paper's tables from the
result files in `results/` and run on any computer (no GPU, no images).

### 1. Data split and MD5 check (Table 3)

```bash
python data/make_split.py --manifest                          # no images needed: split rebuilt from the MD5 list
python data/make_split.py --data-root $DATA_ROOT --check-md5  # split rebuilt from the images; MD5 of every image
```

Both commands print the train / val / test sizes per source (TrashNet 1,766 / 377 / 384, RealWaste 2,509 / 537 / 541,
Garbage-12 2,526 / 539 / 547, total 6,801 / 1,453 / 1,472) and must report every part identical to
`data/split_seed0/`; the second also reports `md5: 0 of 9726 images differ`. Either command exits with status 1 on any
difference. `--seed <s> --write <dir>` writes the split of another seed.

### 2. Model size (Sect. 4.1)

```bash
python -c "from weenet import model as wm; m = wm.build('weenet_z_ds4'); print('%.2f M params, %.2f GMAC' % (sum(p.numel() for p in m.parameters()) / 1e6, wm.count_macs(m) / 1e9))"
```

Prints `2.03 M params, 0.36 GMAC`.

### 3. ImageNet-1k pretraining (GPU; Sect. 4.3)

Fine-tuning starts from WeeNet's own ImageNet-1k weights, which are not included. They are made from the Hugging Face
parquet release `ILSVRC/imagenet-1k`, stored locally at `$HF_IMAGENET`:

```bash
python train/imagenet_prep.py --parquet $HF_IMAGENET/data --pattern 'train-*.parquet' --split train --size 176 \
    --format jpg --quality 95 --out $IMAGENET_PREP
python train/imagenet_prep.py --parquet $HF_IMAGENET/data --pattern 'validation-*.parquet' --split val224 \
    --size 224 --crop-ratio 0.95 --out $IMAGENET_PREP
python train/pretrain_imagenet.py --data $IMAGENET_PREP --epochs 100 --warmup 1 --batch 512 --res 160 \
    --val-split val224 --test-res 224 --label-smoothing 0.1 --out outputs/in1k_weenet
export BACKBONE=outputs/in1k_weenet/backbone.pt
```

`imagenet_prep.py --folder <dir>` replaces `--parquet` for an ImageNet copy with one sub-folder per class. The paper's
backbone reached 55.40 % top-1 at 224 px after 339 min on one RTX 3090 (Sect. 4.3).

### 4. Fine-tuning on the merged set (GPU; Table 4, WeeNet-DO row)

```bash
python train/merged3.py --data-root $DATA_ROOT --models weenet_z_ds4_do --backbone $BACKBONE \
    --seeds 0 --split-seed 0 --regime-name finetune --outdir outputs
```

Recipe (Sect. 4.3): RandomResizedCrop 0.35 to 1, horizontal flip, TrivialAugment, RandomErasing 0.25, MixUp 0.2 or
CutMix 1.0 per batch, label smoothing 0.1, AdamW, learning rate 6e-4, weight decay 0.05, one-cycle schedule, batch 32,
224 px, 63,812 steps (301 epochs of 212 steps; the default `--steps 63750` is rounded up to whole epochs). The result
is `outputs/finetune/seed_0/weenet_z_ds4_do.json` (pooled and per-source test top-1, per-image test predictions, best
and last validation accuracy) and the best-validation checkpoint `outputs/finetune/seed_0/ckpt/weenet_z_ds4_do.pt`.
Training on a GPU is not deterministic: three seeds of one model on the seed-0 split differ by up to 7 test images
(Sect. 10).

A CPU check of the code path with random images (one short epoch, not a training run):

```bash
python train/merged3.py --smoke --models weenet_z_ds4_do
```

### 5. Ablation training (GPU; Table 12, top-1 column)

Each variant changes one component, is trained from scratch (no `--backbone`) with the same recipe and three seeds.
Without `--split-seed`, each seed draws its own 70/15/15 split, shared by the four variants:

```bash
python train/merged3.py --data-root $DATA_ROOT --seeds 0,1,2 --regime-name ablation \
    --models weenet_z_ds4,weenet_z_ds4_cstem,weenet_haar_g1,weenet_z_ds4_dw --outdir outputs
```

`weenet_z_ds4` is the full WeeNet; `weenet_z_ds4_cstem` replaces the Haar stem with two 3x3 stride-2 convolutions
(3 to 16 to 32 channels); `weenet_haar_g1` downsamples with 3x3 stride-2 convolutions and 1x1 shortcuts;
`weenet_z_ds4_dw` uses depthwise-separable blocks. The top-1 of each run is `test_pooled` in
`outputs/ablation/seed_<s>/<model>.json`.

### 6. Checkpoint accuracy and ONNX export (CPU)

```bash
python train/make_eval_crops.py --list data/split_seed0/test.csv --root $DATA_ROOT --out crops \
    --ckpt checkpoints/weenet_do_merged3_seed0.pt --model weenet_z_ds4_do
```

This writes the 1,472 test images as 224x224 crops with the evaluation transform (short side resized to 255 px, center
crop 224) to `crops/crops_u8.npy` and `crops/labels.npy`, runs the checkpoint on them and prints
`PyTorch FP32: 1388 / 1472 correct` (94.29 %, Table 4). The crops are also the input of step 9.

Export the WeeNet-DO checkpoint in four stem forms, and the four ablation variants with random weights (latency does
not depend on the weights):

```bash
CK=checkpoints/weenet_do_merged3_seed0.pt
python weenet/export_onnx.py --model weenet_z_ds4_do --ckpt $CK --out onnx/weenet_do.onnx
python weenet/export_onnx.py --model weenet_z_ds4_do --ckpt $CK --graph haar-ops --out onnx/weenet_do_haarops.onnx
python weenet/export_onnx.py --model weenet_z_ds4_do --ckpt $CK --graph plugin --out onnx/weenet_do_plugin.onnx
python weenet/export_onnx.py --model weenet_z_ds4_do --ckpt $CK --graph plugin-u8 --out onnx/weenet_do_u8.onnx
for m in weenet_z_ds4 weenet_z_ds4_cstem weenet_haar_g1 weenet_z_ds4_dw; do
  python weenet/export_onnx.py --model $m --random --out onnx/abl_$m.onnx
done
```

The default graph (`two-conv`) writes the Haar stem exactly as a 4x4 stride-4 convolution (3 to 96 channels), a ReLU, a
1x1 convolution (96 to 32 channels: the Haar features' 1x1 map, the two BatchNorm layers and the learned 1x1
convolution folded into one) and a ReLU
(Sect. 4.1), so the network has only Conv, Relu, Add, GlobalAveragePool and Gemm nodes; every stock WeeNet latency in
the paper uses it. `haar-ops` exports the stem as written in PyTorch (Sect. 4.1: 10 TensorRT layers, 3.24 ms). `plugin`
replaces the stem by one FusedHaarStem node (normalized FP32 input 1x3x224x224; this engine is not reported in the
paper); `plugin-u8` by one FusedHaarStemU8 node whose input is the raw BGR bytes of a 224x224x3 frame, declared as FP32
1x3x112x112 (same byte count). This 8-bit engine is the plugin engine of the paper. Every export is compared with
PyTorch in ONNX Runtime (relative RMS error at most 1e-5 and equal argmax; for the plugin graphs, the graph after the
plugin node) and stops otherwise.

### 7. Plugin and engines (Jetson)

Copy `jetson/`, `plugin/`, `data/split_seed0/`, `onnx/` and `crops/` to the board, and the first 64 images of
`data/split_seed0/test.csv` (all TrashNet images; `$DATA_ROOT/trashnet` is enough) for steps 11 and 12. On the board,
from the directory that holds them:

```bash
export DATA_ROOT=/path/to/data      # on the board
TRTEXEC=/usr/src/tensorrt/bin/trtexec
make -C plugin                      # -> build/libfused_haar_stem.so (nvcc, sm_53)
mkdir -p engines out
for m in weenet_do weenet_do_haarops abl_weenet_z_ds4 abl_weenet_z_ds4_cstem abl_weenet_haar_g1 abl_weenet_z_ds4_dw; do
  ws=1024; case $m in abl_*) ws=256;; esac
  $TRTEXEC --onnx=onnx/$m.onnx --fp16 --workspace=$ws --saveEngine=engines/$m.plan
done
for m in weenet_do_plugin weenet_do_u8; do
  LD_PRELOAD=build/libfused_haar_stem.so python3 jetson/build_engine.py --onnx onnx/$m.onnx \
      --out engines/$m.plan --plugin build/libfused_haar_stem.so
done
```

The comparison engines were built with `trtexec --fp16 --workspace=1024`, the ablation engines with a 256 MB
workspace. The plugin graphs must be built with `jetson/build_engine.py` (256 MB workspace), which fixes the plugin
output to FP16; it must report `1 plugin layers`. `python3 jetson/engine_layers.py --outdir out/layers
wee=engines/weenet_do.plan` prints the number of layers of an engine (25 for the stock WeeNet engine, Fig. 3).

### 8. Engine latency and energy (Table 4 WeeNet row, Sect. 4.1, Sect. 5.1, Table 12 ms column)

Each engine is warmed up for 3 s and timed for 15 s with `trtexec`; the latency is trtexec's median GPU compute time,
and the energy per image is the mean POM_5V_IN board power over the timed window divided by the throughput
(Sect. 4.4, Eq. (2)). Three rounds, interleaved across engines, then the median of the three round medians:

```bash
for r in 1 2 3; do
  for m in weenet_do weenet_do_haarops; do
    python3 jetson/time_engine.py --plan engines/$m.plan --label $m --round $r --outdir out/latency
  done
  for m in weenet_do_plugin weenet_do_u8; do
    python3 jetson/time_engine.py --plan engines/$m.plan --plugin build/libfused_haar_stem.so \
        --label $m --round $r --outdir out/latency
  done
done
python3 jetson/summarize_rounds.py out/latency

for r in 1 2 3; do
  for m in weenet_z_ds4 weenet_z_ds4_cstem weenet_haar_g1 weenet_z_ds4_dw; do
    python3 jetson/time_engine.py --plan engines/abl_$m.plan --label $m --round $r --outdir out/ablation
  done
done
python3 jetson/summarize_rounds.py out/ablation
```

Paper values: two-conv 3.15 ms and 21.4 mJ (Table 4), Haar ops 3.24 ms (Sect. 4.1), plugin engine with 8-bit input
2.65 ms and 17.5 mJ (Sect. 5.1); ablation (one run with the four engines) 3.19 / 3.23 / 2.74 / 2.93 ms for full WeeNet,
conv stem, 3x3 downsampling and depthwise convolutions (Table 12). `time_engine.py` writes `<label>.r<round>.json`
with the raw trtexec, tegrastats and clock logs; `summarize_rounds.py` exits with an error if a label lacks a round or
a round ran with the median GPU clock below 921.6 MHz.

### 9. Engine accuracy on the board (Sect. 5.1)

```bash
python3 jetson/engine_accuracy.py --plan engines/weenet_do.plan --crops crops/crops_u8.npy \
    --labels crops/labels.npy --label weenet_do --out out/acc_weenet_do.json
LD_PRELOAD=build/libfused_haar_stem.so python3 jetson/engine_accuracy.py --plan engines/weenet_do_u8.plan --u8 \
    --plugin build/libfused_haar_stem.so --crops crops/crops_u8.npy --labels crops/labels.npy \
    --label weenet_do_u8 --out out/acc_weenet_do_u8.json
```

On the same crops as step 6, the paper reports 1,387 of 1,472 correct for both FP16 engines, with identical
predictions, against 1,388 for PyTorch FP32.

### 10. Per-layer profiles (Sect. 4.4 and 6)

trtexec's per-layer profile gives the mean time of each TensorRT layer (2 s warm-up, 10 s, profiling run separate from
the timing run):

```bash
mkdir -p out/profiles
for m in weenet_do:wee abl_weenet_z_ds4:abl_full abl_weenet_z_ds4_cstem:abl_cstem abl_weenet_haar_g1:abl_g1 \
         abl_weenet_z_ds4_dw:abl_dw; do
  $TRTEXEC --loadEngine=engines/${m%%:*}.plan --warmUp=2000 --duration=10 --dumpProfile --separateProfileRun \
      --exportProfile=out/profiles/prof_${m##*:}.json
done
$TRTEXEC --loadEngine=engines/weenet_do_u8.plan --plugins=build/libfused_haar_stem.so --warmUp=2000 --duration=10 \
    --dumpProfile --separateProfileRun --exportProfile=out/profiles/prof_wee_u8.json
```

The file names are those of `results/profiles/`. To analyze them with step 14 together with the shipped profiles of
the baselines, copy `results/profiles/*.json` into a new folder, overwrite the WeeNet files with these and pass that
folder to `analysis/roofline.py --profiles`.

### 11. Frame latency with one 30 fps stream (Table 5)

`jetson/streams.py` feeds K replayed 30 fps streams of 640x480 BGR frames held in memory through the deployed path of
Fig. 1: square center crop, INTER_AREA to 240 px, bilinear resize to the engine input, normalization (inside the plugin
for the 8-bit engine), host-to-device copy, TensorRT execution at batch 1, device-to-host copy and argmax. A frame
misses its deadline if its result is not ready before the next frame of its stream arrives, or if it is dropped because
a newer frame of the same stream arrived first (Sect. 3.2). Each K runs 60 s after a 5 s warm-up (Sect. 4.4).

```bash
python3 jetson/streams.py --plan engines/weenet_do.plan --pre float --list data/split_seed0/test.csv \
    --root $DATA_ROOT --streams 1 --label k1_weenet --out out/streams/k1_weenet.json
python3 jetson/streams.py --plan engines/weenet_do_u8.plan --pre u8 --plugin build/libfused_haar_stem.so \
    --list data/split_seed0/test.csv --root $DATA_ROOT --streams 1 --label k1_weenet_u8 \
    --out out/streams/k1_weenet_u8.json
```

Each run prints the miss rate, the median, P99 and maximum frame latency, the mean GPU clock and the mean board power,
the columns of Table 5 (paper: WeeNet 18.4 / 23.3 ms, 0.0 % missed, 231 MHz, 2.59 W; WeeNet + plugin 18.7 / 23.8 ms,
0.0 %, 190 MHz, 2.20 W). The same command measures any other batch-1 FP16 engine with one input and one output; add
`--res` for an input size other than 224 and `--std-plugins` for engines that use TensorRT's standard plugins.

### 12. Stream capacity of one board (Sect. 7)

Isolated inputs of the estimate (Sect. 4.4): the rate of three preprocessing threads and the median and P99 of the
GPU-thread time per frame, each measured back to back for 15 s without the arrival process:

```bash
python3 jetson/prebench.py --plan engines/weenet_do.plan --pre float --list data/split_seed0/test.csv \
    --root $DATA_ROOT --label wee --out out/capacity/pre_wee.json
python3 jetson/prebench.py --plan engines/weenet_do_u8.plan --pre u8 --plugin build/libfused_haar_stem.so \
    --list data/split_seed0/test.csv --root $DATA_ROOT --label wee_u8 --out out/capacity/pre_wee_u8.json
```

Measured capacity K* (Eq. (4)): the largest K whose miss rate stays below 1 % in every run, with evenly staggered
streams (Eq. (3)). A 60 s sweep from small K upward locates K* (it stops after the first K above 5 % missed frames);
then two 180 s runs at K* and K* + 1 give the value of Table 7:

```bash
python3 jetson/streams.py --plan engines/weenet_do.plan --pre float --list data/split_seed0/test.csv \
    --root $DATA_ROOT --streams 1,2,4,6,7,8,9,10,11,12 --label sweep_wee --out out/sweep_wee.json
python3 jetson/streams.py --plan engines/weenet_do_u8.plan --pre u8 --plugin build/libfused_haar_stem.so \
    --list data/split_seed0/test.csv --root $DATA_ROOT --streams 1,2,4,6,8,9,10,11,12 \
    --label sweep_wee_u8 --out out/sweep_wee_u8.json
for n in 1 2; do
  python3 jetson/streams.py --plan engines/weenet_do.plan --pre float --list data/split_seed0/test.csv \
      --root $DATA_ROOT --streams 8,9 --dur 180 --label wee_$n --out out/capacity/even_wee_$n.json
  python3 jetson/streams.py --plan engines/weenet_do_u8.plan --pre u8 --plugin build/libfused_haar_stem.so \
      --list data/split_seed0/test.csv --root $DATA_ROOT --streams 10,11 --dur 180 --label wee_u8_$n \
      --out out/capacity/even_wee_u8_$n.json
done
```

Paper: K* = 8 for WeeNet and 10 for the plugin engine. The four engines measured after both estimators were fixed
(Table 7, 1x60 runs) come from one 60 s sweep from K = 1 upward (`--streams 1,2,3,4,5,6,7,8`), written as
`even_<key>_1.json`.

Synchronized streams (all K frames arrive at the same instant), 60 s at the staggered capacity (paper: 12.99 % missed
frames for WeeNet at 8 streams, 10.36 % for the plugin engine at 10):

```bash
python3 jetson/streams.py --plan engines/weenet_do.plan --pre float --list data/split_seed0/test.csv \
    --root $DATA_ROOT --streams 8 --phase aligned --label wee_aligned --out out/capacity/aligned_wee.json
```

Batched GPU policy: the GPU thread takes every waiting frame in one launch, on the smallest fixed-batch engine that
holds them. Build engines for B = 2 to 8 from the same ONNX, check them against the batch-1 engine, then sweep K:

```bash
for b in 2 3 4 5 6 7 8; do
  python3 jetson/build_batch.py --onnx onnx/weenet_do.onnx --batch $b --out engines/weenet_do_b$b.plan
done
BP=$(for b in 2 3 4 5 6 7 8; do printf "%s:engines/weenet_do_b%s.plan," $b $b; done)
python3 jetson/verify_batch.py --plans 1:engines/weenet_do.plan,${BP%,} --res 224 --list data/split_seed0/test.csv \
    --root $DATA_ROOT --label weenet_do --out out/verify_weenet_do.json
python3 jetson/streams.py --plan engines/weenet_do.plan --batch-plans ${BP%,} --policy batch --pre float \
    --list data/split_seed0/test.csv --root $DATA_ROOT --streams 7,8,9,10,11,12 \
    --label batch_wee --out out/capacity/batch_wee.json
```

Paper: 11 streams for WeeNet with batching, at a median frame latency of 24.1 ms (Sect. 7). No batched plugin engine
was built. `python analysis/capacity.py out/capacity` (step 15) computes the estimate and K* from these files.

### 13. Engine latency from the shipped timing rounds (Table 4, Table 7)

```bash
python jetson/summarize_rounds.py results/latency/models
python jetson/summarize_rounds.py results/latency/capacity_engines
```

The first command prints the latency and energy per image of ten of the twelve models of Table 4 and of the plugin
engine (Sect. 5.1), from their three timing rounds: for example `weenet_z_ds4_rp 3.151 ms 21.4 mJ`,
`weenet_z_ds4_u8 2.655 ms 17.5 mJ`, `yolo11n_cls 4.178 ms 28.2 mJ`, `mobilevitv2_100 32.160 ms 230.4 mJ`. The rounds of
RepVGG-A0 and FastViT-T8 are not available. The second prints the engines of the capacity study timed in the same way;
`yolo160_plain 3.416 ms` is the engine latency of YOLO11n-cls at 160 px in Table 7.

### 14. Per-layer roofline analysis (Sect. 6: Table 6, Fig. 3)

The ONNX node costs of the WeeNet graphs are recomputed from the exports of step 6 and are identical to the shipped
files:

```bash
mkdir -p out/cost
python analysis/onnx_cost.py onnx/weenet_do.onnx --json out/cost/weenet_do.json
for m in weenet_z_ds4 weenet_z_ds4_cstem weenet_haar_g1 weenet_z_ds4_dw; do
  python analysis/onnx_cost.py onnx/abl_$m.onnx --json out/cost/abl_$m.json
done
for f in out/cost/*.json; do cmp $f results/cost/$(basename $f) && echo "identical: $f"; done
```

Then the analysis of the profiles of the 13 profiled engines (twelve models at 224 px and YOLO11n-cls at 160 px) and the
ablation engines against their roofline bound (P = 471.9 GFLOP/s, Q = 25.6 GB/s):

```bash
python analysis/roofline.py --profiles results/profiles --costs results/cost
```

It prints, against the paper:

- the engine-level bound, 10 to 66 % of the engine latency, and its Spearman correlation with latency, 0.636
  (p = 0.026), against 0.629 for ONNX FLOPs (Sect. 6);
- per engine the number of layers, the summed profile against the engine latency (within 4.2 % for the eleven engines
  of Table 6, +20.7 % for YOLO11n-cls at 224 px and +50 % at 160 px) and the time share of each class (Fig. 3: WeeNet
  25 layers, 86 % dense convolutions; 19 to 37 % non-convolution layers in every engine above 100 layers);
- Table 6: mean non-convolution layer 0.047 ms over 814 layers; dense 75 layers, 30.33 / 17.06 ms, ratio 1.8,
  adjusted 1.6, per engine 1.4-6.2, 5 % memory-bound; pointwise 377, 57.99 / 17.14, 3.4, 2.3, 2.6-4.7, 21 %;
  depthwise 168, 32.72 / 2.16, 15.1, 11.5, 8.1-26.7, 95 %; MatMul / Gemm 22, 1.02 / 0.09, 11.4, 77 %; pooling 14,
  1.24 / 0.18, 6.9, 3.2, 100 %; non-convolution 814 layers, 38.50 ms;
- the input stage of WeeNet: 0.390 ms at 6.4 times its bound, 0.178 ms at 4.3 times and a 0.085 ms format conversion,
  0.65 ms against a bound of 0.10 ms; the plugin layer 0.14 ms; saving 0.52 ms; the rest of the two engines differs by
  0.01 ms;
- the layer-class latency model: errors of +5.3, +1.2, +10.5, -12.5 and +18.6 % on the four ablation engines and
  YOLO11n-cls at 160 px, 4 of 6 ablation pairs in the measured order.

### 15. Capacity estimate against the measured capacity (Sect. 7: Tables 7 and 8)

```bash
python analysis/capacity.py results/capacity
```

It prints Table 7 (for example `WeeNet + plugin 2.65 2.92 1.03 11.42 11 11 10 2x180 6.79 22.6` and `WeeNet 3.15 3.69
1.18 9.04 9 7 8 2x180 6.96 29.0`), the CPU rate (904 to 1625 frames/s with three threads) and the GPU-thread overhead
over the engine latency (0.27 to 0.54 ms), Table 8 (median estimator over the ten engines: 5 exact, 8 within one stream,
5 over, 0 under, bias +0.80, MAE 0.80, max 3, 15 %; P99 estimator: 6, 10, 2, 2, 0.00, 0.40, 1, 7 %; on the four engines
measured afterward: median 4, 4, 0, 0, 0.00, 0.00, 0, 0 %; P99 3, 4, 0, 1, -0.25, 0.25, 1, 6 %), the GPU clock of
YOLO11n-cls at 160 px during its 180 s runs (670 to 805 MHz, GPU-thread time 4.44 to 4.54 ms), the synchronized runs
(10.36 %, 12.99 %, 6.49 %, 1.13 %, 0.04 %, 0.00 %) and the batched capacities (11 for WeeNet at 24.1 ms, at least 16 for
YOLO11n-cls at 160 px at 23.2 ms (K = 17 was not run), 5 for MobileNetV4-Conv-S, 2 for HarDNet-39DS).

### 16. Accuracy and paired tests (Table 4, Sect. 8.1, Table 9)

```bash
python analysis/accuracy.py --ref results/predictions/twelve_models/weenet_z_ds4_do.json \
    results/predictions/twelve_models/*.json
python analysis/accuracy.py results/predictions/further_networks/seed_*/*.json
```

The first command prints the top-1 and macro-F1 of the twelve models (Table 4, for example WeeNet-DO 1,388 correct,
94.29 % and 94.21 %) and the exact McNemar test of each baseline against WeeNet-DO with Holm's correction (Sect. 8.1:
HarDNet-39DS p = 0.016, 0.082 after correction; ResNet-18 0.052, ShuffleNetV2-1.0x 0.245, RepVGG-A0 0.306,
YOLO11n-cls 0.708). The second prints the correct test images of the two networks fine-tuned after Eq. (9) was fixed,
seeds 0 to 2 (Table 9: RegNetX-800MF 1,410 / 1,406 / 1,412, GENet-small 1,406 / 1,398 / 1,398); seeds 1 and 2 use
their own splits.

## Result files

`results/` holds the measurement outputs the paper's tables are computed from, as JSON. Engine keys (`wee`, `wee_u8`,
`yolo224`, `yolo160`, `mnv4`, ...) are listed with their paper names in `analysis/engines.py`. Paths are relative:
images to `$DATA_ROOT`, engines and checkpoints by file name.

```
results/profiles/prof_<key>.json          trtexec per-layer profiles (--exportProfile, step 10) of the twelve models at
                                          224 px and YOLO11n-cls at 160 px (Table 6 and Fig. 3 use eleven of them), the
                                          plugin engine and the four ablation engines
results/cost/<onnx stem>.json             FLOPs, bytes and bound of every ONNX node (onnx_cost.py), same engines
results/capacity/pre_<key>.json           isolated inputs of the capacity estimate, ten engines (jetson/prebench.py)
results/capacity/even_<key>_<n>.json      staggered-stream runs that give K*: repeat n (two 180 s repeats for six
                                          engines, one 60 s sweep for four; even_yolo160_1b.json is the K = 7 run of
                                          repeat 1 of YOLO11n-cls at 160 px)
results/capacity/aligned_<key>.json       synchronized streams, 60 s
results/capacity/batch_<key>.json         batched GPU policy with engines for every batch size from 1 to 8, 60 s
results/latency/models/                   three timing rounds of ten models of Table 4 and of the plugin engine
results/latency/capacity_engines/         three timing rounds of the engines of the capacity study
results/predictions/twelve_models/        per-image test predictions of the twelve models of Table 4, seed 0, seed-0
                                          split (train/merged3.py)
results/predictions/further_networks/     the same for RegNetX-800MF and GENet-small, seeds 0 to 2 (Table 9)
```

Labels used in the file names and contents: `weenet_z_ds4_rp` is the stock WeeNet engine (two-convolution stem
graph), `weenet_z_ds4_u8` the plugin engine, `*_plain` an engine of the capacity study, `wee_b<B>.plan` a fixed-batch
WeeNet engine of the batched policy, and `yo_s0.plan` YOLO11n-cls at 160 px. In the stream and isolated-input files
`"pre": "fast"` is the float preprocessing path (`--pre float`), and `"resize": "a2l"` the deployed resize path
(INTER_AREA to 240 px, then bilinear to the input size). The fields `legacy`, `fan_start`, `fan_end`, `achieved_fps` and
`seed` are recorded but not used by the analysis. In `prof_wee_u8.json` and in the plugin-engine timing rounds the
plugin layer and its library appear as `stem_csmstem` and `libcsm.so`: these are the FusedHaarStemU8 kernel of
`plugin/` and its shared library under the names of the measured build. In `results/predictions/`, `"recipe": "strong"`
and `"regime": "long300"` name the fine-tuning recipe and the 63,812-step schedule of step 4, and `"pretrained"` says
whether ImageNet-1k weights were loaded. The per-round files in `results/latency/capacity_engines/` also record
trtexec's enqueue time (`extra.enqueue_ms`; Sect. 7: 3.32 ms for YOLO11n-cls at 160 px, the median of the round
medians).

## Not included

- **Baselines.** The fine-tuning, ONNX export and engines of the eleven baseline models (timm, torchvision and
  Ultralytics weights) and the two further networks of Table 9 are not in this repository. Their measurement results
  are (`results/`), so steps 13 to 16 recompute the baseline rows of Tables 4, 6, 7, 8 and 9; the measurements
  themselves can be repeated here for the WeeNet engines only. `jetson/time_engine.py`, `jetson/prebench.py`,
  `jetson/streams.py`, `jetson/build_batch.py` and `jetson/verify_batch.py` work on any batch-1 FP16 engine with one
  input and one output.
- **Result files not shipped:** the one-stream runs of Table 5, the timing rounds of RepVGG-A0, FastViT-T8, the
  ablation engines (Table 12) and the two further networks (Table 9), the ablation accuracy runs (Table 12, top-1),
  the matched-latency study (Sect. 8.3, Table 10: YOLO11n-cls at 160 px, three seeds, resize study), the CUDA-graph
  timings (Sect. 8.3), the 20-minute sustained run (Sect. 5.2), the 60 s sweeps that located K* before the 180 s
  repeats (Sect. 7), the timing of the `haar-ops` graph (Sect. 4.1: 3.24 ms) and the later re-timings of WeeNet
  (Sect. 4.4), and the engine-accuracy outputs (Sect. 5.1: 1,387 of 1,472).
- **Analysis code not included:** the rank correlations of Sect. 5.1 (MACs against latency, energy against latency),
  the log-latency fit of Eq. (9) and its prediction interval (Sect. 8.2), the Welch and paired intervals of Sects. 8.3
  and 8.5, the assembly of Table 11 from Tables 4 and 7, and the scripts that draw Figs. 3 to 6.
- **Data preparation not included:** the de-duplication of Garbage-12 against TrashNet, RealWaste and itself (exact
  copies by MD5, near-duplicates by a 64-bit perceptual hash with Hamming distance at most 8; Sect. 4.2), applied to
  the cleaned 13,106-image list of the Garbage-12 release. `data/garbage12_clean_dedup.txt` is its output.
- The ImageNet-1k backbone (`backbone.pt`) and the checkpoints other than the seed-0 WeeNet-DO checkpoint.

## Verification status

Run for this release on a CPU-only machine (macOS, Python 3.12, torch 2.8.0, timm 1.0.29, onnx 1.23.2,
onnxruntime 1.32.0, scipy 1.18.1) in a fresh virtual environment: step 1 (`--manifest`), step 2, the smoke check of
step 4, the eight exports of step 6, and steps 13 to 16 from `results/`, with the outputs quoted above. The cost files
of the five WeeNet graphs written by step 14 are byte-identical to those in `results/cost/`, and the node names of the
exported WeeNet graphs match the layer names of the shipped profiles. Step 1 with `--check-md5` and the crops of step
6 need the images of all three data sets and were not re-run for this release. Training (steps 3 to 5) needs a GPU and
was not re-run from this repository. The Jetson steps (7 to 12) were not run from this repository; their scripts were
only checked for syntax.

## Data sets and third-party work

No images are redistributed here; `data/` holds file lists, the split and MD5 checksums only. The images come from
TrashNet (M. Yang and G. Thung, <https://github.com/garythung/trashnet>, MIT license), RealWaste (S. Single,
S. Iranmanesh and R. Raad, Information 14 (2023) 633, <https://github.com/sam-single/realwaste>) and Garbage
Classification (M. Mohamed, <https://www.kaggle.com/datasets/mostafaabla/garbage-classification>); obtain them from
these sources under their terms. The WeeNet backbone was pretrained on ImageNet-1k, whose images are subject to the
ImageNet terms of access. DO-Conv (J. Cao et al., IEEE Trans. Image Process. 31 (2022) 3726-3736) is reimplemented in
`weenet/model.py`; the original implementation is MIT-licensed. Writing the absolute value of the Haar details as
ReLU(y) + ReLU(-y) follows F. Guth et al. (ICLR 2022).

## Citation

```bibtex
@unpublished{tran_weenet_jetson_nano,
  author = {Hung Tran},
  title  = {Per-layer efficiency, input-stage fusion and stream capacity of {CNN} inference on an embedded {GPU}:
            a {Jetson Nano} case study},
  note   = {Manuscript submitted to the Journal of Systems Architecture}
}
```

## License

The code and the checkpoint are released under the MIT license, see [LICENSE](LICENSE).
