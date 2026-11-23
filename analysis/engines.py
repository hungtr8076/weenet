"""The engines of the paper: short key (used in the file names under results/), name in the paper, stem of the ONNX
graph (and of its cost file results/cost/<stem>.json), and the engine latency in ms (median of the per-round medians of
three timing rounds, jetson/time_engine.py and jetson/summarize_rounds.py).

The WeeNet graphs are those written by weenet/export_onnx.py (onnx/weenet_do.onnx, onnx/abl_<model>.onnx in the README).
Latencies: the twelve models at 224 px are those of the paper's Table 4 (results/latency/models/ holds the rounds of ten
of them); YOLO11n-cls at 160 px is the unrounded median of the rounds of Table 7
(results/latency/capacity_engines/yolo160_plain.r*.json); the four ablation engines are those of Table 12, timed
together with random weights.
"""

T_MS = 1000.0 / 30.0          # frame period of a 30 fps stream (ms)

# key, paper name, ONNX stem, engine latency (ms)
TWELVE = [
    ("wee", "WeeNet", "weenet_do", 3.15),
    ("yolo224", "YOLO11n-cls", "yolo11n_cls_r224", 4.18),
    ("mnv4", "MobileNetV4-Conv-S", "mobilenetv4_conv_small", 6.19),
    ("shufflenet", "ShuffleNetV2-1.0x", "shufflenet_v2_x1_0", 6.69),
    ("repvgg", "RepVGG-A0", "repvgg_a0", 8.70),
    ("hardnet", "HarDNet-39DS", "hardnet39ds", 11.29),
    ("resnet18", "ResNet-18", "resnet18", 11.73),
    ("fastvit", "FastViT-T8", "fastvit_t8", 19.56),
    ("repvit", "RepViT-M0.9", "repvit_m0_9", 20.26),
    ("effformer", "EfficientFormerV2-S0", "efficientformerv2_s0", 20.32),
    ("effnetv2", "EfficientNetV2-B0", "tf_efficientnetv2_b0_s", 20.59),
    ("mobilevit", "MobileViTv2-1.0", "mobilevitv2_100", 32.16),
]
YOLO160 = ("yolo160", "YOLO11n-cls 160 px", "yolo11n_cls_r160", 3.41553)       # 3.42 ms in Table 7
PLUGIN = ("wee_u8", "WeeNet + plugin", None, 2.65)          # FusedHaarStemU8 engine (8-bit input)
ABLATION = [
    ("abl_full", "WeeNet (full)", "abl_weenet_z_ds4", 3.19),
    ("abl_cstem", "Conv stem", "abl_weenet_z_ds4_cstem", 3.23),
    ("abl_g1", "3x3 downsampling", "abl_weenet_haar_g1", 2.74),
    ("abl_dw", "Depthwise convs", "abl_weenet_z_ds4_dw", 2.93),
]
ALL = TWELVE + [YOLO160, PLUGIN] + ABLATION
NAME = {k: n for k, n, _, _ in ALL}
STEM = {k: s for k, _, s, _ in ALL}
LATENCY_MS = {k: ms for k, _, _, ms in ALL}
