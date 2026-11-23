"""WeeNet and WeeNet-DO.

WeeNet: a fixed Haar-scattering stem (224x224x3 -> 21 x 56 x 56, then BN and a learned 1x1 conv to 32 channels), four
stages of dense ResNet basic blocks with depths (1, 2, 2, 1) and widths (32, 64, 128, 256), and a GAP + linear head.
The stride-2 block of stages 1-3 downsamples with a 4x4 stride-2 conv and a 2x2 stride-2 shortcut ("ds4").

WeeNet-DO: the same network with the nine 3x3 body convolutions over-parameterized by DO-Conv (Cao et al., IEEE TIP
2022) during fine-tuning. `fold_do` turns a trained WeeNet-DO into a plain WeeNet with identical outputs, so the
deployed engine is WeeNet's.

`export_graph` replaces the Haar stem by its exact form as a 4x4 stride-4 conv (3 -> 96), a ReLU and a 1x1 conv
(96 -> 21), using |y| = ReLU(y) + ReLU(-y), and merges that 1x1 conv with the BN and the learned 1x1 conv that follow it
into one 1x1 conv (96 -> 32) and its ReLU. This is the "stock" TensorRT graph used for every WeeNet latency number.

Module construction order and random-number use match the code that produced the reported checkpoints, so a model built
here after `torch.manual_seed(s)` has the same initial weights, and state_dict keys are unchanged.
"""
from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F

WIDTHS = (32, 64, 128, 256)
DEPTHS = (1, 2, 2, 1)
HAAR_FEATURES = 21


class HaarScat(nn.Module):
    """Order-1 Haar scattering over 4x4 cells. Per color: the cell mean, the modulus of the 3 Haar details at the 4-pixel
    scale, and the cell average of the modulus of the 3 details at the 2-pixel scale -> 3 * 7 = 21 channels at 1/4
    resolution. No parameters."""

    @staticmethod
    def _haar(x):
        b, c, h, w = x.shape
        u = F.pixel_unshuffle(x, 2).view(b, c, 4, h // 2, w // 2)
        p00, p01, p10, p11 = u[:, :, 0], u[:, :, 1], u[:, :, 2], u[:, :, 3]
        return ((p00 + p01 + p10 + p11) * 0.25, (p00 - p01 + p10 - p11) * 0.25,
                (p00 + p01 - p10 - p11) * 0.25, (p00 - p01 - p10 + p11) * 0.25)

    def forward(self, x):
        a1, h1, v1, d1 = self._haar(x)
        s1 = F.avg_pool2d(torch.cat([h1.abs(), v1.abs(), d1.abs()], 1), 2)
        a2, h2, v2, d2 = self._haar(a1)
        return torch.cat([a2, h2.abs(), v2.abs(), d2.abs(), s1], 1)


class ConvBNAct(nn.Module):
    """Dense conv + BN + ReLU."""

    def __init__(self, cin, cout, k=3, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Block(nn.Module):
    """ResNet basic block: two 3x3 conv + BN + ReLU, residual add, ReLU."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = ConvBNAct(cin, cout, 3, stride)
        self.c2 = ConvBNAct(cout, cout, 3, 1)
        self.proj = None
        if stride != 1 or cin != cout:
            self.proj = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x if self.proj is None else self.proj(x)
        return self.act(self.c2(self.c1(x)) + r)


class DWSep(nn.Module):
    """Depthwise 3x3 + pointwise 1x1, each with BN + ReLU (ablation arm "depthwise")."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False)
        self.bn1 = nn.BatchNorm2d(cin)
        self.pw = nn.Conv2d(cin, cout, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(cout)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.act(self.bn1(self.dw(x)))
        return self.act(self.bn2(self.pw(x)))


class DWBlock(nn.Module):
    """The Block slot built from two depthwise-separable convs; only the convolution type differs from Block."""

    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.c1 = DWSep(cin, cout, stride)
        self.c2 = DWSep(cout, cout, 1)
        self.proj = None
        if stride != 1 or cin != cout:
            self.proj = nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        r = x if self.proj is None else self.proj(x)
        return self.act(self.c2(self.c1(x)) + r)


class WeeNet(nn.Module):
    """Haar stem -> 4 stages of Blocks (the first block of stages 1-3 has stride 2) -> GAP -> linear.
    stem="conv" (ablation): two 3x3 stride-2 conv + BN + ReLU (3 -> 16 -> 32) instead of the Haar stem;
    conv="dw" (ablation): depthwise-separable blocks instead of dense ones."""

    def __init__(self, num_classes=6, widths=WIDTHS, depths=DEPTHS, stem="haar", conv="dense"):
        super().__init__()
        w = list(widths)
        if stem == "haar":
            self.stem = nn.Sequential(HaarScat(), nn.BatchNorm2d(HAAR_FEATURES), ConvBNAct(HAAR_FEATURES, w[0], 1))
        else:
            self.stem = nn.Sequential(ConvBNAct(3, w[0] // 2, 3, 2), ConvBNAct(w[0] // 2, w[0], 3, 2))
        Blk = DWBlock if conv == "dw" else Block
        blocks, cin = [], w[0]
        for i, (cout, d) in enumerate(zip(w, depths)):
            for j in range(d):
                blocks.append(Blk(cin, cout, 2 if (j == 0 and i > 0) else 1))
                cin = cout
        self.body = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(cin, num_classes)

    def forward(self, x):
        return self.head(self.pool(self.body(self.stem(x))).flatten(1))


def _even_downsampling(model: WeeNet):
    """ds4: every stride-2 block gets a 4x4 stride-2 pad-1 conv instead of 3x3, and a 2x2 stride-2 shortcut instead of
    1x1. Built under a forked RNG so the rest of the network keeps the initialization it had."""
    with torch.random.fork_rng(devices=[]):
        for blk in model.body:
            if isinstance(blk, Block) and blk.c1.conv.stride[0] == 2:
                cin, cout = blk.c1.conv.in_channels, blk.c1.conv.out_channels
                blk.c1.conv = nn.Conv2d(cin, cout, 4, 2, 1, bias=False)
                blk.proj[0] = nn.Conv2d(cin, cout, 2, 2, 0, bias=False)
            elif isinstance(blk, DWBlock) and blk.c1.dw.stride[0] == 2:
                cin, cout = blk.c1.dw.in_channels, blk.proj[0].out_channels
                blk.c1.dw = nn.Conv2d(cin, cin, 4, 2, 1, groups=cin, bias=False)
                blk.proj[0] = nn.Conv2d(cin, cout, 2, 2, 0, bias=False)
    return model


class DOConv3x3(nn.Module):
    """DO-Conv in kernel-composition form. `weight` [Cout, Cin, 3, 3] is the plain kernel; `d0` [Cin, 9, 9] = D - I is
    zero at init, so the layer starts as the plain conv. Folded kernel: W'[o, i, k] = sum_m W[o, i, m] D[i, m, k]."""

    def __init__(self, conv: nn.Conv2d):
        super().__init__()
        if conv.kernel_size != (3, 3) or conv.groups != 1 or conv.bias is not None:
            raise ValueError("DOConv3x3 wraps a dense bias-free 3x3 conv")
        self.stride, self.padding = conv.stride, conv.padding
        self.weight = nn.Parameter(conv.weight.detach().clone())
        self.d0 = nn.Parameter(torch.zeros(conv.in_channels, 9, 9))
        self.register_buffer("eye", torch.eye(9), persistent=False)

    def folded(self):
        w = self.weight.flatten(2)                                   # [o, i, m]
        d = self.d0 + self.eye                                       # [i, m, k]
        return torch.einsum("oim,imk->oik", w, d).reshape_as(self.weight)

    def forward(self, x):
        return F.conv2d(x, self.folded(), None, self.stride, self.padding)


MODELS = ("weenet_z_ds4", "weenet_z_ds4_do", "weenet_z_ds4_cstem", "weenet_haar_g1", "weenet_z_ds4_dw")


def build(name="weenet_z_ds4", num_classes=6):
    """weenet_z_ds4 = WeeNet; weenet_z_ds4_do = WeeNet-DO (the names used in the result files). Ablation arms, one change
    each: weenet_z_ds4_cstem = conv stem; weenet_haar_g1 = 3x3 stride-2 downsampling with a 1x1 shortcut;
    weenet_z_ds4_dw = depthwise-separable blocks."""
    if name not in MODELS:
        raise KeyError("unknown model %s (expected one of %s)" % (name, MODELS))
    if name == "weenet_haar_g1":
        return WeeNet(num_classes)
    if name == "weenet_z_ds4_cstem":
        return _even_downsampling(WeeNet(num_classes, stem="conv"))
    if name == "weenet_z_ds4_dw":
        return _even_downsampling(WeeNet(num_classes, conv="dw"))
    m = _even_downsampling(WeeNet(num_classes))
    if name == "weenet_z_ds4_do":
        n = 0
        for blk in m.body:
            for cba in (blk.c1, blk.c2):
                if cba.conv.kernel_size == (3, 3):
                    cba.conv = DOConv3x3(cba.conv)
                    n += 1
        if n != 9:                     # 12 body convs minus the three 4x4 stride-2 convs
            raise RuntimeError("weenet_z_ds4_do: wrapped %d convs, expected 9" % n)
    return m


@torch.no_grad()
def fold_do(model: WeeNet) -> WeeNet:
    """WeeNet-DO -> plain WeeNet with the folded kernels (same function, same engine graph). Returned in eval mode."""
    plain = build("weenet_z_ds4", model.head.out_features)
    sd = {k: v for k, v in model.state_dict().items() if not k.endswith(".d0")}
    for i, blk in enumerate(model.body):
        for c in ("c1", "c2"):
            conv = getattr(blk, c).conv
            if isinstance(conv, DOConv3x3):
                sd["body.%d.%s.conv.weight" % (i, c)] = conv.folded().detach().clone()
    missing, unexpected = plain.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError("fold_do: missing %s unexpected %s" % (missing, unexpected))
    return plain.eval()


def _haar_linear(x):
    """The 48 pre-modulus values per 4x4 cell: mean | H4 V4 D4 | h, v, d of the 2x2 blocks TL, TR, BL, BR."""
    a1, h1, v1, d1 = HaarScat._haar(x)
    a2, h2, v2, d2 = HaarScat._haar(a1)
    blocks = [F.pixel_unshuffle(t, 2) for t in (h1, v1, d1)]
    return torch.cat([a2, h2, v2, d2] + blocks, 1)


@torch.no_grad()
def haar_maps():
    """HaarScat(x) = M @ ReLU(conv(x, [W1; -W1], kernel 4, stride 4)) exactly. W1: (48, 3, 4, 4), M: (21, 96)."""
    eye = torch.eye(48).view(48, 3, 4, 4)
    W1 = _haar_linear(eye)[:, :, 0, 0].T.reshape(48, 3, 4, 4)
    M = torch.zeros(21, 96)
    for c in range(3):
        M[c, c], M[c, 48 + c] = 1.0, -1.0                    # mean: y = ReLU(y) - ReLU(-y)
        for k in range(3):                                   # |H4|, |V4|, |D4|
            j = 3 + 3 * k + c
            M[j, j], M[j, 48 + j] = 1.0, 1.0
        for k in range(3):                                   # mean over the 4 blocks of |h|, |v|, |d| (2-pixel scale)
            for blk in range(4):
                j = 12 + 12 * k + 4 * c + blk
                M[12 + 3 * k + c, j] += 0.25
                M[12 + 3 * k + c, 48 + j] += 0.25
    return W1, M


class HaarAsConvs(nn.Module):
    """The Haar stem as conv 4x4/4 (3 -> 96) + ReLU + conv 1x1 (96 -> 21); equal to HaarScat up to rounding."""

    def __init__(self):
        super().__init__()
        W1, M = haar_maps()
        self.c1 = nn.Conv2d(3, 96, 4, 4, bias=False)
        self.m = nn.Conv2d(96, 21, 1, bias=False)
        with torch.no_grad():
            self.c1.weight.copy_(torch.cat([W1, -W1]))
            self.m.weight.copy_(M[:, :, None, None])

    def forward(self, x):
        return self.m(F.relu(self.c1(x)))


def export_graph(model: WeeNet) -> WeeNet:
    """Copy of a plain WeeNet whose stem is Sequential(conv 4x4/4 (3 -> 96), ReLU, conv 1x1 (96 -> C), ReLU): the 1x1
    conv of HaarAsConvs, the BN after the Haar features and the stem's learned 1x1 conv with its BN are all linear and
    merge into the one 1x1 conv. The whole net is Conv / Relu / Add / GAP / Gemm."""
    m = copy.deepcopy(model).eval()
    if isinstance(m.stem[0], ConvBNAct):        # ablation arm with a conv stem: already Conv / Relu only
        return m
    if not isinstance(m.stem[0], HaarScat):
        raise ValueError("export_graph expects a plain WeeNet (fold WeeNet-DO first)")
    haar, bn1, cba = HaarAsConvs(), m.stem[1], m.stem[2]
    with torch.no_grad():
        s1 = bn1.weight / torch.sqrt(bn1.running_var + bn1.eps)
        t1 = bn1.bias - bn1.running_mean * s1
        s2 = cba.bn.weight / torch.sqrt(cba.bn.running_var + cba.bn.eps)
        t2 = cba.bn.bias - cba.bn.running_mean * s2
        w = cba.conv.weight[:, :, 0, 0]                                    # (C, 21)
        c2 = nn.Conv2d(haar.m.in_channels, w.shape[0], 1)
        c2.weight.copy_(((s2[:, None] * w * s1[None, :]) @ haar.m.weight[:, :, 0, 0])[:, :, None, None])
        c2.bias.copy_(s2 * (w @ t1) + t2)
    m.stem = nn.Sequential(haar.c1, nn.ReLU(), c2, nn.ReLU()).eval()
    return m


def count_macs(model, res=224):
    """Conv2d + Linear multiply-accumulates by forward hooks (the convention of the reported GMAC numbers)."""
    total = [0]

    def hook(mod, inp, out):
        if isinstance(mod, nn.Conv2d):
            total[0] += out.numel() // out.shape[0] * mod.in_channels // mod.groups * mod.kernel_size[0] * mod.kernel_size[1]
        else:
            total[0] += mod.in_features * mod.out_features
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
    with torch.no_grad():
        model.eval()(torch.zeros(1, 3, res, res))
    for h in hs:
        h.remove()
    return total[0]
