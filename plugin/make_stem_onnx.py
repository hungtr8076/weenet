"""Replace the leading 1x1 Conv (+ Relu) of a WeeNet "tail" ONNX graph (input = the 21 Haar features at 56x56) by one
plugin node that takes the image: the Haar-scattering features, the 1x1 conv and the ReLU run in one kernel
(fused_haar_stem.cu).
  FusedHaarStem     the image as FP32 1 x 3 x 224 x 224, already normalized
  FusedHaarStemU8   the image as raw BGR bytes (224 x 224 x 3, as OpenCV decodes it) in a binding declared
                    FP32 1 x 3 x 112 x 112 (the same byte count); the ImageNet normalization moves into the kernel
Called by weenet/export_onnx.py --graph plugin | plugin-u8.
"""
import numpy as np
import onnx
from onnx import helper, numpy_helper

MEAN, STD = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])     # RGB, as in training


def rewrite(src, dst, u8):
    m = onnx.load(src)
    g = m.graph
    init = {i.name: i for i in g.initializer}
    conv = g.node[0]
    if conv.op_type != "Conv" or conv.input[0] != "image":
        raise SystemExit("first node is %s(%s), expected Conv(image)" % (conv.op_type, conv.input[0]))
    relu = [n for n in g.node if n.op_type == "Relu" and n.input[0] == conv.output[0]]
    if len(relu) != 1:
        raise SystemExit("expected exactly one Relu after the first Conv")
    W = numpy_helper.to_array(init[conv.input[1]]).astype(np.float32)
    B = numpy_helper.to_array(init[conv.input[2]]).astype(np.float32)
    if W.shape[1:] != (21, 1, 1) or B.shape[0] != W.shape[0]:
        raise SystemExit("unexpected first conv weights %s / bias %s" % (W.shape, B.shape))
    attrs = dict(cout=int(W.shape[0]), weight=W.reshape(-1).tolist(), bias=B.reshape(-1).tolist())
    if u8:
        attrs["norm"] = np.concatenate([1.0 / (255.0 * STD), -MEAN / STD]).astype(np.float32).tolist()
    stem = helper.make_node("FusedHaarStemU8" if u8 else "FusedHaarStem", ["image"], [relu[0].output[0]],
                            name="stem_fused_haar_stem", **attrs)
    rest = [n for n in g.node if n is not conv and n is not relu[0]]
    del g.node[:]
    g.node.extend([stem] + rest)
    used = {i for n in g.node for i in n.input}
    keep = [i for i in g.initializer if i.name in used]
    del g.initializer[:]
    g.initializer.extend(keep)
    dims = g.input[0].type.tensor_type.shape.dim
    dims[1].dim_value, dims[2].dim_value, dims[3].dim_value = (3, 112, 112) if u8 else (3, 224, 224)
    onnx.save(m, dst)
    return len(g.node)

