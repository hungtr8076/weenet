"""Model factory: WeeNet and WeeNet-DO.

  weenet_z_ds4, weenet_z_ds4_do   WeeNet and WeeNet-DO (weenet/model.py); `backbone` loads WeeNet's own ImageNet weights
                                  (train/pretrain_imagenet.py)
  weenet_z_ds4_cstem, weenet_haar_g1, weenet_z_ds4_dw   ablation arms (one change each), trained from scratch
"""
import torch

from weenet import model as wm


def load_backbone(model, path):
    """Load every tensor of `path` whose name and shape match (the 1000-class head does not). DO-Conv's d0 tensors are
    in no backbone and stay at zero, so WeeNet-DO starts exactly at the pretrained WeeNet."""
    sd = torch.load(path, map_location="cpu")
    msd = model.state_dict()
    keep = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
    need = [k for k in msd if not k.endswith(".d0")]
    skipped = [k for k in msd if k not in keep]
    missing = sorted(k for k in need if k not in keep)
    if missing != ["head.bias", "head.weight"]:
        raise SystemExit("backbone %s: tensors other than the 6-class head do not match: %s" % (path, missing))
    model.load_state_dict(keep, strict=False)
    print("   loaded %d backbone tensors, skipped %s" % (len(keep), skipped), flush=True)


def make_model(name, num_classes, backbone=""):
    if name not in wm.MODELS:
        raise SystemExit("unknown model %s (this code base builds %s only)" % (name, ", ".join(wm.MODELS)))
    m = wm.build(name, num_classes)
    if backbone:
        load_backbone(m, backbone)
    return m
