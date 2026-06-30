import os
from os.path import join

ROOT = os.path.expanduser("~")

_base_ = "../models/convnextv2_cars.py"

# please install the mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
custom_imports = dict(imports=["models.CustomRoI"], allow_failed_imports=False)


model = dict(
    type="MaskRCNN",
    roi_head=dict(
        type="CustomRoI",
    ),
)
