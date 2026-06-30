import os
from os.path import join

ROOT = os.path.expanduser("~")

_base_ = [
    "mmdet::_base_/models/mask-rcnn_r50_fpn.py",
    "mmdet::_base_/datasets/coco_instance.py",
    "mmdet::_base_/schedules/schedule_1x.py",
    "mmdet::_base_/default_runtime.py",
]

classes = ("car",)
data_root = join(ROOT, "db/metapklot")
IMAGE_DIR = join(ROOT, "workspace/data/PKLot2.0/PKLot")
DATASET_TYPE = "CocoDataset"

# please install the mmpretrain
# import mmpretrain.models to trigger register_module in mmpretrain
custom_imports = dict(
    imports=["mmpretrain.models", "models.CustomRoI"], allow_failed_imports=False
)

checkpoint_file = "https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-base_3rdparty-fcmae_in1k_20230104-8a798eaf.pth"  # noqa
image_size = (1024, 1024)

load_from = "https://download.openmmlab.com/mmdetection/v3.0/convnextv2/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco/mask-rcnn_convnext-v2-b_fpn_lsj-3x-fcmae_coco_20230113_110947-757ee2dd.pth"  # noqa

model = dict(
    type="MaskRCNN",
    roi_head=dict(
        # type="CustomRoI",
        bbox_head=dict(num_classes=len(classes)),
        mask_head=dict(num_classes=len(classes)),
    ),
    backbone=dict(
        _delete_=True,
        type="mmpretrain.ConvNeXt",
        arch="base",
        out_indices=[0, 1, 2, 3],
        # TODO: verify stochastic depth rate {0.1, 0.2, 0.3, 0.4}
        drop_path_rate=0.4,
        layer_scale_init_value=0.0,  # disable layer scale when using GRN
        gap_before_final_norm=False,
        use_grn=True,  # V2 uses GRN
        init_cfg=dict(
            type="Pretrained", checkpoint=checkpoint_file, prefix="backbone."
        ),
    ),
    neck=dict(in_channels=[128, 256, 512, 1024]),
    test_cfg=dict(
        rpn=dict(nms=dict(type="nms")),  # TODO: does RPN use soft_nms?
        rcnn=dict(nms=dict(type="soft_nms")),
    ),
)

cutout_sizes = [32, 40, 48, 56, 64, 96, 128, 192, 256]
cutout_shapes = [(n, m) for n in cutout_sizes for m in cutout_sizes]

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=_base_.backend_args),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(
        type="CutOut",
        n_holes=16,
        cutout_shape=cutout_shapes,
    ),
    dict(
        type="RandomResize", scale=image_size, ratio_range=(0.1, 2.0), keep_ratio=False
    ),
    dict(
        type="AutoAugment",
    ),
    dict(
        type="RandomCrop",
        crop_size=image_size,
        recompute_bbox=True,
        allow_negative_crop=True,
    ),
    dict(type="FilterAnnotations", min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    sampler=dict(type="InfiniteSampler", shuffle=True),
    batch_size=1,
    num_workers=16,
    dataset=dict(pipeline=train_pipeline),
)

val_pipeline = [
    dict(backend_args=None, type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
    dict(type="Resize", scale_factor=1.0, keep_ratio=True),
    dict(
        meta_keys=(
            "img_id",
            "img_path",
            "ori_shape",
            "img_shape",
            "scale_factor",
        ),
        type="PackDetInputs",
    ),
]

test_dataloader = dict(
    dataset=dict(pipeline=val_pipeline), batch_size=1, num_workers=16
)
val_dataloader = dict(dataset=dict(pipeline=val_pipeline), batch_size=1, num_workers=16)

max_iter = 100000
val_interval = 5000

train_cfg = dict(
    _delete_=True,
    type="IterBasedTrainLoop",
    max_iters=max_iter,
    val_interval=val_interval,
)

# learning rate
param_scheduler = [
    dict(
        type="LinearLR",
        start_factor=0.00001,
        by_epoch=False,
        begin=0,
        end=int(max_iter * 0.3),
    ),
    dict(
        type="CosineAnnealingLR",
        begin=int(max_iter * 0.3),
        by_epoch=False,
        T_max=max_iter,
        eta_min=1e-9,  # controls final LR
    ),
]

# Enable automatic-mixed-precision training with AmpOptimWrapper.
optim_wrapper = dict(
    type="AmpOptimWrapper",
    constructor="LearningRateDecayOptimizerConstructor",
    paramwise_cfg={
        "decay_rate": 0.95,
        "decay_type": "layer_wise",  # TODO: sweep layer-wise lr decay?
        "num_layers": 12,
    },
    optimizer=dict(
        _delete_=True,
        type="AdamW",
        lr=5e-6,
        betas=(0.9, 0.999),
        weight_decay=0.1,
    ),
    clip_grad=dict(max_norm=1.0, norm_type=2),
)

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    checkpoint=dict(type="CheckpointHook", by_epoch=False, interval=val_interval),
    logger=dict(type="LoggerHook", interval=50, log_metric_by_epoch=False),
)

log_level = "INFO"
log_processor = dict(by_epoch=False)
