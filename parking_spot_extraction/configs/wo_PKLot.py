from os.path import join

classes = ("car",)
ANN_DIR = join("/disk1/lmk22/parking_spot_recognition/metapklot/cjsons/")
IMAGE_DIR = join("/disk1/lmk22/parking_spot_recognition/metapklot/")
DATASET_TYPE = "CocoDataset"
_base_ = "../models/convnextv2_cars.py"

train = join(ANN_DIR, "merged_wo_PKLot_cars.json")
val = join(ANN_DIR, "ufpr04_cars.json")
test = val

backend_args = None

train_dataloader = dict(
    dataset=dict(
        type=DATASET_TYPE,
        data_root=IMAGE_DIR,
        ann_file=train,
        data_prefix=dict(img="."),
    )
)


val_dataloader = dict(
    dataset=dict(
        type=DATASET_TYPE,
        data_root=IMAGE_DIR,
        ann_file=val,
        data_prefix=dict(img="."),
        test_mode=True,
    )
)

val_evaluator = dict(
    type="CocoMetric",
    ann_file=val,
    metric=["bbox", "segm"],
    format_only=False,
    backend_args=backend_args,
)

test_evaluator = val_evaluator
