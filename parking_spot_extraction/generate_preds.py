from mmdet.apis import init_detector
import os
import argparse
from spot import save_predictions

from os.path import join


def main(args):
    prediction_dir = join(args.output_dir, "predictions", args.model, args.exp)
    model_dir = join(args.model_dir, args.exp)
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    for subset in args.subsets:
        print(f"{30 * '*'} Processing subset: {subset} {30 * '*'}")
        coco_input = join(args.annotations_dir, f"{subset}_cars.json")
        output_json = join(prediction_dir, f"{subset}_cars.json")

        model = init_detector(
            args.config,
            join(model_dir, args.model_weight + ".pth"),
            device="cuda",
        )
        save_predictions(args.root, coco_input, output_json, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate predictions for Metapklot dataset"
    )
    parser.add_argument(
        "--config",
        help="Path to the model config file",
        required=False,
        default="configs/custom_ROI.py",
    )
    parser.add_argument("--model", default="mask")
    parser.add_argument("--exp", help="Experiment name", required=True)
    parser.add_argument(
        "--output_dir", help="Path to the output directory", required=True
    )
    parser.add_argument(
        "--annotations_dir", help="Path to the annotations directory", required=True
    )
    parser.add_argument(
        "--root", help="Path to the dataset root directory", required=True
    )
    parser.add_argument(
        "--model_dir", help="Path to the model directory", required=True
    )
    parser.add_argument(
        "--model_weight",
        help="Weight filename without extension",
        default="iter_100000",
    )

    parser.add_argument(
        "--subsets",
        nargs="+",
        default=[
            "cnr-camera-1",
            "cnr-camera-2",
            "cnr-camera-3",
            "cnr-camera-4",
            "cnr-camera-5",
            "cnr-camera-6",
            "cnr-camera-7",
            "cnr-camera-8",
            "cnr-camera-9",
            "pucpr",
            "ufpr04",
            "ufpr05",
            "isshk",
            "qridr",
            "vmlix-vxusd",
        ],
    )

    args = parser.parse_args()
    main(args)


"""
CUDA_VISIBLE_DEVICES=0 nohup python generate_preds.py --output_dir ~/experiments/metapklot/instance_seg/ --annotations_dir ~/db/metapklot/cjsons/ --root ~/db/metapklot/ --model_dir ~/workspace/python/vision/metapklot/work_dirs/ --exp wo_plds_002 > preds_log_wo_plds_002.out &
"""
