import json
import os
from spot import save_prediction_to_graph_data
import argparse
from os.path import join
from non_representative_days import get_non_representative_days


def main(args):
    prediction_dir = join(args.output_dir, "predictions", args.model, args.exp)
    graph_dir = join(args.output_dir, "graphs", args.model, args.exp)
    labelled_spots_dir = join(args.output_dir, "labelled_spots", args.model, args.exp)
    model_dir = join(args.model_dir, args.exp)
    os.makedirs(prediction_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    for subset in args.subsets:
        print(f"{30 * '*'} Processing subset: {subset} {30 * '*'}")
        coco_input = join(args.annotations_dir, f"{subset}_cars.json")
        preds_json = join(prediction_dir, f"{subset}_cars.json")
        spots_input = join(args.annotations_dir, f"{subset}_spots.json")

        non_representative_days = get_non_representative_days(subset)

        graph_structure, labelled_spots_predictions = save_prediction_to_graph_data(
            coco_input, spots_input, preds_json, non_representative_days
        )

        output_graph_file = join(graph_dir, f"{subset}_graph.json")

        os.makedirs(os.path.dirname(output_graph_file), exist_ok=True)

        with open(output_graph_file, "w") as f:
            json.dump(graph_structure, f, indent=2)

        output_labelled_spots_file = join(
            labelled_spots_dir, f"{subset}_labelled_spots.json"
        )
        os.makedirs(os.path.dirname(output_labelled_spots_file), exist_ok=True)

        for pred in labelled_spots_predictions:
            pred.pop("rle", None)  # remove 'rle' key if it exists

        with open(output_labelled_spots_file, "w") as f:
            json.dump(labelled_spots_predictions, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate graph data for GraphSpot prediction"
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
        "--subsets",
        nargs="+",
        default=[
            "qridr",
            "isshk",
            "vmlix-vxusd",
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
        ],
    )

    args = parser.parse_args()
    main(args)

"""
CUDA_VISIBLE_DEVICES=0 nohup python generate_graph.py --output_dir ~/experiments/metapklot/instance_seg/ --annotations_dir ~/db/metapklot/cjsons/ --root ~/db/metapklot/ --model_dir ~/workspace/python/vision/metapklot/work_dirs/ --exp wo_plds_001 > graph_log_wo_plds_001.out &
"""
