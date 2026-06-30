import argparse
import faster_coco_eval

# Replace pycocotools with faster_coco_eval
faster_coco_eval.init_as_pycocotools()
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json
import os
import torch
from torch_geometric.loader import NeighborLoader
from os.path import join
from train_graphspot import HeteroGNN, load_spot_dataset, evaluate
from spot import filter_spot_gt
from copy import deepcopy

from non_representative_days import get_non_representative_days


def main(args):
    graph_dir = join(args.output_dir, "graphs", args.model, args.exp)
    weight_root = join(args.output_dir, "gnn_weights", args.model, args.exp)
    labelled_spots_dir = join(args.output_dir, "labelled_spots", args.model, args.exp)

    if "wo_cnr" in labelled_spots_dir:
        val_subsets = [
            "cnr-camera-1",
            "cnr-camera-2",
            "cnr-camera-3",
            "cnr-camera-4",
            "cnr-camera-5",
            "cnr-camera-6",
            "cnr-camera-7",
            "cnr-camera-8",
            "cnr-camera-9",
        ]
    elif "wo_pklot" in labelled_spots_dir:
        val_subsets = ["pucpr", "ufpr04", "ufpr05"]
    elif "wo_plds" in labelled_spots_dir:
        val_subsets = ["isshk", "qridr", "vmlix-vxusd"]
    else:
        raise ValueError("Please define validation subsets")

    # Load graph structures
    val_data = None

    for subset in val_subsets:
        graph_file = join(graph_dir, f"{subset}_graph.json")

        with open(graph_file, "r") as f:
            graph_structure = json.load(f)

        if val_data is None:
            val_data = graph_structure
        else:
            for key in val_data.keys():
                val_data[key].extend(graph_structure[key])

    val_subset = load_spot_dataset(val_data)

    num_neighbors = {
        ("car", "inside", "spot"): [10, 10, 10],
        ("spot", "rev_inside", "car"): [10, 10, 10],
    }

    test_loader = NeighborLoader(
        val_subset,
        input_nodes=("spot", torch.arange(len(val_subset["spot"].y))),
        num_neighbors={key: (-1, -1, -1) for key in num_neighbors},  # full neighbors
        batch_size=256,
        shuffle=False,
    )
    weight_file = os.path.join(weight_root, "final.pt")

    metadata = (
        ["car", "spot"],
        [("car", "inside", "spot"), ("spot", "rev_inside", "car")],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeteroGNN(metadata, hidden_channels=512, out_channels=1).to(device)
    model.load_state_dict(torch.load(weight_file, map_location="cpu"))
    model.eval()

    pos_spots = evaluate(model, args.annotations_dir, test_loader, device, labelled_spots_dir, val_subsets)

    for subset in val_subsets:
        print(f"{30 * '*'} Processing subset: {subset} {30 * '*'}")
        spots_input = join(args.annotations_dir, f"{subset}_spots.json")

        with open(spots_input, "r") as f:
            spots_data = json.load(f)

        #non_representative_days = get_non_representative_days(subset)
        non_representative_days = get_non_representative_days(subset)

        spots_data = filter_spot_gt(spots_data, non_representative_days)
        
        if "info" not in spots_data:
            spots_data["info"] = {}

        output_labelled_spots_file = (
            f"{labelled_spots_dir}/{subset}_labelled_spots.json"
        )

        # Load ground truth COCO object
        coco_gt = COCO()
        coco_gt.dataset = spots_data
        coco_gt.createIndex()

        # Filter predicitons based on pos_spots
        gnn_spots_predictions = []

        with open(output_labelled_spots_file, "r") as f:
            labelled_spots_predictions = json.load(f)

        for pred in labelled_spots_predictions:
            pred_copy = deepcopy(pred)
            if pred["spot_pred_id"] in pos_spots:
                pred_copy["score"] = pos_spots[pred["spot_pred_id"]]
            gnn_spots_predictions.append(pred_copy)

        for pred in labelled_spots_predictions:
            pred["category_id"] = 1  # Set category_id to 1 for GNN predictions

        coco_dt = coco_gt.loadRes(labelled_spots_predictions)

        coco_eval = COCOeval(
            coco_gt, coco_dt, iouType="segm"
        )  # or 'bbox' if bounding boxes only)
        coco_eval.params.maxDets = [10, 100, 1000]  # instead of default [1, 10, 100]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        print(len(labelled_spots_predictions), "total labelled spots predictions")
        print(len(gnn_spots_predictions), "GNN spots predictions")

        for pred in gnn_spots_predictions:
            pred["category_id"] = 1  # Set category_id to 1 for GNN predictions

        coco_dt = coco_gt.loadRes(gnn_spots_predictions)

        coco_eval = COCOeval(
            coco_gt, coco_dt, iouType="segm"
        )  # or 'bbox' if bounding boxes only
        coco_eval.params.maxDets = [10, 100, 1000]  # instead of default [1, 10, 100]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test GraphSpot GNN on MetaPKLot dataset"
    )

    parser.add_argument("--model", default="mask")
    parser.add_argument("--exp", help="Experiment name", required=True)
    parser.add_argument(
        "--annotations_dir", help="Path to the annotations directory", required=True
    )

    parser.add_argument(
        "--output_dir", help="Path to the output directory", required=True
    )

    args = parser.parse_args()
    main(args)

"""
CUDA_VISIBLE_DEVICES=0 nohup python test_graphspot.py --output_dir ~/experiments/metapklot/instance_seg/ --annotations_dir ~/db/metapklot/cjsons/ --exp wo_plds_002 > test_wo_plds_002.out &
"""
