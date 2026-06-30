import argparse
import json
import os
from copy import deepcopy
from os.path import join
import sys

import cv2
import numpy as np
import torch
from torch_geometric.loader import NeighborLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_graphspot import HeteroGNN, load_spot_dataset, evaluate
from spot import filter_spot_gt
from non_representative_days import get_non_representative_days


def parse_day(day_str: str):
    parts = day_str.strip().split("-")
    if len(parts) != 3:
        raise ValueError("Use day in format YYYY-MM-DD")
    return [int(parts[0]), int(parts[1]), int(parts[2])]


def normalize_subset_name(subset: str) -> str:
    subset = subset.strip()
    if subset.startswith("camera"):
        camera_id = subset.replace("camera", "").strip()
        return f"cnr-camera-{camera_id}"
    return subset


def segmentation_to_rotated_box(segmentation):
    if segmentation is None:
        return None, None

    try:
        from pycocotools import mask as mask_utils

        mask = mask_utils.decode(segmentation)

        if mask.ndim == 3:
            mask = np.max(mask, axis=2)

        mask = mask.astype(np.uint8)

        if mask.sum() == 0:
            return None, None

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )

        if not contours:
            return None, None

        contour = max(contours, key=cv2.contourArea)

        if contour is None or len(contour) < 3:
            return None, None

        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)

        rotated_rect = [
            [float(rect[0][0]), float(rect[0][1])],
            [float(rect[1][0]), float(rect[1][1])],
            float(rect[2]),
        ]

        rotated_box = box.astype(float).tolist()

        return rotated_rect, rotated_box

    except Exception:
        return None, None


def load_pos_spots_for_subset(args, val_subsets, graph_file):
    graph_dir = join(args.output_dir, "graphs", args.model, args.exp)
    weight_root = join(args.output_dir, "gnn_weights", args.model, args.exp)
    labelled_spots_dir = join(args.output_dir, "labelled_spots", args.model, args.exp)

    with open(graph_file, "r", encoding="utf-8") as f:
        graph_structure = json.load(f)

    val_data = graph_structure
    val_subset = load_spot_dataset(val_data)

    test_loader = NeighborLoader(
        val_subset,
        input_nodes=("spot", torch.arange(len(val_subset["spot"].y))),
        num_neighbors={
            ("car", "inside", "spot"): (-1, -1, -1),
            ("spot", "rev_inside", "car"): (-1, -1, -1),
        },
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

    pos_spots = evaluate(
        model,
        args.annotations_dir,
        test_loader,
        device,
        labelled_spots_dir,
        val_subsets,
    )
    return pos_spots


def query_graphspot_values(args):
    day = parse_day(args.day)
    subset = normalize_subset_name(args.subset)

    graph_dir = join(args.output_dir, "graphs", args.model, args.exp)
    labelled_spots_dir = join(args.output_dir, "labelled_spots", args.model, args.exp)

    spots_input = join(args.annotations_dir, f"{subset}_spots.json")
    labelled_file = join(labelled_spots_dir, f"{subset}_labelled_spots.json")
    graph_file = join(graph_dir, f"{subset}_graph.json")

    with open(spots_input, "r", encoding="utf-8") as f:
        spots_data = json.load(f)

    non_representative_days = get_non_representative_days(subset)
    spots_data = filter_spot_gt(spots_data, non_representative_days)

    images_by_id = {img["id"]: img for img in spots_data["images"]}

    with open(labelled_file, "r", encoding="utf-8") as f:
        labelled_spots_predictions = json.load(f)

    pos_spots = load_pos_spots_for_subset(args, [subset], graph_file)

    gnn_spots_predictions = []
    for pred in labelled_spots_predictions:
        pred_copy = deepcopy(pred)
        if pred_copy["spot_pred_id"] in pos_spots:
            pred_copy["score"] = float(pos_spots[pred_copy["spot_pred_id"]])
        gnn_spots_predictions.append(pred_copy)

    matched_image_ids = set()
    for img in spots_data["images"]:
        if list(img.get("date", [])) == day:
            matched_image_ids.add(img["id"])

    result = []
    for pred in gnn_spots_predictions:
        if pred["image_id"] not in matched_image_ids:
            continue

        img = images_by_id.get(pred["image_id"], {})
        rotated_rect, rotated_box = segmentation_to_rotated_box(pred.get("segmentation"))

        result.append({
            "spot_pred_id": pred.get("spot_pred_id"),
            "image_id": pred.get("image_id"),
            "file_name": img.get("file_name"),
            "date": img.get("date"),
            "time": img.get("time"),
            "score": pred.get("score"),
            "label": pred.get("label"),
            "bbox": pred.get("bbox"),
            "rotated_rect": rotated_rect,
            "rotated_box": rotated_box,
            "area": pred.get("area"),
            "norm_length": pred.get("norm_length"),
            "scores": pred.get("scores"),
            "segmentation": pred.get("segmentation"),
        })

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Query GraphSpot output for a specific day"
    )
    parser.add_argument("--model", default="mask")
    parser.add_argument("--exp", required=True)
    parser.add_argument("--annotations_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--subset", required=True, help="Subset folder, e.g. isshk or camera1")
    parser.add_argument("--day", required=True, help="Day in YYYY-MM-DD format")
    parser.add_argument("--out", default="", help="Optional output JSON file")
    args = parser.parse_args()

    values = query_graphspot_values(args)

    print(json.dumps(values, ensure_ascii=False, indent=4))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=4)
        print(f"\nSaved to: {args.out}")


if __name__ == "__main__":
    main()

# Example:
# python get_rotated_rects.py --output_dir /disk1/lmk22/parking_spot_recognition/instance_seg/ --annotations_dir /disk1/lmk22/parking_spot_recognition/metapklot/cjsons/ --exp wo_plds_001 --subset isshk --day 2016-03-23 --out ./out_jsons_isshk_2016-03-23.json
