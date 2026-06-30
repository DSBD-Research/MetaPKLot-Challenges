import faster_coco_eval

# Replace pycocotools with faster_coco_eval
faster_coco_eval.init_as_pycocotools()

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from pycocotools import mask as mask_utils

import numpy as np
import json
from mmdet.apis import inference_detector
import torch
from tqdm import tqdm
import os
from scipy.optimize import linear_sum_assignment
from collections import defaultdict
import cv2

# from quadrilateral_fitter import QuadrilateralFitter
from copy import deepcopy
from torchvision.ops import nms

from math import ceil


def convert_to_coco_format(preds, image_id, use_masks=False):
    coco_results = []
    img_embeddings = []

    bboxes = preds.bboxes.cpu().numpy()
    scores = preds.scores.cpu().numpy()
    if len(scores) > 0:
        masks = (
            preds.masks.cpu().numpy() if use_masks and hasattr(preds, "masks") else None
        )
        if not hasattr(preds, "mask_feats"):
            print("Warning: preds does not have mask_feats, using empty embeddings.")
            print(f"scores: {scores}, has_masks: {hasattr(preds, 'masks')}")
            print(f"masks: {masks.sum()}")

        embeddings = (
            preds.mask_feats.cpu().numpy()
            if use_masks and hasattr(preds, "masks")
            else None
        )

        img_embeddings = preds.metainfo["img_emb"].cpu().numpy()
        # print(f"Image embeddings for {image_id}: {img_embeddings}")

        for i in range(len(bboxes)):
            x1, y1, x2, y2 = bboxes[i]
            coco_bbox = [
                float(x1),
                float(y1),
                float(x2 - x1),
                float(y2 - y1),
            ]  # COCO expects xywh

            coco_dict = {
                "image_id": int(image_id),
                "category_id": 1,
                "bbox": coco_bbox,
                "score": float(scores[i]),
            }

            if masks is not None:
                rle = mask_utils.encode(np.asfortranarray(masks[i].astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("utf-8")  # convert bytes to str
                coco_dict["segmentation"] = rle
                coco_dict["embeddings"] = embeddings[i].tolist()

            coco_results.append(coco_dict)
    return coco_results, img_embeddings


def mask_iou(rle1, rle2):
    if rle1 is None or rle2 is None:
        return 0.0
    """Compute IoU between two binary masks."""
    iou = mask_utils.iou([rle1], [rle2], [False])
    return iou[0][0]


def spot_car_match(preds_t0, preds_t1, iou_thresh=0.5):
    n, m = len(preds_t0), len(preds_t1)
    iou_matrix = np.zeros((n, m))

    for i, p0 in enumerate(preds_t0):
        for j, p1 in enumerate(preds_t1):
            iou_matrix[i, j] = mask_iou(p0["segmentation"], p1["segmentation"])

    # Convert to cost (maximize IoU → minimize -IoU)
    cost = -iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    ious = []
    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_thresh:
            matches.append((i, j))  # index in preds_t0, preds_t1
            ious.append(iou_matrix[i, j])

    unmatched = [set(range(n)), set(range(m))]
    unmatched[0] -= set(
        matches[i][0] for i in range(len(matches))
    )  # unmatched in preds_t0
    unmatched[1] -= set(
        matches[j][1] for j in range(len(matches))
    )  # unmatched in preds_t1
    unmatched = (list(unmatched[0]), list(unmatched[1]))

    return matches, ious, unmatched


def save_predictions(root, coco_input, output_json, model):
    # === Step 1: Load your COCO-style JSON ===
    with open(coco_input, "r") as f:
        coco_data = json.load(f)

    all_results = []
    images_embeddings = {}

    for img_info in tqdm(coco_data["images"]):  # COCO format has "images": [{...}]
        file_name = img_info["file_name"]
        image_id = img_info["id"]

        # Construct full image path
        image_path = os.path.join(root, file_name)

        if not os.path.exists(image_path):
            print(f"[Warning] Image not found: {image_path}")
            continue

        # Run inference
        results = inference_detector(model, image_path)
        coco_results, img_embeddings = convert_to_coco_format(
            results.pred_instances, image_id, use_masks=True
        )

        # Store results
        if coco_results:
            all_results.extend(coco_results)
            images_embeddings[image_id] = img_embeddings.tolist()

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    final_dict = {"images": images_embeddings, "annotations": all_results}
    # === SAVE ALL PREDICTIONS ===
    with open(output_json, "w") as f:
        json.dump(final_dict, f, indent=2)

    print(f"Saved predictions to {output_json}")


def to_rle(segmentation, height, width):
    """Ensure segmentation is in RLE format."""
    if isinstance(segmentation, list):  # polygon
        rle = mask_utils.frPyObjects(segmentation, height, width)
        rle = mask_utils.merge(rle)
    elif isinstance(segmentation, dict):
        return segmentation
    else:
        raise ValueError("Invalid segmentation format")
    return rle


def match_ground_truth(preds, gts, height, width, iou_thresh=0.5):
    if not preds or not gts:
        return [], list(range(len(preds))), list(range(len(gts)))

    # Compute IoU matrix
    rle_preds = [to_rle(pred["segmentation"], height, width) for pred in preds]
    rle_gts = [to_rle(gt["segmentation"], height, width) for gt in gts]
    iou_matrix = mask_utils.iou(
        rle_preds, rle_gts, [False] * len(gts)
    )  # shape: [n_preds, n_gts]

    # Hungarian algorithm maximizes assignment, so we convert IoU to cost
    cost_matrix = 1.0 - iou_matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = []
    unmatched_preds = set(range(len(preds)))
    unmatched_gts = set(range(len(gts)))

    for i, j in zip(row_ind, col_ind):
        if iou_matrix[i, j] >= iou_thresh:
            matches.append((i, j))
            unmatched_preds.discard(i)
            unmatched_gts.discard(j)

    return matches, list(unmatched_preds), list(unmatched_gts)


def label_predictions(spots_data, spots_predictions, iou_thresh=0.5):
    # Build index
    coco_gt = COCO()
    coco_gt.dataset = spots_data
    coco_gt.createIndex()

    # Index GT by image_id
    gt_by_image = {}
    for ann in spots_data["annotations"]:
        gt_by_image.setdefault(ann["image_id"], []).append(ann)

    # Index image size
    size_by_image = {
        img["id"]: (img["height"], img["width"]) for img in spots_data["images"]
    }

    # Group predictions by image
    preds_by_image = {}
    for pred in spots_predictions:
        preds_by_image.setdefault(pred["image_id"], []).append(pred)

    labeled = []
    for img_id, preds in preds_by_image.items():
        gts = gt_by_image.get(img_id, [])
        height, width = size_by_image[img_id]

        matches, unmatched_preds, unmatched_gts = match_ground_truth(
            preds, gts, height, width, iou_thresh
        )

        for i, _ in matches:
            preds[i]["category_id"] = 1  # TP
            preds[i]["label"] = 1  # TP
            labeled.append(preds[i])

        for i in unmatched_preds:
            preds[i]["category_id"] = 0  # FP
            preds[i]["label"] = 0  # FP
            labeled.append(preds[i])

        print(
            f"Image ID {img_id}: {len(matches)} matches, {len(unmatched_preds)} unmatched predictions, {len(unmatched_gts)} unmatched ground truths"
        )

    return labeled


def add_pred(best_pred, pred, union_seg_rle, iou):
    if best_pred.get("id") is None:
        best_pred["segmentation"] = union_seg_rle
        best_pred["merged_ids"].append(pred["id"])
    elif pred.get("id") is None:
        pred["segmentation"] = union_seg_rle
        pred["merged_ids"].append(best_pred["id"])
    else:
        best_pred = {
            "segmentation": union_seg_rle,
            "merged_ids": [best_pred["id"], pred["id"]],
            "iou": iou,
        }


def get_best_pred(gt, pred_annots):
    best_iou = 0.0
    best_pred = None

    for pred in pred_annots:
        pred_seg = pred["segmentation"]

        h, w = pred["segmentation"]["size"]
        gt_rle = mask_utils.merge(mask_utils.frPyObjects(gt["segmentation"], h, w))
        iou = mask_utils.iou([gt_rle], [pred_seg], [False])[0][0]

        if iou > best_iou:
            best_iou = iou
            best_pred, pred = pred, best_pred

        if iou > 0 and pred is not None:
            # Try union
            union_seg_rle = mask_utils.merge([best_pred["segmentation"], pred_seg])
            iou = mask_utils.iou([gt_rle], [union_seg_rle], [False])[0][0]

            if iou > best_iou:
                best_iou = iou
                add_pred(best_pred, pred, union_seg_rle, iou)

    if best_pred:
        improved = True
        while improved:
            improved = False
            for pred in pred_annots:
                if isinstance(pred["id"], int):
                    # Skip if pred is already merged with best_pred
                    if (
                        "merged_ids" in best_pred
                        and pred["id"] in best_pred["merged_ids"]
                    ):
                        continue
                elif isinstance(pred["id"], list):
                    # Skip if pred is already merged with best_pred
                    if "merged_ids" in best_pred and any(
                        id in best_pred["merged_ids"] for id in pred["id"]
                    ):
                        continue

                # Try union
                union_seg_rle = mask_utils.merge(
                    [best_pred["segmentation"], pred["segmentation"]]
                )

                iou = mask_utils.iou([union_seg_rle], [gt_rle], [False])[0][0]

                if iou > best_iou:
                    # Update best_pred to be a virtual merged prediction
                    add_pred(best_pred, pred, union_seg_rle, iou)
                    # best_pred["segmentation"] = union_seg_rle
                    best_iou = iou
                    improved = True
                    break  # try again from start with new best_pred

    seg = best_pred["segmentation"] if best_pred else None
    return best_iou, seg


def generate_spot_tracks(gt_json, pred_json, iou_thresh, coco_eval=False):
    gt = COCO(gt_json)  # COCO-style GT annotations
    # Nested dict: date → time → list of predictions
    grouped_by_date_time = defaultdict(lambda: defaultdict(list))
    unique_pred_id = (
        int(gt.dataset["images"][0]["id"] / 1000000000) * 1000000000
    )  # Ensure car predictions IDs are in the same range as the current subset

    image_info_by_id = {img["id"]: img for img in gt.dataset["images"]}
    with open(pred_json) as f:
        predictions = json.load(f)

    # Check if predictions is a dict or a list
    if isinstance(predictions, dict):
        predictions = predictions["annotations"]

    predictions = [p for p in predictions if p["score"] >= 0.5]

    # Group predictions by date and time
    for pred in tqdm(predictions):
        img_id = pred["image_id"]
        img_info = image_info_by_id[img_id]

        date = tuple(img_info["date"])  # (YYYY, MM, DD)
        time = tuple(img_info["time"])  # (hh, mm, ss)

        pred["id"] = unique_pred_id  # assign unique ID
        pred["date"] = date

        # pred["segmentation"] = segmentation_to_convex_polygon(pred["segmentation"])

        if pred["segmentation"] is None:
            print(
                f"!!! Warning: No segmentation for prediction {pred['id']} in image {img_id}"
            )
            continue

        unique_pred_id += 1
        grouped_by_date_time[date][time].append(pred)

    # Filter out predictions with None segmentation
    predictions = [pred for pred in predictions if pred["segmentation"] is not None]

    # Skipping invalid predictions
    predictions_dict = {}
    for p in predictions:
        if isinstance(p, dict) and "id" in p:
            predictions_dict[p["id"]] = p
        else:
            print(f"Skipping invalid prediction: {p}")

    # Sort each time slot by prediction score (descending)
    for date in grouped_by_date_time:
        for time in grouped_by_date_time[date]:
            grouped_by_date_time[date][time].sort(key=lambda p: -p["score"])

    # Generate tracks for each date
    subset_tracks = defaultdict(list)  # date -> list of tracks
    for date in tqdm(grouped_by_date_time, desc="Generating tracks..."):
        pred_tracks = daily_spot_heatmap_V2(
            grouped_by_date_time[date], iou_thresh
        )  # Dict[time -> list of predictions]

        subset_tracks[date] = pred_tracks

    return subset_tracks, predictions_dict


def segmentation_to_convex_polygon(rle):
    """
    Convert a segmentation (COCO polygon or RLE) into a convex hull polygon.
    """

    mask = mask_utils.decode(rle)
    height, width = rle["size"]

    # Get contours from the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None  # No valid contour

    # Get the largest contour
    contour = max(contours, key=cv2.contourArea)

    # Compute the convex hull
    hull = cv2.convexHull(contour)

    # Convert to COCO polygon format (flattened list of x, y coords)
    polygon = hull.squeeze()
    polygon = [float(x) for x in polygon.flatten()]

    if len(polygon) < 6:
        return rle

    rle = mask_utils.frPyObjects([polygon], height, width)
    rle = mask_utils.merge(rle)
    return rle


def segmentation_to_rotated_polygon(rle):
    """
    Convert a segmentation (RLE) into a rotated bounding box polygon.
    """
    height, width = rle["size"]

    # Compute rotated bounding box (min area)
    rect = get_rotated_bbox(rle)
    box = cv2.boxPoints(rect)  # 4 corner points
    box = np.int0(box).tolist()  # Convert to list of points

    # Flatten the list of points into COCO polygon format
    polygon = [float(coord) for point in box for coord in point]

    if not isinstance(polygon[0], list):
        polygon = [polygon]

    rle = mask_utils.frPyObjects(polygon, height, width)
    rle = mask_utils.merge(rle)
    return rle


def get_rotated_bbox(rle):
    # Convert to binary mask
    mask = mask_utils.decode(rle)
    height, width = rle["size"]

    # Get contours from the binary mask
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None  # No valid contour

    # Get largest contour
    contour = max(contours, key=cv2.contourArea)

    # Compute rotated bounding box (min area)
    rect = cv2.minAreaRect(contour)  # returns ((cx, cy), (w, h), angle)

    return rect


def add_prediction_to_track(pred):
    first = {
        "ids": [pred["id"]],
        "image_ids": [pred["image_id"]],
        "scores": [pred["score"]],
        "segmentation": pred["segmentation"],
        "segmentations": [pred["segmentation"]],
    }

    return deepcopy(first)


def merge_masks(rle0, rle1):
    # Decode to binary masks
    mask0 = mask_utils.decode(rle0)
    mask1 = mask_utils.decode(rle1)

    # Merge using logical OR
    merged_mask = (mask0 | mask1).astype(np.uint8)  # ensure uint8 for encoding

    # Encode back to RLE
    return mask_utils.encode(np.asfortranarray(merged_mask))


def merge_tracks_by_iou(tracks, iou_thresh=0.5):
    original_n = len(tracks)

    while True:
        n = len(tracks)
        iou_matrix = np.zeros((n, n))

        for i, p0 in enumerate(tracks):
            for j, p1 in enumerate(tracks):
                if i >= j:
                    continue
                iou_matrix[i, j] = mask_iou(p0["segmentation"], p1["segmentation"])

        # Convert to cost (maximize IoU → minimize -IoU)
        cost = -iou_matrix
        row_ind, col_ind = linear_sum_assignment(cost)

        matches = []
        for i, j in zip(row_ind, col_ind):
            if i < j and iou_matrix[i, j] >= iou_thresh:
                matches.append((i, j, iou_matrix[i, j]))

        # Sort matches by IoU descending
        matches.sort(key=lambda x: -x[2])

        # Display sorted matches
        merged = set()
        removed = set()
        for i, j, iou in matches:
            if i in merged or j in merged:
                continue  # skip if either has been merged already

            # Merge ids
            tracks[i]["ids"].extend(tracks[j]["ids"])
            tracks[i]["image_ids"].extend(tracks[j]["image_ids"])
            tracks[i]["scores"].extend(tracks[j]["scores"])

            # Merge segmentations
            tracks[i]["segmentation"] = merge_masks(
                tracks[i]["segmentation"], tracks[j]["segmentation"]
            )

            # remove j from tracks
            removed.add(j)

            # Mark i and j as merged
            merged.add(i)
            merged.add(j)

        # Filter out removed tracks
        new_tracks = [t for i, t in enumerate(tracks) if i not in removed]

        if len(new_tracks) == n:
            break

        tracks = new_tracks

    print(f"Reduced from {original_n} to {len(new_tracks)} tracks after merging")
    return new_tracks


def rle_majority_vote(rles, vote_ratio=0.5):
    """
    Majority-vote combine a list of RLE masks into a single RLE.

    Args:
        rles (list[dict]): COCO RLEs with equal HxW.
        vote_ratio (float): Fraction in [0,1]. 0.5 = simple majority.
                            Threshold = ceil(vote_ratio * len(rles))

    Returns:
        dict: COCO RLE (json-serializable: counts as str)
    """
    if not rles:
        return None

    k = max(1, ceil(vote_ratio * len(rles)))  # votes needed
    # decode to boolean arrays (HxW)
    masks = [_decode_bool_rle(r) for r in rles]  # your existing helper
    counts = np.sum(masks, axis=0)  # integer image
    voted = counts >= k  # boolean image

    # back to RLE; ensure Fortran order for pycocotools
    rle = mask_utils.encode(np.asfortranarray(voted.astype(np.uint8)))
    # make JSON-friendly
    rle["counts"] = rle["counts"].decode("ascii")
    return rle


def daily_spot_heatmap_V2(preds_by_time: dict, iou_thresh: float):
    """Generate spot heatmap from daily predictions using Hungarian matching.

    Args:
        preds_by_time (dict): Nested dict with date -> time -> list of predictions
        iou_thresh (float): IoU threshold for matching

    Returns:
        list: List of tracks, each track is a dict with merged segmentations
    """

    daily_images = sorted(preds_by_time.keys())
    pred_tracks = []  # list of lists of predictions
    used_preds = defaultdict(set)  # time -> set of used indices

    # initialize tracks with first time step
    for i, pred in enumerate(preds_by_time[daily_images[0]]):
        pred_tracks += [add_prediction_to_track(pred)]
        used_preds[daily_images[0]].add(pred["id"])  # mark as used

    # iterate over consecutive pairs of time steps
    for cur_image in daily_images[1:]:
        # preds_t0 = pred_tracks  # previous tracks
        # preds_t1 = preds_by_time[t1]
        matches, _, _ = spot_car_match(
            pred_tracks, preds_by_time[cur_image], iou_thresh
        )
        # print(f"Matching between pred_tracks and t1: {len(matches)} matches found")

        # Extend existing tracks
        for i, j in matches:
            i0 = pred_tracks[i]
            i1 = preds_by_time[cur_image][j]

            if i1["id"] in used_preds[cur_image]:
                continue  # skip reused preds

            # Find the track for pred_tracks[i0]
            used_preds[cur_image].add(i1["id"])  # mark as used
            i0["segmentation"] = merge_masks(i0["segmentation"], i1["segmentation"])
            i0["segmentations"] += [i1["segmentation"]]
            i0["ids"] += [i1["id"]]
            i0["image_ids"] += [i1["image_id"]]
            i0["scores"] += [i1["score"]]

        # Start new tracks from unmatched predictions in cur_image
        for j, pred in enumerate(preds_by_time[cur_image]):
            i1 = preds_by_time[cur_image][j]
            if i1["id"] not in used_preds[cur_image]:
                pred_tracks += [add_prediction_to_track(pred)]
                used_preds[cur_image].add(i1["id"])

    for track in pred_tracks:
        non_zero_pixels = int(_decode_bool_rle(track["segmentation"]).sum())
        sum_values = sum(
            [_decode_bool_rle(seg).sum() for seg in track["segmentations"]]
        )
        track["score"] = (
            (sum_values / non_zero_pixels) / len(daily_images)
            if non_zero_pixels > 0
            else 0
        )

        track["norm_length"] = len(track["scores"]) / len(daily_images)
        track["segmentation"] = segmentation_to_rotated_polygon(track["segmentation"])

    return pred_tracks


def _decode_bool_rle(rle):
    """Decode COCO RLE (or list of RLEs) -> boolean mask."""
    m = mask_utils.decode(rle)  # HxW (uint8 {0,1}) or HxWxN
    if m.ndim == 3:
        m = np.max(m, axis=2)  # collapse parts if any slipped through
    return m.astype(bool)


def get_first_image_by_day(coco_data, add_annotations=False, non_working_days=None):
    # Group images by date
    images_by_date = defaultdict(list)
    for img in coco_data["images"]:
        date = tuple(img["date"])
        images_by_date[date].append(img)

    # Get the first image for each day
    first_images = []
    first_image_ids = set()
    for date, imgs in images_by_date.items():
        if non_working_days and date in non_working_days:
            continue
        imgs.sort(key=lambda x: x["time"])  # sort by time within the day
        first_img = imgs[0]
        first_images.append(first_img)
        first_image_ids.add(first_img["id"])

    first_annotations = []  # Initialize empty list for annotations

    # If annotations are requested, filter them
    if add_annotations:
        # Filter annotations corresponding to the selected images
        first_annotations = [
            ann
            for ann in coco_data["annotations"]
            if ann["image_id"] in first_image_ids and ann["segmentation"] is not None
        ]

    # Create new COCO-style structure
    new_coco = {
        "images": first_images,
        "annotations": first_annotations,
        "categories": coco_data.get("categories", []),
    }

    return new_coco


def update_areas(coco_data):
    # Build image metadata lookup
    image_info_by_id = {img["id"]: img for img in coco_data["images"]}

    updated_annotations = []
    missing_dims = 0

    for ann in coco_data["annotations"]:
        img_info = image_info_by_id.get(ann["image_id"])

        if img_info is None:
            print(f"[Warning] Image ID {ann['image_id']} not found in 'images'")
            continue

        height = img_info.get("height")
        width = img_info.get("width")

        if height is None or width is None:
            print(f"[Warning] Missing dimensions for image_id {ann['image_id']}")
            missing_dims += 1
            continue

        seg = ann["segmentation"]
        if not seg:
            ann["area"] = 0.0
            continue

        rle = mask_utils.frPyObjects(seg, height, width)
        rle = mask_utils.merge(rle)
        area = float(mask_utils.area(rle))
        ann["area"] = area
        updated_annotations.append(ann)

    coco_data["annotations"] = updated_annotations

    if missing_dims > 0:
        print(
            f"[Note] Skipped {missing_dims} annotations due to missing image dimensions."
        )

    return coco_data


def filter_spot_gt(spots_data, non_working_days=None):
    spots_data = get_first_image_by_day(
        spots_data, add_annotations=True, non_working_days=non_working_days
    )
    spots_data["categories"] = [{"id": 1, "name": "spot"}]
    for ann in spots_data["annotations"]:
        ann["category_id"] = 1

    if "info" not in spots_data:
        spots_data["info"] = {}

    return update_areas(spots_data)


def save_prediction_to_graph_data(
    coco_input, spots_input, output_json, non_working_days=None
):
    """
    Save spot predictions in COCO format and spots data to a graph data dictionary.

    Args:
        coco_input (dict): COCO dictionary with ground truth car annotations.
        spots_input (dict): Spots dictionary with ground truth annotations.
        output_json (dict): COCO dictionary with car predictions.
        non_working_days (list, optional): List of non-working days. Defaults to None.

    Returns:
        dict: Graph data to be saved into nodes and edges.
        list: List of labelled spots predictions.
    """

    with open(coco_input, "r") as f:
        coco_data = json.load(f)

    with open(spots_input, "r") as f:
        spots_data = json.load(f)

    spots_data = filter_spot_gt(spots_data, non_working_days)

    subset_tracks, predictions_dict = generate_spot_tracks(
        coco_input, output_json, iou_thresh=0.5
    )

    with open(output_json) as f:
        image_embeddings_by_id = json.load(f)["images"]

    spots_day_to_id_dict = {
        tuple(img["date"]): img["id"] for img in spots_data["images"]
    }
    spots_predictions = []

    spot_pred_id = (
        int(spots_data["images"][0]["id"] / 1000000000) * 1000000000
    )  # Ensure spot predictions IDs are in the same range as the current subset

    for date, tracks in subset_tracks.items():
        if date not in spots_day_to_id_dict:
            print(f"Warning: No matching image for date {date}")
            continue
        matching_image_id = spots_day_to_id_dict[date]

        for track in tracks:
            track["spot_pred_id"] = spot_pred_id  # Ensure unique ID

            segmentation = track["segmentation"]
            if segmentation is None:
                print(
                    f"Warning: No segmentation for track {track['spot_pred_id']} on date {date}"
                )
                continue

            # print(f"Old segmentation: {segmentation}")
            # segmentation = segmentation_to_approx_quad_polygon(
            #     segmentation, convert_to_rle=True
            # )
            # print(f"New segmentation: {segmentation}")

            spot_prediction = {
                "spot_pred_id": track["spot_pred_id"],
                "image_id": matching_image_id,
                "category_id": 1,  # Assuming category_id for spots is 1
                "score": track["score"],
                "scores": track["scores"],
                "norm_length": track["norm_length"],
                "bbox": mask_utils.toBbox(segmentation).tolist(),
                "area": mask_utils.area(segmentation).item(),
                "segmentation": segmentation,
            }

            spot_pred_id += 1
            spots_predictions.append(spot_prediction)

    print(f"Number of spot predictions: {len(spots_predictions)}")
    print(f"Number of images in spots data: {len(spots_data['images'])}")
    print(f"Number of annotations in spots data: {len(spots_data['annotations'])}")

    if "info" not in spots_data:
        spots_data["info"] = {}

    # Load ground truth COCO object
    coco_gt = COCO()
    coco_gt.dataset = spots_data
    coco_gt.createIndex()

    print(f"{20 * '*'} COCO evaluation - spots_predictions {20 * '*'}")

    coco_dt = coco_gt.loadRes(spots_predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.maxDets = [10, 100, 1000]  # instead of default [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    labelled_spots_predictions = label_predictions(
        spots_data, spots_predictions, iou_thresh=0.5
    )

    # labelled_spots_predictions = apply_nms(labelled_spots_predictions)

    print(f"{20 * '*'} COCO evaluation - labelled_spots_predictions {20 * '*'}")

    coco_dt = coco_gt.loadRes(labelled_spots_predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, iouType="segm")
    coco_eval.params.maxDets = [10, 100, 1000]  # instead of default [1, 10, 100]
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # filtered_spots_predictions = [
    #     pred for pred in labelled_spots_predictions if pred["label"] == 1
    # ]

    from collections import defaultdict

    predictions_by_date = defaultdict(list)

    for pred in predictions_dict.values():
        date = pred["date"]  # e.g., (2015, 11, 12)
        # pred["segmentation"] = segmentation_to_convex_polygon(pred["segmentation"])
        predictions_by_date[date].append(pred)

    # Output: date → list of annotations
    spots_data_by_date = defaultdict(list)

    # Step 1: Map image_id → date
    image_id_to_date = {img["id"]: tuple(img["date"]) for img in spots_data["images"]}

    # Step 2: Assign annotations to their image's date
    for ann in spots_data["annotations"]:
        img_id = ann["image_id"]
        if img_id in image_id_to_date:
            date = image_id_to_date[img_id]
            spots_data_by_date[date].append(ann)

    # Create image nodes

    image_nodes = list()

    image_dims_by_id = {
        image_info["id"]: {"height": image_info["height"], "width": image_info["width"]}
        for image_info in coco_data["images"]
    }

    for image_info in coco_data["images"]:
        if str(image_info["id"]) in image_embeddings_by_id:
            image_node = {
                "id": image_info["id"],
                "file_name": image_info["file_name"],
                "date": image_info["date"],
                "time": image_info["time"],
                "width": image_info["width"],
                "height": image_info["height"],
                "embeddings": image_embeddings_by_id[str(image_info["id"])],
            }
            image_nodes += [image_node]

    # Initialize car nodes and relationships
    # These will be used to create the graph structure
    # Each car node will have an ID, bounding box, score, and embeddings
    # Relationships will link car nodes to images they appear in

    car_nodes = list()
    car_from_image = list()

    for pred in predictions_dict.values():
        width = image_dims_by_id[pred["image_id"]]["width"]
        height = image_dims_by_id[pred["image_id"]]["height"]

        if pred["segmentation"] is None:
            print(
                f"Warning: No segmentation for prediction in image {pred['image_id']}"
            )
            continue

        rotated_bbox = get_rotated_bbox(pred["segmentation"])
        if rotated_bbox is None:
            print(
                f"Warning: Could not compute rotated bbox for prediction in image {pred['image_id']}"
            )
            continue
        rotated_bbox = [*rotated_bbox[0], *rotated_bbox[1], rotated_bbox[2] / 90.0]
        rotated_bbox[0] /= width
        rotated_bbox[1] /= height
        rotated_bbox[2] /= width
        rotated_bbox[3] /= height

        car_node = {
            "id": pred["id"],
            # "bbox": pred["bbox"],
            "score": pred["score"],
            "segmentation": pred["segmentation"],
            "embeddings": (
                pred["embeddings"] if "embeddings" in pred else pred["embedding"]
            ),
        }
        if isinstance(pred["segmentation"]["counts"], bytes):
            # Decode RLE counts if they are in bytes format
            pred["segmentation"]["counts"] = pred["segmentation"]["counts"].decode(
                "utf-8"
            )

        car_from_image += [
            {
                "car_id": pred["id"],
                "image_id": pred["image_id"],
                "rotated_bbox": rotated_bbox,
            }
        ]
        car_nodes += [car_node]

    labelled_spots_predictions_dict = {
        pred["spot_pred_id"]: pred for pred in labelled_spots_predictions
    }

    # Create spot nodes and relationships between car nodes and spot nodes
    spot_nodes = list()
    car_inside_spot = list()

    car_id_to_embedding = {
        car["id"]: car["embeddings"] if "embeddings" in car else car["embedding"]
        for car in car_nodes
    }

    for date, tracks in subset_tracks.items():
        if date not in spots_day_to_id_dict:
            print(f"Warning: No matching image for date {date}")
            continue
        for track in tracks:
            labelled_spot = labelled_spots_predictions_dict[track["spot_pred_id"]]

            car_embeddings = []

            for car_id in track["ids"]:
                car_inside_spot += [
                    {"car_id": car_id, "spot_id": track["spot_pred_id"]}
                ]
                if car_id in car_id_to_embedding:
                    car_embeddings.append(car_id_to_embedding[car_id])
                else:
                    print(f"[WARNING] Missing embedding for car_id {car_id}")
                    continue

            car_embeddings_tensor = torch.tensor(car_embeddings, dtype=torch.float)
            spot_embedding = car_embeddings_tensor.mean(dim=0).tolist()

            if isinstance(labelled_spot["segmentation"]["counts"], bytes):
                labelled_spot["segmentation"]["counts"] = labelled_spot["segmentation"][
                    "counts"
                ].decode("utf-8")

            spot_score_stats = [
                np.average(labelled_spot["scores"]),
                np.std(labelled_spot["scores"]),
                np.min(labelled_spot["scores"]),
                np.max(labelled_spot["scores"]),
                len(labelled_spot["scores"]),
                track["norm_length"],
            ]

            spot_node = {
                "id": track["spot_pred_id"],
                "date": date,
                "label": labelled_spot["label"],
                "segmentation": labelled_spot["segmentation"],
                "score": track["score"],
                "area": labelled_spot["area"],
                "score_stats": spot_score_stats,  # Add more statistics as needed
                "embeddings": spot_embedding,
            }

            spot_nodes += [spot_node]

    graph_structure = {
        "car_nodes": car_nodes,
        "image_nodes": image_nodes,
        "spot_nodes": spot_nodes,
        "car_from_image": car_from_image,
        "car_inside_spot": car_inside_spot,
    }

    for pred in labelled_spots_predictions:
        pred.pop("rle", None)

    return graph_structure, labelled_spots_predictions


def apply_nms(predictions, iou_threshold=0.7):
    """Apply Non-Maximum Suppression to the predictions."""
    if not predictions:
        return []

    # Group by image + category
    from collections import defaultdict

    grouped = defaultdict(list)
    for pred in predictions:
        key = (pred["image_id"], pred["category_id"])
        grouped[key].append(pred)

    # Apply NMS
    iou_threshold = 0.5  # can adjust
    filtered_preds = []

    for (image_id, category_id), boxes in grouped.items():
        if len(boxes) == 0:
            continue

        # Extract boxes and scores
        boxes_tensor = torch.tensor([b["bbox"] for b in boxes], dtype=torch.float32)
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        boxes_xyxy = boxes_tensor.clone()
        boxes_xyxy[:, 2] = boxes_xyxy[:, 0] + boxes_xyxy[:, 2]
        boxes_xyxy[:, 3] = boxes_xyxy[:, 1] + boxes_xyxy[:, 3]

        scores = torch.tensor([b["score"] for b in boxes], dtype=torch.float32)

        keep_indices = nms(boxes_xyxy, scores, iou_threshold)

        for idx in keep_indices:
            filtered_preds.append(boxes[idx])

    return filtered_preds
