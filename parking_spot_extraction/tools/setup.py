import os
import sys
import json
from glob import glob
from pathlib import Path
from collections import defaultdict
import shutil

import cv2
import numpy as np

def clip(v, lo, hi):
    return max(lo, min(v, hi))

def polygon_area(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    pts = pts[np.isfinite(pts).all(axis=1)]
    x = pts[:, 0]
    y = pts[:, 1]

    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

def polygon_to_crop(poly, x_off, y_off, crop_w, crop_h):
    pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)

    if len(pts) < 3:
        return None, None

    pts[:, 0] -= x_off
    pts[:, 1] -= y_off

    pts[:, 0] = np.clip(pts[:, 0], 0, crop_w - 1)
    pts[:, 1] = np.clip(pts[:, 1], 0, crop_h - 1)

    pts = pts[np.isfinite(pts).all(axis=1)]

    pts = np.asarray(pts, dtype=np.float32)

    return pts, pts.flatten().tolist()

def bbox_from_points(pts, crop_w, crop_h):
    pts = np.asarray(pts, dtype=np.float32)

    x_min = float(np.min(pts[:, 0]))
    y_min = float(np.min(pts[:, 1]))
    x_max = float(np.max(pts[:, 0]))
    y_max = float(np.max(pts[:, 1]))

    x_min = np.clip(x_min, 0, crop_w - 1)
    y_min = np.clip(y_min, 0, crop_h - 1)
    x_max = np.clip(x_max, 0, crop_w - 1)
    y_max = np.clip(y_max, 0, crop_h - 1)

    return [
        float(x_min),
        float(y_min),
        float(x_max - x_min),
        float(y_max - y_min),
    ]

def get_best_rotated_rect(last_best_rotated_rect, x_off, y_off):
    (cx, cy), (w, h), angle = last_best_rotated_rect

    cx = float(cx) - float(x_off)
    cy = float(cy) - float(y_off)

    return ((cx, cy), (float(w), float(h)), float(angle))

def adjust_annotation(annotation, x_off, y_off, crop_w, crop_h, keep_best_rotated_rect=True):
    new_segmentations = []
    all_pts = []
    total_area = 0.0

    for poly in annotation["segmentation"]:
        pts, flat = polygon_to_crop(poly, x_off, y_off, crop_w, crop_h)
        if flat is None:
            continue

        if len(flat) < 6:
            continue

        area = polygon_area(pts)
        if not np.isfinite(area) or area <= 1e-6:
            continue

        new_segmentations.append(flat)
        all_pts.append(pts)
        total_area += area

    if not new_segmentations:
        return None

    merged = np.vstack(all_pts).astype(np.float32)

    new_annotation = dict(annotation)
    new_annotation["segmentation"] = new_segmentations
    new_annotation["bbox"] = bbox_from_points(merged, crop_w, crop_h)
    new_annotation["area"] = float(total_area)

    if keep_best_rotated_rect and "best_rotated_rect" in annotation:
        new_annotation["best_rotated_rect"] = get_best_rotated_rect(
            annotation["best_rotated_rect"], x_off=x_off, y_off=y_off
        )

    return new_annotation

def crop_image(image_path, out_path, x1, y1, w, h):
    image = cv2.imread(image_path)
    if image is None:
        print(f"WARNING: could not read image {image_path}")
        return

    img_h, img_w = image.shape[:2]
    x1 = clip(int(x1), 0, img_w - 1)
    y1 = clip(int(y1), 0, img_h - 1)
    x2 = clip(int(x1 + w), 0, img_w)
    y2 = clip(int(y1 + h), 0, img_h)

    cropped_image = image[y1:y2, x1:x2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cropped_image)

def process_coco_json(data):
    anns_by_image = defaultdict(list)
    for ann in data.get("annotations", []):
        anns_by_image[ann["image_id"]].append(ann)

    new_images = []
    new_annotations = []

    for image_dict in data.get("images", []):
        x1, y1, w, h = image_dict["annotationsRectangle"]
        crop_w, crop_h = int(w), int(h)

        new_image_dict = dict(image_dict)
        new_image_dict["width"] = crop_w
        new_image_dict["height"] = crop_h
        new_image_dict["annotationsRectangle"] = [0, 0, crop_w, crop_h]
        new_images.append(new_image_dict)

        image_id = image_dict["id"]
        for ann in anns_by_image.get(image_id, []):
            adjusted = adjust_annotation(
                ann,
                x1, y1,
                crop_w, crop_h,
                keep_best_rotated_rect=("best_rotated_rect" in ann)
            )
            if adjusted is not None:
                new_annotations.append(adjusted)

    out = dict(data)
    out["images"] = new_images
    out["annotations"] = new_annotations

    return out

def merge_coco_files(inputs, out_file, default_category_name):
    merged = {
        "info": {"description": f"Merged {' + '.join(p.stem for p in inputs)} annotations"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": None,
    }

    next_img_id = 1
    next_ann_id = 1

    for f in inputs:
        if not f.exists():
            print(f"WARNING: missing file {f}")
            continue

        data = json.loads(f.read_text(encoding="utf-8"))

        if merged["categories"] is None:
            merged["categories"] = data.get("categories", [])

        image_id_map = {}
        for img in data.get("images", []):
            old_id = img["id"]
            new_img = dict(img)
            new_img["id"] = next_img_id
            image_id_map[old_id] = next_img_id
            next_img_id += 1
            merged["images"].append(new_img)

        for ann in data.get("annotations", []):
            old_image_id = ann.get("image_id")
            if old_image_id not in image_id_map:
                continue

            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_id_map[old_image_id]
            next_ann_id += 1
            merged["annotations"].append(new_ann)

    if merged["categories"] is None or len(merged["categories"]) == 0:
        merged["categories"] = [{"id": 1, "name": default_category_name, "supercategory": "vehicle"}]

    out_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved: {out_file}")
    print(f"images={len(merged['images'])} anns={len(merged['annotations'])}")

if len(sys.argv) != 4:
    print("Usage: python crop_annotationsRectangle.py <MetaPKLot-Dataset> <PLds_converted> <output_directory>")
    sys.exit(1)

metapklot = sys.argv[1]
plds_converted = sys.argv[2]
output_directory = sys.argv[3]

metapklot += "/" if not metapklot.endswith("/") else ""
plds_converted += "/" if not plds_converted.endswith("/") else ""
output_directory += "/" if not output_directory.endswith("/") else ""

os.makedirs(output_directory, exist_ok=True)
os.makedirs(output_directory + "cjsons", exist_ok=True)

cars_jsons_path = sorted(glob(f"{metapklot}annotations/original/cars/**/*.json", recursive=True))
spots_jsons_path = sorted(glob(f"{metapklot}annotations/original/spots/**/*.json", recursive=True))

cropped_cars_jsons_path = sorted(glob(f"{metapklot}annotations/cropped/cars/**/*.json", recursive=True))
cropped_spots_jsons_path = sorted(glob(f"{metapklot}annotations/cropped/spots/**/*.json", recursive=True))

for spot_json_path in spots_jsons_path:
    with open(spot_json_path, "r", encoding="utf-8") as f:
        spot_data = json.load(f)

    for image_dict in spot_data.get("images", []):
        filename = image_dict["file_name"]
        x1, y1, w, h = image_dict["annotationsRectangle"]

        base_image_path = plds_converted if "PLds" in filename else metapklot
        image_path = f"{base_image_path}{filename}"
        out_path = f"{output_directory}{filename}"

        crop_image(image_path, out_path, x1, y1, w, h)

# for spot_json_path, car_json_path in zip(spots_jsons_path, cars_jsons_path):
#     with open(spot_json_path, "r", encoding="utf-8") as f:
#         spot_data = json.load(f)

#     with open(car_json_path, "r", encoding="utf-8") as f:
#         car_data = json.load(f)

#     spot_data = process_coco_json(spot_data)
#     car_data = process_coco_json(car_data)

#     spot_output_path = os.path.join(output_directory + "cjsons", os.path.basename(spot_json_path))
#     car_output_path = os.path.join(output_directory + "cjsons", os.path.basename(car_json_path))

#     with open(spot_output_path, "w", encoding="utf-8") as f:
#         json.dump(spot_data, f, ensure_ascii=False, indent=2)

#     with open(car_output_path, "w", encoding="utf-8") as f:
#         json.dump(car_data, f, ensure_ascii=False, indent=2)

for spot_json_path in cropped_spots_jsons_path:
    shutil.copy(spot_json_path, os.path.join(output_directory + "cjsons", os.path.basename(spot_json_path)))
for car_json_path in cropped_cars_jsons_path:
    shutil.copy(car_json_path, os.path.join(output_directory + "cjsons", os.path.basename(car_json_path)))

cjsons = Path(f"{output_directory}cjsons")

merge_coco_files(
    inputs=[
        cjsons / "vxusd_spots.json",
        cjsons / "vmlix_spots.json",
    ],
    out_file=cjsons / "vmlix-vxusd_spots.json",
    default_category_name="car",
)

merge_coco_files(
    inputs=[
        cjsons / "vxusd_cars.json",
        cjsons / "vmlix_cars.json",
    ],
    out_file=cjsons / "vmlix-vxusd_cars.json",
    default_category_name="car",
)

out_files = [
    cjsons / "merged_wo_PLds_cars.json",
    cjsons / "merged_wo_PKLot_cars.json",
    cjsons / "merged_wo_CNR_cars.json",
]

exclude_prefixes = [
    ["isshk", "qridr", "vxusd", "vmlix", "vmlix-vxusd"],
    ["ufpr04", "ufpr05", "pucpr"],
    ["cnr"],
]

for i, out_file in enumerate(out_files):
    inputs = []
    for p in sorted(cjsons.glob("*_cars.json")):
        name = p.name.lower()
        if name.startswith("merged_"):
            continue
        if any(name.startswith(pref.lower()) for pref in exclude_prefixes[i]):
            continue
        inputs.append(p)

    merged = {
        "info": {"description": "Merged cars annotations without PLds"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": None,
    }

    next_img_id = 1
    next_ann_id = 1

    for f in inputs:
        data = json.loads(f.read_text(encoding="utf-8"))

        if merged["categories"] is None:
            merged["categories"] = data.get("categories", [])

        image_id_map = {}
        for img in data.get("images", []):
            old_id = img["id"]
            new_img = dict(img)
            new_img["id"] = next_img_id
            image_id_map[old_id] = next_img_id
            next_img_id += 1
            merged["images"].append(new_img)

        for ann in data.get("annotations", []):
            old_image_id = ann.get("image_id")
            if old_image_id not in image_id_map:
                continue

            new_ann = dict(ann)
            new_ann["id"] = next_ann_id
            new_ann["image_id"] = image_id_map[old_image_id]
            next_ann_id += 1
            merged["annotations"].append(new_ann)

    if merged["categories"] is None or len(merged["categories"]) == 0:
        merged["categories"] = [{"id": 1, "name": "car", "supercategory": "vehicle"}]

    out_file.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
