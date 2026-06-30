import json
import sys
import os
from glob import glob
import cv2
import numpy as np

MIN_CONFIDENCE = 0.5

def calculate_iou(rect1, rect2):
    intersect_type, intersect_pts = cv2.rotatedRectangleIntersection(rect1, rect2)
    
    if intersect_type == cv2.INTERSECT_NONE or intersect_pts is None or len(intersect_pts) < 3:
        return 0.0

    hull = cv2.convexHull(intersect_pts)
    intersect_area = cv2.contourArea(hull)

    area1 = rect1[1][0] * rect1[1][1]
    area2 = rect2[1][0] * rect2[1][1]

    union_area = area1 + area2 - intersect_area

    if union_area <= 0:
        return 0.0

    return intersect_area / union_area

def draw_horizontal_hatch(image, box, color=(255, 0, 0), step=6):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [box], 255)

    x, y, w, h = cv2.boundingRect(box)

    pattern = np.zeros_like(image)
    for cy in range(y, y + h, step):
        cv2.line(pattern, (x, cy), (x + w, cy), color, 1)

    pattern_inside = cv2.bitwise_and(pattern, pattern, mask=mask)

    gray_pattern = cv2.cvtColor(pattern_inside, cv2.COLOR_BGR2GRAY)
    image[gray_pattern > 0] = color


if len(sys.argv) != 6:
    print("Usage: python write_rotated_rects.py <metapklot> <plds_converted> <input_json> <gt_json> <out_path>")
    sys.exit(1)

metapklot_dir = sys.argv[1]
plds_converted_dir = sys.argv[2]
input_json_path = sys.argv[3]
gt_json_path = sys.argv[4]
out_path = sys.argv[5]

os.makedirs(out_path, exist_ok=True)

metapklot_dir += "/" if not metapklot_dir.endswith("/") else ""
plds_converted_dir += "/" if not plds_converted_dir.endswith("/") else ""

with open(input_json_path, "r", encoding="utf-8") as f:
    input_json = json.load(f)

image_path_0 = input_json[0]["file_name"]
day = image_path_0.split("/")[-2]
dataset = image_path_0.split("/")[0]
subset = image_path_0.split("/")[1]

annotations_gt = []
gt_rects = []
json_gt = json.load(open(gt_json_path, "r", encoding="utf-8"))

for image in json_gt["images"]:
    if image["file_name"] == image_path_0:
        image_id = image["id"]
        for annotation in json_gt["annotations"]:
            if annotation["image_id"] == image_id:
                gt_rr = annotation['best_rotated_rect']
                annotations_gt.append(gt_rr)
                
                rect = (
                    (float(gt_rr[0][0]), float(gt_rr[0][1])),
                    (float(gt_rr[1][0]), float(gt_rr[1][1])),
                    float(gt_rr[2]),
                )
                gt_rects.append(rect)

base_image_path = plds_converted_dir if dataset.lower() == "plds" else metapklot_dir
all_images = glob(f"{base_image_path}{dataset}/{subset}/**/*.jpg", recursive=True)
images_of_the_day = [img for img in all_images if day in img]

valid_predictions = []
for spot_dict in input_json:
    if float(spot_dict.get("score", 0.0)) >= MIN_CONFIDENCE:
        rotated_rect = spot_dict.get("rotated_rect")
        if rotated_rect:
            rect = (
                (float(rotated_rect[0][0]), float(rotated_rect[0][1])),
                (float(rotated_rect[1][0]), float(rotated_rect[1][1])),
                float(rotated_rect[2]),
            )
            valid_predictions.append(rect)


for image in images_of_the_day:
    opened_image = cv2.imread(image)

    if opened_image is None:
        continue

    matched_gt_indices = set()
    matched_pred_indices = set()
    
    for i, gt_rect in enumerate(gt_rects):
        for j, pred_rect in enumerate(valid_predictions):
            iou = calculate_iou(gt_rect, pred_rect)
            if iou > 0.5:
                matched_gt_indices.add(i)
                matched_pred_indices.add(j)

    red_color = (0, 0, 255)
    blue_color = (255, 0, 0)
    magenta_color = (255, 0, 255)

    overlay = opened_image.copy()
    has_unmatched_pred = False

    for j, pred_rect in enumerate(valid_predictions):
        if j not in matched_pred_indices:
            box = cv2.boxPoints(pred_rect).astype(int)
            cv2.fillPoly(overlay, [box], red_color)
            has_unmatched_pred = True

    if has_unmatched_pred:
        alpha = 0.3
        cv2.addWeighted(overlay, alpha, opened_image, 1.0 - alpha, 0, opened_image)

    for i, gt_rect in enumerate(gt_rects):
        if i not in matched_gt_indices:
            box = cv2.boxPoints(gt_rect).astype(int)
            draw_horizontal_hatch(opened_image, box, color=blue_color, step=6)
            cv2.polylines(opened_image, [box], True, blue_color, 2)

    for j, pred_rect in enumerate(valid_predictions):
        box = cv2.boxPoints(pred_rect).astype(int)
        
        if j in matched_pred_indices:
            cv2.polylines(opened_image, [box], True, magenta_color, 2)
        else:
            cv2.polylines(opened_image, [box], True, red_color, 2)
            cv2.line(opened_image, tuple(box[0]), tuple(box[2]), red_color, 2)
            cv2.line(opened_image, tuple(box[1]), tuple(box[3]), red_color, 2)

    out_file = os.path.join(
        out_path,
        image.split("/")[-1].replace(".jpg", "_rotated_rects.jpg")
    )
    cv2.imwrite(out_file, opened_image)
