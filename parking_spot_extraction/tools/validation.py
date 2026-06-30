import os
import sys
import json
import cv2
import numpy as np
import random

def draw_segmentation_overlay(image, annotations, alpha=0.35):
    overlay = image.copy()

    for ann in annotations:
        color = (0, 255, 0)
        segs = ann.get("segmentation", [])

        for poly in segs:
            if not poly or len(poly) < 6:
                continue

            pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
            pts = np.round(pts).astype(np.int32)

            cv2.fillPoly(overlay, [pts], color)
            cv2.polylines(image, [pts], isClosed=True, color=(0, 0, 255), thickness=2)

        if "bbox" in ann and ann["bbox"]:
            x, y, w, h = ann["bbox"]
            x, y, w, h = map(int, [x, y, w, h])
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)

        if "best_rotated_rect" in ann and ann["best_rotated_rect"]:
            rect = ann["best_rotated_rect"]
            (cx, cy), (rw, rh), angle = rect

            box = cv2.boxPoints(((float(cx), float(cy)), (float(rw), float(rh)), float(angle)))
            box = np.round(box).astype(np.int32)
            cv2.polylines(image, [box], isClosed=True, color=(0, 255, 255), thickness=2)

    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return blended

def find_image_info(data, image_id=None, file_name=None):
    for img in data.get("images", []):
        if image_id is not None and img.get("id") == image_id:
            return img
        if file_name is not None and img.get("file_name") == file_name:
            return img
    return None

def main():
    if len(sys.argv) != 3:
        print("Usage:")
        print("python3 validate_segmentation.py <metapklot> <json_path>")
        sys.exit(1)

    metapklot = sys.argv[1]
    metapklot += "/" if not metapklot.endswith("/") else ""
    json_path = sys.argv[2]

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    image_ids = [img["id"] for img in data.get("images", [])]

    image_id = random.choice(image_ids)
    file_name = None
    for image in data['images']:
        if image['id'] == image_id:
            file_name = image['file_name']
            break

    image_path = f"{metapklot}{file_name}"

    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        sys.exit(1)

    annotations = [ann for ann in data.get("annotations", []) if ann.get("image_id") == image_id]

    vis = draw_segmentation_overlay(image, annotations, alpha=0.35)

    os.makedirs("./teste", exist_ok=True)
    output_path = f"./teste/validation_{os.path.basename(file_name)}"
    cv2.imwrite(output_path, vis)

    print(f"Saved visualization to: {output_path}")
    print(f"Image: {file_name}")
    print(f"Annotations drawn: {len(annotations)}")

if __name__ == "__main__":
    main()
