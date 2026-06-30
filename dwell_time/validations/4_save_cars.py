from collections import defaultdict
from copy import copy
from tqdm import tqdm
import cv2 as cv
import argparse
import hashlib
import json
import os

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def get_spot_id(annotation):
    rotated_rect = annotation["best_rotated_rect"]

    flat = []
    for item in rotated_rect:
        if isinstance(item, list):
            flat.extend(item)
        else:
            flat.append(item)

    flat_rounded = [round(x, 4) for x in flat]
    rect_string = ','.join(map(str, flat_rounded))
    rect_hash = hashlib.sha256(rect_string.encode()).hexdigest()
    unique_id = rect_hash[:10]

    return unique_id

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def crop_rect(image, rect):
    center, size, angle = rect[0:3]
    height, width = image.shape[0:2]

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    spot_rot = cv.warpAffine(image, rotation_matrix, (width, height))
    spot_crop = cv.getRectSubPix(spot_rot, [int(u) for u in size], center)

    if angle >= 45:
        spot_crop = cv.rotate(spot_crop, cv.ROTATE_90_CLOCKWISE)

    return spot_crop


def get_car_images(dest_folder, artifact):
    for car in tqdm(artifact, desc="Cropping images", unit="img"):
        car_id, annotations, images = car

        for annotation in annotations:
            image = next((img for img in images if img["id"] == annotation["image_id"]), None)
            if image is None:
                continue

            date = '-'.join([str(item).zfill(2) if isinstance(item, int) else str(item) for item in image["date"]])
            car_name = f"{car_id}#{date}"

            cv_image = cv.imread(image['file_name'])
            rotated_rect = annotation["best_rotated_rect"]

            cropped_car = crop_rect(cv_image, rotated_rect)

            spot_id = get_spot_id(annotation)
            os.makedirs(f"{dest_folder}/{spot_id}", exist_ok=True)
            cv.imwrite(f"{dest_folder}/{spot_id}/{car_name}.jpg", cropped_car)


def get_crop_artifact(ds_path, file):
    with open(f"{ds_path}/{file}", "r") as json_file:
        data = json.load(json_file)

    cars_id = []
    cars_annotations = defaultdict(list)
    for annotation in data["annotations"]:
        if annotation["car_id"] != -1 and annotation["car_id"] != -2:
            cars_id.append(annotation["car_id"])
            cars_annotations[annotation["car_id"]].append(annotation)
    cars_id = sorted(set(cars_id))

    artifact = []
    for car_id in cars_id:
        annotations = cars_annotations[car_id]

        image_ids = [annotation["image_id"] for annotation in annotations]
        images = [copy(image) for image in data["images"] if image.get("id") in image_ids]     

        # unique_by_date = {"".join([str(i) for i in image.get("date")]): image for image in images}
        unique_by_date = {}
        for image in images:
            if "".join([str(i) for i in image.get("date")]) not in unique_by_date:
                unique_by_date["".join([str(i) for i in image.get("date")])] = image

        images = list(unique_by_date.values())
    
        for image in images:
            image["file_name"] = f"{ds_path}/{image['file_name']}"

        artifact.append((car_id, annotations, images))

    return artifact


def main(args):
    assert os.path.exists(args.source), "invalid source dataset"

    print(f"Saving cars image for {args.name} in {args.dest}/{args.name}")
    os.makedirs(f"{args.dest}/{args.name}", exist_ok=True)

    jsons = [file for file in os.listdir(args.source) if ".json" in file]
    for file in jsons:
        subset = file.split("_")[0].upper()
        dest_folder = f"{args.dest}/{args.name}/{subset}"

        print(f"Cropping subset {subset}...")
        os.makedirs(dest_folder, exist_ok=True)

        artifact = get_crop_artifact(args.source, file)
        get_car_images(dest_folder, artifact)

    print(f"Dataset {args.name} completed!") 


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", "-s", type=str, required=True)
    parser.add_argument("--dest", "-d", type=str, required=True)
    parser.add_argument("--name", "-n", type=str, required=True)

    main(parser.parse_args())


# python3 4_save_cars.py -s /home/tteuz/Desktop/mestrado/meta_pklot/datasets/organized/PKLot -d /home/tteuz/Desktop/mestrado/meta_pklot/datasets/temp -n PKLot

# python3 4_save_cars.py -s /home/tteuz/Desktop/mestrado/meta_pklot/datasets/organized/CNRParkEXT -d /home/tteuz/Desktop/mestrado/meta_pklot/datasets/temp -n CNRParkEXT

# python3 4_save_cars.py -s /home/tteuz/Desktop/mestrado/meta_pklot/datasets/organized/PLds -d /home/tteuz/Desktop/mestrado/meta_pklot/datasets/temp -n PLds