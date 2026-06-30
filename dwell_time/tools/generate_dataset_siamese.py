from collections import defaultdict
from tqdm import tqdm
import cv2 as cv
import argparse
import hashlib
import json
import os

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def treat_empty_folders(path):
    folders = os.listdir(path)

    for folder in folders:
        images_qty = len(os.listdir(f"{path}/{folder}"))

        if images_qty == 0:
            os.rmdir(f"{path}/{folder}")
            print(f"removing {path}/{folder}")


def get_spot_id(annotation):
    rotated_rect = annotation["best_rotated_rect"]
    flat = [round(x, 4) for item in rotated_rect for x in (item if isinstance(item, list) else [item])]

    rect_string = ','.join(map(str, flat))
    return hashlib.sha256(rect_string.encode()).hexdigest()[:10]

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


def crop_image(cv_image, dest_folder, crop_info):
    annotations, date, time  = crop_info

    for annotation in annotations:
        if annotation["car_id"] != -1 and annotation["car_id"] != -2:
            car_save_path = f"{date}/{date}_{time}#{annotation['image_id']}#{get_spot_id(annotation)}#{annotation['car_id']}"
            rotated_rect = annotation["best_rotated_rect"]
        
            cropped = crop_rect(cv_image, rotated_rect)

            cv.imwrite(f"{dest_folder}/{car_save_path}.jpg", cropped)


def get_images(source_path, dest_folder, artifacts):
    images, annotations = artifacts

    for image in tqdm(images, desc="Cropping images", unit="img"):
        cv_image = cv.imread(f"{source_path}/{image['file_name']}")

        date = '-'.join([str(item).zfill(2) if isinstance(item, int) else str(item) for item in image["date"]])
        time = '_'.join([str(item).zfill(2) if isinstance(item, int) else str(item) for item in image["time"]])

        os.makedirs(f"{dest_folder}/{date}", exist_ok=True)

        crop_info = (annotations[image["id"]], date, time)
        crop_image(cv_image, dest_folder, crop_info)


def get_crop_artifacts(json_path):
    with open(json_path, "r") as file:
        data = json.load(file)

    annotations = defaultdict(list)
    for annotation in data["annotations"]:
        annotations[annotation["image_id"]].append(annotation)

    return data["images"], dict(annotations)


def crop_plds(args):
    jsons_groups = [["qridr_spots.json"], ["vxusd_spots.json", "vmlix_spots.json"], ["isshk_spots.json"]]

    for json_group in jsons_groups:
        subsets = [file.split("_")[0].upper() for file in json_group]
        final_subset = "_".join(subsets)

        dest_folder = f"{args.dest}/{args.name}/{final_subset}"
        print(f"Cropping subset {final_subset}...")
        os.makedirs(dest_folder, exist_ok=True)

        for file in json_group:
            artifacts = get_crop_artifacts(f"{args.source}/{file}")
            get_images("/".join(args.source.split("/")[:-1]), dest_folder, artifacts)
        
        print("Deleting empty folders...")
        treat_empty_folders(dest_folder)


def generic_crop(args):
    jsons = [file for file in os.listdir(args.source) if ".json" in file]
    for file in jsons:
        subset = file.split("_")[0].upper()
        dest_folder = f"{args.dest}/{args.name}/{subset}"

        print(f"Cropping subset {subset}...")
        os.makedirs(dest_folder, exist_ok=True)

        artifacts = get_crop_artifacts(f"{args.source}/{file}")
        get_images("/".join(args.source.split("/")[:-1]), dest_folder, artifacts)

        print("Deleting empty folders...")
        treat_empty_folders(dest_folder)


def main(args):
    assert os.path.exists(args.source), "invalid source dataset"

    print(f"Generating dataset for siamese network training {args.name} in {args.dest}/{args.name}")
    os.makedirs(f"{args.dest}/{args.name}", exist_ok=True)

    if "PLds" in args.source:
        crop_plds(args)
    else:
        generic_crop(args)

    print(f"Dataset {args.name} completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--source", "-s", type=str, required=True)
    parser.add_argument("--dest", "-d", type=str, required=True)
    parser.add_argument("--name", "-n", type=str, required=True)

    main(parser.parse_args())

# Generate PKLot dataset: 
# python3 generate_dataset_siamese.py -s /home/tteuz/Desktop/mestrado/meta_pklot/datasets/organized/PKLot -d /home/tteuz/Desktop/mestrado/meta_pklot/datasets/siamese -n PKLot

# Generate CNRPark-EXT dataset: 
# python3 generate_dataset_siamese.py -s /home/tteuz/Desktop/mestrado/meta_pklot/datasets/organized/CNRPark-EXT -d /home/tteuz/Desktop/mestrado/meta_pklot/datasets/siamese -n CNRPark-EXT

# Generate PLds dataset: 
# python3 generate_dataset_siamese.py -s /home/tteuz/Desktop/mestrado/meta_pklot/datasets/organized/PLds -d /home/tteuz/Desktop/mestrado/meta_pklot/datasets/siamese -n PLds