import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ensemble.models.ensemble import ensemble_model
from dataset.data_loader import data_loader
from collections import defaultdict
from tools.logger import logger
from datetime import datetime
import numpy as np
import cv2 as cv
import importlib
import argparse
import hashlib
import torch
import json
import uuid
import time

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


def get_car_id(car_id, date):
    return hashlib.sha256(f"{car_id},{','.join(map(str, date))}".encode()).hexdigest()[:10]


def get_spot_id(rotated_rect, date):
    flat = [round(x, 4) for item in rotated_rect for x in (item if isinstance(item, list) else [item])]
    combined = flat + date

    combined_string = ','.join(map(str, combined))
    return hashlib.sha256(combined_string.encode()).hexdigest()[:10]


def get_models(args, classif_module, classif_config, siamese_module, siamese_config):
    classif_device = classif_config["config"]["device"]

    classif_path = os.path.join(classif_config["path"], args.classif_ensemble)
    classif_state_dicts = [torch.load(os.path.join(classif_path, model), map_location=classif_device) for model in os.listdir(classif_path)]
    classif_ensemble = ensemble_model(classif_module, classif_state_dicts, classif_config["config"])

    siamese_path = siamese_config["path"]
    siamese_model = siamese_module.model(pre_trained=False, config=siamese_config["config"])
    siamese_model.load_state_dict(torch.load(os.path.join(siamese_path, f"{args.siamese_model}.pt"), map_location=siamese_config["config"]["device"]))

    with open(os.path.join(siamese_path, f"{args.siamese_model}.json"), "r") as f:
        siamese_threshold = json.load(f)["metrics"]["threshold"]

    return (classif_ensemble, 0.5), (siamese_model, siamese_threshold)

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def compare_spots(siamese_set, ds_loader, actual_spots, previous_spots):
    siamese_model, siamese_threshold = siamese_set
    siamese_loader = ds_loader.get_siamese_loader(actual_spots, previous_spots)

    _, _ = siamese_model.predict(siamese_loader, siamese_threshold)


def classify_spots(classif_set, ds_loader, spots_imgs):
    classif_ensemble, classif_threshold = classif_set
    classif_loader = ds_loader.get_classif_loader(spots_imgs)

    _, _ = classif_ensemble.predict(classif_loader, classif_threshold)


def crop_image(image, annotations):
    cv_image = cv.imread(image['file_name'])
    return [crop_rect(cv_image, ann["best_rotated_rect"]) for ann in annotations]


def get_test_artifacts(ds_path, file):
    with open(f"{ds_path}/{file}", "r") as json_file:
        data = json.load(json_file)

    id_to_date = {img["id"]: img["date"] for img in data["images"]}
    annotations = defaultdict(list)

    for ann in data["annotations"]:
        if ann["car_id"] == -2:
            ann["category_id"] = 0
        
        date = id_to_date[ann["image_id"]]
        ann["spot_id"] = get_spot_id(ann["best_rotated_rect"], date)
        ann["car_id"] = get_car_id(ann["car_id"], date) if ann["category_id"] == 1 else -1
        annotations[ann["image_id"]].append(ann)

    for img in data["images"]:
        img["file_name"] = f"{"/".join(ds_path.split("/")[:-1])}/{img['file_name']}"

    images = sorted(data["images"], key=lambda x: datetime(*x["date"], *x["time"]))

    return images, dict(annotations)


def execute(config, classif_module, siamese_module, args):
    output_name = f"test_{config['dataset']['path'].split('/')[-1].lower()}_{str(uuid.uuid4()).replace('-', '')[:5]}"
    exp_logger = logger(f"_results/{args.save}", output_name)

    classif_config, siamese_config = config["models"]["classification"], config["models"]["siamese"]

    exp_logger.add(["experiment", "subset"], config["dataset"]["subset"])
    exp_logger.add(["experiment", "models", "classification"], {"model": args.classif_ensemble, "module": classif_config["module"]})
    exp_logger.add(["experiment", "models", "siamese"], {"model": args.siamese_model, "module": siamese_config["module"]})

    classif_set, siamese_set = get_models(args, classif_module, classif_config, siamese_module, siamese_config)    
    ds_loader = data_loader(config["dataset"]["img_size"], config["experiment"]["num_workers"])

    jsons = [file for file in os.listdir(config["dataset"]["path"]) if ".json" in file]
    subset_json = next((json for json in jsons if config["dataset"]["subset"] in json), None)
    images, annotations = get_test_artifacts(config["dataset"]["path"], subset_json)

    previous_spots = crop_image(images[0], annotations[images[0]["id"]])
    total_spots, crop_times, classify_times, compare_times = 0, [], [], []

    start_global_time = time.time()
    for image in images[1:]:
        start_crop = time.time()
        actual_spots = crop_image(image, annotations[image["id"]])
        end_crop = time.time()

        start_classify = time.time()
        classify_spots(classif_set, ds_loader, actual_spots)
        end_classify = time.time()

        start_compare = time.time()
        compare_spots(siamese_set, ds_loader, actual_spots, previous_spots)
        end_compare = time.time()

        spots_qty = len(actual_spots)
        crop_times.append((end_crop - start_crop) / spots_qty)
        classify_times.append((end_classify - start_classify) / spots_qty)
        compare_times.append((end_compare - start_compare) / spots_qty)

        total_spots += spots_qty
        previous_spots = actual_spots
    
    end_global_time = time.time()

    exp_logger.add(["times", "global"], ((end_global_time - start_global_time) / total_spots))
    exp_logger.add(["times", "crop"], {"avg": np.average(crop_times), "std": np.std(crop_times)})
    exp_logger.add(["times", "classify"], {"avg": np.average(classify_times), "std": np.std(classify_times)})
    exp_logger.add(["times", "compare"], {"avg": np.average(compare_times), "std": np.std(compare_times)})

    print("Saving result file")
    exp_logger.save()


def main(args):
    assert os.path.exists(args.config), "Invalid config"

    config = json.load(open(args.config, "r"))
    classif_module = importlib.import_module(config["models"]["classification"]["module"])
    siamese_module = importlib.import_module(config["models"]["siamese"]["module"])

    execute(config, classif_module, siamese_module, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dwell time pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--classif_ensemble", "-ce", type=str, required=True)
    parser.add_argument("--siamese_model", "-sm", type=str, required=True)
    parser.add_argument("--save", "-s", type=str, required=True)

    main(parser.parse_args())
