import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from ensemble.models.ensemble import ensemble_model
from dataset.data_loader import data_loader
from collections import defaultdict
from tools.logger import logger
from datetime import datetime
from tqdm import tqdm
import cv2 as cv
import importlib
import argparse
import hashlib
import torch
import json
import uuid
import math



# -------------------------------------------------- HELPERS ---------------------------------------------------------

def get_car_id(car_id, date):
    return hashlib.sha256(f"{car_id},{','.join(map(str, date))}".encode()).hexdigest()[:10]


def get_spot_id(rotated_rect, date):
    flat = [round(x, 4) for item in rotated_rect for x in (item if isinstance(item, list) else [item])]
    combined = flat + date

    combined_string = ','.join(map(str, combined))
    return hashlib.sha256(combined_string.encode()).hexdigest()[:10]


def get_duration(pred):
    return (pred[1] - pred[0]).total_seconds() / 60


def get_longest(preds):
    return max(preds, key=lambda x: x[1] - x[0])


def get_intersection(gt, preds):
    gt_start, gt_end = gt

    return [(ps, pe) for ps, pe in preds if ps <= gt_end and pe >= gt_start]


def crop_rect(image, rect):
    center, size, angle = rect[0:3]
    height, width = image.shape[0:2]

    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1)
    spot_rot = cv.warpAffine(image, rotation_matrix, (width, height))
    spot_crop = cv.getRectSubPix(spot_rot, [int(u) for u in size], center)

    if angle >= 45:
        spot_crop = cv.rotate(spot_crop, cv.ROTATE_90_CLOCKWISE)

    return spot_crop


def get_spots(image, annotations):
    cv_image = cv.imread(image['file_name'])
    return [crop_rect(cv_image, ann["best_rotated_rect"]) for ann in annotations]


def get_gt_image_info(annotations):
    return {ann["spot_id"]: ann["car_id"] for ann in annotations}


def get_gt(images, annotations):
    gt = defaultdict(lambda: {"actual": [], "preds": []})

    prev_img_info = get_gt_image_info(annotations[images[0]["id"]])
    prev_date = datetime(*images[0]["date"], *images[0]["time"])

    for spot_id, car_id in prev_img_info.items():
        if car_id != -1:
            gt[spot_id]["actual"].append(prev_date)

    for image in images[1:]:
        cur_img_info = get_gt_image_info(annotations[image["id"]])
        cur_date = datetime(*image["date"], *image["time"])

        for spot_id, cur_car in cur_img_info.items():
            prev_car = prev_img_info.get(spot_id, None)

            if cur_car != -1:
                if prev_car is None or prev_car == -1 or prev_car == cur_car:
                    gt[spot_id]["actual"].append(cur_date)
                else:
                    if len(gt[spot_id]["actual"]) > 1:
                        gt[spot_id]["preds"].append((min(gt[spot_id]["actual"]), max(gt[spot_id]["actual"])))
                    gt[spot_id]["actual"] = [cur_date]

            elif prev_car is not None and prev_car != -1:
                if len(gt[spot_id]["actual"]) > 1:
                    gt[spot_id]["preds"].append((min(gt[spot_id]["actual"]), max(gt[spot_id]["actual"])))
                gt[spot_id]["actual"].clear()

        prev_img_info = cur_img_info

    for info in gt.values():
        if len(info["actual"]) > 1:
            info["preds"].append((min(info["actual"]), max(info["actual"])))
        info["actual"].clear()

    return {k: sorted(v["preds"], key=lambda x: x[0]) for k, v in gt.items()}


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
        img["file_name"] = f"{ds_path}/{img['file_name']}"

    images = sorted(data["images"], key=lambda x: datetime(*x["date"], *x["time"]))
    return images, dict(annotations)


def get_models(args, classif_module, classif_config, siamese_module, siamese_config):
    return (None, None), (None, None)

# -------------------------------------------------- HELPERS ---------------------------------------------------------

def get_metrics(gt, predicted):
    n = total_cars = sum(len(v) for v in gt.values())
    mae = rsme = acc = 0

    for spot_id, gt_times in gt.items():
        pred_list = predicted.get(spot_id, {}).get("preds", [])

        for time in gt_times:
            intersections = get_intersection(time, pred_list)

            if intersections:
                longest = get_longest(intersections)
                pred = get_duration(longest)

                if longest != time:
                    print(spot_id, longest, time, end="\n")
                
                pred_list.remove(longest)
            else:
                pred = 0

            gt_duration = get_duration(time)
            error = gt_duration - pred

            mae += abs(error)
            rsme += math.pow((error), 2)

            if error == 0:
                acc += 1
        
    print("\n")
    for spot_id, values in predicted.items():
        if len(values["preds"]) > 0:
            print(spot_id, values)

    for values in predicted.values():
        remaining_preds = values["preds"]
        n += len(remaining_preds)

        for pred in remaining_preds:
            pred = get_duration(pred)

            mae += pred
            rsme += math.pow(pred, 2)

    return (mae / n), math.sqrt(rsme / n), (acc / total_cars)


def compare_cars(siamese_set, ds_loader, actual, previous):
    return actual["img"] == previous["img"]


def classify_spots(classif_set, ds_loader, spots_imgs):
    classif_ensemble, classif_threshold = classif_set
    classif_loader = ds_loader.get_classif_loader(spots_imgs)

    preds, _ = classif_ensemble.predict(classif_loader, classif_threshold)
    return [int(pred) for pred in preds]


def get_image_info(image, annotations, classif_set, ds_loader, using_classification):
    anns = annotations[image["id"]]
    spot_ids = [ann["spot_id"] for ann in anns]
    imgs = [ann["car_id"] for ann in anns]

    if using_classification:
        labels = classify_spots(classif_set, ds_loader, imgs)
    else:
        labels = [ann["category_id"] for ann in anns]

    return {spot_id: {"label": spot_label, "img": spot_img} for spot_id, spot_label, spot_img in zip(spot_ids, labels, imgs)}


def get_predicted_dwell_time(config, classif_set, siamese_set, ds_loader, images, annotations, subset):
    dwell_time = defaultdict(lambda: {"actual": [], "preds": []})
    using_classification = config["experiment"]["using_classification"]

    prev_img = get_image_info(images[0], annotations, classif_set, ds_loader, using_classification)
    prev_date = datetime(*images[0]["date"], *images[0]["time"])

    for spot_id, info in prev_img.items():
        if info["label"] == 1:
            dwell_time[spot_id]["actual"].append(prev_date)
    
    for image in tqdm(images[1:], desc=f"Computing dwell time for {subset}", unit="img"):
        cur_img = get_image_info(image, annotations, classif_set, ds_loader, using_classification)
        cur_date = datetime(*image["date"], *image["time"])

        for spot_id, cur_info in cur_img.items():
            prev_info = prev_img.get(spot_id, None)

            if prev_info is None:
                if cur_info["label"] == 1:
                    dwell_time[spot_id]["actual"].append(cur_date)

            else:
                prev_label = prev_info["label"]
                cur_label = cur_info["label"]

                if prev_label == 0 and cur_label == 1:
                    dwell_time[spot_id]["actual"].append(cur_date)
                
                elif prev_label == 1 and cur_label == 0:
                    if len(dwell_time[spot_id]["actual"]) > 1:
                        dwell_time[spot_id]["preds"].append((min(dwell_time[spot_id]["actual"]), max(dwell_time[spot_id]["actual"])))
                    dwell_time[spot_id]["actual"].clear()
                
                elif prev_label == 1 and cur_label == 1:
                    if compare_cars(siamese_set, ds_loader, cur_info, prev_info):
                        dwell_time[spot_id]["actual"].append(cur_date)
                    else:
                        if len(dwell_time[spot_id]["actual"]) > 1:
                            dwell_time[spot_id]["preds"].append((min(dwell_time[spot_id]["actual"]), max(dwell_time[spot_id]["actual"])))
                        dwell_time[spot_id]["actual"] = [cur_date]
            
        prev_img = cur_img

    for spot_id, info in dwell_time.items():
        if len(info["actual"]) > 1:
            info["preds"].append((min(info["actual"]), max(info["actual"])))
            info["actual"].clear()

    return dwell_time


def execute(config, classif_module, siamese_module, args):
    output_name = f"test_{config['dataset']['path'].split('/')[-1].lower()}_{str(uuid.uuid4()).replace('-', '')[:5]}"
    exp_logger = logger(f"_results/{args.save}", output_name)

    classif_config, siamese_config = config["models"]["classification"], config["models"]["siamese"]

    exp_logger.add(["experiment", "models", "classification"], {"model": args.classif_model, "module": classif_config["module"]})
    exp_logger.add(["experiment", "models", "siamese"], {"model": args.siamese_model, "module": siamese_config["module"]})
    exp_logger.add(["experiment", "using_classification"], config["experiment"]["using_classification"])

    classif_set, siamese_set = get_models(args, classif_module, classif_config, siamese_module, siamese_config)
    ds_loader = data_loader(config["dataset"]["img_size"], config["experiment"]["num_workers"])

    jsons = [file for file in os.listdir(config["dataset"]["path"]) if ".json" in file]
    for file in jsons:
        subset = file.split("_")[0].upper()
        print(subset)

        images, annotations = get_test_artifacts(config['dataset']['path'], file)
        gt = get_gt(images, annotations)
        
        # for spot_id, values in gt.items():
        #     for interval in values:
        #         if interval[0] == interval[1]:
        #             print(f"Invalid interval for spot {spot_id}: {interval}")

        predicted = get_predicted_dwell_time(config, classif_set, siamese_set, ds_loader, images, annotations, subset)
        
        # for spot_id, values in predicted.items():
        #     for interval in values:
        #         if interval[0] == interval[1]:
        #             print(f"Invalid interval for spot {spot_id}: {interval}")
        
        mae, rsme, accuracy = get_metrics(gt, predicted)
        print(mae, rsme, accuracy)
        print()


def main(args):
    assert os.path.exists(args.config), "Invalid config"
    assert torch.cuda.is_available(), "Cuda unavailable"

    config = json.load(open(args.config, "r"))
    classif_module = importlib.import_module(config["models"]["classification"]["module"])
    siamese_module = importlib.import_module(config["models"]["siamese"]["module"])

    execute(config, classif_module, siamese_module, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dwell time pipeline", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--classif_model", "-cm", type=str, required=True)
    parser.add_argument("--siamese_model", "-sm", type=str, required=True)
    parser.add_argument("--save", "-s", type=str, required=True)

    main(parser.parse_args())