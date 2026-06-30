from datetime import datetime, timedelta
from collections import defaultdict
import subprocess
import argparse
import json
import os

def get_gt_first_way(ds_path):
    ground_truth = {}

    for file in filter(lambda f: f.endswith(".json"), os.listdir(ds_path)):
        with open(os.path.join(ds_path, file), "r") as json_file:
            data = json.load(json_file)

        annotations = defaultdict(list)
        for annotation in data["annotations"]:
            if annotation["category_id"] == 1 and annotation["car_id"] != -2:
                annotations[annotation["image_id"]].append(annotation)
        images = sorted(data["images"], key=lambda x: datetime(*x["date"], *x["time"]))

        cars_time = defaultdict(list)
        for image in images:
            date = datetime(*image["date"], *image["time"])

            for annotation in annotations[image["id"]]:
                cars_time[annotation["car_id"]].append(date)

        subset = file.split("_")[0].upper()
        ground_truth[subset] = {car_id: max(times) - min(times) for car_id, times in cars_time.items()}

    return ground_truth


def get_gt_second_way(ds_path):
    ground_truth = {}

    for file in filter(lambda f: f.endswith(".json"), os.listdir(ds_path)):
        with open(os.path.join(ds_path, file), "r") as json_file:
            data = json.load(json_file)

        annotations = defaultdict(list)
        for annotation in data["annotations"]:
            if annotation["category_id"] == 1 and annotation["car_id"] != -2:
                annotations[annotation["image_id"]].append(annotation)
        images = sorted(data["images"], key=lambda x: datetime(*x["date"], *x["time"]))
 
        previous_image = (datetime(*images[0]["date"], *images[0]["time"]), [annotation["car_id"] for annotation in annotations[images[0]["id"]]])
        cars_time = {car_id: timedelta(minutes=0) for car_id in previous_image[1]}
        
        for image in images[1:]:
            date = datetime(*image["date"], *image["time"])
            cars_in_image = [annotation["car_id"] for annotation in annotations[image["id"]]]

            for car in cars_in_image:
                if car not in cars_time.keys():
                    cars_time[car] = timedelta(minutes=0)

                if car in previous_image[1]:
                    cars_time[car] += date - previous_image[0]

            previous_image = (date, cars_in_image)
        
        subset = file.split("_")[0].upper()
        ground_truth[subset] = cars_time
    
    return ground_truth


def main(args):
    first_gt, second_gt = get_gt_first_way(args.dataset), get_gt_second_way(args.dataset)
    first_gt_in_min, second_gt_in_min = {}, {}
    gt_diff = {}

    for subset, cars in first_gt.items():
        first_gt_in_min[subset] = {key: (value.total_seconds() / 60) for key, value in cars.items()}
    
    for subset, cars in second_gt.items():
        second_gt_in_min[subset] = {key: (value.total_seconds() / 60) for key, value in cars.items()}

    for subset in first_gt_in_min.keys():
        first_way_cars, second_way_cars = first_gt_in_min[subset], second_gt_in_min[subset]
        diff_cars = {}

        gt_diff[subset] = {}
        gt_diff[subset]["cars_qty"] = { "first_way": len(first_way_cars), "second_way": len(second_way_cars) }

        for car in first_way_cars:
            if first_way_cars[car] != second_way_cars[car]:
                command = f"find /home/tteuz/Desktop/mestrado/datasets/siamese/{args.dataset.split('/')[-1]} -type f -iname '*#{car}.jpg*'"
                result = subprocess.run(command, capture_output=True, text=True, shell=True)

                details = []
                images = [image for image in result.stdout.split("\n")[:-1]]
                for image in images:
                    details.append({"img_path": image, "img_id": image.split("/")[-1].split("#")[-2], "day": image.split("/")[-2]})

                days = [image.split("/")[-2] for image in result.stdout.split("\n")[:-1]]
                days = set(days)

                diff_cars[car] = {"first_way": first_way_cars[car], "second_way": second_way_cars[car], "days": list(days), "details": details}
        
        gt_diff[subset]["cars"] = diff_cars
    
    with open(f"{args.dataset.split('/')[-1]}.json", "w") as file:
        json.dump(gt_diff, file, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GT check", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataset", "-d", type=str, required=True)

    main(parser.parse_args())

# python3 siamese_gt_diffs.py -d /home/tteuz/Desktop/mestrado/datasets/organized/PKLot

# python3 siamese_gt_diffs.py -d /home/tteuz/Desktop/mestrado/datasets/organized/CNRParkEXT

# python3 siamese_gt_diffs.py -d /home/tteuz/Desktop/mestrado/datasets/organized/PLds