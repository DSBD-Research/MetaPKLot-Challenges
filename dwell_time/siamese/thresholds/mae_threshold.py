import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'));

from pipeline.main import get_test_artifacts, get_gt, get_spots, compare_cars, get_metrics
from pipeline.dataset.data_loader import data_loader
from datetime import date, datetime

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

class mae_threshold():
    def __init__(self, dates, config):
        self.origin_dataset = [dataset_path.replace("siamese", "organized") for dataset_path in config["train"]]
        self.dates = dates

        self.mae_datasets = {}
        self.__prepare_mae_dataset()


    def __get_image_info(self, image, annotations):
        spots_ids = [annotation["spot_id"] for annotation in annotations[image["id"]]]
        spots_imgs = get_spots(image, annotations[image["id"]])
        spots_labels = [annotation["category_id"] for annotation in annotations[image["id"]]]

        return {spot_id: {"label": spot_label, "img": spot_img} for spot_id, spot_label, spot_img in zip(spots_ids, spots_labels, spots_imgs)}


    def __get_prepared_images(self, images, annotations):
        prepared_images = []

        for image in images:
            image["spots"] = self.__get_image_info(image, annotations)
            prepared_images.append(image)
        
        return prepared_images


    def __prepare_mae_dataset(self):
        jsons = [os.path.join(path, json_file) for path in self.origin_dataset for json_file in os.listdir(path) if ".json" in json_file]

        artifacts = {}
        for json_file in jsons:
            ds_path, file = "/".join(json_file.split("/")[:-1]), json_file.split("/")[-1]
            artifacts[file.split("_")[0].upper()] = get_test_artifacts(ds_path, file)
        
        for subset, values in artifacts.items():
            subset_images, annotations = values
            subset_dates = self.dates[subset]

            images = [image for image in subset_images if date(*image["date"]) in subset_dates]
            gt = get_gt(images, annotations)
            prepared_images = self.__get_prepared_images(images, annotations)

            self.mae_datasets[subset] = {
                "images": prepared_images,
                "annotations": annotations,
                "gt": gt
            }
    

    def __get_predicted_dwell_time(self, siamese_set, ds_loader, images, annotations, subset):
        all_spots_ids = set([annotation["spot_id"] for image in images for annotation in annotations[image["id"]]])
        dwell_time = {spot_id: {"actual": [], "preds": []} for spot_id in all_spots_ids}

        # Getting the info for the first image
        previous_img = images[0]["spots"]
        img_date = datetime(*images[0]["date"], *images[0]["time"])

        for spot_id, spot_info in previous_img.items():
            if spot_info["label"] == 1:
                dwell_time[spot_id]["actual"].append(img_date)

        # Going thru the other images
        for image in images[1:]:
            img_info = image["spots"]
            img_date = datetime(*image["date"], *image["time"])
            
            for spot_id, spot_info in img_info.items():
                if spot_id not in previous_img.keys():
                    if spot_info["label"] == 1:
                        dwell_time[spot_id]["actual"].append(img_date)
                
                elif previous_img[spot_id]["label"] == 1 and spot_info["label"] == 0:
                    if len(dwell_time[spot_id]["actual"]) > 1:
                        dwell_time[spot_id]["preds"].append((min(dwell_time[spot_id]["actual"]), max(dwell_time[spot_id]["actual"])))
                    dwell_time[spot_id]["actual"] = []

                elif previous_img[spot_id]["label"] == 0 and spot_info["label"] == 1:
                    dwell_time[spot_id]["actual"].append(img_date)

                elif previous_img[spot_id]["label"] == 1 and spot_info["label"] == 1:
                    same_car = compare_cars(siamese_set, ds_loader, spot_info, previous_img[spot_id])

                    if same_car:
                        dwell_time[spot_id]["actual"].append(img_date)
                    else:
                        if len(dwell_time[spot_id]["actual"]) > 1:
                            dwell_time[spot_id]["preds"].append((min(dwell_time[spot_id]["actual"]), max(dwell_time[spot_id]["actual"])))
                        dwell_time[spot_id]["actual"] = [img_date]
            
            previous_img = img_info
        
        for spot_id, spot_info in dwell_time.items():
            if len(spot_info["actual"]) > 1:
                dwell_time[spot_id]["preds"].append((min(dwell_time[spot_id]["actual"]), max(dwell_time[spot_id]["actual"])))
                dwell_time[spot_id]["actual"] = []

        return dwell_time


    def get_threshold(self, siamese_model, img_size, num_workers):
        ds_loader = data_loader(img_size, num_workers)
        maes = {}

        for threshold in THRESHOLDS:
            siamese_set = (siamese_model, threshold)
            total_gt, total_predicted = {}, {}

            for subset, values in self.mae_datasets.items():
                images, annotations, gt = values["images"], values["annotations"], values["gt"]

                predicted = self.__get_predicted_dwell_time(siamese_set, ds_loader, images, annotations, subset)
                total_gt.update(gt), total_predicted.update(predicted)

            mae, _, _ = get_metrics(total_gt, total_predicted)
            maes[threshold] = mae
        
        return min(maes, key=maes.get)