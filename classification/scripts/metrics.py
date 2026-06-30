import sys
import os
import numpy as np
import json
from statistics import mean, stdev
from datetime import datetime

CAMERAS = ["ufpr04", "ufpr05", "pucpr", "camera1", "camera2", "camera3", "camera4", "camera5", "camera6", "camera7", "camera8", "camera9", "isshk", "vxusd,vmlix", "qridr"]

def main(results_folder, distill_n):
    n_executions = len(os.listdir(results_folder))

    macro_f1_ensemble = []
    macro_f1_distill = []
    macro_f1_distill_general = []
    accuracies_ensemble = []
    accuracies_distill = []
    accuracies_general = []

    total_samples_ensemble = []
    total_samples_distill = []
    total_samples_general = []

    for exec_id in range(1, n_executions + 1):

        ###### ENSEMBLE ######
        distill_configs_json = json.load(open(f"{results_folder}execution{exec_id}/distill/distill{distill_n}/distill_configs.json"))
        train_val_days = distill_configs_json['train_days'] + distill_configs_json['val_days']

        ensemble_days = train_val_days # Used for general
        distill_days_per_camera = {}

        tests_json_paths = os.listdir(f"{results_folder}execution{exec_id}/test/jsons")

        image_path_to_preds_and_target = {}
        cameras = set()
        for test_json_path in tests_json_paths:
            json_file = json.load(open(f"{results_folder}execution{exec_id}/test/jsons/{test_json_path}"))

            for i in range(len(json_file['predictions'])):
                pred = json_file['predictions'][i]
                target = json_file['labels'][i]
                im_path =  json_file['images_paths'][i]

                if im_path not in image_path_to_preds_and_target:
                    image_path_to_preds_and_target[str(im_path)] = {}
                    image_path_to_preds_and_target[str(im_path)]['preds'] = []
                    image_path_to_preds_and_target[str(im_path)]['target'] = target

                image_path_to_preds_and_target[str(im_path)]['preds'].append(pred)

                cameras.add(json_file['images_paths'][i].split("/")[-4])

        for im_path_key in image_path_to_preds_and_target.keys():
            image_path_to_preds_and_target[str(im_path_key)]['preds'] = mean(image_path_to_preds_and_target[str(im_path_key)]['preds'])

        cameras = list(cameras)
        dates_per_camera = {}
        for im_path in image_path_to_preds_and_target.keys():
            camera = im_path.split("/")[-4]
            date = im_path.split("/")[-3]

            if camera not in dates_per_camera:
                dates_per_camera[str(camera)] = set()
            
            dates_per_camera[str(camera)].add(date)

        for camera in dates_per_camera.keys():
            distill_days_per_camera[str(camera)] = len(dates_per_camera[str(camera)]) - ensemble_days # Used for general

            dates_per_camera[str(camera)] = sorted(dates_per_camera[str(camera)], key=lambda date: datetime.strptime(date, "%Y-%m-%d"))
            dates_per_camera[str(camera)] = dates_per_camera[str(camera)][:train_val_days]

        total_per_subset_ensemble = {}
        correct_per_subset_ensemble = {}
        tp_per_subset_class1_ensemble = {}
        fp_per_subset_class1_ensemble = {}
        fn_per_subset_class1_ensemble = {}
        tp_per_subset_class0_ensemble = {}
        fp_per_subset_class0_ensemble = {}
        fn_per_subset_class0_ensemble = {}
        for im_path in image_path_to_preds_and_target.keys():
            camera = im_path.split("/")[-4]
            if im_path.split("/")[-3] not in dates_per_camera[str(camera)]:
                continue

            if camera not in total_per_subset_ensemble:
                total_per_subset_ensemble[str(camera)] = 0
                correct_per_subset_ensemble[str(camera)] = 0
                tp_per_subset_class1_ensemble[str(camera)] = 0
                fp_per_subset_class1_ensemble[str(camera)] = 0
                fn_per_subset_class1_ensemble[str(camera)] = 0
                tp_per_subset_class0_ensemble[str(camera)] = 0
                fp_per_subset_class0_ensemble[str(camera)] = 0
                fn_per_subset_class0_ensemble[str(camera)] = 0

            pred = image_path_to_preds_and_target[str(im_path)]['preds']
            target = image_path_to_preds_and_target[str(im_path)]['target']

            total_per_subset_ensemble[str(camera)] += 1

            if target == 1:
                if pred >= 0.5:
                    correct_per_subset_ensemble[str(camera)] += 1
                    tp_per_subset_class1_ensemble[str(camera)] += 1
                elif pred < 0.5:
                    fn_per_subset_class1_ensemble[str(camera)] += 1
                    fp_per_subset_class0_ensemble[str(camera)] += 1
            elif target == 0:
                if pred < 0.5:
                    correct_per_subset_ensemble[str(camera)] += 1
                    tp_per_subset_class0_ensemble[str(camera)] += 1
                elif pred >= 0.5:
                    fn_per_subset_class0_ensemble[str(camera)] += 1
                    fp_per_subset_class1_ensemble[str(camera)] += 1
        
        macro_f1_per_camera = {}
        accuracy_per_camera = {}
        for camera in total_per_subset_ensemble.keys():
            precision_class1 = tp_per_subset_class1_ensemble[str(camera)] / (tp_per_subset_class1_ensemble[str(camera)] + fp_per_subset_class1_ensemble[str(camera)])
            recall_class1 = tp_per_subset_class1_ensemble[str(camera)] / (tp_per_subset_class1_ensemble[str(camera)] + fn_per_subset_class1_ensemble[str(camera)])
            f1_class1 = 2 * (1 / ((1 / precision_class1) + (1 / recall_class1)))

            precision_class0 = tp_per_subset_class0_ensemble[str(camera)] / (tp_per_subset_class0_ensemble[str(camera)] + fp_per_subset_class0_ensemble[str(camera)])
            recall_class0 = tp_per_subset_class0_ensemble[str(camera)] / (tp_per_subset_class0_ensemble[str(camera)] + fn_per_subset_class0_ensemble[str(camera)])
            f1_class0 = 2 * (1 / ((1 / precision_class0) + (1 / recall_class0)))

            macro_f1_per_camera[camera.lower()] = (f1_class1 + f1_class0) / 2
            accuracy_per_camera[camera.lower()] = correct_per_subset_ensemble[str(camera)] / total_per_subset_ensemble[str(camera)]

        macro_f1_ensemble.append(macro_f1_per_camera)
        accuracies_ensemble.append(accuracy_per_camera)
        total_samples_ensemble.append(total_per_subset_ensemble)

        ###### DISTILL ######
        jsons_paths = os.listdir(f"{results_folder}execution{exec_id}/distill/distill{distill_n}/jsons")
        thresholds = json.load(open(f"{results_folder}execution{exec_id}/distill/distill{distill_n}/thresholds.json"))

        total_per_subset_distill = {}
        correct_per_subset_distill = {}
        tp_per_subset_class1_distill = {}
        fp_per_subset_class1_distill = {}
        fn_per_subset_class1_distill = {}
        tp_per_subset_class0_distill = {}
        fp_per_subset_class0_distill = {}
        fn_per_subset_class0_distill = {}
        for json_path in jsons_paths:
            if "student" in json_path:
                continue

            json_file = json.load(open(f"{results_folder}execution{exec_id}/distill/distill{distill_n}/jsons/{json_path}"))

            th = get_threshold(json_path, thresholds)

            for i in range(len(json_file["predictions"])):
                pred = json_file["predictions"][i]
                target = json_file["labels"][i]
                camera = json_path.split("/")[-1].split(".")[0]

                if camera not in total_per_subset_distill:
                    total_per_subset_distill[str(camera)] = 0
                    correct_per_subset_distill[str(camera)] = 0
                    tp_per_subset_class1_distill[str(camera)] = 0
                    fp_per_subset_class1_distill[str(camera)] = 0
                    fn_per_subset_class1_distill[str(camera)] = 0
                    tp_per_subset_class0_distill[str(camera)] = 0
                    fp_per_subset_class0_distill[str(camera)] = 0
                    fn_per_subset_class0_distill[str(camera)] = 0

                total_per_subset_distill[str(camera)] += 1

                if target == 1:
                    if pred >= th:
                        correct_per_subset_distill[str(camera)] += 1
                        tp_per_subset_class1_distill[str(camera)] += 1
                    elif pred < th:
                        fn_per_subset_class1_distill[str(camera)] += 1
                        fp_per_subset_class0_distill[str(camera)] += 1
                elif target == 0:
                    if pred < th:
                        correct_per_subset_distill[str(camera)] += 1
                        tp_per_subset_class0_distill[str(camera)] += 1
                    elif pred >= th:
                        fn_per_subset_class0_distill[str(camera)] += 1
                        fp_per_subset_class1_distill[str(camera)] += 1

        macro_f1_per_camera = {}
        accuracy_per_camera = {}
        for camera in total_per_subset_distill.keys():
            precision_class1 = tp_per_subset_class1_distill[str(camera)] / (tp_per_subset_class1_distill[str(camera)] + fp_per_subset_class1_distill[str(camera)])
            recall_class1 = tp_per_subset_class1_distill[str(camera)] / (tp_per_subset_class1_distill[str(camera)] + fn_per_subset_class1_distill[str(camera)])
            f1_class1 = 2 * (1 / ((1 / precision_class1) + (1 / recall_class1)))

            precision_class0 = tp_per_subset_class0_distill[str(camera)] / (tp_per_subset_class0_distill[str(camera)] + fp_per_subset_class0_distill[str(camera)])
            recall_class0 = tp_per_subset_class0_distill[str(camera)] / (tp_per_subset_class0_distill[str(camera)] + fn_per_subset_class0_distill[str(camera)])
            f1_class0 = 2 * (1 / ((1 / precision_class0) + (1 / recall_class0)))

            macro_f1_per_camera[camera.lower()] = (f1_class1 + f1_class0) / 2
            accuracy_per_camera[camera.lower()] = correct_per_subset_distill[str(camera)] / total_per_subset_distill[str(camera)]

        macro_f1_distill.append(macro_f1_per_camera)
        accuracies_distill.append(accuracy_per_camera)
        total_samples_distill.append(total_per_subset_distill)

        ###### GENERAL ######
        macro_f1_per_camera = {}
        accuracy_per_camera = {}
        tp_per_subset_class1_general = {}
        fp_per_subset_class1_general = {}
        fn_per_subset_class1_general = {}
        tp_per_subset_class0_general = {}
        fp_per_subset_class0_general = {}
        fn_per_subset_class0_general = {}
        total_per_subset_general = {}
        for camera in total_per_subset_distill.keys():
            tp_per_subset_class1_general[str(camera)] = tp_per_subset_class1_ensemble[str(camera)] + tp_per_subset_class1_distill[str(camera)]
            fp_per_subset_class1_general[str(camera)] = fp_per_subset_class1_ensemble[str(camera)] + fp_per_subset_class1_distill[str(camera)]
            fn_per_subset_class1_general[str(camera)] = fn_per_subset_class1_ensemble[str(camera)] + fn_per_subset_class1_distill[str(camera)]
            tp_per_subset_class0_general[str(camera)] = tp_per_subset_class0_ensemble[str(camera)] + tp_per_subset_class0_distill[str(camera)]
            fp_per_subset_class0_general[str(camera)] = fp_per_subset_class0_ensemble[str(camera)] + fp_per_subset_class0_distill[str(camera)]
            fn_per_subset_class0_general[str(camera)] = fn_per_subset_class0_ensemble[str(camera)] + fn_per_subset_class0_distill[str(camera)]
            total_per_subset_general[str(camera)] = total_per_subset_ensemble[str(camera)] + total_per_subset_distill[str(camera)]
        
            precision_class1 = tp_per_subset_class1_general[str(camera)] / (tp_per_subset_class1_general[str(camera)] + fp_per_subset_class1_general[str(camera)])
            recall_class1 = tp_per_subset_class1_general[str(camera)] / (tp_per_subset_class1_general[str(camera)] + fn_per_subset_class1_general[str(camera)])
            f1_class1 = 2 * (1 / ((1 / precision_class1) + (1 / recall_class1)))

            precision_class0 = tp_per_subset_class0_general[str(camera)] / (tp_per_subset_class0_general[str(camera)] + fp_per_subset_class0_general[str(camera)])
            recall_class0 = tp_per_subset_class0_general[str(camera)] / (tp_per_subset_class0_general[str(camera)] + fn_per_subset_class0_general[str(camera)])
            f1_class0 = 2 * (1 / ((1 / precision_class0) + (1 / recall_class0)))

            macro_f1_per_camera[camera.lower()] = (f1_class1 + f1_class0) / 2

            total_general = total_per_subset_ensemble[str(camera)] + total_per_subset_distill[str(camera)]
            correct_general = correct_per_subset_ensemble[str(camera)] + correct_per_subset_distill[str(camera)]
            accuracy_per_camera[camera.lower()] = correct_general / total_general

        macro_f1_distill_general.append(macro_f1_per_camera)
        accuracies_general.append(accuracy_per_camera)
        total_samples_general.append(total_per_subset_general)

    overall_macrof1_ens = 0
    overall_macrof1_dis = 0
    overall_macrof1_gen = 0
    overall_acc_ens = 0
    overall_acc_dis = 0
    overall_acc_gen = 0
    total_samples_ensemble_ = 0
    total_samples_distill_ = 0
    total_samples_general_ = 0
    for camera in CAMERAS:
        cameras = macro_f1_ensemble[0].keys()
        cameras = [c.lower() for c in cameras]
        if camera.lower() not in cameras:
            continue

        macro_f1_ens = [m[camera] for m in macro_f1_ensemble]
        macro_f1_dis = [m[camera] for m in macro_f1_distill]
        macro_f1_gen = [m[camera] for m in macro_f1_distill_general]
        acc_ens = [a[camera] for a in accuracies_ensemble]
        acc_dis = [a[camera] for a in accuracies_distill]
        acc_gen = [a[camera] for a in accuracies_general]

        total_samples_ensemble_camera = mean([t[camera.upper()] for t in total_samples_ensemble])
        total_samples_distill_camera = mean([t[camera.upper()] for t in total_samples_distill])
        total_samples_general_camera = mean([t[camera.upper()] for t in total_samples_general])

        total_samples_ensemble_ += total_samples_ensemble_camera
        total_samples_distill_ += total_samples_distill_camera
        total_samples_general_ += total_samples_general_camera

        mean_macro_f1_ens = mean(macro_f1_ens)
        mean_macro_f1_dis = mean(macro_f1_dis)
        mean_macro_f1_gen = mean(macro_f1_gen)
        mean_acc_ens = mean(acc_ens)
        mean_acc_dis = mean(acc_dis)
        mean_acc_gen = mean(acc_gen)

        overall_macrof1_ens += mean_macro_f1_ens * total_samples_ensemble_camera
        overall_macrof1_dis += mean_macro_f1_dis * total_samples_distill_camera
        overall_macrof1_gen += mean_macro_f1_gen * total_samples_general_camera
        overall_acc_ens += mean_acc_ens * total_samples_ensemble_camera
        overall_acc_dis += mean_acc_dis * total_samples_distill_camera
        overall_acc_gen += mean_acc_gen * total_samples_general_camera

        print(f"Camera: {camera}")
        print(f"Ensemble Macro f1: {mean_macro_f1_ens:.3f} ± {stdev(macro_f1_ens):.3f}, Accuracy: {mean_acc_ens:.3f} ± {stdev(acc_ens):.3f}")
        print(f"Distill Macro f1: {mean_macro_f1_dis:.3f} ± {stdev(macro_f1_dis):.3f}, Accuracy: {mean_acc_dis:.3f} ± {stdev(acc_dis):.3f}")
        print(f"General Macro f1: {mean_macro_f1_gen:.3f} ± {stdev(macro_f1_gen):.3f}, Accuracy: {mean_acc_gen:.3f} ± {stdev(acc_gen):.3f}")
        print("")

    overall_macrof1_ens /= total_samples_ensemble_
    overall_macrof1_dis /= total_samples_distill_
    overall_macrof1_gen /= total_samples_general_
    overall_acc_ens /= total_samples_ensemble_
    overall_acc_dis /= total_samples_distill_
    overall_acc_gen /= total_samples_general_

    print("Overall")
    print(f"Ensemble Macro f1: {overall_macrof1_ens:.3f}, Accuracy: {overall_acc_ens:.3f}")
    print(f"Distill Macro f1: {overall_macrof1_dis:.3f}, Accuracy: {overall_acc_dis:.3f}")
    print(f"General Macro f1: {overall_macrof1_gen:.3f}, Accuracy: {overall_acc_gen:.3f}")
    
def get_threshold(json_path, thresholds):
    camera = json_path.split("/")[-1].split(".")[0]

    return thresholds[str(camera)]       

if __name__ == "__main__":
    if len (sys.argv) != 3:
        print("Usage: python macro_f1.py <results_folder> <distill_n>")
        sys.exit(1)

    results_folder = sys.argv[1]
    distill_n = int(sys.argv[2])

    if not os.path.exists(results_folder):
        print(f"Error: The folder {results_folder} does not exist.")
        sys.exit(1)

    results_folder += "/" if not results_folder.endswith("/") else ""

    main(results_folder, distill_n)
