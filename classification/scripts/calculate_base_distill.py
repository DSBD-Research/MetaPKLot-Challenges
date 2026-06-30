import sys
import json
import os
from statistics import mean, stdev

def main(results_folder_path, distill):
    n_executions = len(os.listdir(results_folder_path))

    base_accuracys = []
    distill_accuracys = []
    base_macro_f1s = []
    distill_macro_f1s = []

    for execution in range(1, n_executions + 1):
        jsons_paths = os.listdir(results_folder_path + f"execution{execution}/distill/distill{distill}/jsons")
        thresholds = json.load(open(results_folder_path + f"execution{execution}/distill/distill{distill}/thresholds.json"))

        base_total = {}
        base_correct = {}
        distill_total = {}
        distill_correct = {}

        base_tp_class0 = {}
        base_fp_class0 = {}
        base_fn_class0 = {}
        base_tp_class1 = {}
        base_fp_class1 = {}
        base_fn_class1 = {}
        distill_tp_class0 = {}
        distill_fp_class0 = {}
        distill_fn_class0 = {}
        distill_tp_class1 = {}
        distill_fp_class1 = {}
        distill_fn_class1 = {}

        for json_path in jsons_paths:
            json_file = json.load(open(results_folder_path + f"execution{execution}/distill/distill{distill}/jsons/{json_path}"))

            th = get_threshold(json_path, thresholds)

            for pred, label in zip(json_file["predictions"], json_file["labels"]):

                if "student" in json_path:
                    if json_path not in base_total:
                        base_total[json_path] = 0
                        base_correct[json_path] = 0
                        base_tp_class0[json_path] = 0
                        base_fp_class0[json_path] = 0
                        base_fn_class0[json_path] = 0
                        base_tp_class1[json_path] = 0
                        base_fp_class1[json_path] = 0
                        base_fn_class1[json_path] = 0

                    base_total[json_path] += 1
                    if (pred >= th and label == 1) or (pred < th and label == 0):
                        base_correct[json_path] += 1

                    if pred >= th and label == 1:
                        base_tp_class1[json_path] += 1
                    elif pred >= th and label == 0:
                        base_fp_class1[json_path] += 1
                        base_fn_class0[json_path] += 1
                    elif pred < th and label == 0:
                        base_tp_class0[json_path] += 1
                    elif pred < th and label == 1:
                        base_fp_class0[json_path] += 1
                        base_fn_class1[json_path] += 1

                else:
                    if json_path not in distill_total:
                        distill_total[json_path] = 0
                        distill_correct[json_path] = 0
                        distill_tp_class0[json_path] = 0
                        distill_fp_class0[json_path] = 0
                        distill_fn_class0[json_path] = 0
                        distill_tp_class1[json_path] = 0
                        distill_fp_class1[json_path] = 0
                        distill_fn_class1[json_path] = 0

                    distill_total[json_path] += 1
                    if (pred >= th and label == 1) or (pred < th and label == 0):
                        distill_correct[json_path] += 1

                    if pred >= th and label == 1:
                        distill_tp_class1[json_path] += 1
                    elif pred >= th and label == 0:
                        distill_fp_class1[json_path] += 1
                        distill_fn_class0[json_path] += 1
                    elif pred < th and label == 0:
                        distill_tp_class0[json_path] += 1
                    elif pred < th and label == 1:
                        distill_fp_class0[json_path] += 1
                        distill_fn_class1[json_path] += 1
            
        # Base
        total_base_images = sum(base_total.values())
        accuracy = 0
        macro_f1 = 0
        for json_path in base_total:
            accuracy += (base_correct[json_path] / base_total[json_path]) * base_total[json_path] / total_base_images

            precision_class0 = base_tp_class0[json_path] / (base_tp_class0[json_path] + base_fp_class0[json_path])
            recall_class0 = base_tp_class0[json_path] / (base_tp_class0[json_path] + base_fn_class0[json_path])
            f1_class0 = 2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0)

            precision_class1 = base_tp_class1[json_path] / (base_tp_class1[json_path] + base_fp_class1[json_path])
            recall_class1 = base_tp_class1[json_path] / (base_tp_class1[json_path] + base_fn_class1[json_path])
            f1_class1 = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1)

            macro_f1 += ((f1_class0 + f1_class1) / 2) * base_total[json_path] / total_base_images
        
        base_accuracys.append(accuracy)
        base_macro_f1s.append(macro_f1)

        # Distill
        total_distill_images = sum(distill_total.values())
        accuracy = 0
        macro_f1 = 0
        for json_path in distill_total:
            accuracy += (distill_correct[json_path] / distill_total[json_path]) * distill_total[json_path] / total_distill_images

            precision_class0 = distill_tp_class0[json_path] / (distill_tp_class0[json_path] + distill_fp_class0[json_path])
            recall_class0 = distill_tp_class0[json_path] / (distill_tp_class0[json_path] + distill_fn_class0[json_path])
            f1_class0 = 2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0)

            precision_class1 = distill_tp_class1[json_path] / (distill_tp_class1[json_path] + distill_fp_class1[json_path])
            recall_class1 = distill_tp_class1[json_path] / (distill_tp_class1[json_path] + distill_fn_class1[json_path])
            f1_class1 = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1)

            macro_f1 += ((f1_class0 + f1_class1) / 2) * distill_total[json_path] / total_distill_images
        
        distill_accuracys.append(accuracy)
        distill_macro_f1s.append(macro_f1)

    print("\nBase")
    print(f"Accuracy: {mean(base_accuracys):.3f} ± {stdev(base_accuracys):.3f}")
    print(f"Macro F1: {mean(base_macro_f1s):.3f} ± {stdev(base_macro_f1s):.3f}")

    print("\nDistill")
    print(f"Accuracy: {mean(distill_accuracys):.3f} ± {stdev(distill_accuracys):.3f}")
    print(f"Macro F1: {mean(distill_macro_f1s):.3f} ± {stdev(distill_macro_f1s):.3f}")


def get_threshold(json_path, thresholds):
    if "student" in json_path:
        return thresholds["student"]
    else:
        camera = json_path.split("/")[-1].split(".")[0]
        return thresholds[str(camera)]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python calculate_base_distill.py <results_folder> <distill>")
        sys.exit(1)

    results_folder_path = sys.argv[1]
    distill = int(sys.argv[2])

    results_folder_path += "/" if results_folder_path[-1] != "/" else ""

    main(results_folder_path, distill)
