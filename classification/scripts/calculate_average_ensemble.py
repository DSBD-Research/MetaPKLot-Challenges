import sys
import json
import os
from statistics import mean, stdev

def get_camera(image_path):
    splitted_image_path = image_path.split("/")
    for field_idx in range(len(splitted_image_path)):
        if splitted_image_path[field_idx] == "PKLot" or splitted_image_path[field_idx] == "CNRPark-EXT" or splitted_image_path[field_idx] == "PLds":
            return str(splitted_image_path[field_idx + 1])
        
def main(results_folder_path, mode, eer_threshold):
    n_executions = len(os.listdir(results_folder_path))

    accuracys = []
    macro_f1s = []

    for execution in range(1, n_executions + 1):
        execution_threshold = 0.5
        if eer_threshold:
            thresholds_json_file = json.load(open(results_folder_path + f"/execution{execution}/train/thresholds.json"))
            execution_threshold = float(thresholds_json_file["val_total"])

        jsons_paths = os.listdir(results_folder_path + f"/execution{execution}/test/jsons")

        preds_per_subset = {}
        targets_per_subset = {}

        json_file = json.load(open(results_folder_path + f"/execution{execution}/test/jsons/val_total.json"))

        for pred, label, image_path in zip(json_file["predictions"], json_file["labels"], json_file["images_paths"]):
            subset = get_subset(image_path)

            if subset not in preds_per_subset:
                preds_per_subset[subset] = []
                targets_per_subset[subset] = []

            targets_per_subset[subset].append(label)
            preds_per_subset[subset].append(pred)

        total_images_per_subset = {}
        correct_per_subset = {}

        tp_class0_per_subset = {}
        fp_class0_per_subset = {}
        fn_class0_per_subset = {}
        tp_class1_per_subset = {}
        fp_class1_per_subset = {}
        fn_class1_per_subset = {}

        for subset in preds_per_subset:
            correct_per_subset[subset] = 0
            total_images_per_subset[subset] = 0

            tp_class0_per_subset[subset] = 0
            fp_class0_per_subset[subset] = 0
            fn_class0_per_subset[subset] = 0
            tp_class1_per_subset[subset] = 0
            fp_class1_per_subset[subset] = 0
            fn_class1_per_subset[subset] = 0

            for pred, label in zip(preds_per_subset[subset], targets_per_subset[subset]):
                prediction = 1 if pred >= execution_threshold else 0

                total_images_per_subset[subset] += 1
                if prediction == label:
                    correct_per_subset[subset] += 1

                if label == 0:
                    if prediction == 0:
                        tp_class0_per_subset[subset] += 1
                    else:
                        fn_class0_per_subset[subset] += 1
                        fp_class1_per_subset[subset] += 1
                else:
                    if prediction == 1:
                        tp_class1_per_subset[subset] += 1
                    else:
                        fn_class1_per_subset[subset] += 1
                        fp_class0_per_subset[subset] += 1

        accuracy = 0
        macro_f1 = 0
        for subset in correct_per_subset:
            accuracy += (correct_per_subset[subset] / total_images_per_subset[subset]) * total_images_per_subset[subset] / sum(total_images_per_subset.values())

            precision_class0 = tp_class0_per_subset[subset] / (tp_class0_per_subset[subset] + fp_class0_per_subset[subset])
            recall_class0 = tp_class0_per_subset[subset] / (tp_class0_per_subset[subset] + fn_class0_per_subset[subset])
            f1_class0 = 2 * (precision_class0 * recall_class0) / (precision_class0 + recall_class0)

            precision_class1 = tp_class1_per_subset[subset] / (tp_class1_per_subset[subset] + fp_class1_per_subset[subset])
            recall_class1 = tp_class1_per_subset[subset] / (tp_class1_per_subset[subset] + fn_class1_per_subset[subset])
            f1_class1 = 2 * (precision_class1 * recall_class1) / (precision_class1 + recall_class1)

            macro_f1 += ((f1_class0 + f1_class1) / 2) * total_images_per_subset[subset] / sum(total_images_per_subset.values())
        
        accuracys.append(accuracy)
        macro_f1s.append(macro_f1)

    print(f"Accuracy: {mean(accuracys):.3f} ± {stdev(accuracys):.3f}")
    print(f"Macro F1: {mean(macro_f1s):.3f} ± {stdev(macro_f1s):.3f}")


def get_subset(image_path):
    return image_path.split("/")[-4].lower()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python calculate_average_ensemble.py <results_folder> <mode: all | total> <use_eer_threshold: True | False>")
        sys.exit(1)

    results_folder_path = sys.argv[1]
    mode = sys.argv[2]
    use_eer_threshold = sys.argv[3]

    if use_eer_threshold == "True":
        use_eer_threshold = True
    elif use_eer_threshold == "False":
        use_eer_threshold = False

    if mode != "total" and use_eer_threshold:
        raise ValueError("You can only use the EER threshold with the total mode")

    main(results_folder_path, mode, use_eer_threshold)
