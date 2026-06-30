import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from tools.utils import get_cm
from datetime import datetime
import numpy as np
import argparse
import json
import os

def print_results(results, file):
    results.sort(key=lambda result: result["test"]["average"]["accuracy"])

    thresholds = [result["metrics"]["threshold"] for result in results]
    val_loss = [result["best_model"]["loss"] for result in results]
    epochs = [result["best_model"]["epoch_id"] for result in results]
    accuracies = [result["test"]["average"]["accuracy"] for result in results]
    cms = [get_cm(result["test"]["average"]["cm"]) for result in results]

    avg_threshold, std_threshold = np.average(thresholds), np.std(thresholds)
    avg_val_loss, std_val_loss = np.average(val_loss), np.std(val_loss)
    avg_epoch, std_epoch = int(np.average(epochs)), np.std(epochs)
    avg_accuracy, std_accuracy = np.average(accuracies), np.std(accuracies)
    avg_cm = np.round(sum(cms) / len(cms))

    file.write("-------------------------------[ALL RESULTS]--------------------------------\n")
    file.write(" - ".join(f"{threshold:.4f}" for threshold in thresholds) + " [THRESHOLDS]\n")
    file.write(" - ".join(f"{val_loss:.4f}" for val_loss in val_loss) + " [VAL LOSS]\n")
    file.write(" - ".join(f"{epoch:6d}" for epoch in epochs) + " [EPOCHS]\n")
    file.write(" - ".join(f"{acc:.4f}" for acc in accuracies) + " [ACCURACY]\n")

    file.write("\n-------------------------------[AVG RESULTS]--------------------------------\n")
    file.write(" [AVG]  [STD]".ljust(50) + "Confusion matrix - Rounded\n")
    file.write(f"{avg_threshold:.4f} {std_threshold:.4f} [THRESHOLD]".ljust(60) + f"{avg_cm[1][1]:6.0f}".ljust(10) + f"{avg_cm[1][0]:6.0f}\n")
    file.write(f"{avg_val_loss:.4f} {std_val_loss:.4f} [VAL LOSS]".ljust(60) + f"{avg_cm[0][1]:6.0f}".ljust(10) + f"{avg_cm[0][0]:6.0f}\n")
    file.write(f"{avg_epoch:6d} {std_epoch:.4f} [EPOCH]\n")
    file.write(f"{avg_accuracy:.4f} {std_accuracy:.4f} [ACCURACY]\n")

    file.write("\n--------------------------------[BY SUBSET]---------------------------------\n")
    subsets = set()
    for subset in results[0]["test"]["subsets"]:
        subsets.add(subset)

    results_by_subset = {subset: {'accuracy': [], 'std_accuracy': [], 'cm': [[0, 0], [0, 0]]} for subset in subsets}
    results_by_subset = dict(sorted(results_by_subset.items()))

    for result in results:
        for subset in result["test"]["subsets"]:
            results_by_subset[subset]["accuracy"].append(result["test"]["subsets"][subset]["accuracy"])
            results_by_subset[subset]["std_accuracy"].append(result["test"]["subsets"][subset]["accuracy"])
            results_by_subset[subset]["cm"] += get_cm(result["test"]["subsets"][subset]["cm"])
    
    for subset in results_by_subset:
        results_by_subset[subset]["accuracy"] = np.average(results_by_subset[subset]["accuracy"])
        results_by_subset[subset]["std_accuracy"] = np.std(results_by_subset[subset]["std_accuracy"])
        results_by_subset[subset]["cm"] = np.round(results_by_subset[subset]["cm"] / len(cms))

    for subset in results_by_subset:
        accuracy = results_by_subset[subset]["accuracy"]
        std_accuracy = results_by_subset[subset]["std_accuracy"]
        cm = results_by_subset[subset]["cm"]

        file.write(f"{subset}:".ljust(50))
        file.write("Confusion matrix - Rounded\n")

        file.write(f"{accuracy:.4f} [ACCURACY]".ljust(60))
        file.write(f"{cm[1][1]:6.0f}".ljust(10) + f"{cm[1][0]:6.0f}\n")
        file.write(f"{std_accuracy:.4f} [STD_ACCURACY]".ljust(60))
        file.write(f"{cm[0][1]:6.0f}".ljust(10) + f"{cm[0][0]:6.0f}\n")
        file.write("\n")


def main(args):
    assert os.path.exists(args.files), "Invalid results files path"

    config = json.load(open(args.config, "r"))
    results = [json.load(open(f"{args.files}/{file}", "r")) for file in os.listdir(args.files) if ".json" in file]
    train_datasets = "_".join([train.split("/")[-1] for train in config["datasets"]["train"]])
    summary_name = args.files.split("/")[1]

    begin_date = datetime.strptime(args.begin_date, '%Y-%m-%d--%H:%M:%S')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d--%H:%M:%S')
    time_spent = end_date - begin_date

    with open(f"_summaries/{summary_name}.txt", "w") as file:
        file.write(f"Begin: {args.begin_date} End: {args.end_date} -- Time spent: {time_spent} \n")

        file.write("\n--------------------------------[RUN CONFIG]--------------------------------\n")
        file.write(f"Epochs: {config['experiment']['epochs']} - Early stop: {config['experiment']['early_stop']} - ")
        file.write(f"Batch size: {config['experiment']['batch_size']} - ")
        file.write(f"Image size: {str(config['experiment']['img_size'])}\n")
        file.write(f"Train: {train_datasets} - Test: {config['datasets']['test'].split('/')[-1]} - ")
        file.write(f"Split: {str(config['datasets']['split'])}\n")

        file.write("\n-------------------------------[MODEL CONFIG]-------------------------------\n")
        file.write(f"Model: {config['model']['module'].split('.')[-1]} - ")
        file.write(f"Loss: {config['model']['config']['loss']}\n")
        file.write(f"Optimizer: {config['model']['config']['optimizer']} - ")
        file.write(f"Learning Rate: {config['model']['config']['lr']}\n\n")
        file.write("############################################################################\n\n")

        print_results(results, file)
        file.write("############################################################################\n\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification training summary", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", "-f", type=str, required=True)
    parser.add_argument("--begin_date", "-bd", type=str, required=True)
    parser.add_argument("--end_date", "-ed", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)

    main(parser.parse_args())