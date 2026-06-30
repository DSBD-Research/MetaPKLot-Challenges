from datetime import datetime
import numpy as np
import argparse
import json
import os

def print_results(results, file):
    results.sort(key=lambda result: result["metrics"]["total"]["accuracy"])

    maes = [result["metrics"]["total"]["mae"] for result in results]
    rsmes = [result["metrics"]["total"]["rsme"] for result in results]
    accuracies = [result["metrics"]["total"]["accuracy"] for result in results]

    avg_mae, std_mae = np.average(maes), np.std(maes)
    avg_rsme, std_rsme = np.average(rsmes), np.std(rsmes)
    avg_accuracy, std_accuracy = np.average(accuracies), np.std(accuracies)

    file.write("-------------------------------[ALL RESULTS]--------------------------------\n")
    file.write(" - ".join(f"{threshold:.4f}" for threshold in maes) + " [MAE]\n")
    file.write(" - ".join(f"{val_loss:.4f}" for val_loss in rsmes) + " [RSME]\n")
    file.write(" - ".join(f"{accuracy:.5f}" for accuracy in accuracies) + " [ACCURACY]\n")

    file.write("\n-------------------------------[AVG RESULTS]--------------------------------\n")
    file.write(" [AVG]  [STD] \n")
    file.write(f"{avg_mae:.4f} {std_mae:.4f} [MAE]\n")
    file.write(f"{avg_rsme:.4f} {std_rsme:.4f} [RSME]\n")
    file.write(f"{avg_accuracy:.5f} {std_accuracy:.4f} [ACCURACY]\n")

    file.write("\n--------------------------------[BY SUBSET]---------------------------------\n")
    subsets = set()
    for subset in list(results[0]["metrics"].keys())[:-1]:
        subsets.add(subset)    

    subsets = sorted(list(subsets))
    results_by_subset = {subset: {'accuracies': [], 'maes': [], 'rsmes': []} for subset in subsets}

    for result in results:
        for subset in subsets:
            results_by_subset[subset]["maes"].append(result["metrics"][subset]["mae"])
            results_by_subset[subset]["rsmes"].append(result["metrics"][subset]["rsme"])
            results_by_subset[subset]["accuracies"].append(result["metrics"][subset]["accuracy"])
    
    for subset, values in results_by_subset.items():
        avg_mae, std_mae = np.average(values["maes"]), np.std(values["maes"])
        avg_rsme, std_rsme = np.average(values["rsmes"]), np.std(values["rsmes"])
        avg_accuracy, std_accuracy = np.average(values["accuracies"]), np.std(values["accuracies"])

        file.write(f"{subset}: \n")

        file.write(f"[AVG] {avg_mae:.4f} - [STD] {std_mae:.4f} [MAE]\n")
        file.write(f"[AVG] {avg_rsme:.4f} - [STD] {std_rsme:.4f} [RSME]\n")
        file.write(f"[AVG] {avg_accuracy:.5f} - [STD] {std_accuracy:.4f} [ACCURACY]\n\n")


def main(args):
    assert os.path.exists(args.files), "Invalid results files path"

    config = json.load(open(args.config, "r"))
    results = [json.load(open(f"{args.files}/{file}", "r")) for file in os.listdir(args.files) if ".json" in file]
    summary_name = args.files.split("/")[1]

    begin_date = datetime.strptime(args.begin_date, '%Y-%m-%d--%H:%M:%S')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d--%H:%M:%S')
    time_spent = end_date - begin_date
    
    with open(f"_summaries/{summary_name}.txt", "w") as file:
        file.write(f"Begin: {args.begin_date} End: {args.end_date} -- Time spent: {time_spent} \n")

        file.write("\n--------------------------------[RUN CONFIG]--------------------------------\n")
        file.write(f"Test dataset: {config['dataset']['path'].split('/')[-1]} - ")
        file.write(f"Using classification: {config['experiment']['using_classification']}\n\n")

        file.write("############################################################################\n\n")

        print_results(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline summary", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", "-f", type=str, required=True)
    parser.add_argument("--begin_date", "-bd", type=str, required=True)
    parser.add_argument("--end_date", "-ed", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)

    main(parser.parse_args())
