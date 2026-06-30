from datetime import datetime
import numpy as np
import argparse
import json
import os

def print_results(results, file):
    results.sort(key=lambda result: result["times"]["global"])

    globals = [result["times"]["global"] for result in results]
    crops = [result["times"]["crop"]["avg"] for result in results]
    classifies = [result["times"]["classify"]["avg"] for result in results]
    comparies = [result["times"]["compare"]["avg"] for result in results]

    avg_globals, std_globals = np.average(globals), np.std(globals)
    avg_crops, std_crops = np.average(crops), np.std(crops)
    avg_classifies, std_classifies = np.average(classifies), np.std(classifies)
    avg_comparies, std_comparies = np.average(comparies), np.std(comparies)

    file.write("-------------------------------[ALL RESULTS]--------------------------------\n")
    file.write(" - ".join(f"{g:.6f}" for g in globals) + " [GLOBAL]\n")
    file.write(" - ".join(f"{c:.6f}" for c in crops) + " [CROP]\n")
    file.write(" - ".join(f"{c:.6f}" for c in classifies) + " [CLASSIFY]\n")
    file.write(" - ".join(f"{c:.6f}" for c in comparies) + " [COMPARE]\n")

    file.write("\n-------------------------------[AVG RESULTS]--------------------------------\n")
    file.write("  [AVG]   [STD] \n")
    file.write(f"{avg_globals:.6f} {std_globals:.5f} [GLOBAL]\n")
    file.write(f"{avg_crops:.6f} {std_crops:.5f} [CROP]\n")
    file.write(f"{avg_classifies:.6f} {std_classifies:.5f} [CLASSIFY]\n")
    file.write(f"{avg_comparies:.6f} {std_comparies:.5f} [COMPARE]\n")

    file.write("\n--------------------------------[COMPLETE]---------------------------------\n")
    file.write(f"\nClassify: {avg_classifies:.6f}s - Compare: {avg_comparies:.6f}s - Others: {(avg_globals - (avg_classifies + avg_comparies)):.6f}s\n")
    file.write(f"Update dwell time: {(avg_globals):.6f}s per spot")


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
        file.write(f"Subset: {config['dataset']['subset'].upper()}\n\n")

        file.write("############################################################################\n\n")

        print_results(results, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline summary", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--files", "-f", type=str, required=True)
    parser.add_argument("--begin_date", "-bd", type=str, required=True)
    parser.add_argument("--end_date", "-ed", type=str, required=True)
    parser.add_argument("--config", "-c", type=str, required=True)

    main(parser.parse_args())