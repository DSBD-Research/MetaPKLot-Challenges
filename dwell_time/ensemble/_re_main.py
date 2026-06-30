import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import confusion_matrix, accuracy_score
from dataset.data_loader import data_loader
from models.ensemble import ensemble_model
from tools.logger import logger
import numpy as np
import importlib
import argparse
import torch
import json

def test(ensemble, test_subsets, config, threshold, exp_logger):
    print(f"Testing ensemble")
    
    final_cm = [[0, 0], [0, 0]]
    for key, subset in test_subsets.items():
        test_loader = torch.utils.data.DataLoader(subset, batch_size=config["experiment"]["batch_size"], shuffle=False, num_workers=config["experiment"]["num_workers"], pin_memory=True)
        preds, labels = ensemble.predict(test_loader, threshold)

        accuracy = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)
        final_cm += cm

        exp_logger.add(["test", "subsets", f"{key}"], {"accuracy": accuracy, "cm": str(cm)})
    exp_logger.add(["test", "average"], {"accuracy": (np.trace(final_cm) / np.sum(final_cm)), "cm": str(final_cm)})


def execute(module, config, args):
    output_name = args.ensemble.split('/')[-1]
    exp_logger = logger(f"_results/{args.save}", output_name)

    print(f"\nStarting experiment [RERUN: {output_name}][TEST: {config["datasets"]["test"].split("/")[-1]}]\n")

    exp_logger.add(["experiment", "model"], config["model"]["config"])
    exp_logger.add(["experiment", "model", "module"], config["model"]["module"])
    exp_logger.add(["experiment", "run"], config["experiment"])
    exp_logger.add("dataset", {"test": config["datasets"]["test"].split("/")[-1]})

    torch.cuda.set_device(config["model"]["config"]["device"])
    ds_loader = data_loader(config["experiment"]["img_size"])
    test_subsets = ds_loader.load_test_set(config["datasets"]["test"])

    state_dicts = [torch.load(f"{args.ensemble}/{model}", map_location=config["model"]["config"]["device"]) for model in os.listdir(args.ensemble)]
    test_ensemble = ensemble_model(module, state_dicts, config["model"]["config"])
    test(test_ensemble, test_subsets, config, 0.5, exp_logger)

    print("Saving result file")
    exp_logger.save()


def main(args):
    assert os.path.exists(args.config), "Invalid config"
    assert torch.cuda.is_available(), "Cuda unavailable"

    config = json.load(open(args.config, "r"))
    module = importlib.import_module(config["model"]["module"])

    execute(module, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ensemble Model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--ensemble", "-e", type=str, required=True)
    parser.add_argument("--save", "-s", type=str, required=True)

    main(parser.parse_args())