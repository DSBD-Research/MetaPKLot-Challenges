import sys, os; sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sklearn.metrics import confusion_matrix, accuracy_score
from dataset.data_leveler import data_leveler
from tools.utils import get_roc_auc, get_eer
from dataset.data_loader import data_loader
from tools.logger import logger
import numpy as np
import importlib
import argparse
import torch
import json

def test(model, test_subsets, config, threshold, exp_logger):
    print(f"Testing model")
    
    final_cm = [[0, 0], [0, 0]]
    for key, subset in test_subsets.items():
        test_loader = torch.utils.data.DataLoader(subset, batch_size=config["experiment"]["batch_size"], shuffle=False, num_workers=config["experiment"]["num_workers"], pin_memory=True)
        preds, labels = model.predict(test_loader, threshold)

        accuracy = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)
        final_cm += cm

        exp_logger.add(["test", "subsets", f"{key}"], {"accuracy": accuracy, "cm": str(cm)})
    exp_logger.add(["test", "average"], {"accuracy": (np.trace(final_cm) / np.sum(final_cm)), "cm": str(final_cm)})


def get_threshold(model, val_ds, config, exp_logger):
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config["experiment"]["batch_size"], shuffle=False, num_workers=config["experiment"]["num_workers"], pin_memory=True)

    probs, labels = model.probabilities(val_loader)
    probs = np.array([prob[1] for prob in probs])

    roc, auc = get_roc_auc(probs, labels)
    threshold, eer = get_eer(roc)

    exp_logger.add("metrics", {"auc": float(auc), "eer": float(eer), "threshold": float(threshold)})
    return threshold
            

def execute(module, config, args):
    output_name = args.model.split('/')[-1].replace(".pt", "")
    exp_logger = logger(f"_results/{args.save}", output_name)

    print(f"\nStarting experiment [RERUN: {output_name}][TEST: {config["datasets"]["test"].split("/")[-1]}]\n")

    exp_logger.add(["experiment", "model"], config["model"]["config"])
    exp_logger.add(["experiment", "model", "module"], config["model"]["module"])
    exp_logger.add(["experiment", "run"], config["experiment"])
    exp_logger.add("dataset", {"test": config["datasets"]["test"].split("/")[-1], "train_val_split": config["datasets"]["split"]})

    torch.cuda.set_device(config["model"]["config"]["device"])
    ds_loader, ds_leveler = data_loader(config["experiment"]["img_size"]), data_leveler()

    _, val_ds = ds_loader.load_train_val_sets(config["datasets"]["train"], config["datasets"]["split"])
    val_ds = ds_leveler.flatten_dataset(val_ds)
    test_subsets = ds_loader.load_test_set(config["datasets"]["test"])

    test_model = module.model(pre_trained=False, config=config["model"]["config"])
    test_model.load_state_dict(torch.load(args.model, map_location=config["model"]["config"]["device"]))

    threshold = get_threshold(test_model, val_ds, config, exp_logger)
    test(test_model, test_subsets, config, threshold, exp_logger)

    print("Saving result file")
    exp_logger.save()


def main(args):
    assert os.path.exists(args.config), "Invalid config"
    assert torch.cuda.is_available(), "Cuda unavailable"

    config = json.load(open(args.config, "r"))
    module = importlib.import_module(config["model"]["module"])

    execute(module, config, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classification Model", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--config", "-c", type=str, required=True)
    parser.add_argument("--model", "-m", type=str, required=True)
    parser.add_argument("--save", "-s", type=str, required=True)

    main(parser.parse_args())