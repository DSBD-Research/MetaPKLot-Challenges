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
import uuid
import math
import copy

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


def train(model, train_ds, val_ds, config, exp_logger, output_name):
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=config["experiment"]["num_workers"], pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=config["experiment"]["batch_size"], shuffle=False, num_workers=config["experiment"]["num_workers"], pin_memory=True)
    best_state_dict, best_loss, epoch_id, early_stop_count = None, math.inf, -1, 0
    epochs, early_stop = config["experiment"]["epochs"], config["experiment"]["early_stop"]

    for epoch in range(epochs):
        print(f"Epoch [{epoch + 1}/{epochs}] Initializing training epoch", end=" ")

        model.train(train_loader)
        val_loss = model.validate(val_loader)

        if val_loss < best_loss:
            best_state_dict = copy.deepcopy(model.get_state_dict())
            best_loss = val_loss
            epoch_id = (epoch + 1)
            early_stop_count = 0
        else:
            early_stop_count += 1
        
        print(f" - val_loss: {val_loss:.4f} - best_loss: {best_loss:.4f} - early_stop_count: {early_stop_count}/{early_stop}")
        exp_logger.add(["epochs", f"epoch_{epoch + 1}"], val_loss)

        if early_stop_count == early_stop:
            print(f"{early_stop} attempts - val_loss not decreasing - early stopping")
            break
    
    exp_logger.add("best_model", {"loss": best_loss, "epoch_id": epoch_id, "model": f"{output_name}.pt"})
    return best_state_dict
            

def execute(module, config, args):
    train_datasets = "_".join([train.split("/")[-1] for train in config["datasets"]["train"]])
    output_name = f"model_{train_datasets.lower()}_{str(uuid.uuid4()).replace('-', '')[:5]}"
    exp_logger = logger(f"_results/{args.save}", output_name)

    print(f"\nStarting experiment [MODEL: {config['model']['module'].split('.')[-1]}][TRAIN: {train_datasets}][TEST: {config["datasets"]["test"].split("/")[-1]}]\n")

    exp_logger.add(["experiment", "model"], config["model"]["config"])
    exp_logger.add(["experiment", "model", "module"], config["model"]["module"])
    exp_logger.add(["experiment", "run"], config["experiment"])
    exp_logger.add("dataset", {"train": train_datasets, "test": config["datasets"]["test"].split("/")[-1], "train_val_split": config["datasets"]["split"]})

    torch.cuda.set_device(config["model"]["config"]["device"])
    ds_loader, ds_leveler = data_loader(config["experiment"]["img_size"]), data_leveler()

    train_ds, val_ds = ds_loader.load_train_val_sets(config["datasets"]["train"], config["datasets"]["split"])
    train_ds, val_ds = ds_leveler.flatten_dataset(train_ds), ds_leveler.flatten_dataset(val_ds)
    test_subsets = ds_loader.load_test_set(config["datasets"]["test"])

    train_model = module.model(config=config["model"]["config"])
    best_state_dict = train(train_model, train_ds, val_ds, config, exp_logger, output_name)

    test_model = module.model(pre_trained=False, config=config["model"]["config"])
    test_model.load_state_dict(best_state_dict)

    threshold = get_threshold(test_model, val_ds, config, exp_logger)
    test(test_model, test_subsets, config, threshold, exp_logger)

    print("Saving best model")
    torch.save(best_state_dict, f"_models/{args.save}/{output_name}.pt")

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
    parser.add_argument("--save", "-s", type=str, required=True)

    main(parser.parse_args())