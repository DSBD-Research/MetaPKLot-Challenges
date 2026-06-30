import torch
import yaml
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

from src.Model import Model

def main():
    args = get_args()
    num_executions = get_num_executions(args.train_dir)

    for execution_idx in range(1, num_executions + 1):
        if args.execution is not None and args.execution != execution_idx:
            continue
        
        model_config_path = args.train_dir + f"/execution{execution_idx}/train/config.yaml"
        model_config = yaml.safe_load(open(model_config_path, "r"))

        handle_config(model_config)

        model = Model(model_config, "test", args.gpu)
        model.set_test_output_path(args.train_dir + f"/execution{execution_idx}/test/")
        model.set_mean_std_json(args.train_dir + f"/execution{execution_idx}/train/mean_std.json")
        model.create_transformer()
        model.make_test_dataloaders()
        model.set_test_models_paths(args.train_dir + f"/execution{execution_idx}/train/models/")
        model.set_thresholds(args.train_dir + f"/execution{execution_idx}/train/thresholds.json", args.use_roc_eer_thresholds)
        
        test(model)

def test(model):
    model.write_on_log("Start testing...")

    scaler = torch.amp.GradScaler()

    if model.get_models_to_train() == "ensemble":
        l_predictions = []
        l_gd_labels = []

    for model_idx in range(model.get_n_models()):
        model.load_model(model_idx)
        model.set_to_eval()

        model.write_on_log(f"Testing {model.get_model_name()}...")

        predictions = []
        gd_labels = []
        images_paths = []
        with torch.no_grad():
            for images, labels, im_path in model.get_test_loader():
                images_, labels_ = images.to(model.get_device()), labels.to(model.get_device())

                with torch.amp.autocast("cuda"):
                    outputs = model.predict(images_)
                
                class_1_prob = torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy()

                predictions.extend(class_1_prob)
                gd_labels.extend(labels_.cpu().numpy())
                images_paths.extend(im_path)

        if model.get_models_to_train() == "ensemble":
            l_predictions.append(predictions)
            l_gd_labels.append(gd_labels)

        model.save_test_json(predictions, gd_labels, images_paths)
        model.save_single_metrics(predictions, gd_labels)

    if model.get_models_to_train() == "ensemble":
        model.save_ensemble_metrics(l_predictions, l_gd_labels)

    model.write_on_log("Done.")

def handle_config(config):
    config["datasets_path"] += "/" if not config["datasets_path"].endswith("/") else ""
    config["output_dir"] += "/" if not config["output_dir"].endswith("/") else ""

def get_num_executions(train_dir):
    return len(os.listdir(train_dir))

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", type=str, help="Path to the train dir", required=True)
    parser.add_argument("--use_roc_eer_thresholds", type=str, help="True or False", required=True)
    parser.add_argument("--execution", type=int, help="Execution number", required=False)
    parser.add_argument("--gpu", type=int, help="GPU index to use", required=True)

    args = parser.parse_args()

    args.train_dir += "/" if not args.train_dir.endswith("/") else ""

    return args

if __name__ == "__main__":
    main()
