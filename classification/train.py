from torchvision.transforms import v2
import torch
import yaml
import torch.nn as nn
import torch.optim as optim
import argparse
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
import numpy as np
import os

from src.Model import Model

def main():
    args = get_args()
    config = load_config(args)

    model = Model(config, "train", args.gpu, args.pretrained_model)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.get_parameters(), lr=0.0001)

    for i_dataloader in range(model.get_num_dataloaders()):
        model.set_dataloader(i_dataloader)
        
        train(model, criterion, optimizer)

        model.clear()
        optimizer = optim.Adam(model.get_parameters(), lr=0.0001)
        criterion = nn.CrossEntropyLoss()

def train(model, criterion, optimizer):
    model.write_on_log(f"Start training... {model.get_model_name()}")

    scaler = torch.amp.GradScaler()

    model.set_to_train()
    best_validation_loss = float("inf")
    best_y = None
    best_y_prob = None
    for epoch in range(model.get_num_epochs()):
        model.write_on_log(f"Epoch {epoch + 1}/{model.get_num_epochs()}")
        model.set_to_train()

        # Training
        total_loss = 0
        for images, labels, _ in model.get_train_loader():
            optimizer.zero_grad()

            with torch.amp.autocast("cuda"):
                images_, labels_ = images.to(model.get_device()), labels.to(model.get_device())

                outputs = model.predict(images_)
                loss = criterion(outputs, labels_)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss = criterion(outputs, labels_)
            total_loss += loss.item()
        
        training_loss = total_loss / len(model.get_train_loader())

        # Validation
        model.set_to_eval()
        correct_predictions = 0
        total_predictions = 0
        total_loss = 0

        y = []
        y_prob = []
        with torch.no_grad():
            for images, labels, _ in model.get_val_loader():
                images_, labels_ = images.to(model.get_device()), labels.to(model.get_device())

                with torch.amp.autocast("cuda"):
                    outputs = model.predict(images_)
                    loss = criterion(outputs, labels_)
                total_loss += loss.item()

                predicted = torch.max(outputs.data, 1)[1]
                total_predictions += labels_.size(0)
                correct_predictions += (predicted == labels_).sum().item()

                y_prob.extend(torch.nn.functional.softmax(outputs, dim=1)[:, 1].cpu().numpy())
                y.extend(labels_.cpu().numpy())

        validation_loss = total_loss / len(model.get_val_loader())
        validation_accuracy = correct_predictions / total_predictions

        model.write_on_log(f"Training loss: {training_loss}")
        model.write_on_log(f"Validation loss: {validation_loss}")
        model.write_on_log(f"Validation accuracy: {validation_accuracy}")
        
        if validation_loss < best_validation_loss:
            model.write_on_log(f"{validation_loss} < {best_validation_loss}. Saving model...")

            best_validation_loss = validation_loss
            model.save_model()

            best_y = y
            best_y_prob = y_prob

        model.write_on_log("")

    eer_threshold = plot_roc_curve(best_y, best_y_prob, model.get_model_name(), model.get_output_dir())
    model.save_on_thresholds_json(eer_threshold)

def plot_roc_curve(y, y_prob, model_name, model_output_dir):
    fpr, tpr, thresholds = roc_curve(y, y_prob)

    plt.plot(fpr, tpr, color='darkorange')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC curve for {model_name}')
    os.makedirs(f"{model_output_dir}plots/", exist_ok=True)
    plt.savefig(f"{model_output_dir}plots/{model_name}_roc_curve.png")
    plt.clf()

    eer_threshold = compute_eer_threshold(fpr, tpr, thresholds)

    return eer_threshold

def compute_eer_threshold(fpr, tpr, thresholds):
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)

    return thresholds[min_index]

def load_config(args):
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    config['datasets_path'] += "/" if not config['datasets_path'].endswith("/") else ""
    config['output_dir'] += "/" if not config['output_dir'].endswith("/") else ""

    output_dir = config.get('output_dir')
    if not os.path.exists(output_dir):
        os.makedirs(f"{output_dir}execution1/train")
        config['output_dir'] = f"{output_dir}execution1/train/"
        return config

    n_folders = os.listdir(output_dir)
    config['output_dir'] = f"{output_dir}execution{len(n_folders) + 1}/train/"
    os.makedirs(config['output_dir'])

    return config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the config file", required=True)
    parser.add_argument("--gpu", type=int, help="GPU index to use", required=True)
    parser.add_argument("--pretrained_model", type=str, help="Path to the pretrained model", required=False)
    args = parser.parse_args()

    return args

if __name__ == "__main__":
    main()
