from torchvision import models
from torch.utils.data import Dataset, DataLoader
from time import localtime, strftime
import torch
import torchvision.transforms.v2 as v2
import shutil
import torch.nn as nn
import os
import json
import argparse
from glob import glob
from torchvision.io import read_image, ImageReadMode

from src.customNNLarge import customNNLarge
from src.customNNSmall import customNNSmall
from src.Hochuli import Hochuli

class PKLotDataset(Dataset):
    def __init__(self, mode, transformer, num_channels, train_dir=None, val_dir=None):
        self.images = None
        self.labels = None
        self.transformer = transformer
        self.num_channels = num_channels

        if mode == "train":
            self.images = glob(f"{train_dir}/**/*.JPEG", recursive=True)
            class_names = sorted(set(img.split("/")[-3] for img in self.images))
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
            self.labels = [self.class_to_idx[img.split("/")[-3]] for img in self.images]

        elif mode == "val":
            with open(f"{val_dir}val_annotations.txt", "r") as f:
                lines = f.readlines()

            image_name_to_class = {}
            for line in lines:
                l = line.split("\t")
                image_name_to_class[l[0]] = l[1]

            class_names = sorted(set(image_name_to_class.values()))
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(class_names)}
            
            self.images = glob(f"{val_dir}/**/*.JPEG", recursive=True)
            self.labels = [self.class_to_idx[image_name_to_class[img.split("/")[-1]]] for img in self.images]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        _img = self.images[idx]
        
        if self.num_channels == 1:
            image = read_image(_img, ImageReadMode.GRAY)
        else:
            image = read_image(_img, ImageReadMode.RGB)

        image = self.transformer(image)

        return image, self.labels[idx]


def main():
    args = get_args()

    model = build_model(args.model_type, args.train_dir, args.resize_shape, args.num_channels)

    mean, std = calculate_mean_std(args.train_dir, args.resize_shape, args.num_channels, args.gpu, args.val_batch_size, args.num_workers)

    save_mean_std(args.output_dir, mean, std)

    transformer = create_transformer(args.resize_shape, mean, std)

    train_dataloader, val_dataloader = create_dataloaders(args.train_dir, args.val_dir, transformer, args.train_batch_size, 
                                                          args.val_batch_size, args.num_workers, args.num_channels)

    train(model, train_dataloader, val_dataloader, args.epochs, args.gpu, args.output_dir)

def train(model, train_dataloader, val_dataloader, epochs, gpu, output_dir):
    def write_on_log(output_dir, text):
        time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())
        mode = "w" if not os.path.exists(os.path.join(output_dir, "log.txt")) else "a"

        with open(os.path.join(output_dir, "log.txt"), mode) as file:
            file.write(time + text + "\n")

    model = model.to(f"cuda:{gpu}")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    best_validation_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        write_on_log(output_dir, f"Epoch {epoch + 1}/{epochs}")

        total_loss = 0
        for images, labels in train_dataloader:
            images = images.to(f"cuda:{gpu}")
            labels = labels.to(f"cuda:{gpu}")

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        
        write_on_log(output_dir, f"Training loss: {total_loss / len(train_dataloader)}")

        model.eval()

        correct_predictions = 0
        total_predictions = 0
        total_loss = 0
        for images, labels in val_dataloader:
            images = images.to(f"cuda:{gpu}")
            labels = labels.to(f"cuda:{gpu}")

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            correct_predictions += (preds == labels).sum().item()
            total_predictions += len(labels)

            loss = criterion(outputs, labels)
            total_loss += loss.item()

        validation_loss = total_loss / len(val_dataloader)
        accuracy = correct_predictions / total_predictions

        write_on_log(output_dir, f"Validation loss: {validation_loss}")
        write_on_log(output_dir, f"Validation ccuracy: {accuracy}")

        if validation_loss < best_validation_loss:
            write_on_log(output_dir, f"{validation_loss} < {best_validation_loss}. Saving model")

            best_validation_loss = validation_loss
            torch.save(model.state_dict(), os.path.join(output_dir, "model.pth"))

        write_on_log(output_dir, "")

def create_dataloaders(train_dir, val_dir, transformer, train_batch_size, val_batch_size, num_workers, num_channels):
    train_dataset = PKLotDataset("train", transformer, num_channels=num_channels, train_dir=train_dir)
    val_dataset = PKLotDataset("val", transformer, num_channels=num_channels, val_dir=val_dir)

    train_dataloader = DataLoader(train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    return train_dataloader, val_dataloader

def create_transformer(resize_shape, mean, std):
    transformer = v2.Compose([
        v2.Resize((resize_shape, resize_shape)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean, std)
    ])

    return transformer

def save_mean_std(output_dir, mean, std):
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "mean_std.json"), "w") as f:
        json.dump({"mean": mean, "std": std}, f)

def calculate_mean_std(train_dir, resize_shape, num_channels, gpu, val_batch_size, num_workers):
    transformer = v2.Compose([
        v2.Resize((resize_shape, resize_shape)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True)
    ])

    dataset = PKLotDataset("train", transformer, num_channels, train_dir=train_dir)
    dataloader = DataLoader(dataset, batch_size=val_batch_size, shuffle=False, num_workers=num_workers)

    mean = torch.zeros(num_channels, device=f"cuda:{gpu}")
    std = torch.zeros(num_channels, device=f"cuda:{gpu}")
    total_samples = 0
    for images, _, in dataloader:
        images = images.to(f"cuda:{gpu}")
        batch_samples = images.size(0)
        total_samples += batch_samples
        for i in range(num_channels):
            mean[i] += images[:, i, :, :].mean().item() * batch_samples
            std[i] += images[:, i, :, :].std().item() * batch_samples

    mean = [m.cpu().item() / total_samples for m in mean]
    std = [s.cpu().item() / total_samples for s in std]

    return mean, std

def build_model(model_type, train_dir, resize_shape, num_channels):
    if model_type == "customNNLarge":
        model = customNNLarge(num_classes=len(os.listdir(train_dir)), input_shape=(num_channels, resize_shape, resize_shape))
    elif model_type == "customNNSmall":
        model = customNNSmall(num_classes=len(os.listdir(train_dir)), input_shape=(num_channels, resize_shape, resize_shape))
    elif model_type == "hochuli":
        model = Hochuli(num_classes=len(os.listdir(train_dir)), input_shape=(num_channels, resize_shape, resize_shape)) 
    else:
        raise ValueError("Invalid model type")

    return model

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, help="customNNLarge or customNNSmall", required=True)
    parser.add_argument("--gpu", type=int, help="GPU index to use", required=True)
    parser.add_argument("--train_batch_size", type=int, help="Batch size", required=True)
    parser.add_argument("--val_batch_size", type=int, help="Batch size", required=True)
    parser.add_argument("--num_workers", type=int, help="Num workers", required=True)
    parser.add_argument("--epochs", type=int, help="Epochs", required=True)
    parser.add_argument("--resize_shape", type=int, help="Resize shape. 32", required=True)
    parser.add_argument("--num_channels", type=int, help="Number of channels", required=True)
    parser.add_argument("--train_dir", type=str, help="Train directory", required=True)
    parser.add_argument("--val_dir", type=str, help="Validation directory", required=True)
    parser.add_argument("--output_dir", type=str, help="Output directory", required=True)

    args = parser.parse_args()

    args.train_dir += "/" if not args.train_dir.endswith("/") else ""
    args.val_dir += "/" if not args.val_dir.endswith("/") else ""
    args.output_dir += "/" if not args.output_dir.endswith("/") else ""

    return args


if __name__ == "__main__":
    main()
