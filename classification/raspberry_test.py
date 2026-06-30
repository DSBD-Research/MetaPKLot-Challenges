import torch
import argparse
import os
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
import time

from src.customNNSmall import customNNSmall
from src.PKLotDataset import PKLotDataset

def main():
    args = get_args()

    subsets = get_subsets(args.dataset_path)

    transformer = get_transformer()

    device = torch.device("cpu")

    dataset = PKLotDataset("all", subsets=subsets, transformer=transformer, gray_scale=True)

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = customNNSmall(input_shape=(1, 32, 32), num_classes=2)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)

    test(model, dataloader, args.output_dir)

def test(model, dataloader, output_dir):
    model.eval()

    total_images = 0

    init_time = time.time()

    with torch.no_grad():
        for im, _, _ in dataloader:
            total_images += len(im)

            _ = model(im)
    
    end_time = time.time()

    write_on_log(output_dir + "test.log", f"Total images: {total_images}")
    write_on_log(output_dir + "test.log", f"Total time: {end_time - init_time}")
    write_on_log(output_dir + "test.log", f"Images per second: {total_images / (end_time - init_time)}")

def write_on_log(log_file, message):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    with open(log_file, "a") as f:
        f.write(message + "\n")

def get_transformer():
    return v2.Compose([
        v2.Resize((32, 32)),
        v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ]),
    ])

def get_subsets(dataset_path):
    datasets = os.listdir(dataset_path)

    subsets = []
    for dataset in datasets:
        _subsets = os.listdir(dataset_path + dataset)
        subsets.extend([dataset_path + dataset + "/" + subset for subset in _subsets])

    return subsets

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, help="model.pt", required=True)
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset", required=True)
    parser.add_argument("--output_dir", type=str, help="Path to the output dir", required=True)
    parser.add_argument("--batch_size", type=int, help="Batch size", required=True)
    parser.add_argument("--num_workers", type=int, help="Number of workers", required=True)

    args = parser.parse_args()

    args.dataset_path += "/" if not args.dataset_path.endswith("/") else ""
    args.output_dir += "/" if not args.output_dir.endswith("/") else ""

    return args

if __name__ == "__main__":
    main()