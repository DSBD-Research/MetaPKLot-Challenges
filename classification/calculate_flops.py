from calflops import calculate_flops
import torch
from torchvision import models
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as v2
import torch.nn as nn
import sys

from src.PKLotDataset import PKLotDataset
from src.customNNLarge import customNNLarge
from src.customNNSmall import customNNSmall
from src.Hochuli import Hochuli

mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def main(model_name, subset, batch_size):
    model = load_model_by_model_name(model_name)
    transformer = load_transformer_by_model_name(model_name)
    dataloader = create_dataloader(subset, transformer, batch_size)

    if model is None:
        print(f"Model name not found")
        sys.exit(1)
    
    model.eval()

    image_batch = get_image_batch(dataloader)
    batch_shape = tuple(image_batch.shape)

    flops, _, _ = calculate_flops(model, batch_shape, output_as_string=False, print_detailed=False, print_results=False)

    print(f"FLOPs per image: {flops // batch_size}")

def load_model_by_model_name(model_name):
    model = None

    match model_name:
        case "mobilenet_v3_large":
            model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
            model.classifier[3] = nn.Linear(model.classifier[0].out_features, 2)

        case "mobilenet_v3_small":
            model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
            model.classifier[3] = nn.Linear(model.classifier[0].out_features, 2)

        case "vit_b_16":
            model = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT)
            model.heads.head = torch.nn.Linear(in_features=model.heads.head.in_features, out_features=2)

        case "hochuli":
            model = Hochuli()

        case "customNNLarge":
            model = customNNLarge()

        case "customNNSmall":
            model = customNNSmall()

    model.to("cuda:0")

    return model

def load_transformer_by_model_name(model_name):
    transform_resize = None

    match model_name:
        case "mobilenet_v3_large":
            transform_resize = [128, 128]

        case "mobilenet_v3_small":
            transform_resize = [128, 128]

        case "vit_b_16":
            transform_resize = [224, 224]
        
        case "hochuli":
            transform_resize = [32, 32]
        
        case "customNNLarge":
            transform_resize = [32, 32]

        case "customNNSmall":
            transform_resize = [32, 32]


    if transform_resize is None:
        print(f"Model name not found")
        sys.exit(1)


    transformer = v2.Compose([
                v2.Resize(transform_resize),
                v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.float32, scale=True)
                ]),
                v2.Normalize(mean=mean, std=std)
            ])

    return transformer

def create_dataloader(subset, transformer, batch_size):
    dataset = PKLotDataset("all", [subset], transformer)

    return DataLoader(dataset, batch_size=batch_size)

def get_image_batch(dataloader):
    with torch.no_grad():
        for images, _, _ in dataloader:
            images_ = images.to("cuda:0")

            return images_

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python calculate_flops.py <model> <subset_path> <batch_size>")
        sys.exit(1)
    
    model_name = str(sys.argv[1])
    subset = str(sys.argv[2])
    batch_size = int(sys.argv[3])

    main(model_name, subset, batch_size)
