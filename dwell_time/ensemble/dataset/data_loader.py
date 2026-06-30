from torchvision import transforms, datasets
from datetime import datetime
from itertools import chain
import torch
import math
import os

class data_loader():
    def __init__(self, img_size):
        self.aug_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(img_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomHorizontalFlip(p=0.1),
            transforms.RandomAutocontrast(p=0.1),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=0.1),
            transforms.RandomApply([transforms.RandomCrop(size=img_size)], p=0.1)
        ])

        self.normal_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(img_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    def __get_pytorch_dataset(self, subsets, transform):
        dataset_list = []

        for subset in subsets:
            _, path = subset
            dataset_list.append(datasets.ImageFolder(path, transform))
        
        return torch.utils.data.ConcatDataset(dataset_list)
    

    def __get_dates(self, ds_path):
        dates = os.listdir(ds_path)
        formated_dates = [(datetime.strptime(date, '%Y-%m-%d').date(), f"{ds_path}/{date}") for date in dates]
        
        return sorted(formated_dates, key=lambda x: x[0])


    def load_train_val_sets(self, ds_paths=[]):
        for path in ds_paths:
            assert os.path.exists(path), "Invalid dataset"

        datasets = []
        for ds_path in ds_paths:
            subsets = [os.path.join(ds_path, subset) for subset in os.listdir(ds_path)]
            dates = list(chain.from_iterable([self.__get_dates(subset) for subset in subsets]))
            
            datasets.append(self.__get_pytorch_dataset(dates, self.aug_transform))

        return datasets
    

    def load_test_set(self, ds_path=""):
        assert os.path.exists(ds_path), "Invalid dataset"

        datasets = {}
        subsets = os.listdir(ds_path)

        for subset in subsets:
            dates = self.__get_dates(f"{ds_path}/{subset}")
            datasets[subset] = self.__get_pytorch_dataset(dates, self.normal_transform)

        return datasets