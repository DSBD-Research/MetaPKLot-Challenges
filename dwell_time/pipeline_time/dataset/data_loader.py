from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import torch

class classif_dataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels = images, labels
        self.transform = transform


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]


class siamese_dataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels = images, labels
        self.transform = transform


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img1 = self.transform(self.images[index][0])
        img2 = self.transform(self.images[index][1])

        return (img1, img2), self.labels[index]


class data_loader():
    def __init__(self, img_size, num_workers):
        self.transformer = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(img_size, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.num_workers = num_workers

    
    def get_classif_loader(self, spots_imgs):
        labels = np.zeros(len(spots_imgs), dtype=int)
        dataset = classif_dataset(spots_imgs, labels, self.transformer)

        return torch.utils.data.DataLoader(dataset, batch_size=len(spots_imgs), shuffle=False, num_workers=self.num_workers, pin_memory=True)
        
    
    def get_siamese_loader(self, actual_spots, previous_spots):
        labels = np.zeros(len(actual_spots), dtype=int)
        dataset = siamese_dataset(list(zip(previous_spots, actual_spots)), labels, self.transformer)

        return torch.utils.data.DataLoader(dataset, batch_size=len(actual_spots), shuffle=False, num_workers=self.num_workers, pin_memory=True)