from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from glob import glob
from math import ceil
import os
from random import shuffle

class PKLotDataset(Dataset):
    def __init__(self, mode, subsets=None, transformer=None, normalize_train_data=None, gray_scale=None, distill_train_days=5, distill_val_days=2, images_paths=None):
        self.images = []
        self.targets = []
        self.transformer = transformer
        self.gray_scale = gray_scale
        
        if mode == "all":
            for subset in subsets:
                self.images += glob(subset + '/**/*.jpg', recursive=True)

            for image in self.images:
                if "empty" in image:
                    self.targets.append(0)
                else:
                    self.targets.append(1)

        elif mode == "first_70":
            for subset in subsets:
                days = sorted(os.listdir(subset))
                days_formated = [int("".join(day.split("-"))) for day in days]
                ordered_days = [day for _, day in sorted(zip(days_formated, days))]

                first_70 = ceil(len(ordered_days) * 0.7)

                for day in ordered_days[:first_70]:
                    day_images = glob(subset + '/' + day + '/**/*.jpg', recursive=True)
                    self.images += day_images
                    self.targets += [0 if "empty" in image else 1 for image in day_images]
        
        elif mode == "last_30":
            for subset in subsets:
                days = sorted(os.listdir(subset))
                days_formated = [int("".join(day.split("-"))) for day in days]
                ordered_days = [day for _, day in sorted(zip(days_formated, days))]

                first_70 = ceil(len(ordered_days) * 0.7)

                for day in ordered_days[first_70:]:
                    day_images = glob(subset + '/' + day + '/**/*.jpg', recursive=True)
                    self.images += day_images
                    self.targets += [0 if "empty" in image else 1 for image in day_images]

        elif mode == "distill_train":
            for subset in subsets:
                days = sorted(os.listdir(subset))
                days_formated = [int("".join(day.split("-"))) for day in days]
                ordered_days = [day for _, day in sorted(zip(days_formated, days))]

                for day in ordered_days[:distill_train_days]:
                    day_images = glob(subset + '/' + day + '/**/*.jpg', recursive=True)
                    self.images += day_images
                    self.targets += [0 if "empty" in image else 1 for image in day_images]
        
        elif mode == "distill_val":
            for subset in subsets:
                days = sorted(os.listdir(subset))
                days_formated = [int("".join(day.split("-"))) for day in days]
                ordered_days = [day for _, day in sorted(zip(days_formated, days))]

                for day in ordered_days[distill_train_days:distill_train_days + distill_val_days]:
                    day_images = glob(subset + '/' + day + '/**/*.jpg', recursive=True)
                    self.images += day_images
                    self.targets += [0 if "empty" in image else 1 for image in day_images]

        elif mode == "distill_test":
            for subset in subsets:
                days = sorted(os.listdir(subset))
                days_formated = [int("".join(day.split("-"))) for day in days]
                ordered_days = [day for _, day in sorted(zip(days_formated, days))]

                for day in ordered_days[distill_train_days + distill_val_days:]:
                    day_images = glob(subset + '/' + day + '/**/*.jpg', recursive=True)
                    self.images += day_images
                    self.targets += [0 if "empty" in image else 1 for image in day_images]
        
        elif mode == "post_threshold":
            for image_path in images_paths:
                self.images.append(image_path)
                if "empty" in image_path:
                    self.targets.append(0)
                else:
                    self.targets.append(1)
        
        if normalize_train_data:
            self.images, self.targets = self.normalize_data(self.images, self.targets)
    
    def normalize_data(self, images, targets):
        zips = list(zip(images, targets))
        shuffle(zips)
        images, targets = zip(*zips)

        num_targets_1 = 0
        num_targets_0 = 0
        for target in targets:
            if target == 1:
                num_targets_1 += 1
            else:
                num_targets_0 += 1
        
        num_min = min(num_targets_0, num_targets_1)

        new_images = []
        new_targets = []
        targets_1 = 0
        targets_0 = 0
        for i in range(len(images)):
            if targets[i] == 1 and targets_1 < num_min:
                targets_1 += 1
                new_images.append(images[i])
                new_targets.append(targets[i])
            elif targets[i] == 0 and targets_0 < num_min:
                targets_0 += 1
                new_images.append(images[i])
                new_targets.append(targets[i])
        
        return new_images, new_targets

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if not self.gray_scale:
            image = read_image(self.images[idx])
        else:
            image = read_image(self.images[idx], mode=ImageReadMode.GRAY)
        target = self.targets[idx]

        if self.transformer:
            image = self.transformer(image)

        return image, target, self.images[idx]
