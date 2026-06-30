from torch.utils.data import Dataset
from collections import defaultdict
from torchvision import transforms
from datetime import datetime
from PIL import Image
import random
import math
import os

class triplet_dataset(Dataset):
    def __init__(self, triplets, transform):
        self.triplets = triplets
        self.transform = transform

    
    def __len__(self):
        return len(self.triplets)
    

    def __getitem__(self, index):
        anchor = self.transform(Image.open(self.triplets[index][0]))
        positive = self.transform(Image.open(self.triplets[index][1]))
        negative = self.transform(Image.open(self.triplets[index][2]))

        return (anchor, positive, negative)


class test_dataset(Dataset):
    def __init__(self, images, labels, transform):
        self.images, self.labels = images, labels
        self.transform = transform


    def __len__(self):
        return len(self.images)


    def __getitem__(self, index):
        img1 = self.transform(Image.open(self.images[index][0]))
        img2 = self.transform(Image.open(self.images[index][1]))

        return (img1, img2), self.labels[index]


class data_loader():
    def __init__(self, config):
        self.aug_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(config["img_size"], antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=0.1)
        ])
        self.normal_transform = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(config["img_size"], antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.config = config
        self.sets = dict()
        self.val_dates = defaultdict(list)

        self.__get_train_val_sets()
        self.__treat_train_val_intersection()
        self.__get_test_set()


    def __process_images_by_spot(self, image_list):
        images = [os.path.join(date, image) for _, date in image_list for image in os.listdir(date)]
        spot_ids = set([image.split("#")[-2] for image in images])
        images_by_spot = {spot_id: defaultdict(list) for spot_id in spot_ids}

        for image in images:
            split_image = image.split("#")
            spot_id, car_id = split_image[-2], split_image[-1].replace(".jpg", "")

            images_by_spot[spot_id][car_id].append(image)

        final = {}
        for spot_id, cars in images_by_spot.items():
            temp_cars = {car_id: images for car_id, images in cars.items() if len(images) >= 2}
            
            if len(temp_cars.keys()) >= 2:
                final[spot_id] = temp_cars
            
        return final
    

    def __process_images_by_car(self, image_list):
        images = [os.path.join(date, image) for _, date in image_list for image in os.listdir(date)]
        image_dict = defaultdict(list)
        
        for image in images:
            image_dict[image.split("#")[-1].replace(".jpg", "")].append(image)
            
        return {key: value for key, value in image_dict.items() if len(value) >= 2}


    def __get_dates(self, ds_path):
        dates = os.listdir(ds_path)
        formated_dates = [(datetime.strptime(date, '%Y-%m-%d').date(), f"{ds_path}/{date}") for date in dates]
        
        return sorted(formated_dates, key=lambda x: x[0])

    
    def __get_train_val_sets(self):
        subsets = [os.path.join(ds_path, subset) for ds_path in self.config["train"] for subset in os.listdir(ds_path)]
        train_half, val_half = [], []

        for subset in subsets:
            dates = self.__get_dates(subset)
            divider = math.ceil(len(dates) * self.config["split"])
            train_half.extend(dates[:divider])
            val_half.extend(dates[divider:])
        
        self.sets["train_spots"], self.sets["val_spots"] = self.__process_images_by_spot(train_half), self.__process_images_by_spot(val_half)
        self.sets["train_cars"], self.sets["val_cars"] = self.__process_images_by_car(train_half), self.__process_images_by_car(val_half)
        
        for date in val_half:
            self.val_dates[date[1].split("/")[-2]].append(date[0])


    def __treat_train_val_intersection(self):
        intersection = set(self.sets["train_cars"]).intersection(set(self.sets["val_cars"]))
        for car in intersection:
            self.sets["val_cars"][car].extend(self.sets["train_cars"][car])
            del self.sets["train_cars"][car]

        train_cars_spots = {car: spot for spot, cars in self.sets["train_spots"].items() for car in cars}
        train_cars = [car for cars in self.sets["train_spots"].values() for car in cars]

        val_cars_spots = {car: spot for spot, cars in self.sets["val_spots"].items() for car in cars}
        val_cars = [car for cars in self.sets["val_spots"].values() for car in cars]

        intersection = set(train_cars).intersection(val_cars)
        for car in intersection:
            train_spot, val_spot = train_cars_spots[car], val_cars_spots[car]
            
            self.sets["val_spots"][val_spot][car].extend(self.sets["train_spots"][train_spot][car])
            del self.sets["train_spots"][train_spot][car]

            if len(self.sets["train_spots"][train_spot]) <= 1:
                del self.sets["train_spots"][train_spot]


    def __get_test_set(self):
        datasets = {}

        subsets = os.listdir(self.config["test"])
        for subset in subsets:
            dates = self.__get_dates(f"{self.config["test"]}/{subset}")
            datasets[subset] = self.__process_images_by_spot(dates)
        
        self.sets["test"] = datasets
    

    def __get_triplet(self, cars, possible_choices, images):
        neg_car = random.choice(possible_choices)

        triplet_list = random.sample(images, 2)
        triplet_list.append(random.choice(cars[neg_car]))

        return tuple(triplet_list)
    

    def __get_pair(self, cars, possible_choices, images):
        neg_car = random.choice(possible_choices)

        pos_pair = tuple(random.sample(images, 2))
        neg_pair = (random.choice(images), random.choice(cars[neg_car]))

        return [pos_pair, neg_pair]


    def get_train_ds(self):
        origin_spots, origin_cars = self.sets["train_spots"], self.sets["train_cars"]
        triplets = []

        if self.config["full_difficult"]:
            for cars in origin_spots.values():
                cars_list = list(cars.keys())

                for car, images in cars.items():
                    possible_choices = cars_list.copy()
                    possible_choices.remove(car)

                    triplets.append(self.__get_triplet(cars, possible_choices, images))
        else:
            for car, images in origin_cars.items():
                spot_id = images[0].split("#")[-2]

                force_get_by_cars = spot_id not in origin_spots.keys() or car not in origin_spots[spot_id].keys()
                if random.randint(0, 1) or force_get_by_cars:
                    possible_choices = list(origin_cars.keys())
                    possible_choices.remove(car)

                    triplets.append(self.__get_triplet(origin_cars, possible_choices, images))
                else:
                    possible_choices = list(origin_spots[spot_id].keys())
                    possible_choices.remove(car)

                    triplets.append(self.__get_triplet(origin_spots[spot_id], possible_choices, images))
        
        return triplet_dataset(triplets, self.aug_transform)


    def get_val_ds(self):
        origin_spots, origin_cars = self.sets["val_spots"], self.sets["val_cars"]
        triplets = []

        if self.config["full_difficult"]:
            for cars in origin_spots.values():
                cars_list = list(cars.keys())

                for car, images in cars.items():
                    possible_choices = cars_list.copy()
                    possible_choices.remove(car)

                    triplets.append(self.__get_triplet(cars, possible_choices, images))
                    triplets.append(self.__get_triplet(cars, possible_choices, images))
        else:
            for car, images in origin_cars.items():
                spot_id = images[0].split("#")[-2]

                force_get_by_cars = spot_id not in origin_spots.keys() or car not in origin_spots[spot_id].keys()
                if random.randint(0, 1) or force_get_by_cars:
                    possible_choices = list(origin_cars.keys())
                    possible_choices.remove(car)

                    triplets.append(self.__get_triplet(origin_cars, possible_choices, images))
                    triplets.append(self.__get_triplet(origin_cars, possible_choices, images))
                else:
                    possible_choices = list(origin_spots[spot_id].keys())
                    possible_choices.remove(car)

                    triplets.append(self.__get_triplet(origin_spots[spot_id], possible_choices, images))
                    triplets.append(self.__get_triplet(origin_spots[spot_id], possible_choices, images))
        
        return triplet_dataset(triplets, self.normal_transform)


    def get_test_subsets(self):
        origin, datasets = self.sets["test"], {}

        for subset, spots in origin.items():
            pairs, labels = [], []

            for cars in spots.values():
                cars_list = list(cars.keys())
                
                for car, images in cars.items():
                    possible_choices = cars_list.copy()
                    possible_choices.remove(car)
                    
                    pairs.extend(self.__get_pair(cars, possible_choices, images))
                    pairs.extend(self.__get_pair(cars, possible_choices, images))
                    labels.extend([0, 1, 0, 1])

            datasets[subset] = test_dataset(pairs, labels, self.normal_transform)

        return datasets