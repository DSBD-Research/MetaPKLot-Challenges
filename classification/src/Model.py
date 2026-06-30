from torchvision import models
from torch.utils.data import DataLoader
from time import localtime, strftime
import torch
import torchvision.transforms.v2 as v2
import shutil
import torch.nn as nn
import os
import json
from sklearn.metrics import roc_auc_score

from .PKLotDataset import PKLotDataset
from .customNNLarge import customNNLarge
from .customNNSmall import customNNSmall
from .Hochuli import Hochuli

class Model():
    def __init__(self, config, mode, gpu_index, pretrained_model_path=None):
        self.mode = mode
        self.config = config
        self.config_path = str(config.get('config_path'))
        self.gpu_index = int(gpu_index)
        self.output_dir = str(config.get('output_dir'))
        self.transform_resize = tuple(config.get('transform_resize'))
        self.normalize = bool(config.get('normalize'))
        self.train_val_datasets = list(config.get('train_val_datasets'))
        self.test_datasets = list(config.get('test_datasets'))
        self.batch_size = int(config.get('batch_size'))
        self.num_workers = int(config.get('num_workers'))
        self.model_type = str(config.get('model_type'))
        self.num_epochs = int(config.get('num_epochs'))
        self.pretrained = bool(config.get('pretrained'))
        self.data_augmentation = bool(config.get('data_augmentation'))
        self.data_augmentation_prob = float(config.get('data_augmentation_prob'))
        self.datasets_path = str(config.get('datasets_path'))
        self.val_batch_size = int(config.get('val_batch_size'))
        self.unfrozen_layers = str(config.get('unfrozen_layers'))
        self.models_to_train = str(config.get('models_to_train'))
        self.normalize_train_data = bool(config.get('normalize_train_data'))
        self.gray_scale = bool(config.get('gray_scale'))
        self.training_mode = str(config.get('training_mode'))
        self.pretrained_model_path = pretrained_model_path
        self.weights = None

        self.load_device()

        if mode == "train":
            self.create_train_output_dir()

            if len(list(config.get('mean_std'))) == 0:
                self.calculate_mean_std()
            else:
                self.mean = list(config.get('mean_std')[0])
                self.std = list(config.get('mean_std')[1])
            self.save_mean_std_json()

            self.create_transformer()
            self.load_weights()
            self.create_model()
            self.make_train_dataloaders()
    
    def write_on_log(self, text):
        if self.mode == "train":
            out_dir = self.output_dir
        elif self.mode == "test":
            out_dir = self.test_output_path
        elif self.mode == "distill":
            out_dir = self.distill_output_path

        time = strftime("%Y-%m-%d %H:%M:%S - ", localtime())
        mode = "w" if not os.path.exists(os.path.join(out_dir, "log.txt")) else "a"

        with open(os.path.join(out_dir, "log.txt"), mode) as file:
            file.write(time + text + "\n")
    
    def match_model(self):
        if self.gray_scale:
            input_shape = (1, self.transform_resize[0], self.transform_resize[1])
        else:
            input_shape = (3, self.transform_resize[0], self.transform_resize[1])
        
        model = None
        match self.model_type:
            case "mobilenet_v3_small":
                model = models.mobilenet_v3_small(weights=self.weights)
                model.classifier[3] = nn.Linear(model.classifier[0].out_features, 2)
            case "mobilenet_v3_large":
                model = models.mobilenet_v3_large(weights=self.weights)
                model.classifier[3] = nn.Linear(model.classifier[0].out_features, 2)
            case "customNNLarge":
                model = customNNLarge(input_shape=input_shape)
                if self.weights is not None:
                    model.load_state_dict(self.weights)
            case "customNNSmall":
                try:
                    model = customNNSmall(input_shape=input_shape)
                    if self.weights is not None:
                        model.load_state_dict(self.weights)
                except: # 200 classes
                    model = customNNSmall(num_classes=200, input_shape=input_shape)
                    if self.weights is not None:
                        model.load_state_dict(self.weights)
                    model.fc2 = nn.Linear(in_features=model.fc1.out_features, out_features=2)
            case "hochuli":
                model = Hochuli(input_shape=input_shape)
                if self.weights is not None:
                    model.load_state_dict(self.weights)
            case "vit_b_16":
                model = models.vit_b_16(weights=self.weights)
                model.heads.head = torch.nn.Linear(in_features=model.heads.head.in_features, out_features=2)
            case _:
                raise ValueError(f"Model type {self.model_type} not found.")
        
        return model

    def create_train_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        shutil.copy(self.config_path, os.path.join(self.output_dir, "config.yaml"))

    def create_distill_output_dir(self):
        os.makedirs(self.output_dir, exist_ok=True)
        shutil.copy(self.config_path, os.path.join(self.output_dir, "config.yaml"))

    def load_device(self):
        self.device = torch.device(f"cuda:{self.gpu_index}")
    
    def calculate_mean_std(self):
        subsets = []
        for dataset in self.train_val_datasets:
            _subsets = os.listdir(self.datasets_path + dataset)
            subsets.extend([self.datasets_path + dataset + "/" + subset for subset in _subsets])

        transformer = v2.Compose([
            v2.Resize(self.transform_resize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        dataset = PKLotDataset("first_70", subsets, transformer, normalize_train_data=False, gray_scale=self.gray_scale)
        loader = DataLoader(dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, pin_memory=True)

        if self.gray_scale:
            num_channels = 1
        else:
            num_channels = 3

        mean = torch.zeros(num_channels, device=self.device)
        std = torch.zeros(num_channels, device=self.device)
        total_samples = 0
        for images, _, _ in loader:
            images = images.to(self.device)
            batch_samples = images.size(0)
            total_samples += batch_samples
            for i in range(num_channels):
                mean[i] += images[:, i, :, :].mean().item() * batch_samples
                std[i] += images[:, i, :, :].std().item() * batch_samples

        self.mean = [m.cpu().item() / total_samples for m in mean]
        self.std = [s.cpu().item() / total_samples for s in std]
    
    def save_mean_std_json(self):
        with open(os.path.join(self.output_dir, "mean_std.json"), "w") as file:
            json.dump({"mean": [float(m) for m in self.mean], "std": [float(s) for s in self.std]}, file)
    
    def set_mean_std_json(self, json_path):
        json_file = json.load(open(json_path, "r"))
        
        self.mean, self.std = json_file["mean"], json_file["std"]

    def create_transformer(self):
        if not self.normalize:
            if not self.data_augmentation:
                self.transformer = v2.Compose([
                    v2.Resize(self.transform_resize),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)
                    ])
                ])
            else:
                self.transformer = v2.Compose([
                    v2.Resize(self.transform_resize),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)
                    ]),
                    v2.RandomHorizontalFlip(p=self.data_augmentation_prob),
                    v2.RandomAutocontrast(p=self.data_augmentation_prob),
                    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=self.data_augmentation_prob),
                    v2.RandomApply([v2.RandomCrop(size=self.transform_resize)], p=self.data_augmentation_prob)

                ])
        else:
            if not self.data_augmentation:
                self.transformer = v2.Compose([
                    v2.Resize(self.transform_resize),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)
                    ]),
                    v2.Normalize(mean=self.mean, std=self.std)
                ])
            else:
                self.transformer = v2.Compose([
                    v2.Resize(self.transform_resize),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)
                    ]),
                    v2.Normalize(mean=self.mean, std=self.std),
                    v2.RandomHorizontalFlip(p=self.data_augmentation_prob),
                    v2.RandomAutocontrast(p=self.data_augmentation_prob),
                    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=self.data_augmentation_prob),
                    v2.RandomApply([v2.RandomCrop(size=self.transform_resize)], p=self.data_augmentation_prob)
                ])

    def create_student_transformer(self, model_type, use_data_augmentation, augmentation_prob):
        resize = None
        if model_type == "mobilenet_v3_large" or model_type == "mobilenet_v3_small":
            resize = (128, 128)
        elif model_type == "customNNLarge" or model_type == "customNNSmall" or model_type == "hochuli":
            resize = (32, 32)
        elif model_type == "vit_b_16":
            resize = (224, 224)

        if not self.normalize:
            if use_data_augmentation == "False":
                self.student_transformer = v2.Compose([
                    v2.Resize(resize),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)
                    ])
                ])
            elif use_data_augmentation == "True":
                self.student_transformer = v2.Compose([
                    v2.Resize(resize),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)
                    ]),
                    v2.RandomHorizontalFlip(p=augmentation_prob),
                    v2.RandomAutocontrast(p=augmentation_prob),
                    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=augmentation_prob),
                    v2.RandomApply([v2.RandomCrop(size=resize)], p=augmentation_prob)
                ])
            else:
                raise ValueError(f"Invalid value for use_data_augmentation: {use_data_augmentation}")
        else:
            if use_data_augmentation == "False":
                self.student_transformer = v2.Compose([
                    v2.Resize(resize),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)
                    ]),
                    v2.Normalize(mean=self.student_mean, std=self.student_std)
                ])
            elif use_data_augmentation == "True":
                self.student_transformer = v2.Compose([
                    v2.Resize(resize),
                    v2.Compose([
                        v2.ToImage(),
                        v2.ToDtype(torch.float32, scale=True)
                    ]),
                    v2.Normalize(mean=self.student_mean, std=self.student_std),
                    v2.RandomHorizontalFlip(p=augmentation_prob),
                    v2.RandomAutocontrast(p=augmentation_prob),
                    v2.RandomApply([v2.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5))], p=augmentation_prob),
                    v2.RandomApply([v2.RandomCrop(size=resize)], p=augmentation_prob)
                ])
            else:
                raise ValueError(f"Invalid value for use_data_augmentation: {use_data_augmentation}")

    def save_student_mean_std_json(self):
        with open(os.path.join(self.distill_output_path, "student_mean_std.json"), "w") as file:
            json.dump({"mean": [float(m) for m in self.student_mean], "std": [float(s) for s in self.student_std]}, file)
    
    def load_weights(self):
        if self.model_type == "mobilenet_v3_small":
            self.weights = models.MobileNet_V3_Small_Weights.DEFAULT if self.pretrained else None
        elif self.model_type == "mobilenet_v3_large":
            self.weights = models.MobileNet_V3_Large_Weights.DEFAULT if self.pretrained else None
        elif self.model_type == "vit_b_16":
            self.weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if self.pretrained else None
        elif self.pretrained_model_path is not None and (self.model_type == "customNNLarge" or self.model_type == "customNNSmall" or self.model_type == "hochuli"):
            self.weights = torch.load(self.pretrained_model_path, map_location=self.device, weights_only=True)

    def create_model(self):
        self.load_weights()
        self.model = self.match_model()

        self.model.to(self.device)

        for param in self.model.parameters():
            param.requires_grad = False
        
        if self.unfrozen_layers == "all":
            for param in self.model.parameters():
                param.requires_grad = True
        elif self.unfrozen_layers == "classifier":
            if self.model_type == "vit_b_16":
                for param in self.model.heads.head.parameters():
                    param.requires_grad = True
            else:
                for param in self.model.classifier.parameters():
                    param.requires_grad = True
        elif self.unfrozen_layers == "classifier_and_last_conv":
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            for param in self.model.features[-1].parameters():
                param.requires_grad = True

    def freeze_student_layers(self):
        if self.student_model_type == "customNNLarge" or self.student_model_type == "customNNSmall" or self.student_model_type == "hochuli":
            for param in self.student_model.parameters():
                param.requires_grad = True
            
            return

        for param in self.student_model.parameters():
            param.requires_grad = False

        if self.unfrozen_layers == "all":
            for param in self.student_model.parameters():
                param.requires_grad = True
        elif self.unfrozen_layers == "classifier":
            if self.student_model_type == "vit_b_16":
                for param in self.student_model.heads.head.parameters():
                    param.requires_grad = True
            else:
                for param in self.student_model.classifier.parameters():
                    param.requires_grad = True
        elif self.unfrozen_layers == "classifier_and_last_conv":
            for param in self.student_model.classifier.parameters():
                param.requires_grad = True
            for param in self.student_model.features[-1].parameters():
                param.requires_grad = True
    
    def make_train_dataloaders(self):
        self.dataloaders = []

        subsets = []
        for dataset in self.train_val_datasets:
            _subsets = os.listdir(self.datasets_path + dataset)
            subsets.extend([self.datasets_path + dataset + "/" + subset for subset in _subsets])

        if self.models_to_train == "total":
            train_dataset = PKLotDataset("first_70", subsets, self.transformer, normalize_train_data=self.normalize_train_data,
                                         gray_scale=self.gray_scale)
            val_dataset = PKLotDataset("last_30", subsets, self.transformer, normalize_train_data=self.normalize_train_data, gray_scale=self.gray_scale)

            self.dataloaders.append([DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True),
                                    DataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)])

            self.train_val_datasets = ["total"]
            self.val_subsets = ["total"]

            return


        if self.training_mode == "total_ensemble":
            if self.models_to_train != "ensemble":
                print("Training mode is total_ensemble, but models_to_train is not ensemble.")
                exit(1)

            self.train_val_datasets = []
            self.val_subsets = []
            for i_subset in range(len(subsets)):
                self.train_val_datasets.append([])
                self.val_subsets.append([])
                self.dataloaders.append([])
                for j_subset in range(len(subsets)):
                    if i_subset == j_subset:
                        self.val_subsets[i_subset].append(subsets[j_subset])
                    else:
                        self.train_val_datasets[i_subset].append(subsets[j_subset])
                
                train_dataset = PKLotDataset("all", self.train_val_datasets[i_subset], 
                                            self.transformer, normalize_train_data=self.normalize_train_data, gray_scale=self.gray_scale)
                val_dataset = PKLotDataset("all", self.val_subsets[i_subset], self.transformer, normalize_train_data=self.normalize_train_data,
                                            gray_scale=self.gray_scale)

                train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True,
                                            num_workers=self.num_workers, pin_memory=True)
                val_dataloader = DataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=False, 
                                            num_workers=self.num_workers, pin_memory=True)

                self.dataloaders[i_subset].append(train_dataloader)
                self.dataloaders[i_subset].append(val_dataloader)
            
            self.train_val_datasets.append("total")
            self.val_subsets.append("total")
            
            train_dataset = PKLotDataset("first_70", subsets, self.transformer, normalize_train_data=self.normalize_train_data,
                                        gray_scale=self.gray_scale)
            val_dataset = PKLotDataset("last_30", subsets, self.transformer, normalize_train_data=self.normalize_train_data, gray_scale=self.gray_scale)

            self.dataloaders.append([DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True),
                                    DataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)])

        elif self.training_mode == "partial_ensemble":
            if len(self.train_val_datasets) != 2:
                print("Training mode is partial_ensemble, but train_val_datasets doesn't have 2 elements.")
                exit(1)

            self.val_subsets = []

            self.val_subsets.append("total")
            
            train_dataset = PKLotDataset("first_70", subsets, self.transformer, normalize_train_data=self.normalize_train_data,
                                        gray_scale=self.gray_scale)
            val_dataset = PKLotDataset("last_30", subsets, self.transformer, normalize_train_data=self.normalize_train_data, gray_scale=self.gray_scale)

            self.dataloaders.append([DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True),
                                    DataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)])
            
            _subsets_0 = os.listdir(self.datasets_path + self.train_val_datasets[0])
            _subsets_0 = [self.datasets_path + self.train_val_datasets[0] + "/" + subset for subset in _subsets_0]
            _subsets_1 = os.listdir(self.datasets_path + self.train_val_datasets[1])
            _subsets_1 = [self.datasets_path + self.train_val_datasets[1] + "/" + subset for subset in _subsets_1]

            train_dataset = PKLotDataset("all", _subsets_0, self.transformer, normalize_train_data=self.normalize_train_data,
                                            gray_scale=self.gray_scale)
                
            val_dataset = PKLotDataset("all", _subsets_1, self.transformer, normalize_train_data=self.normalize_train_data, 
                                        gray_scale=self.gray_scale)
                
            self.dataloaders.append([DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True),
                                    DataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)])
            
            train_dataset = PKLotDataset("all", _subsets_1, self.transformer, normalize_train_data=self.normalize_train_data,
                                            gray_scale=self.gray_scale)
            
            val_dataset = PKLotDataset("all", _subsets_0, self.transformer, normalize_train_data=self.normalize_train_data,
                                        gray_scale=self.gray_scale)
            
            self.dataloaders.append([DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=True),
                                    DataLoader(val_dataset, batch_size=self.val_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=True)])
            
            self.val_subsets.append(self.train_val_datasets[0] + "_partial_ensemble")
            self.val_subsets.append(self.train_val_datasets[1] + "_partial_ensemble")

    def make_test_dataloaders(self):
        self.dataloaders = []
    
        subsets = []
        for dataset in self.test_datasets:
            _subsets = os.listdir(self.datasets_path + dataset)
            subsets.extend([self.datasets_path + dataset + "/" + subset for subset in _subsets])
        
        test_dataset = PKLotDataset("all", subsets, self.transformer, normalize_train_data=False, gray_scale=self.gray_scale)
        self.dataloaders.append(DataLoader(test_dataset, shuffle=False, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True))

        self.set_dataloader()

    def make_distill_dataloaders(self, train_days, val_days, grayscale):
        self.student_train_dataloaders = []
        self.student_val_dataloaders = []
        self.student_test_dataloaders = []
        self.student_subsets_names = []

        if grayscale == "True":
            grayscale = True
        else:
            grayscale = False

        for test_dataset_ in self.test_datasets:
            subsets = os.listdir(self.datasets_path + test_dataset_)

            self.student_subsets_names.extend(subsets)
            
            '''
            Using grayscale on val and test because it's only used for validating and testing the student models
            the train is used to get post threshold images, so it can't be grayscale
            '''
            for subset in subsets:
                train_dataset = PKLotDataset("distill_train", [self.datasets_path + test_dataset_ + "/" + subset], self.transformer,
                                            normalize_train_data=self.normalize_train_data, gray_scale=self.gray_scale, 
                                            distill_train_days=train_days, distill_val_days=val_days)
                
                val_dataset = PKLotDataset("distill_val", [self.datasets_path + test_dataset_ + "/" + subset], self.student_transformer,
                                            normalize_train_data=self.normalize_train_data, gray_scale=grayscale,
                                            distill_train_days=train_days, distill_val_days=val_days)
                
                test_dataset = PKLotDataset("distill_test", [self.datasets_path + test_dataset_ + "/" + subset], self.student_transformer,
                                            normalize_train_data=False, gray_scale=grayscale, 
                                            distill_train_days=train_days, distill_val_days=val_days)

                self.student_train_dataloaders.append(DataLoader(train_dataset, 
                                                                 batch_size=self.batch_size, shuffle=True, 
                                                                 num_workers=self.num_workers, pin_memory=True))
                self.student_val_dataloaders.append(DataLoader(val_dataset, 
                                                                 batch_size=self.val_batch_size, shuffle=False, 
                                                                 num_workers=self.num_workers, pin_memory=True))
                self.student_test_dataloaders.append(DataLoader(test_dataset, 
                                                                 batch_size=self.batch_size, shuffle=False,
                                                                 num_workers=self.num_workers, pin_memory=True))

        subsets = []
        for dataset in self.train_val_datasets:
            _subsets = os.listdir(self.datasets_path + dataset)
            subsets.extend([self.datasets_path + dataset + "/" + subset for subset in _subsets])

        student_pretrain_train_dataset = PKLotDataset("first_70", subsets, self.student_transformer, 
                                                      normalize_train_data=self.normalize_train_data, gray_scale=grayscale)
        student_pretrain_val_dataset = PKLotDataset("last_30", subsets, self.student_transformer, 
                                                      normalize_train_data=self.normalize_train_data, gray_scale=grayscale)

        self.student_pretrain_train_dataloader = DataLoader(student_pretrain_train_dataset, batch_size=self.batch_size, 
                                                            shuffle=True, num_workers=self.num_workers, pin_memory=True)
        self.student_pretrain_val_dataloader = DataLoader(student_pretrain_val_dataset, batch_size=self.val_batch_size, 
                                                            shuffle=False, num_workers=self.num_workers, pin_memory=True)

    def create_post_threshold_distill_train_dataloader(self, images_paths, grayscale):
        if grayscale == "True":
            grayscale = True
        else:
            grayscale = False

        post_threshold_dataset = PKLotDataset(mode="post_threshold", transformer=self.student_transformer, 
                                              images_paths=images_paths, gray_scale=grayscale)
        self.post_threshold_train_dataloader = DataLoader(post_threshold_dataset, batch_size=self.batch_size, 
                                              shuffle=True, num_workers=self.num_workers, pin_memory=True)

    def get_post_threshold_train_dataloader(self):
        return self.post_threshold_train_dataloader

    def get_student_pretrain_train_dataloader(self):
        return self.student_pretrain_train_dataloader

    def get_student_pretrain_val_dataloader(self):
        return self.student_pretrain_val_dataloader
    
    def load_student_weights(self):
        self.student_weights = None
        if self.student_model_type == "mobilenet_v3_small":
            self.student_weights = models.MobileNet_V3_Small_Weights.DEFAULT if self.pretrained else None
        elif self.student_model_type == "mobilenet_v3_large":
            self.student_weights = models.MobileNet_V3_Large_Weights.DEFAULT if self.pretrained else None
        elif self.student_model_type == "vit_b_16":
            self.student_weights = models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1 if self.pretrained else None
        elif self.pretrained_model_path is not None and (self.student_model_type == "customNNLarge" or self.student_model_type == "customNNSmall" or self.student_model_type == "hochuli"):
            self.student_weights = torch.load(self.pretrained_model_path, map_location=self.device, weights_only=True)
    
    def load_student_model_type(self, student_model_type, grayscale):
        self.student_model_type = student_model_type
        self.load_student_weights()

        channels = 1 if grayscale == "True" else 3
        if student_model_type == "customNNLarge":
            if self.student_weights is not None:
                self.student_model = customNNLarge(num_classes=200, input_shape=(channels, 32, 32))
                self.student_model.load_state_dict(self.student_weights)
                self.student_model.fc2 = nn.Linear(in_features=self.student_model.fc1.out_features, out_features=2)
            else:
                self.student_model = customNNLarge(input_shape=(channels, 32, 32))
        elif student_model_type == "customNNSmall":
            if self.student_weights is not None:
                self.student_model = customNNSmall(num_classes=200, input_shape=(channels, 32, 32))
                self.student_model.load_state_dict(self.student_weights)
                self.student_model.fc2 = nn.Linear(in_features=self.student_model.fc1.out_features, out_features=2)
            else:
                self.student_model = customNNSmall(input_shape=(channels, 32, 32))
        elif student_model_type == "hochuli":
            if self.student_weights is not None:
                self.student_model = Hochuli(num_classes=200, input_shape=(channels, 32, 32))
                self.student_model.load_state_dict(self.student_weights)
                self.student_model.fc2 = nn.Linear(in_features=self.student_model.fc1.out_features, out_features=2)
            else:
                self.student_model = Hochuli(input_shape=(channels, 32, 32))
        elif student_model_type == "mobilenet_v3_small":
            self.student_model = models.mobilenet_v3_small(weights=self.student_weights)
            self.student_model.classifier[3] = nn.Linear(self.student_model.classifier[0].out_features, 2)
        elif student_model_type == "mobilenet_v3_large":
            self.student_model = models.mobilenet_v3_large(weights=self.student_weights)
            self.student_model.classifier[3] = nn.Linear(self.student_model.classifier[0].out_features, 2)
        elif student_model_type == "vit_b_16":
            self.student_model = models.vit_b_16(weights=self.student_weights)
            self.student_model.heads.head = torch.nn.Linear(in_features=self.student_model.heads.head.in_features, out_features=2)
        else:
            raise ValueError(f"Student model type {student_model_type} not found.")
        
        self.freeze_student_layers()
        self.student_model.to(self.device)

    def calculate_mean_std_student(self, model_type, grayscale):
        if grayscale == "True":
            grayscale = True
        elif grayscale == "False":
            grayscale = False

        if self.pretrained and not grayscale:
            self.student_mean, self.student_std = self.mean, self.std
            return
        
        self.write_on_log("Calculating mean and std for student model.")
        
        subsets = []
        for dataset in self.train_val_datasets:
            _subsets = os.listdir(self.datasets_path + dataset)
            subsets.extend([self.datasets_path + dataset + "/" + subset for subset in _subsets])

        if model_type == "mobilenet_v3_small" or model_type == "mobilenet_v3_large":
            resize = (128, 128)
        elif model_type == "customNNLarge" or model_type == "customNNSmall" or model_type == "hochuli":
            resize = (32, 32)

        transformer = v2.Compose([
            v2.Resize(resize),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True)
        ])

        dataset = PKLotDataset("first_70", subsets, transformer, normalize_train_data=False, gray_scale=grayscale)
        loader = DataLoader(dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, pin_memory=True)

        if grayscale:
            num_channels = 1
        else:
            num_channels = 3

        mean = torch.zeros(num_channels, device=self.device)
        std = torch.zeros(num_channels, device=self.device)
        total_samples = 0
        for images, _, _ in loader:
            images = images.to(self.device)
            batch_samples = images.size(0)
            total_samples += batch_samples
            for i in range(num_channels):
                mean[i] += images[:, i, :, :].mean().item() * batch_samples
                std[i] += images[:, i, :, :].std().item() * batch_samples

        self.student_mean = [m.cpu().item() / total_samples for m in mean]
        self.student_std = [s.cpu().item() / total_samples for s in std]
    
    def save_student_eer_threshold(self, eer_threshold):
        if os.path.exists(f"{self.distill_output_path}thresholds.json"):
            json_file = json.load(open(f"{self.distill_output_path}thresholds.json", "r"))
            json_file[self.student_model_name] = float(eer_threshold)
        else:
            json_file = {self.student_model_name: float(eer_threshold)}
        
        with open(f"{self.distill_output_path}thresholds.json", "w") as file:
            json.dump(json_file, file)

    def load_student_base_model(self, grayscale):
        self.load_student_model_type(self.student_model_type, grayscale)
        self.student_model.load_state_dict(torch.load(self.student_base_model_path, weights_only=True, map_location=self.device))
        self.student_model.to(self.device)

    def save_student_base_eer_threshold(self, eer_threshold):
        if os.path.exists(f"{self.distill_output_path}thresholds.json"):
            json_file = json.load(open(f"{self.distill_output_path}thresholds.json", "r"))
            json_file["student"] = float(eer_threshold)
        else:
            json_file = {"student": float(eer_threshold)}
        
        with open(f"{self.distill_output_path}thresholds.json", "w") as file:
            json.dump(json_file, file)

    def set_dataloader(self, i=0):
        if self.mode == "train":
            self.train_loader = self.dataloaders[i][0]
            self.val_loader = self.dataloaders[i][1]
            if self.training_mode == "total_ensemble":
                if i == self.get_num_dataloaders() - 1:
                    self.model_name = "total"
                else:
                    self.model_name = f"val_{str(self.val_subsets[i][0].split('/')[-1])}"
            elif self.training_mode == "partial_ensemble":
                    self.model_name = f"val_{str(self.val_subsets[i])}"

        elif self.mode == "test":
            self.test_loader = self.dataloaders[i]

        elif self.mode == "distill":
            self.student_train_loader = self.student_train_dataloaders[i]
            self.student_val_loader = self.student_val_dataloaders[i]
            self.student_test_loader = self.student_test_dataloaders[i]
            self.student_model_name = self.student_subsets_names[i]

        elif self.mode == "test":
            self.test_loader = self.dataloaders[i]

        elif self.mode == "distill":
            self.student_train_loader = self.student_train_dataloaders[i]
            self.student_val_loader = self.student_val_dataloaders[i]
            self.student_test_loader = self.student_test_dataloaders[i]
            self.student_model_name = self.student_subsets_names[i]
    
    def get_distill_output_path(self):
        return self.distill_output_path

    def get_student_model_name(self):
        return self.student_model_name
    
    def get_student_model(self):
        return self.student_model
    
    def get_student_model_parameters(self):
        return self.student_model.parameters()

    def get_distill_num_loaders(self):
        return len(self.student_subsets_names)
    
    def set_student_model(self, idx, grayscale):
        self.student_model_path = self.distill_output_path + f"models/{self.student_subsets_names[idx]}.pth"
        self.load_student_model_type(self.student_model_type, grayscale)
        self.student_model.load_state_dict(torch.load(self.student_model_path, weights_only=True, map_location=self.device))
        self.student_model.to(self.device)

    def get_num_dataloaders(self):
        return len(self.dataloaders)
    
    def save_model(self):
        os.makedirs(os.path.join(self.output_dir, "models/"), exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, f"models/{self.model_name}.pth"))
    
    def save_student_model(self):
        os.makedirs(os.path.join(self.distill_output_path, "models/"), exist_ok=True)
        torch.save(self.student_model.state_dict(), os.path.join(self.distill_output_path, f"models/{self.student_model_name}.pth"))

    def save_student_base_model(self):
        os.makedirs(os.path.join(self.distill_output_path, "models/"), exist_ok=True)
        torch.save(self.student_model.state_dict(), os.path.join(self.distill_output_path, f"models/student.pth"))
        self.student_base_model_path = self.distill_output_path + "models/student.pth"

    def set_student_base_model(self, student_base_model, grayscale):
        self.load_student_model_type(student_base_model, grayscale)
        self.student_model.load_state_dict(torch.load(self.student_base_model_path, weights_only=True, map_location=self.device))
        self.student_model.to(self.device)
        self.freeze_student_layers()

    def set_student_model_to_train(self):
        self.student_model.train()
    
    def set_student_model_to_eval(self):
        self.student_model.eval()
    
    def predict_student_model(self, images):
        return self.student_model(images)

    def save_distill_configs(self, train_days, val_days, threshold, student_model_type, grayscale):
        with open(os.path.join(self.distill_output_path, "distill_configs.json"), "w") as file:
            json.dump({"train_days": int(train_days), "val_days": int(val_days), "threshold": float(threshold),
                    "student_model_type": str(student_model_type), "grayscale": str(grayscale)}, file)

    def get_student_train_dataloader(self):
        return self.student_train_loader

    def get_student_val_dataloader(self):
        return self.student_val_loader
    
    def get_student_test_dataloader(self):
        return self.student_test_loader

    def get_parameters(self):
        return self.model.parameters()
    
    def get_student_parameters(self):
        return self.student_model.parameters()
    
    def set_to_train(self):
        self.model.train()

    def set_to_eval(self):
        self.model.eval()
    
    def get_num_epochs(self):
        return self.num_epochs
    
    def get_train_loader(self):
        return self.train_loader
    
    def get_val_loader(self):
        return self.val_loader
    
    def get_device(self):
        return self.device

    def get_data_augmentation(self):
        return self.data_augmentation
    
    def get_data_augmentation_prob(self):
        return self.data_augmentation_prob
    
    def load_distill_teacher_model(self, i):
        self.model = self.match_model()
        self.model.load_state_dict(torch.load(self.teacher_models_paths[i], weights_only=True, map_location=self.device))
        self.model_name = self.teacher_models_paths[i].split("/")[-1].split(".")[0]

        self.model.to(self.device)
        self.model.eval()

    def load_distill_total_teacher_model(self):
        self.model = self.match_model()

        i = None
        for n, mod in enumerate(self.teacher_models_paths):
            if "total" in mod:
                i = n

        self.model.load_state_dict(torch.load(self.teacher_models_paths[i], weights_only=True, map_location=self.device))
        self.model_name = self.teacher_models_paths[i].split("/")[-1].split(".")[0]

        self.model.to(self.device)
        self.model.eval()
    
    def predict(self, images):
        return self.model(images)
    
    def get_transform_resize(self):
        return self.transform_resize
    
    def clear(self):
        del self.model
        torch.cuda.empty_cache()
        self.create_model()

    def get_model_name(self):
        return self.model_name
    
    def calculate_executions(self):
        self.n_executions = len(os.listdir(self.models_path))
    
    def set_test_dir(self, output_dir, n):
        self.test_dir = output_dir + f"execution{n}/test/"
        os.makedirs(self.test_dir, exist_ok=True)

    def set_test_models_paths(self, models_path):
        models = os.listdir(models_path)
        self.path_to_each_test_model = [f"{models_path}{model}" for model in models]

    def get_n_models(self):
        return len(self.path_to_each_test_model)

    def get_n_teacher_distill_models(self):
        return len(self.teacher_models_paths)
    
    def save_pseudo_targets(self, pseudo_targets):
        with open(f"{self.distill_output_path}pseudo_targets.json", "w") as file:
            json.dump(pseudo_targets, file)
    
    def load_model(self, i):
        self.model = self.match_model()

        self.model.load_state_dict(torch.load(self.path_to_each_test_model[i], weights_only=True, map_location=self.device))
        self.model_name = self.path_to_each_test_model[i].split("/")[-1].split(".")[0]

        self.model.to(self.device)
        self.model.eval()

        self.eer_threshold = self.thresholds[self.model_name]

    def set_thresholds(self, thresholds_json_path, use_roc_eer_thresholds):
        thresholds_json = json.load(open(thresholds_json_path, "r"))
        if use_roc_eer_thresholds == "True":
            self.thresholds = {model: thresholds_json[model] for model in thresholds_json}
        elif use_roc_eer_thresholds == "False":
            self.thresholds = {model: 0.5 for model in thresholds_json}

    def get_test_loader(self):
        return self.test_loader
    
    def get_output_dir(self):
        return self.output_dir
    
    def set_test_output_path(self, test_output_path):
        self.test_output_path = test_output_path
        os.makedirs(self.test_output_path, exist_ok=True)
    
    def set_teacher_models_paths(self, models_path):
        models = os.listdir(models_path)
        self.teacher_models_paths = [models_path + model for model in models]

    def set_distill_output_path(self, distill_output_path, distill_n):
        os.makedirs(distill_output_path, exist_ok=True)

        self.distill_output_path = distill_output_path + f"distill{distill_n}/"
        os.makedirs(self.distill_output_path, exist_ok=True)
    
    def save_test_json(self, predictions, labels, images_paths):
        predictions = [float(prediction) for prediction in predictions]
        labels = [int(label) for label in labels]
        images_paths = [str(image_path) for image_path in images_paths]

        os.makedirs(f"{self.test_output_path}jsons/", exist_ok=True)
        with open(f"{self.test_output_path}jsons/{self.model_name}.json", "w") as file:
            json.dump({"predictions": predictions, "labels": labels, "images_paths": images_paths}, file)

    def save_test_student_model_json(self, predictions, labels):
        predictions = [float(prediction) for prediction in predictions]
        labels = [int(label) for label in labels]

        os.makedirs(f"{self.distill_output_path}jsons/", exist_ok=True)
        with open(f"{self.distill_output_path}jsons/{self.student_model_name}.json", "w") as file:
            json.dump({"predictions": predictions, "labels": labels}, file)

    def save_test_student_base_model_json(self, predictions, labels):
        predictions = [float(prediction) for prediction in predictions]
        labels = [int(label) for label in labels]

        os.makedirs(f"{self.distill_output_path}jsons/", exist_ok=True)
        with open(f"{self.distill_output_path}jsons/student_{self.student_model_name}.json", "w") as file:
            json.dump({"predictions": predictions, "labels": labels}, file)

    def set_execution(self, execution):
        self.execution = execution
        self.execution_path = self.output_dir + f"execution{execution}/"

        models = os.listdir(self.execution_path + "models")
        self.models_paths = [self.execution_path + "models/" + model for model in models]
    
    def get_models_paths(self):
        return self.models_paths
    
    def get_thresholds_dataloader(self):
        return self.thresholds_dataloader
    
    def get_models_to_train(self):
        return self.models_to_train
    
    def save_on_thresholds_json(self, eer_threshold):
        eer_threshold = float(eer_threshold)
        
        if os.path.exists(f"{self.output_dir}thresholds.json"):
            json_file = json.load(open(f"{self.output_dir}thresholds.json", "r"))
            json_file[self.model_name] = eer_threshold
        else:
            json_file = {self.model_name: eer_threshold}
        
        with open(f"{self.output_dir}thresholds.json", "w") as file:
            json.dump(json_file, file)
    
    def get_student_threshold_json(self):
        return json.load(open(f"{self.distill_output_path}thresholds.json", "r"))

    def save_single_metrics(self, predictions, labels):
        accuracy = self.calculate_single_accuracy(predictions, labels, self.eer_threshold)
        roc = self.calculate_roc(predictions, labels)

        dict_ = {
            "accuracy": accuracy,
            "roc": roc,
            "threshold": self.eer_threshold
        }

        self.append_on_json(f"{self.test_output_path}metrics.json", self.model_name, dict_)

    def save_single_metrics_distill(self, predictions, labels, threshold=0.5):
        accuracy = self.calculate_single_accuracy(predictions, labels, threshold)
        roc = self.calculate_roc(predictions, labels)

        dict_ = {
            "accuracy": accuracy,
            "roc": roc,
            "threshold": threshold,
        }

        self.append_on_json(f"{self.distill_output_path}metrics.json", self.student_model_name, dict_)

    def save_single_metrics_base_distill(self, predictions, labels, threshold=0.5):
        accuracy = self.calculate_single_accuracy(predictions, labels, threshold)
        roc = self.calculate_roc(predictions, labels)

        dict_ = {
            "accuracy": accuracy,
            "roc": roc,
            "threshold": threshold,
        }

        self.append_on_json(f"{self.distill_output_path}metrics.json", f"student_{self.student_model_name}", dict_)

    def save_ensemble_metrics(self, l_predictions, l_gd_labels):
        accuracy_prob_sum, roc_prob_sum = self.calculate_prob_sum_metrics(l_predictions, l_gd_labels)

        dict_ = {
            "accuracy_prob_sum": accuracy_prob_sum,
            "roc_prob_sum": roc_prob_sum,
        }

        self.append_on_json(f"{self.test_output_path}metrics.json", "ensemble", dict_)

    def calculate_prob_sum_metrics(self, l_probs, l_labels):
        prob_sum = []
        for prob in l_probs[0]:
            prob_sum.append(prob)
        for probs in l_probs[1:]:
            for i, prob in enumerate(probs):
                prob_sum[i] += prob
        
        num_models = len(l_probs)

        predictions = [prob / num_models for prob in prob_sum]
        labels = l_labels[0]

        accuracy = self.calculate_single_accuracy(predictions, labels, threshold=0.5)
        roc = self.calculate_roc(predictions, labels)

        return accuracy, roc

    def calculate_roc(self, predictions, labels):
        return roc_auc_score(labels, predictions)

    def calculate_single_accuracy(self, predictions, labels, threshold):
        predictions = [1 if prediction >= threshold else 0 for prediction in predictions]
        accuracy = sum([1 if prediction == label else 0 for prediction, label in zip(predictions, labels)]) / len(predictions)

        return accuracy

    def append_on_json(self, result_json_path, model_name, dict_):
        if os.path.exists(result_json_path):
            result_json = json.load(open(result_json_path, "r"))
        else:
            result_json = {}

        result_json[model_name] = dict_

        with open(result_json_path, "w") as file:
            json.dump(result_json, file)
