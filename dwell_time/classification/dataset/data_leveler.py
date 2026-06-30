from collections import Counter
import random
import torch
import copy

FLATTEN_LINIAR = 5

# Works only with Pytorch ConcatDatasets
class data_leveler():
    def __remove_images_from_dataset(self, dataset, indexes_to_remove):
        current_index = 0

        temp_dataset = []
        for _, subset in enumerate(dataset.datasets):
            subset_len = len(subset)
            indexes = [index - current_index for index in indexes_to_remove if current_index <= index < current_index + subset_len]

            temp_subset = copy.deepcopy(subset)
            for index in sorted(indexes, reverse=True):
                del temp_subset.imgs[index]
                del temp_subset.targets[index]
            temp_subset.samples = temp_subset.imgs

            temp_dataset.append(temp_subset)
            current_index += subset_len
        
        return torch.utils.data.ConcatDataset(temp_dataset)


    def __get_fewer_class_indexes(self, dataset, class_index):
        current_index = 0
        indexes = []

        for _, subset in enumerate(dataset.datasets):
            class_indexes = [current_index + index for index, label in enumerate(subset.targets) if label == class_index]
            indexes.extend(class_indexes)

            current_index += len(subset)
        
        return indexes


    def __flatten_dataset(self, dataset, flatter_object):
        indexes = self.__get_fewer_class_indexes(dataset, flatter_object[1])
        indexes_to_remove = random.sample(indexes, flatter_object[0])

        return self.__remove_images_from_dataset(dataset, indexes_to_remove)


    def __get_flatter_object(self, dataset):
        class_0, class_1 = 0, 0

        for subset in dataset.datasets:
            class_count = Counter(subset.targets)
            class_0 += class_count[0]
            class_1 += class_count[1]
        
        diff = abs(class_0 - class_1)
        diff_perc = round((diff / (class_0 + class_1)) * 100)

        return (diff, 0, diff_perc) if class_0 > class_1 else (diff, 1, diff_perc)


    def flatten_dataset(self, dataset):
        flatter_object = self.__get_flatter_object(dataset)

        return dataset if flatter_object[2] < FLATTEN_LINIAR else self.__flatten_dataset(dataset, flatter_object)