import sys

sys.path.append('.')
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_set import VOCDataSet, CityscapesSegmentation
from utils.util import make_data_list, DataTransform
from utils.custom_transforms import DataCustomTransform
import numpy as np


class VOCDataLoader(BaseDataLoader):

    def __init__(self, data_dir, phase, batch_size, shuffle=True, num_workers=1):
        color_mean = (0.485, 0.456, 0.406)
        color_std = (0.229, 0.224, 0.225)

        tf = DataTransform(input_size=256, color_mean=color_mean, color_std=color_std)

        self.data_set = VOCDataSet(data_dir=data_dir, transform=tf, phase=phase)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_workers = num_workers

        super(VOCDataLoader, self).__init__(dataset=self.data_set, batch_size=self.batch_size, shuffle=self.shuffle,
                                            num_workers=self.num_workers)


class CityScapesDataLoader(BaseDataLoader):

    def __init__(self, data_dir, phase, batch_size, shuffle=True, num_workers=1):

        tf = DataCustomTransform(base_size=1024, size=475)

        self.data_set = CityscapesSegmentation(root=data_dir, transform=tf, split=phase)

        self.batch_size = batch_size
        self.shuffle = shuffle

        self.num_workers = num_workers

        super(CityScapesDataLoader, self).__init__(dataset=self.data_set, batch_size=self.batch_size,
                                                   shuffle=self.shuffle,
                                                   num_workers=self.num_workers)
