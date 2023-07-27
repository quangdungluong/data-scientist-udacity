import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from randaugment import Cutout, RandAugment, ImageNetPolicy
import os


class ChestXRayDatasetTest:
    def __init__(self, data_dir, image_size):
        image_transformation = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        self.image_datasets = datasets.ImageFolder(os.path.join(
            data_dir, "test"), image_transformation)

    def setup_data(self, batch_size):
        dataset_sizes = len(self.image_datasets)
        dataloaders = DataLoader(self.image_datasets, batch_size=batch_size,
                                 shuffle=False, num_workers=4)
        return dataset_sizes, dataloaders
