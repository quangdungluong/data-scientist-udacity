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
        """
        The function "setup_data" takes in a batch size as a parameter and returns the dataset sizes and
        dataloaders for the given batch size.
        
        :param batch_size: The batch size is the number of samples that will be propagated through the
        network at once. It is a hyperparameter that determines the number of samples in each mini-batch
        during training
        :return: two values: dataset_sizes and dataloaders.
        """
        dataset_sizes = len(self.image_datasets)
        dataloaders = DataLoader(self.image_datasets, batch_size=batch_size,
                                 shuffle=False, num_workers=4)
        return dataset_sizes, dataloaders
