import torch
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from randaugment import Cutout, RandAugment, ImageNetPolicy
import os


class ChestXRayDataset:
    def __init__(self, data_dir, image_size):
        image_transformation = {
            'train': transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                Cutout(size=16),
                RandAugment(),
                ImageNetPolicy(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])]),
            'val': transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])
            ])
        }

        self.image_datasets = {x: datasets.ImageFolder(os.path.join(
            data_dir, x), image_transformation[x]) for x in ['train', 'val']}

    def setup_data(self, batch_size):
        dataset_sizes = {x: len(self.image_datasets[x]) for x in ['train', 'val']}
        dataloaders = {
            x: DataLoader(self.image_datasets[x], batch_size=batch_size,
                          shuffle=(x == 'train'), num_workers=4) for x in ['train', 'val']
        }
        return dataset_sizes, dataloaders
