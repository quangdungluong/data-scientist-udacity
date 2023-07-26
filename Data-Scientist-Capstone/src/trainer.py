import torch
from tqdm import tqdm
import torch.nn as nn
from src.model import ChestXRayModel
from timm.loss import LabelSmoothingCrossEntropy
from torch.optim import SGD
from src.dataset import ChestXRayDataset

class ChestXRayClassifier():
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])

    def setup_training(self):
        model = ChestXRayModel(self.num_classes)
        loss_criteria = LabelSmoothingCrossEntropy()
        optimizer = SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        return model, loss_criteria, optimizer, device

    def train_model(self):
        [model, loss_criteria, optimizer, device] = self.setup_training()
        dataset_sizes, dataloaders = ChestXRayDataset(self.data_dir, self.image_size).setup_data(self.batch_size)
        num_epochs = self.max_epochs
        for epoch in range(num_epochs):
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

            running_loss = 0.0
            running_correct = 0

            for i, (inputs, labels) in tqdm(enumerate(dataloaders[phase])):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = loss_criteria(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_correct += torch.sum(preds == labels.data).item()

        torch.save(model.state_dict(), self.model_path)
        return model