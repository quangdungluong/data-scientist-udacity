import torch
from tqdm import tqdm
from src.model import ChestXRayModel
from timm.loss import LabelSmoothingCrossEntropy
from torch.optim import SGD
from src.dataset import ChestXRayDataset
from copy import deepcopy

class ChestXRayClassifier():
    def __init__(self, **args):
        for key in args:
            setattr(self, key, args[key])

    def setup_training(self):
        model = ChestXRayModel(self.num_classes)
        loss_criteria = LabelSmoothingCrossEntropy()
        optimizer = SGD(model.parameters(), lr=self.lr, momentum=self.momentum, weight_decay=self.weight_decay)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        return model, loss_criteria, optimizer, device

    def train_model(self):
        [model, loss_criteria, optimizer, device] = self.setup_training()
        best_model_wts = deepcopy(model.state_dict())
        best_acc = 0.

        dataset_sizes, dataloaders = ChestXRayDataset(self.data_dir, self.image_size).setup_data(self.batch_size)
        num_epochs = self.max_epochs
        for epoch in range(num_epochs):
            print("="*5 + f"Epoch {epoch+1}/{num_epochs}" + "="*5)
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
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_correct / dataset_sizes[phase]
            print(f"{phase.upper()}: Loss: {epoch_loss:.3f} - Accuracy: {epoch_acc:.3f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = deepcopy(model.state_dict())
                torch.save(model.state_dict(), self.model_path)

        model.load_state_dict(best_model_wts)
        return model