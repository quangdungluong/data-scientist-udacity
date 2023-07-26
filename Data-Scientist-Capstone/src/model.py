import timm
from collections import OrderedDict
import torch.nn as nn


class ChestXRayModel(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.num_classes = num_classes
        self.model = timm.create_model("resnet50", pretrained=True)
        self.classifier = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(self.model.fc.in_features, 512)),
            ('relu1', nn.ReLU()),
            ('fc2', nn.Linear(512, self.num_classes))
        ]))
        self.model.fc = self.classifier

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = ChestXRayModel(5)