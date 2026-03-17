# model.py

import torch.nn as nn
from torchvision import models

class XrayModel(nn.Module):

    def __init__(self):
        super(XrayModel, self).__init__()

        self.model = models.resnet18(pretrained=True)

        # change last layer for 2 classes
        self.model.fc = nn.Linear(512, 2)

    def forward(self, x):
        return self.model(x)
    