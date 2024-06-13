import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout

class BestModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layer1 = Linear(n_features, 9)
        self.act1 = ReLU()
        self.drop1 = Dropout(0.15)
        self.layer2 = Linear(9,13)
        self.act2 = ReLU()
        self.layer3 = Linear(13,11)
        self.act3 = ReLU()
        self.output = Linear(11,1)

    def forward(self,x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.layer2(x)
        x = self.act2(x)
        x = self.layer3(x)
        x = self.act3(x)
        x = self.output(x)
        return x