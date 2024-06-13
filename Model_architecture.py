import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Dropout

class BestModel(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.layer1 = Linear(n_features, 20)
        self.act1 = ReLU()
        self.drop1 = Dropout(0.06)
        self.layer2 = Linear(20,10)
        self.act2 = ReLU()
        self.layer3 = Linear(10,15)
        self.act3 = ReLU()
        self.output = Linear(15,1)

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