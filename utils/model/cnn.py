import torch as th
import torch.nn as nn
from typing import Callable,List

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, padding = 1)
        self.pool = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(in_features=7 * 7 * 64, out_features=10, bias=False)
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        '''
        Args:
            x (th.Tensor) : Input tensor, shape (batch_size, 1, 28, 28)
            
        Returns:
            th.Tensor: Output tensor, shape (batch_size, 10)
        '''
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1 )
        x = self.linear(x)
        return x
    
def make_layers(cfg : List, batch_norm : bool = False) : 
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(nn.MaxPool2d(kernel_size= 2, stride= 2))
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size= 3, padding = 1)
            if batch_norm :
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
            
    return nn.Sequential(*layers)


class VGG(nn.Module):
    def __init__(self, features, num_classes = 10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
    def forward(self, x) :
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x