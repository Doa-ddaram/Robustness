import torch as th
import torch.nn as nn
from typing import Callable,List,Dict, Union, Any

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
    
def make_layers_CNN(cfg : List, batch_norm : bool = False) : 
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

cfgs: Dict[str, List[Union[str, int]]] = {
    "A": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "B": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "D": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "E": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}

class VGG(nn.Module):
    def __init__(self, features, num_classes = 10):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096,1000),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1000, num_classes),
        )
        self.flatten = nn.Flatten()
    def forward(self, x) :
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

def _vgg(cfg: str, batch_norm: bool, progress: bool, **kwargs: Any) -> VGG:
    model = VGG(make_layers_CNN(cfgs[cfg], batch_norm=batch_norm), **kwargs)
    return model

def vgg16(*, progress: bool = True, **kwargs: Any) -> VGG:
    return _vgg("D", False, progress, **kwargs)