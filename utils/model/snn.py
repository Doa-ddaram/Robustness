import torch as th
import torch.nn as nn
from ..spikingjelly.spikingjelly.activation_based import neuron, surrogate, functional, layer
from typing import List

class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.layer1 = nn.Sequential(
            layer.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
            neuron.LIFNode(tau = 2.0, surrogate_function = surrogate.ATan()),
            layer.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            layer.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            neuron.LIFNode(tau = 2.0, surrogate_function = surrogate.ATan()),
            layer.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.linear1 = nn.Linear(in_features=7 * 7 * 64, out_features=10, bias=False)
       # functional.set_step_mode(self, step_mode = 'm')
    def forward(self, x):
        #x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x

# class SNN_VGG(nn.Module):
#     def __init__(self):
#         super(SNN_VGG, self).__init__()
#         self.linear1 = layer.Linear(256 * 8 * 8, 512)
#         self.linear2 = layer.Linear(512, 10)
#         self.layer1 = layer.Sequential(
#             layer.Conv2d(3, 32, kernel_size= 3, padding = 1),
#             neuron.LIFNode(2.0, surrogate_function= surrogate.ATan())
#         )
#         self.layer2 = nn.Sequential(
#             layer.Conv2d(32, 64, kernel_size= 3, padding= 1),
#             neuron.LIFNode(2.0, surrogate_function= surrogate.ATan()),
#             layer.MaxPool2d(kernel_size=2, stride=2)
#         )
#         self.layer3 = nn.Sequential(
#             layer.Conv2d(64, 128, kernel_size= 3, padding = 1),
#             neuron.LIFNode(2.0, surrogate_function= surrogate.ATan())
#         )
#         self.layer4 = nn.Sequential(
#             layer.Conv2d(128, 256, kernel_size= 3, padding= 1),
#             neuron.LIFNode(2.0, surrogate_function= surrogate.ATan()),
#             layer.MaxPool2d(kernel_size= 2, stride= 2)
#         )
#         self.layer5 = nn.Sequential(
#             layer.Linear(256 * 8 * 8, 512),
#             neuron.LIFNode(2.0, surrogate_function= surrogate.ATan()),
#             layer.Dropout(0.5)
#         )
#     def forward(self, x) :
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = x.view(x.size(0), -1)
#         x = self.layer5(x)
#         x = self.linear2(x)
#         return x

def make_layers_SNN(cfg : List, batch_norm : bool = False) : 
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers.append(layer.MaxPool2d(kernel_size= 2, stride= 2))
        else:
            conv2d = layer.Conv2d(in_channels, v, kernel_size= 3, padding = 1)
            if batch_norm :
                layers += [conv2d, layer.BatchNorm2d(v), neuron.LIFNode()]
            else:
                layers += [conv2d, neuron.IFNode()]
            in_channels = v
            
    return nn.Sequential(*layers)


class SNN_VGG(nn.Module):
    def __init__(self, features, num_classes = 10):
        super(SNN_VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            neuron.LIFNode(),
            nn.Dropout(),
            nn.Linear(512,512),
            neuron.LIFNode(),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )
    def forward(self, x) :
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x