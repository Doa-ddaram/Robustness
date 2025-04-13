import torch.nn as nn
import torch as th
from .spikingjelly.spikingjelly.activation_based import neuron, surrogate, functional, layer

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=7 * 7 * 64, out_features=10, bias=False)
    def forward(self, x : th.Tensor) -> th.Tensor:
        '''
        Args:
            x (th.Tensor) : Input tensor, shape (batch_size, 1, 28, 28)
            
        Returns:
            th.Tensor: Output tensor, shape (batch_size, 10)
        '''
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x
    
class SNN(nn.Module):
    def __init__(self, T : int = 20):
        super(SNN, self).__init__()
        self.T = T
        self.layer1 = nn.Sequential(
            layer.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
            neuron.IFNode(),
            layer.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            layer.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            neuron.IFNode(),
            layer.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.flatten = layer.Flatten()
        self.linear = layer.Linear(in_features=7 * 7 * 64, out_features=10, bias=False)
        functional.set_step_mode(self, step_mode = 'm')
        
    def forward(self, x : th.Tensor) -> th.Tensor:
        '''
        Args:
            x (th.Tensor) : Input tensor, shape (batch_size, 1, 28, 28)
            
        Returns:
            th.Tensor: Output tensor, shape (batch_size, 10)
        '''
        x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x