import torch.nn as nn
from ..spikingjelly.spikingjelly.activation_based import layer, neuron

class SNN_STDP(nn.Module):
    def __init__(self):
        super(SNN_STDP, self).__init__()
        self.cv1 = layer.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1)
        self.sn1 = neuron.IFNode()
        self.pool1 = layer.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = layer.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.sn2 = neuron.IFNode()
        self.pool2 = layer.MaxPool2d(kernel_size = 2, stride = 2)
        self.linear1 = layer.Linear(in_features=7 * 7 * 64, out_features=10, bias=False)
        
    def forward(self, x):
        x = self.cv1(x)
        x = self.sn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.sn2(x)
        x = self.pool2(x)
        x = x.view(x.size(0),x.size(1), -1)
        x = self.linear1(x)
        return x