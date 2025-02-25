import torch as th
import torch.nn as nn
from ..spikingjelly.spikingjelly.activation_based import neuron, surrogate, functional, layer

class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        #self.T = T
        self.conv1 = layer.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1)
        self.sn = neuron.LIFNode(tau = 2.0, surrogate_function = surrogate.ATan())
        self.pool = layer.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = layer.Conv2d(32, 64, kernel_size= 3, stride= 1, padding= 1)

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
        #x = self.conv1(x)
        #print(x.shape)
        #x = self.sn(x)
        #x = self.pool(x)
        #x = self.conv2(x)
        #x = self.sn(x)
        #x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x

class SNN_VGG(nn.Module):
    def __init__(self):
        super(SNN_VGG, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size= 3, padding= 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size= 3, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size= 3, padding = 1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size= 3, padding = 1)
        self.linear1 = nn.Linear(256 * 8 * 8, 512)
        self.linear2 = nn.Linear(512, 10)
        self.pool = nn.MaxPool2d(kernel_size= 2, stride = 2)
        self.dropout = nn.Dropout(0.5)
        self.sn = neuron.LIFNode(surrogate_function= surrogate.ATan())
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size= 3, padding = 1),
            neuron.LIFNode(2.0, surrogate_function= surrogate.ATan())
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size= 3, padding= 1),
            neuron.LIFNode(2.0, surrogate_function= surrogate.ATan()),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size= 3, padding = 1),
            neuron.LIFNode(2.0, surrogate_function= surrogate.ATan())
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size= 3, padding= 1),
            neuron.LIFNode(2.0, surrogate_function= surrogate.ATan()),
            nn.MaxPool2d(kernel_size= 2, stride= 2)
        )
        self.layer5 = nn.Sequential(
            nn.Linear(256 * 8 * 8, 512),
            neuron.LIFNode(2.0, surrogate_function= surrogate.ATan()),
            nn.Dropout(0.5)
        )
    def forward(self, x) :
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = x.view(x.size(0), -1)
        x = self.layer5(x)
        x = self.linear2(x)
        return x