import torch.nn as nn
from ..spikingjelly.spikingjelly.activation_based import layer, neuron, surrogate, functional

class SNN_STDP(nn.Module):
    def __init__(self):
        super(SNN_STDP, self).__init__()
        self.conv1 = layer.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1)
        self.sn1 = neuron.IFNode()
        self.pool1 = layer.MaxPool2d(kernel_size = 2, stride = 2)
        self.conv2 = layer.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.sn2 = neuron.IFNode()
        self.pool2 = layer.MaxPool2d(kernel_size = 2, stride = 2)
        self.linear1 = layer.Linear(in_features=7 * 7 * 64, out_features=10, bias=False)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.sn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.sn2(x)
        x = self.pool2(x)
        x = x.view(x.size(0),x.size(1), -1)
        x = self.linear1(x)
        return x
    
class SNN_STDP_VGG(nn.Module):
    def __init__(self):
        super(SNN_STDP_VGG, self).__init__()
        self.conv1 = layer.Conv2d(3, 32, kernel_size= 3, padding= 1)
        self.conv2 = layer.Conv2d(32, 64, kernel_size= 3, padding = 1)
        self.conv3 = layer.Conv2d(64, 128, kernel_size= 3, padding = 1)
        self.conv4 = layer.Conv2d(128, 256, kernel_size= 3, padding = 1)
        self.linear1 = layer.Linear(256 * 8 * 8, 512)
        self.linear2 = layer.Linear(512, 10)
        self.pool = layer.MaxPool2d(kernel_size= 2, stride = 2)
        self.dropout = layer.Dropout(0.5)
        self.sn = neuron.IFNode()

    def forward(self, x) :
        x = self.conv1(x)
        x = self.sn(x)
        x = self.conv2(x)
        x = self.sn(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.sn(x)
        x = self.conv4(x)
        x = self.sn(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.sn(x)
        x = self.dropout(x)
        x = self.layer5(x)
        x = self.linear2(x)
        return x

def sequential_forward(sequential, x_seq):
    assert isinstance(sequential, nn.Sequential)
    out = x_seq
    for i in range(len(sequential)):
        m = sequential[i]
        if isinstance(m, neuron.BaseNode):
            out = m(out)
        else:
            out = functional.seq_to_ann_forward(out, m)
    return out