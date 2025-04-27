import torch.nn as nn
import torch as th
from .spikingjelly.spikingjelly.activation_based import neuron, surrogate, functional, layer


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(), nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features=7 * 7 * 64, out_features=10, bias=False)

    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (th.Tensor) : Input tensor, shape (batch_size, 1, 28, 28)

        Returns:
            th.Tensor: Output tensor, shape (batch_size, 10)
        """
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.flatten(x)
        x = self.linear(x)
        return x


class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.layer = nn.Sequential(
            layer.Conv2d(1, 16, kernel_size=5, stride=1, padding=2, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),
            layer.Conv2d(16, 16, kernel_size=5, stride=1, padding=2, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),
            layer.Flatten(),
            layer.Linear(16 * 7 * 7, 64, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Linear(64, 10, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
        )
        functional.set_step_mode(self, step_mode="m")
        
    def forward(self, x: th.Tensor) -> th.Tensor:
        """
        Args:
            x (th.Tensor) : Input tensor, shape (batch_size, 1, 28, 28)

        Returns:
            th.Tensor: Output tensor, shape (batch_size, 10)
        """
        x = x.permute(1, 0, 2, 3, 4)
        x = self.layer(x)
        return x

class SpikingVGG16(nn.Module):
    def __init__(self, batch_norm: bool = False, num_classes: int = 10, init_weights: bool = True):
        super(SpikingVGG16, self).__init__()
        list = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
        self.features = self._make_layers(list, batch_norm)
        self.avgpool = layer.AdaptiveAvgPool2d((7, 7))
        self.flatten = layer.Flatten()
        self.classifier = nn.Sequential(
            layer.Linear(512 * 7 * 7, 4096),
            neuron.IFNode(),
            layer.Dropout(p = 0.5),
            layer.Linear(4096, 4096),
            neuron.IFNode(),
            layer.Dropout(p = 0.5),
            layer.Linear(4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()
        functional.set_step_mode(self, step_mode='m')
        
    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def _make_layers(cfg, batch_norm):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = layer.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, layer.BatchNorm2d(v), neuron.IFNode()]
                else:
                    layers += [conv2d, neuron.IFNode()]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)