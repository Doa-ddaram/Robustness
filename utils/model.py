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
    def __init__(self, T: int = 20):
        super(SNN, self).__init__()
        self.T = T
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

class SNN_CIFAR10(nn.Module):
    def __init__(self, T: int = 20):
        super().__init__()
        self.T = T

        self.layer = nn.Sequential(
            layer.Conv2d(3, 64, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(64, 64, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(64, 128, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(128, 128, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(128, 256, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(256, 256, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(256, 256, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(256, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(512, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(512, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),

            layer.Conv2d(512, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(512, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(512, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),
            layer.AdaptiveAvgPool2d((7, 7))
        )

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(512 * 7 * 7, 4096),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Dropout(p=0.5),
            layer.Linear(4096, 4096),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Dropout(p=0.5),
            layer.Linear(4096, 10)
        )

        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        x = self.layer(x)
        if isinstance(self.layer[-1], layer.AdaptiveAvgPool2d):
            x = th.flatten(x, 2)  # [T, B, 512*7*7]
        x = self.classifier(x)       # [T, B, num_classes]
        return x
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.block = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
        )
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.block(x)
        if self.downsample is not None:
            identity = self.downsample(x)
        identity = identity.detach()
        out += identity
        return out


class SpikingResNet18(nn.Module):
    def __init__(self, T=20, num_classes=10):
        super().__init__()
        self.T = T
        self.in_channels = 64
        self.init_layer = nn.Sequential(
            layer.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.Sigmoid())
        )
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.flatten = layer.Flatten()
        self.fc = layer.Linear(512, num_classes)

        functional.set_step_mode(self, step_mode='m')

    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                layer.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )
        layers = [BasicBlock(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.permute(1, 0, 2, 3, 4)
        x = self.init_layer(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
