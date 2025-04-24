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
        super(SNN_CIFAR10, self).__init__()
        self.T = T
        self.layer = nn.Sequential(
            # Conv Block 1
            layer.Conv2d(3, 64, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(64, 64, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),

            # Conv Block 2
            layer.Conv2d(64, 128, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(128, 128, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),

            # Conv Block 3
            layer.Conv2d(128, 256, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(256, 256, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(256, 256, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),

            # Conv Block 4
            layer.Conv2d(256, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(512, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(512, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.MaxPool2d(2, 2),

            # Conv Block 5
            layer.Conv2d(512, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(512, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.Conv2d(512, 512, kernel_size=3, padding=1),
            neuron.LIFNode(tau=2.0, surrogate_function=surrogate.Sigmoid()),
            layer.AdaptiveAvgPool2d((1, 1))
        )

        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(512, 10)
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
        x = self.classifier(x)
        return x
