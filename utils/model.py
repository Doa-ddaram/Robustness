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

tau = 1
class ConvBNLIFBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            layer.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding, bias=False),
            layer.BatchNorm2d(out_channels),
            neuron.LIFNode(
                v_threshold=1.0, v_reset=0.0, tau= 1.0,
                surrogate_function=surrogate.ATan(),
                detach_reset=True
            )
        )

    def forward(self, x):
        return self.block(x)


class SNNVGG(nn.Module):
    def __init__(self, cfg, num_classes=10):
        super().__init__()
        cfg = [128, 128, 'M',
             256, 256, 'M',
             256, 256, 256, 'M',
             512, 512, 512, 'M',
             512, 512, 512]
        self.features = self._make_layers(cfg)
        self.avgpool = layer.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            layer.Flatten(),
            layer.Linear(cfg[-1], num_classes)
        )

        self._initialize_weights()
        functional.set_step_mode(self, step_mode='m')

    def forward(self, x):  # x shape: [T, B, C, H, W]
        x = x.permute(1, 0, 2, 3, 4)  # to [B, T, C, H, W]
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [layer.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [ConvBNLIFBlock(in_channels, v, tau=self.tau)]
                in_channels = v
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, layer.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, layer.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)