from dataclasses import dataclass
from typing import Type, Callable, List
import torch.nn as nn
import torch as th
from torch.utils.data import DataLoader
import torch.optim as optim


@dataclass
class Config:
    network: Type[nn.Module] = None
    optimizer: Type[optim.Optimizer] = None
    train_loader: DataLoader[tuple[th.Tensor, th.Tensor]] = None
    test_loader: DataLoader[tuple[th.Tensor, th.Tensor]] = None
    loss_fn: Callable[[th.Tensor, th.Tensor], th.Tensor] = None
    stdp_learners: List = None
    parameters_gd: List = None
    lr: float = 1e-2
    seed: int = 0
    num_workers: int = 8
    batch_size: int = 32
    num_epochs: int = 50
    device: str = "cuda"
    method: str = "CNN"
    data_set: str = "MNIST"
    attack: bool = False
    save: bool = False
    load: bool = False
    epsilon: float = 0.05
