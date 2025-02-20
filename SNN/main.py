import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, surrogate,functional, encoding
from Adversial import generate_adversial_image, save_image
class SNN(nn.Module):
    def __init__(self):
        super(SNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
            neuron.LIFNode(tau = 2.0, surrogate_function = surrogate.ATan()),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
            neuron.LIFNode(tau = 2.0, surrogate_function = surrogate.ATan()),
            nn.MaxPool2d(kernel_size = 2, stride = 2)
        )
        self.linear1 = nn.Linear(in_features=7 * 7 * 64, out_features=10, bias=False)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        return x


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
net = SNN().to(device)
print(len(list(net.children()))
MNIST_train = MNIST(root = '.', download = True, train = True, transform = transforms.ToTensor())
MNIST_test = MNIST(root = '.', download = True, train = False, transform = transforms.ToTensor())
total_params = sum(p.numel() for p in net.parameters())
print(total_params)

batch_size = 32
num_workers = 4
learning_rate = 1e-2
train_loader = DataLoader(
    MNIST_train, batch_size = batch_size, shuffle = True, num_workers = num_workers
)
test_loader = DataLoader(
    MNIST_test, batch_size = batch_size, shuffle = False, num_workers = num_workers
)
num_epochs = 10
T = 20
encoder = encoding.PoissonEncoder()
Loss_function= nn.CrossEntropyLoss().to(device)
Optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
for epoch in range(num_epochs):
    total_Loss_train = 0
    total_acc_train = 0
    net.train()
    for i, (data, target) in tqdm(enumerate(iter(train_loader))):
        data = data.to(device)
        target = target.to(device)
        target_onehot = F.one_hot(target, 10).float()
        Optimizer.zero_grad()
        y_hat_t, y_hat = 0, 0
        for t in range(T):
            encode = encoder(data)
            y_hat_t += net(encode)
        y_hat = y_hat_t / T
        Loss = F.mse_loss(y_hat, target_onehot)
        Loss.backward()
        Optimizer.step()
        total_Loss_train += Loss.item()
        pred_target = y_hat.argmax(1)
        total_acc_train += (pred_target == target).sum()
        functional.reset_net(net)
    Loss_train = total_Loss_train / (60000 / batch_size)
    acc_train = (total_acc_train / 60000) * 100 
    print(f'{epoch} epoch\'s of Loss : {Loss_train}, accuracy rate : {acc_train}')

        
    print('done')
    net.eval()
    
    total_Loss_test = 0
    total_acc_test = 0
    with th.no_grad():
        for i, (data, target) in tqdm(enumerate(iter(test_loader))):
            data = data.to(device)
            target = target.to(device)
            target_onehot = F.one_hot(target, 10).float()
            y_hat_t, y_hat = 0, 0
            for t in range(T):
                encode = encoder(data)
                y_hat_t += net(encode)
            y_hat = y_hat_t / T
            Loss = F.mse_loss(y_hat, target_onehot)
            total_Loss_test += Loss.item()
            pred_target = y_hat.argmax(1)
            total_acc_test += (pred_target == target).sum()
            functional.reset_net(net)
        Loss_test = total_Loss_test / (10000/32)
        acc_test = (total_acc_test / 10000) * 100
        
    adversarial_image = generate_adversial_image(net, test_image, test_label, epsilon = 0.1)
    save(test_image, adversarial_image, './images/comparison_image.png', test_label)
    print(f"{epoch} epoch\'s loss: {Loss_test}, accuracy rate : {acc_test}")