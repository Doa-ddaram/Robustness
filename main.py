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
from adversial_image import generate_adversial_image, save_image
from advertorch.attacks import GradientAttack

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
        self.linear1 = nn.Linear(in_features=7 * 7 * 64, out_features=10, bias=False)
    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = x.view(x.size(0), -1 )
        x = self.linear1(x)
        return x


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
net = CNN().to(device)
total_params = sum(p.numel() for p in net.parameters())
print(total_params)
MNIST_train = MNIST(root = '.', download = True, train = True, transform = transforms.ToTensor())
MNIST_test = MNIST(root = '.', download = True, train = False, transform = transforms.ToTensor())

batch_size = 32
num_workers = 4
learning_rate = 1e-3
train_loader = DataLoader(
    MNIST_train, batch_size = batch_size, shuffle = True, num_workers = num_workers
)
test_loader = DataLoader(
    MNIST_test, batch_size = batch_size, shuffle = False, num_workers = num_workers
)
num_epochs = 10
Loss_function= nn.CrossEntropyLoss().to(device)
Optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
adversary = GradientAttack(net, loss_fn = nn.CrossEntropyLoss(reduction="sum"),
                           eps = 1/255, clip_min = 0.0, clip_max = 1.0, targeted = False)
for epoch in range(num_epochs):
    total_Loss_train = 0
    total_acc_train = 0
    net.train()
    for i, (data, target) in tqdm(enumerate(iter(train_loader))):
        data = data.to(device)
        target = target.to(device)
        adversial_data = adversary.perturb(data, target)
        Optimizer.zero_grad()
        y_hat = net(adversial_data)
        Loss = Loss_function(y_hat, target)
        Loss.backward()
        Optimizer.step()
        total_Loss_train += Loss.item()
        pred_target = y_hat.argmax(1)
        total_acc_train += (pred_target == target).sum()
    Loss_train = total_Loss_train / (60000 / batch_size)
    acc_train = (total_acc_train / 60000) * 100 
    print(f'{epoch} epoch\'s of Loss : {Loss_train}, accuracy rate : {acc_train}')

        
    print('done')
    net.eval()
    test_image, test_label = next(iter(test_loader))
    test_image, test_label = test_image[0].to(device), test_label[0].to(device)
    
    adversarial_image = generate_adversial_image(net, test_image, test_label, epsilon = 0.1)
    save_image(test_image, adversarial_image, './images/comparison_image.png', test_label)
    
    total_Loss_test = 0
    total_acc_test = 0
    with th.no_grad():
        for i, (data, target) in tqdm(enumerate(iter(test_loader))):
            data = data.to(device)
            target = target.to(device)
            target_onehot = F.one_hot(target, 10).float()
            y_hat = net(data)
            Loss = Loss_function(y_hat, target_onehot)
            total_Loss_test += Loss.item()
            pred_target = y_hat.argmax(1)
            total_acc_test += (pred_target == target).sum()
        Loss_test = total_Loss_test / (10000/32)
        acc_test = (total_acc_test / 10000) * 100
    print(f"{epoch} epoch\'s loss: {Loss_test}, accuracy rate : {acc_test}")
    