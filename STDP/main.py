import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
from torch.autograd import Variable
import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import wandb
import matplotlib.pyplot as plt
from spikingjelly.activation_based import neuron, surrogate,functional, encoding, learning, layer
import pdb

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
        self.sn3 = neuron.IFNode()
    def forward(self, x):
        x = self.cv1(x)
        x = self.sn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.sn2(x)
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.linear1(x)
        x = self.sn3(x)
        return x


device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
net = SNN_STDP().to(device)
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

step_mode = 'm'
tau_pre = 2.
tau_post = 10.
def f_weight(x):
    return th.clamp(x, -1, 1.)
encoder = encoding.PoissonEncoder()
Loss_function= nn.CrossEntropyLoss().to(device)


for epoch in range(num_epochs):
    total_Loss_train = 0
    total_acc_train = 0
    net.train()
    instances_stdp = (layer.Linear,
                      layer.Conv2d,
                      layer.MaxPool2d)
    stdp_learners = []
    for i, (data, target) in tqdm(enumerate(iter(train_loader))):
        data = data.to(device)
        target = target.to(device)
        data_batch, target_batch = Variable(data), Variable(target)
        data_batch = data_batch.unsqueeze(0).repeat(8, 1, 1, 1, 1)
        for i, layers in enumerate(net.modules()):
            if isinstance(layers, instances_stdp):
                if i + 2 < len(list(net.modules())) :
                    sn_layer = list(net.modules())[i+1]
                stdp_learners.append(
                    learning.STDPLearner(
                        step_mode = step_mode,
                        synapse = layers,
                        sn = sn_layer,
                        tau_pre = tau_pre,
                        tau_post = tau_post,
                        f_pre = f_weight,
                        f_post = f_weight
                    )
                )
        parameters_stdp = []
        for modules in net.modules():
            if isinstance(modules, instances_stdp):
                for parameters in modules.parameters():
                    parameters_stdp.append(parameters)
        parameters_stdp_set = set(parameters_stdp)
        parameters_gd = []
        for parameters in net.parameters():
            if parameters not in parameters_stdp_set:
                parameters_gd.append(parameters)
        Optimizer_stdp = th.optim.SGD(parameters_stdp, lr = learning_rate, momentum = 0.)
        Optimizer_gd = th.optim.Adam(net.parameters(), lr = learning_rate)
        Optimizer_stdp.zero_grad()
        Optimizer_gd.zero_grad()
        y_hat_batch = net(data_batch).mean(0)
        print(y_hat_batch)
        Loss = Loss_function(y_hat_batch, target_batch)


        Loss.backward()

        for i in range(stdp_learners.__len__()):
            stdp_learners[i].step(on_grad = True)
        Optimizer_stdp.step()
        Optimizer_gd.step()
        functional.reset_net(net)
        total_Loss_train += Loss.item()
        pred_target = y_hat_batch.argmax(1)
        total_acc_train += (pred_target == target).sum()
    Loss_train = total_Loss_train / (60000 / batch_size)
    acc_train = (total_acc_train / 60000) * 100 
    print(f'{epoch} epoch\'s of Loss : {Loss_train}, accuracy rate : {acc_train}')

        
    print('done')
    # net.eval()
    # total_Loss_test = 0
    # total_acc_test = 0
    # with th.no_grad():
    #     for i, (data, target) in tqdm(enumerate(iter(test_loader))):
    #         data = data.to(device)
    #         target = target.to(device)
    #         target_onehot = F.one_hot(target, 10).float()
    #         y_hat_t, y_hat = 0, 0
    #         for t in range(T):
    #             encode = encoder(data)
    #             y_hat_t += net(encode)
    #         y_hat = y_hat_t / T
    #         Loss = F.mse_loss(y_hat, target_onehot)
    #         total_Loss_test += Loss.item()
    #         pred_target = y_hat.argmax(1)
    #         total_acc_test += (pred_target == target).sum()
    #         functional.reset_net(net)
    #     Loss_test = total_Loss_test / (10000/32)
    #     acc_test = (total_acc_test / 10000) * 100
    # print(f"{epoch} epoch\'s loss: {Loss_test}, accuracy rate : {acc_test}")