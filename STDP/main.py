import torch as th
import torch.nn as nn
#import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision

import numpy as np
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import wandb
from spikingjelly.activation_based import neuron, surrogate,functional, learning, layer

time_steps = 10
batch_size = 32
learning_rate = 1e-2
num_workers = 4
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

wandb.init(
    project="STDP",
    config={
        "time_steps": time_steps,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
    },
)


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
    def forward(self, x):
        x = self.cv1(x)
        x = self.sn1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.sn2(x)
        x = self.pool2(x)
        x = x.view(x.size(0),x.size(1), -1)
        x = self.linear1(x)
        return x


net = SNN_STDP().to(device)

MNIST_train = MNIST(root = '.', download = True, train = True, transform = transforms.ToTensor())
MNIST_test = MNIST(root = '.', download = True, train = False, transform = transforms.ToTensor())

total_params = sum(p.numel() for p in net.parameters())
print(total_params)

train_loader = DataLoader(
    MNIST_train, batch_size = batch_size, shuffle = True, num_workers = num_workers
)

test_loader = DataLoader(
    MNIST_test, batch_size = batch_size, shuffle = False, num_workers = num_workers
)

def train(net, num_epochs, loss, learning_rate, dataloader):
    net.train()
    net = net.to(device)
    if th.cuda.device_count() > 1 :
        net = nn.DataParallel(net)
    instances_stdp = ( layer.Linear,
                        layer.Conv1d,
                        layer.BatchNorm2d,
                        layer.Conv2d,
                        layer.MaxPool2d,)
    stdp_learners = []

    step_mode = 'm'
    tau_pre = 2.
    tau_post = 10.
    def f_weight(x):
        return th.clamp(x, -1, 1.)
    functional.set_step_mode(net, 'm')

    for epoch in range(num_epochs):
        total_Loss_train = 0
        total_acc_train = 0
        for i, (data, target) in tqdm(enumerate(iter(dataloader))):
            data, target = data.to(device), target.to(device)    
            data, target = Variable(data), Variable(target)
            data = data.unsqueeze(0).repeat(8, 1, 1, 1, 1)
            for i, layers in enumerate(net.children()):
                if isinstance(layers, instances_stdp):
                    if i + 2 < len(list(net.children())) :
                        sn_layer = list(net.children())[i+1]
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
            for module in net.modules():
                if isinstance(module, instances_stdp):
                    for parameters in module.parameters():
                        parameters_stdp.append(parameters)
            parameters_stdp_set = set(parameters_stdp)
            
            parameters_gd = []
            for parameters in net.parameters():
                if parameters not in parameters_stdp_set:
                    parameters_gd.append(parameters)
            y_hat = net(data).mean(0)
            Loss = loss(y_hat, target)
            Optimizer_stdp = th.optim.SGD(parameters_stdp, lr = learning_rate, momentum = 0.)
            Optimizer_gd = th.optim.Adam(net.parameters(), lr = learning_rate)
            Optimizer_stdp.zero_grad()
            Optimizer_gd.zero_grad()

            Loss.backward()
        
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].step(on_grad = True)
            Optimizer_stdp.step()
            Optimizer_gd.step()
            functional.reset_net(net)
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].reset()
            total_Loss_train += Loss.item()
            pred_target = y_hat.argmax(1)
            total_acc_train += (pred_target == target).sum()
        Loss_train = total_Loss_train / (60000 / batch_size)
        acc_train = (total_acc_train / 60000) * 100 
        
        print(f'{epoch} epoch\'s of Loss : {Loss_train}, accuracy rate : {acc_train}')
        wandb.log(
            {
                "acc_train" : acc_train 
            }
        )
    return acc_train
    
def test(net, learning_rate, dataloader):
    net.eval()
    net = net.to(device)
    if th.cuda.device_count() > 1 :
        net = nn.DataParallel(net)
    total_Loss_test = 0
    total_acc_test = 0
    for i, (data, target) in tqdm (enumerate(iter(dataloader))):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target)
        data = data.unsqueeze(0).repeat(8, 1, 1, 1, 1)
        y_hat = net(data).mean(0)
        total_acc_test += (y_hat.max(1)[1] == target.to(device)).float().sum().item()
        total_Loss_test += target.numel()
        functional.reset_net(net)
    test_accuracy = total_acc_test/ total_Loss_test
    return test_accuracy
train(
    net = net, num_epochs = time_steps, loss = nn.CrossEntropyLoss(), 
    learning_rate = learning_rate, dataloader = train_loader
    )
test = test(net, learning_rate, test_loader)
wandb.log(
    {
        'evaluate' : test
    }
)

