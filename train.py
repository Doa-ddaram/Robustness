import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from utils.model.cnn import CNN
from utils.model.snn import SNN
from utils.model.stdp import SNN_STDP
from tqdm.auto import tqdm
import argparse
from utils.spikingjelly.spikingjelly.activation_based import functional, layer, learning
from utils.Adversarial.nes import *
from utils.Adversarial.adversial_image import save_image
from typing import Tuple
from torch.autograd import Variable
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--net', type = str, help = "Input Model type (CNN or SNN or STDP)")
parser.add_argument('-t', type = int, help = 'training time step')
parser.add_argument('--seed', type = int, help = 'fixed random seed')
args = parser.parse_args()

def train_CNN(
    net : nn.Module,  
    data_loader : DataLoader,
    loss_fn,
    num_epochs : int = 10,
    length : int = 60000
          ) -> None : 
    for epoch in range(num_epochs):
        total_loss, total_acc = 0, 0
        net.train()
        for i, (data, target) in tqdm(enumerate(iter(data_loader))):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_hat = net(data)
            loss = loss_fn(y_hat, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred_target = y_hat.argmax(1)
            total_acc += (pred_target == target).sum()
        loss = total_loss / length
        acc = (total_acc / length) * 100 
        print(f'{epoch + 1} epoch\'s of Loss : {loss}, accuracy rate : {acc}')
        
def test_CNN(
    net : nn.Module,
    data_loader : DataLoader, 
    loss_fn,
    length : int = 10000,
             ) -> Tuple[float, float]: 
    total_loss, total_acc = 0, 0
    net.eval()
    with th.no_grad():
        for i, (data, target) in tqdm(enumerate(iter(data_loader))):
            data, target = data.to(device), target.to(device)
            adv_imgs = nes(net, data, target, iteration= 5, sample_size=10)
            y_hat = net(adv_imgs)
            save_image(data,adv_imgs, './images/comparison_image_cnn.png', target)
            loss = loss_fn(y_hat, target)
            total_loss += loss.item()
            pred_target = y_hat.argmax(1)
            total_acc += (pred_target == target).sum()
        loss = total_loss / length
        acc = (total_acc / length) * 100
    return loss, acc

def train_SNN(
    net : nn.Module,  
    data_loader : DataLoader,
    loss_fn,
    num_epochs : int = 10,
    length : int = 60000
          ) -> None : 
    for epoch in range(num_epochs):
        total_loss, total_acc = 0, 0
        net.train()
        for i, (data, target) in tqdm(enumerate(iter(data_loader))):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            y_hat = net(data)
            loss = loss_fn(y_hat, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred_target = y_hat.argmax(1)
            total_acc += (pred_target == target).sum()
            functional.reset_net(net)
        loss = total_loss / length
        acc = (total_acc / length) * 100 
        print(f'{epoch + 1} epoch\'s of Loss : {loss}, accuracy rate : {acc}')

def test_SNN(
    net : nn.Module,
    data_loader : DataLoader, 
    loss_fn,
    length : int = 10000,
             ) -> Tuple[float, float]: 
    net.eval()
    total_loss, total_acc = 0, 0
    with th.no_grad():
        for i, (data, target) in tqdm(enumerate(iter(data_loader))):
            data, target = data.to(device), target.to(device)
            adv_imgs = nes(net, data, target, iteration= 5, sample_size=10)
            y_hat = net(adv_imgs)
            save_image(data,adv_imgs, './images/comparison_image_snn.png', target)
            loss = loss_fn(y_hat, target)
            total_loss += loss.item()
            pred_target = y_hat.argmax(1)
            total_acc += (pred_target == target).sum()
            functional.reset_net(net)
        loss = total_loss / length
        acc = (total_acc / length) * 100
    return loss, acc

def train_STDP(
    net : nn.Module,  
    data_loader : DataLoader,
    loss_fn,
    num_epochs : int = 10,
    length : int = 60000
          ) -> None :
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
        total_loss, total_acc = 0, 0
        for i, (data, target) in tqdm(enumerate(iter(data_loader))):
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
            loss = loss_fn(y_hat, target)
            optimizer_stdp = th.optim.SGD(parameters_stdp, lr = learning_rate, momentum = 0.)
            optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
            optimizer_stdp.zero_grad()
            optimizer.zero_grad()

            loss.backward()
        
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].step(on_grad = False)
            optimizer_stdp.step()
            optimizer.step()
            functional.reset_net(net)
            for i in range(stdp_learners.__len__()):
                stdp_learners[i].reset()
            total_loss += loss.item()
            pred_target = y_hat.argmax(1)
            total_acc += (pred_target == target).sum()
        loss = total_loss / length
        acc = (total_acc / length) * 100 
        
        print(f'{epoch} epoch\'s of Loss : {loss}, accuracy rate : {acc}')

def test_STDP(
    net : nn.Module,
    data_loader : DataLoader, 
    loss_fn,
    length : int = 10000,
             ) -> Tuple[float, float]: 
    net.eval()
    net = net.to(device)
    if th.cuda.device_count() > 1 :
        net = nn.DataParallel(net)
    total_loss, total_acc = 0, 0
    for i, (data, target) in tqdm (enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)
        data, target = Variable(data), Variable(target) # Yet, not use adversarial attack. because of unique nature of stdp. 
        data = data.unsqueeze(0).repeat(8, 1, 1, 1, 1)
        y_hat = net(data).mean(0)
        loss = loss_fn(y_hat, target)
        total_loss += loss.item()
        pred_target = y_hat.argmax(1)
        total_acc += (pred_target == target).sum()
        functional.reset_net(net)
    loss = total_loss / length
    acc = (total_acc / length) * 100
    return loss, acc

if __name__ == "__main__":
    num_epochs = args.t
    batch_size = 32
    num_workers = 4
    learning_rate = 1e-2
    seed = args.seed
    
    np.random.seed(seed)
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    MNIST_train = MNIST(root = './data', download = True, train = True, transform = transforms.ToTensor())
    MNIST_test = MNIST(root = './data', download = True, train = False, transform = transforms.ToTensor())
    train_len = len(MNIST_train)
    test_len = len(MNIST_test)
    train_loader = DataLoader(
        MNIST_train, batch_size = batch_size, shuffle = True, num_workers = num_workers
    )
    test_loader = DataLoader(
        MNIST_test, batch_size = batch_size, shuffle = False, num_workers = num_workers
    )
    Loss_function= nn.CrossEntropyLoss().to(device)
    if args.net == 'CNN':
        net = CNN().to(device)
        optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
        train_CNN(
            net = net,
            num_epochs = num_epochs,
            data_loader= train_loader,
            loss_fn = Loss_function,
            length = train_len
            )
        test_loss, test_acc = test_CNN(
            net = net,
            data_loader=test_loader,
            loss_fn = Loss_function,
            length = test_len
            )
    elif args.net == 'SNN':
        net = SNN().to(device)
        optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
        train_SNN(
            net = net,
            num_epochs = num_epochs,
            data_loader= train_loader,
            loss_fn = Loss_function, 
            length = train_len
            )
        test_loss, test_acc = test_SNN(
            net = net,
            data_loader = test_loader,
            loss_fn = Loss_function,
            length = test_len
            )
    else:
        net = SNN_STDP().to(device)
        optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
        train_STDP(
            net = net,
            num_epochs= num_epochs,
            data_loader = train_loader,
            loss_fn = Loss_function,
            length = train_len
            )
        test_loss, test_acc = test_STDP(
            net = net,
            data_loader = test_loader,
            loss_fn = Loss_function,
            length = test_len
        )
    # total_params = sum(p.numel() for p in net.parameters())
    # print(total_params)    
    print(f'test loss = {test_loss}, and test accuracy = {test_acc}')
