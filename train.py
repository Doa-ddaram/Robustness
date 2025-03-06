import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg16
from torchvision.datasets import MNIST, CIFAR10
from utils.model.cnn import CNN,VGG, make_layers_CNN
from utils.model.snn import SNN, SNN_VGG, make_layers_SNN
from utils.model.stdp import SNN_STDP
from tqdm.auto import tqdm
import argparse
from utils.spikingjelly.spikingjelly.activation_based import functional, layer, learning, encoding, neuron, surrogate
from utils.spikingjelly.spikingjelly.activation_based.model import spiking_vgg
from utils.Adversarial.adversial_image import *
from typing import Tuple, Callable
from torch.autograd import Variable
import wandb
import os
import random

def implement_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', type = str, help = "Input Model such of (CNN or SNN or STDP)")
    parser.add_argument('-t', type = int, help = 'training time step')
    parser.add_argument('--seed', type = int, help = 'fixed random seed')
    parser.add_argument('--dset', type = str, help = 'input dataset.')

    parser.add_argument('--attack', action=argparse.BooleanOptionalAction, help = 'enable or disable attack, type = bool')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, help = 'Saved')
    parser.add_argument('--epsilon', type = float, default = None, help = 'if Adv attack, Must be typing. type float')
    args = parser.parse_args()
    return args

def train_CNN(
    net : nn.Module,  
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]],
    loss_fn : Callable[[th.Tensor,th.Tensor], th.Tensor],
    save : bool = False
          ) -> None : 
    total_loss, total_acc = 0, 0
    net.train()
    length = 0
    for i, (data, target) in tqdm(enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_hat = net(data)
        loss = loss_fn(y_hat, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred_target = y_hat.argmax(1)
        total_acc += (pred_target == target).sum().item()
        length += len(target)
    loss = total_loss / length
    acc = (total_acc / length) * 100
    if save:
        th.save(net.state_dict(), f"./saved/{net}_{args.dset}.pt")
    return loss, acc
                    
def test_CNN(
    net : nn.Module,
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]], 
    loss_fn : Callable[[th.Tensor, th.Tensor], th.Tensor],
    attack : bool = False,
    epsilon : float = 0.05
             ) -> Tuple[float, float]: 
    total_acc = 0
    total_loss = 0
    net.eval()
    length = 0
    for i, (data, target) in tqdm(enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)
        if attack :
            adv_imgs= generate_adversial_image(net, data, target, epsilon = epsilon)
            save_image(data,adv_imgs, f'./images/comparison_image_{args.net}_{args.dset}.png', target)
            data = adv_imgs
        with th.no_grad():
            y_hat = net(data)
            loss = loss_fn(y_hat, target)
            total_loss += loss.item()
            pred_target = y_hat.argmax(1)
            total_acc += (pred_target == target).sum().item()
            length += len(target)
    total_loss /= length
    total_acc = total_acc / length * 100
    return total_loss, total_acc

def train_SNN(
    net : nn.Module,  
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]],
    save : bool = False
          ) -> None : 
    T = 20
    encoder = encoding.PoissonEncoder()
    total_loss, total_acc = 0, 0
    net.train()
    length = 0
    for i, (data, target) in tqdm(enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        target_onehot = F.one_hot(target, 10).float()
        y_hat = 0.0
        for _ in range(T):
            encode = encoder(data)
            y_hat += net(encode)
        y_hat /= T
        loss = F.mse_loss(y_hat, target_onehot)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred_target = y_hat.argmax(1)
        total_acc += (pred_target == target).sum().item()
        length += len(target)
        functional.reset_net(net)
    loss = total_loss / length
    acc = (total_acc / length) * 100 
    if save:
        th.save(net.state_dict(), f"./saved/{net}_{args.dset}.pt")
    return loss, acc
            
def test_SNN(
    net : nn.Module,
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]],
    attack : bool = False,
    epsilon : float = 0.05
             ) -> Tuple[float, float]: 
    encoder = encoding.PoissonEncoder()
    T = 20
    total_acc = 0
    total_loss = 0
    net.eval()
    length = 0
    for i, (data, target) in tqdm(enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)
        target_onehot = F.one_hot(target, 10).float()
        was_training = net.training
        net.train()
        if attack:
            with th.enable_grad():
                adv_imgs= generate_adversial_image(net, data, target, epsilon = epsilon)
            save_image(data, adv_imgs, f'./images/comparison_image_{args.net}_{args.dset}.png', target)
            data = adv_imgs
        net.train(was_training)
        with th.no_grad():
            y_hat = 0.0
            for _ in range(T):
                encode = encoder(data)
                y_hat += net(encode)
            y_hat /= T
            loss = F.mse_loss(y_hat, target_onehot)
            total_loss += loss.item()
            pred_target = y_hat.argmax(1)
            total_acc += (pred_target == target).sum().item()
            length += len(target)
            functional.reset_net(net)
    total_loss /= length
    total_acc = total_acc / length * 100
    return total_loss, total_acc

def train_STDP(
    net : nn.Module,  
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]],
    loss_fn : Callable[[th.Tensor, th.Tensor], th.Tensor],
    save : bool = False
        ) -> None :
    net.train()
    instances_stdp = (layer.Conv2d,)
    stdp_learners = []
    net = net.to(device)
    step_mode = 'm'
    tau_pre = 2.
    tau_post = 10.
    def f_weight(x):
        return th.clamp(x, -1, 1.)
    functional.set_step_mode(net, 'm')
    total_loss, total_acc = 0, 0
    length = 0
    for i, (data, target) in tqdm(enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)    
        data = data.unsqueeze(0).repeat(20, 1, 1, 1, 1)
        for i, layers in enumerate(net.children()):
            if isinstance(layers, nn.Sequential):
                for j, layer_in in enumerate(layers):
                    if isinstance(layer_in, neuron.BaseNode):
                        syn = layers[j-1]
                        sn_layer = layer_in
                        stdp_learners.append(
                            learning.STDPLearner(
                                step_mode = step_mode,
                                synapse = syn,
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
        total_acc += (pred_target == target).sum().item()
        length += len(target)
    loss = total_loss / length
    acc = (total_acc / length) * 100
    if save:
        th.save(net.state_dict(), f"./saved/{net}_{args.dset}.pt")
    return loss, acc
        
def test_STDP(
    net : nn.Module,
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]], 
    loss_fn : Callable[[th.Tensor, th.Tensor], th.Tensor],
    attack : bool = False,
    epsilon : float = 0.05
             ) -> Tuple[float, float]:
    T = 20
    net.eval()
    net = net.to(device)
    total_acc = 0
    total_loss = 0
    length = 0
    for i, (data, target) in tqdm (enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)
        data = data.unsqueeze(0).repeat(T, 1, 1, 1, 1)
        if attack:
            was_training = net.training
            net.train()
            with th.enable_grad():
                adv_imgs = generate_adversial_image(net, data, target, epsilon = epsilon)
            save_image(data.mean(0), adv_imgs.mean(0), f'./images/comparison_image_{args.net}_{args.dset}.png', target)
            data = adv_imgs
            if not was_training:
                net.eval()
        data, target = Variable(data), Variable(target) 
        y_hat = net(data).mean(0)
        loss = loss_fn(y_hat, target)
        total_loss += loss.item()
        pred_target = y_hat.argmax(1)
        total_acc += (pred_target == target).sum().item()
        functional.reset_net(net)
        length += len(target)
    total_loss /= length
    total_acc = total_acc / length * 100
    return total_loss, total_acc

if __name__ == "__main__":
    args = implement_parser()
    num_epochs = args.t
    batch_size = 32
    num_workers = 4
    learning_rate = 1e-2
    seed = args.seed
    save = args.save

    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.use_deterministic_algorithms(True)
    random.seed(seed)
    
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    config = {
            'dataset' : args.dset,
            'batch_size' : batch_size,
            'num_epochs' : num_epochs,
            'learning_rate' : learning_rate,
            'seed' : seed,
            'epsilon' : args.epsilon
        }
    
    if args.dset == 'MNIST' :
        MNIST_train = MNIST(
            root = './data', download = True, train = True, transform = transforms.ToTensor()
        )
        MNIST_test = MNIST(
            root = './data', download = True, train = False, transform = transforms.ToTensor()
        )
        
        train_loader = DataLoader(
            MNIST_train, batch_size = batch_size, shuffle = True, num_workers = num_workers
        )
        test_loader = DataLoader(
            MNIST_test, batch_size = batch_size, shuffle = False, num_workers = num_workers
        )
    
    else:
        transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(32, padding = 2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
        
        CIFAR10_train = CIFAR10(
            root = './data/CIFAR10', download= True, train= True, transform= transform
        )
        CIFAR10_test = CIFAR10(
            root= './data/CIFAR10', download=True, train = False, transform=transform
        )
        
        train_loader = DataLoader(
            CIFAR10_train, batch_size = batch_size, shuffle=True, num_workers= num_workers
        )
        test_loader = DataLoader(
            CIFAR10_test, batch_size = batch_size, shuffle=False, num_workers= num_workers
        )
    
    Loss_function= nn.CrossEntropyLoss().to(device)
    
    wandb.init(project = args.dset, config = config, name = args.dset + '_' + args.net)
    
    attack = args.attack
    
    if args.net == 'CNN':
        if args.dset == 'MNIST':
            net = CNN().to(device)
            optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
        else:
            net = vgg16().to(device)
            optimizer = th.optim.SGD(net.parameters(), lr = learning_rate)
        for epoch in range(num_epochs):
            loss, acc = train_CNN(
                net = net,
                data_loader= train_loader,
                loss_fn = Loss_function,
                save = save
                )
            wandb.log({
                    "train loss" : loss,
                    "train acc" : acc
                },
                    step = epoch
                    )
            print(f'{epoch + 1} epoch\'s of Loss : {loss}, accuracy rate : {acc}')
            if (epoch+ 1) % 10 == 0:
                if attack :
                    adv_loss, adv_acc = test_CNN(
                        net = net, 
                        data_loader=test_loader, 
                        loss_fn = Loss_function, 
                        attack = True,
                        epsilon = args.epsilon
                        )
                    wandb.log({
                        "attack loss" : adv_loss,
                        "attack acc" : adv_acc   
                    },
                        step = epoch
                        )
                    print(f'adv acc of {epoch+1} : {adv_acc}, and adv loss of {epoch+1} : {adv_loss}')
                clean_loss, clean_acc = test_CNN(
                    net = net, 
                    data_loader= test_loader, 
                    loss_fn = Loss_function, 
                    attack = False)
                wandb.log({
                    "clean loss" : clean_loss,
                    "clean acc" : clean_acc
                },
                    step = epoch
                    )
                print(f'clean acc of {epoch+1} : {clean_acc}, and clean loss of {epoch+1} : {clean_loss}')
    elif args.net == 'SNN':
        if args.dset == 'MNIST':
            net = SNN().to(device)
        else:
            # cfg_vgg16 = [64, 64, 'M', 
            #  128, 128, 'M', 
            #  256, 256, 256, 'M', 
            #  512, 512, 512, 'M', 
            #  512, 512, 512, 'M']
            # net = SNN_VGG(make_layers_SNN(cfg_vgg16, batch_norm=False)).to(device)
            net = spiking_vgg.spiking_vgg11(num_classes = 10, spiking_neuron = neuron.LIFNode, surrogate_function = surrogate.ATan()).to(device)
        optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
        for epoch in range(num_epochs):
            loss, acc = train_SNN(
                net = net,
                data_loader= train_loader,
                save = save
                )
            wandb.log({
                    "train loss" : loss,
                    "train acc" : acc
                },
                    step = epoch
                    )
            print(f'{epoch + 1} epoch\'s of Loss : {loss}, accuracy rate : {acc}')
            if (epoch + 1) % 10 == 0:
                if attack :
                    adv_loss, adv_acc = test_SNN(
                        net = net, 
                        data_loader=test_loader,  
                        attack = True,
                        epsilon = args.epsilon
                        )
                    wandb.log({
                        "attack loss" : adv_loss,
                        "attack acc" : adv_acc   
                    },
                        step = epoch
                    )
                    print(f'adv acc of {epoch+1} : {adv_acc}, and adv loss of {epoch+1} : {adv_loss}')
                clean_loss, clean_acc = test_SNN(
                    net = net, 
                    data_loader= test_loader,
                    attack = False)
                wandb.log({
                    "clean loss" : clean_loss,
                    "clean acc" : clean_acc
                },
                    step = epoch
                    )
                print(f'clean acc of {epoch+1} : {clean_acc}, and clean loss of {epoch+1} : {clean_loss}')
    else:
        if args.dset == 'MNIST':
            net = SNN_STDP().to(device)
        else :
            net = spiking_vgg.spiking_vgg16(num_classes = 10, spiking_neuron = neuron.LIFNode, surrogate_function = surrogate.ATan()).to(device)
        for epoch in range(num_epochs):
            loss, acc = train_STDP(
                net = net,
                data_loader = train_loader,
                loss_fn = Loss_function,
                save = save
                )
            wandb.log({
                    "train loss" : loss,
                    "train acc" : acc
                },
                    step = epoch
                    )
            print(f'{epoch + 1} epoch\'s of Loss : {loss}, accuracy rate : {acc}')
            if (epoch + 1) % 10 == 0:
                if attack :
                    adv_loss, adv_acc = test_STDP(
                        net = net, 
                        data_loader = test_loader,
                        loss_fn = Loss_function,
                        attack = True,
                        epsilon = args.epsilon
                        )
                    wandb.log({
                        "attack loss" : adv_loss,
                        "attack acc" : adv_acc   
                    },
                        step = epoch
                    )
                    print(f'adv acc of {epoch+1} : {adv_acc}, and adv loss of {epoch+1} : {adv_loss}')
                clean_loss, clean_acc = test_STDP(
                    net = net, 
                    data_loader = test_loader, 
                    loss_fn = Loss_function,
                    attack = False)
                wandb.log({
                    "clean loss" : clean_loss,
                    "clean acc" : clean_acc
                },
                    step = epoch
                    )
                print(f'clean acc of {epoch+1} : {clean_acc}, and clean loss of {epoch+1} : {clean_loss}')
    # total_params = sum(p.numel() for p in net.parameters())
    # print(total_params)