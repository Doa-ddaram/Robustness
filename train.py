import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import vgg16
from torchvision.datasets import MNIST, CIFAR10
from utils.model.cnn import CNN,VGG, make_layers
from utils.model.snn import SNN, SNN_VGG
from utils.model.stdp import SNN_STDP
from tqdm.auto import tqdm
import argparse
from utils.spikingjelly.spikingjelly.activation_based import functional, layer, learning, encoding, neuron, surrogate
from utils.spikingjelly.spikingjelly.activation_based.model import spiking_vgg
from utils.Adversarial.nes import *
from utils.Adversarial.adversial_image import generate_adversial_image, save_image
from typing import Tuple, Callable
from torch.autograd import Variable

parser = argparse.ArgumentParser()

parser.add_argument('--net', type = str, help = "Input Model type (CNN or SNN or STDP)")
parser.add_argument('-t', type = int, help = 'training time step')
parser.add_argument('--seed', type = int, help = 'fixed random seed')
parser.add_argument('--dset', type = str, help = 'input dataset.')

parser.add_argument('--attack', action=argparse.BooleanOptionalAction, help = 'enable or disable attack, type = bool')
args = parser.parse_args()

def train_CNN(
    net : nn.Module,  
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]],
    loss_fn : Callable[[th.Tensor,th.Tensor], th.Tensor],
    num_epochs : int = 10,
    attack : bool = False
          ) -> None : 
    for epoch in range(num_epochs):
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
            total_acc += (pred_target == target).sum()
            length += len(target)
        loss = total_loss / length
        acc = (total_acc / length) * 100
        print(f'{epoch + 1} epoch\'s of Loss : {loss}, accuracy rate : {acc}')
        if (epoch+ 1) % 10 == 0:
            if attack :
                adv_loss, adv_acc = test_CNN(net, test_loader, loss_fn = Loss_function, attack = True)
                print(f'adv acc of {epoch+1} : {adv_acc}, and adv loss of {epoch+1} : {adv_loss}')
            clean_loss, clean_acc = test_CNN(net, test_loader, loss_fn = Loss_function, attack = False)
            print(f'clean acc of {epoch+1} : {clean_acc}, and clean loss of {epoch+1} : {clean_loss}')
            
                    
def test_CNN(
    net : nn.Module,
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]], 
    loss_fn : Callable[[th.Tensor, th.Tensor], th.Tensor],
    attack : bool = False
             ) -> Tuple[float, float]: 
    total_acc = 0
    total_loss = 0
    net.eval()
    length = 0
    for i, (data, target) in tqdm(enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)
        if attack :
            adv_imgs= generate_adversial_image(net, data, target)
            save_image(data,adv_imgs, './images/comparison_image_cnn.png', target)
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
    num_epochs : int = 10,
    attack : bool = False
          ) -> None : 
    
    T = 20
    encoder = encoding.PoissonEncoder()
    for epoch in range(num_epochs):
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
        print(f'{epoch + 1} epoch\'s of Loss : {loss}, accuracy rate : {acc}')
        if (epoch +1) % 10 == 0:
            if attack :
                adv_loss, adv_acc = test_SNN(net, test_loader, attack = True)
                print(f'adv acc of {epoch+1} : {adv_acc}, and adv loss of {epoch+1} : {adv_loss}')
            clean_loss, clean_acc = test_SNN(net, test_loader, attack = False)
            print(f'clean acc of {epoch+1} : {clean_acc}, and clean loss of {epoch+1} : {clean_loss}')
def test_SNN(
    net : nn.Module,
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]],
    attack : bool = False
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
                adv_imgs= generate_adversial_image(net, data, target)
            save_image(data,adv_imgs, './images/comparison_image_snn.png', target)
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
    num_epochs : int = 10
        ) -> None :
    
    net.train()
    instances_stdp = ( layer.Linear,
                        layer.Conv2d,)
    stdp_learners = []
    net = net.to(device)
    step_mode = 'm'
    tau_pre = 2.
    tau_post = 10.
    def f_weight(x):
        return th.clamp(x, -1, 1.)
    # if th.cuda.device_count() > 1 :
    #     net = nn.DataParallel(net)
    functional.set_step_mode(net, 'm')
    for epoch in range(num_epochs):
        total_loss, total_acc = 0, 0
        length = 0
        for i, (data, target) in tqdm(enumerate(iter(data_loader))):
            data, target = data.to(device), target.to(device)    
            data = data.unsqueeze(0).repeat(20, 1, 1, 1, 1)
            for i, layers in enumerate(net.children()):
                print(i, layers)
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
            print(stdp_learners)
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
            length += len(target)
        loss = total_loss / length
        acc = (total_acc / length) * 100 
        print(f'{epoch + 1} epoch\'s of Loss : {loss}, accuracy rate : {acc}')

def test_STDP(
    net : nn.Module,
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]], 
    loss_fn : Callable[[th.Tensor, th.Tensor], th.Tensor],
    attack : bool = False
             ) -> Tuple[float, float]: 
    net.eval()
    net = net.to(device)
    if th.cuda.device_count() > 1 :
        net = nn.DataParallel(net)
    total_acc = 0
    total_loss = 0
    length = 0
    for i, (data, target) in tqdm (enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)
        data = data.unsqueeze(0).repeat(20, 1, 1, 1, 1)
        if attack:
            data = nes_stdp(net, data, target, iteration= 5, sample_size=10)
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
    num_epochs = args.t
    batch_size = 32
    num_workers = 2
    learning_rate = 1e-2
    seed = args.seed
    
    th.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    
    device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    
    if args.dset == 'MNIST' :
        MNIST_train = MNIST(root = './data', download = True, train = True, transform = transforms.ToTensor())
        MNIST_test = MNIST(root = './data', download = True, train = False, transform = transforms.ToTensor())
        
        train_loader = DataLoader(
            MNIST_train, batch_size = batch_size, shuffle = True, num_workers = num_workers
        )
        test_loader = DataLoader(
            MNIST_test, batch_size = batch_size, shuffle = False, num_workers = num_workers
        )
    
    else:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding = 4),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
        
        CIFAR10_train = CIFAR10(root = './data/CIFAR10', download= True, train= True, transform= transform)
        CIFAR10_test = CIFAR10(root= './data/CIFAR10', download=True, train = False, transform=transform)
        
        train_loader = DataLoader(CIFAR10_train, batch_size = batch_size, shuffle=True, num_workers= num_workers)
        test_loader = DataLoader(CIFAR10_test, batch_size = batch_size, shuffle=False, num_workers= num_workers)
    
    Loss_function= nn.CrossEntropyLoss().to(device)
    if args.net == 'CNN':
        if args.dset == 'MNIST':
            net = CNN().to(device)
            optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
        else:
            net = vgg16().to(device)
            optimizer = th.optim.SGD(net.parameters(), lr = learning_rate)
        train_CNN(
            net = net,
            num_epochs = num_epochs,
            data_loader= train_loader,
            loss_fn = Loss_function,
            attack = args.attack
            )
    elif args.net == 'SNN':
        if args.dset == 'MNIST':
            net = SNN().to(device)
        else:
            net = spiking_vgg.spiking_vgg16(spiking_neuron = neuron.LIFNode, surrogate_function = surrogate.ATan()).to(device)
        optimizer = th.optim.Adam(net.parameters(), lr = learning_rate)
        train_SNN(
            net = net,
            num_epochs = num_epochs,
            data_loader= train_loader,
            attack = args.attack
            )

    else:
        if args.dset == 'MNIST':
            net = SNN_STDP().to(device)
        else :
            net = spiking_vgg.spiking_vgg16(spiking_neuron = neuron.LIFNode, surrogate_function = surrogate.ATan()).to(device)

        stdp_learners = []

        step_mode = 'm'
        tau_pre = 2.
        tau_post = 10.
        def f_weight(x):
            return th.clamp(x, -1, 1.)

        functional.set_step_mode(net, 'm')
        '''
        In SNN and CNN, Can put a test process inside a train process,
        but STDP still doesn't this work, so temporarily devide train process and test process. 
        '''
        train_STDP(
            net = net,
            num_epochs= num_epochs,
            data_loader = train_loader,
            loss_fn = Loss_function
            )
        clean_loss, clean_acc = test_STDP(
            net = net, 
            data_loader= test_loader, 
            loss_fn= Loss_function, 
            attack = False
        )
        adv_loss, adv_acc = test_STDP(
            net = net,
            data_loader = test_loader,
            loss_fn = Loss_function,
            attack = True
        )
        # if args.attack :
        #     adv_loss, adv_acc = test_STDP(
        #     net = net,
        #     data_loader = test_loader,
        #     loss_fn = Loss_function,
        #     attack = True
        # )
        #     print(f'adv loss, adv acc = {adv_loss}, {adv_acc}')
        # else:
        #     clean_loss, clean_acc = test_STDP(
        #     net = net, 
        #     data_loader= test_loader, 
        #     loss_fn= Loss_function, 
        #     attack = False
        # )
        #     print(f'clean loss, clean acc = {clean_acc}, {clean_loss}')
        print(f'clean loss, clean acc = {clean_acc}, {clean_loss}')
        print(f'adv loss, adv acc = {adv_loss}, {adv_acc}')
    # total_params = sum(p.numel() for p in net.parameters())
    # print(total_params)    

