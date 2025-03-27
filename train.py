import torch as th
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from utils.train.cnn import train_evaluate_cnn
from utils.train.snn import train_evaluate_snn
from utils.train.stdp import train_evaluate_stdp
import argparse
import wandb

def implement_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--net', type = str, help = "Input Model such of (CNN or SNN or STDP)")
    parser.add_argument('-t', type = int, help = 'training time step')
    parser.add_argument('--seed', type = int, help = 'fixed random seed')
    parser.add_argument('--dset', type = str, help = 'input dataset.')
    parser.add_argument('--batch_size', type = int, default = 32, help = 'batch size')
    parser.add_argument('--optimizer', type = str, default = 'adam', help = 'optimizer function')
    parser.add_argument('--loss_funtion', type = str, default = 'cross_entropy')
    parser.add_argument('--device', type = str, default = 'cuda', help = 'cuda or cpu')
    parser.add_argument('--learning_rate', type = float, default = 1e-2, help = 'hyperparamter learning rate')

    parser.add_argument('--attack', action=argparse.BooleanOptionalAction, help = 'enable or disable attack, type = bool')
    parser.add_argument('--save', action=argparse.BooleanOptionalAction, help = 'Saved')
    parser.add_argument('--epsilon', type = float, default = None, help = 'if Adv attack, Must be typing. type float')
    parser.add_argument('--indicate', action=argparse.BooleanOptionalAction, help = 'wandb indicate')
    
    args = parser.parse_args()
    return args

def manual_seed(seed : int = 42):
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.use_deterministic_algorithms(True, warn_only = True)
    
if __name__ == "__main__":
    args = implement_parser()
    
    num_epochs = args.t
    batch_size = 32
    num_workers = 4
    lr = args.learning_rate
    seed = args.seed
    save = args.save
    attack = args.attack
    data_set = args.dset.upper()
    epsilon = args.epsilon
    device = args.device
    network = args.net.upper()
    indicate = args.indicate
    
    if indicate:
        config = {
            'dataset' : data_set,
            'batch_size' : batch_size,
            'num_epochs' : num_epochs,
            'learning_rate' : lr,
            'seed' : seed,
            'epsilon' : epsilon
        }
    
        wandb.init(project = data_set,
                group = network,
                config = config,
                name = data_set + '_' + network)
        
    if data_set == 'MNIST' :
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
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding = 2),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        
        CIFAR10_train = CIFAR10(
            root = './data/CIFAR10', download= True, train= True, transform= transform_train
        )
        CIFAR10_test = CIFAR10(
            root= './data/CIFAR10', download=True, train = False, transform=transform_test
        )
        
        train_loader = DataLoader(
            CIFAR10_train, batch_size = batch_size, shuffle=True, num_workers= num_workers
        )
        test_loader = DataLoader(
            CIFAR10_test, batch_size = batch_size, shuffle=False, num_workers= num_workers
        )
    
    Loss_function= nn.CrossEntropyLoss().to(device)

    if network == 'CNN':
        loss_acc = train_evaluate_cnn(data_set, lr, num_epochs, train_loader, test_loader, Loss_function, epsilon, attack, save, device)
    elif network == 'SNN':
        loss_acc = train_evaluate_snn(data_set, lr, num_epochs, train_loader, test_loader, Loss_function, epsilon, attack, save, device)
    else:
        loss_acc = train_evaluate_stdp(data_set, lr, num_epochs, train_loader, test_loader, Loss_function, epsilon, attack, save, device)
    
    if indicate:
        for i in range(len(loss_acc[0])):
                if attack:
                    wandb.log({
                                    "attack loss" : loss_acc[4],
                                    "attack acc" : loss_acc[5]   
                                },
                                    step = i
                                    )
                wandb.log({
                            "training loss" : loss_acc[0],
                            "trainin acc" : loss_acc[1],
                            "clean loss" : loss_acc[2],
                            "clean acc" : loss_acc[3]
                        },
                            step = i
                )