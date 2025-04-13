import torch as th
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import MNIST, CIFAR10
from utils.train import train_evaluate
import argparse
import wandb
from utils.config import Config

def implement_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--net", type=str, help="Input Model such of (CNN or SNN or STDP)")
    parser.add_argument("-t", type=int, help="training time step")
    parser.add_argument("--seed", type=int, help="fixed random seed")
    parser.add_argument("--dset", type=str, help="input dataset.")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--learning_rate", type=float, default=1e-2, help="hyperparamter learning rate")
    parser.add_argument("--num_workers", type=int, default=8, help="number of worker")
    parser.add_argument("--epsilon", type=float, default=None, help="if Adv attack, Must be typing. type float")
    parser.add_argument("--attack", action=argparse.BooleanOptionalAction, help="enable or disable attack, type = bool")

    parser.add_argument(
        "--load", action=argparse.BooleanOptionalAction, help="whether load model parameter, type = bool"
    )
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, help="Saved")

    args = parser.parse_args()
    return Config(
        method=args.net.upper(),
        data_set=args.dset.upper(),
        num_epochs=args.t,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        lr=args.learning_rate,
        device=args.device,
        save=args.save,
        load=args.load,
        attack=args.attack,
        epsilon=args.epsilon,
    )


def manual_seed(seed: int = 42) -> None:
    th.manual_seed(seed)
    th.cuda.manual_seed(seed)
    th.cuda.manual_seed_all(seed)
    th.use_deterministic_algorithms(True, warn_only=True)


if __name__ == "__main__":
    args = implement_parser()

    # config = {
    #     'dataset' : args.data_set,
    #     'batch_size' : args.batch_size,
    #     'num_epochs' : args.num_epochs,
    #     'learning_rate' : args.lr,
    #     'seed' : args.seed,
    #     'epsilon' : args.epsilon
    # }

    # wandb.init(project = args.data_set,
    #         group = args.method,
    #         config = config,
    #         name = args.data_set + '_' + args.method)

    manual_seed(args.seed)

    if args.data_set == "MNIST":
        MNIST_train = MNIST(root="./data", download=True, train=True, transform=transforms.ToTensor())
        MNIST_test = MNIST(root="./data", download=True, train=False, transform=transforms.ToTensor())

        train_loader = DataLoader(MNIST_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(MNIST_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    else:
        transform_train = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=2),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        transform_test = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        CIFAR10_train = CIFAR10(root="./data/CIFAR10", download=True, train=True, transform=transform_train)
        CIFAR10_test = CIFAR10(root="./data/CIFAR10", download=True, train=False, transform=transform_test)

        train_loader = DataLoader(CIFAR10_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(CIFAR10_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    Loss_function = nn.CrossEntropyLoss().to(args.device)

    args.loss_fn = Loss_function
    args.train_loader = train_loader
    args.test_loader = test_loader
    loss_acc = train_evaluate(args)