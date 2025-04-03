import torch as th
import torch.nn as nn
import torch.optim as optim
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from dataclasses import dataclass, replace
from .spikingjelly.spikingjelly.activation_based import functional, learning, neuron, layer
from .spikingjelly.spikingjelly.activation_based.model import spiking_vgg
from typing import Callable, List, Type, Tuple
from tqdm.auto import tqdm
from .model import CNN, SNN
from .adversial_image import *
import wandb


@dataclass
class Config:
    network : Type[nn.Module]
    optimizer : Type[optim.Optimizer]
    train_loader : DataLoader[tuple[th.Tensor, th.Tensor]] = None
    test_loader : DataLoader[tuple[th.Tensor, th.Tensor]] = None
    loss_fn : Callable[[th.Tensor, th.Tensor], th.Tensor] = None
    stdp_learners : List = None
    parameters_stdp : List = None
    lr : float = 1e-2
    seed : int = 0
    num_workers : int = 4
    batch_size : int = 32
    num_epochs : int = 50
    device : str = 'cuda'
    method : str = 'CNN'
    data_set : str = 'MNIST'
    attack : bool = False
    save : bool = False
    load : bool = False
    epsilon : float = 0.05

# Training function
def train_model(config : Config) -> Tuple[float, float]:
    net = config.network
    net.train()
    total_acc, total_loss, length = 0, 0, 0
    optimizer_stdp = th.optim.SGD(config.parameters_stdp, lr=config.lr) if config.parameters_stdp else None

    for i, (data, target) in tqdm(enumerate(iter(config.train_loader))):
        data, target = data.to(config.device), target.to(config.device)
        config.optimizer.zero_grad()
        if config.method == 'STDP' and optimizer_stdp:
            optimizer_stdp.zero_grad()
        
        y_hat = net(data).mean(0) if config.method != 'CNN' else net(data)
        loss = config.loss_fn(y_hat, target)
        loss.backward()
        
        if config.method == 'STDP' and config.stdp_learners:
            for learner in config.stdp_learners:
                with th.no_grad():
                    learner.step(on_grad=False)
            optimizer_stdp.step()
        
        config.optimizer.step()
        if config.method != 'CNN':
            print('hello')
            functional.reset_net(net)
        total_loss += loss.item()
        total_acc += (y_hat.argmax(1) == target).sum().item()
        length += len(target)
    
    return total_loss / length, (total_acc / length) * 100

def evaluate_model(config : Config
    ) -> Tuple[float, float]:
    net = config.network
    net.eval()
    if config.load:
        net.load_state_dict(th.load(f'./saved/{config.method.lower()}_{config.data_set}.pt'))
    total_loss, total_acc = 0, 0
    length = 0
    for i, (data, target) in tqdm(enumerate(iter(config.test_loader))):
        data, target = data.to(config.device), target.to(config.device)
        if config.attack:
            was_training = net.training
            net.train()
            with th.enable_grad():
                adv_imgs = generate_adversial_image(net, data, target, config.epsilon)
            save_image(
                data, adv_imgs, f'./images/comparison_image_{config.method.lower()}_{config.data_set}.png', target
                )
            data = adv_imgs
            if not was_training:
                 net.eval()
        y_hat = net(data).mean(0) if config.method != 'CNN' else net(data)
        loss = config.loss_fn(y_hat, target)
        total_loss += loss.item()
        pred_target = y_hat.argmax(1)
        total_acc += (pred_target == target).sum().item()
        if config.method != 'CNN':
            functional.reset_net(net)
        length += len(target)
    return total_loss / length, (total_acc / length) * 100

# Unified training function and evaluate function
def train_evaluate(config : Config) -> None:
    if config.method == 'CNN':
        net = CNN().to(config.device) if config.data_set == 'MNIST' else vgg16().to(config.device)
    else:
        net = SNN(T=20).to(config.device) if config.data_set == 'MNIST' else spiking_vgg.spiking_vgg16(
            num_classes=10, spiking_neuron=neuron.IFNode
        ).to(config.device)
    config.optimizer = th.optim.Adam(net.parameters(), lr = config.lr)
    stdp_learners, parameters_stdp = [], []
    
    if config.method == 'STDP':
        for layers in net.modules():
            if isinstance(layers, nn.Sequential):
                for i, layer_in in enumerate(layers):
                    if isinstance(layer_in, neuron.BaseNode):
                        synapse = layers[i - 1]
                        stdp_learners.append(
                            learning.STDPLearner(
                                step_mode='m', synapse=synapse, sn=layer_in,
                                tau_pre=2., tau_post=10., f_pre=lambda x: th.clamp(x, -1, 1.), f_post=lambda x: th.clamp(x, -1, 1.)
                            )
                        )
        for module in net.modules():
            if isinstance(module, layer.Linear):
                parameters_stdp.extend(module.parameters())
    config.network = net
    attack = config.attack
    if attack:
        adv_config = replace(config, attack = True)
        config = replace(config, attack = False)
    for epoch in range(config.num_epochs):
        if config.load :
            if attack :
                epoch_adv_loss, epoch_adv_acc = evaluate_model(adv_config)
                print(f'{epoch + 1} epoch - adv_loss: {epoch_adv_loss:.4f}, adv_acc: {epoch_adv_acc:.2f}%')
                # wandb.log({
                #          "attack loss" : adv_loss,
                #          "attack acc" : adv_acc   
                #      },
                #          step = epoch + 1
                #      )
            epoch_clean_loss, epoch_clean_acc = evaluate_model(config)
            print(f'{epoch + 1} epoch - clean_loss: {epoch_clean_loss:.4f}, clean_acc: {epoch_clean_acc:.2f}%')
            # wandb.log({
            #          "clean loss" : clean_loss,
            #          "clean acc" : clean_acc
            #      },
            #          step = epoch
            #          )
        else:
            epoch_loss, epoch_acc = train_model(config)
            print(f'{epoch + 1} epoch - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%')
            # wandb.log({
            #              "attack loss" : epoch_loss,
            #              "attack acc" : epoch_acc   
            #          },
            #              step = epoch + 1
            #          )
            if config.save:
                th.save(net.state_dict(), f'./saved/{config.method.lower()}_{config.data_set}.pt')
            if attack :
                epoch_adv_loss, epoch_adv_acc = evaluate_model(adv_config)
                # wandb.log({
                #          "attack loss" : adv_loss,
                #          "attack acc" : adv_acc   
                #      },
                #          step = epoch + 1
                #      )
                print(f'{epoch + 1} epoch - adv_loss: {epoch_adv_loss:.4f}, adv_acc: {epoch_adv_acc:.2f}%')
                config.attack = False
            epoch_clean_loss, epoch_clean_acc = evaluate_model(config)
            print(f'{epoch + 1} epoch - clean_loss: {epoch_clean_loss:.4f}, clean_acc: {epoch_clean_acc:.2f}%')
            # wandb.log({
            #          "clean loss" : clean_loss,
            #          "clean acc" : clean_acc
            #      },
            #          step = epoch
            #          )
