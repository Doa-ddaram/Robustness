import torch.nn as nn
import torch as th
from torch.utils.data import DataLoader
from typing import Callable, List,Tuple
from tqdm.auto import tqdm
from utils.model.snn import SNN
from utils.train.snn import test_SNN
from ..spikingjelly.spikingjelly.activation_based import functional, learning, neuron, layer
from ..spikingjelly.spikingjelly.activation_based.model import spiking_vgg

def train_STDP(
    net : nn.Module,
    optimizer,
    data_loader : DataLoader[tuple[th.Tensor, th.Tensor]],
    loss_fn : Callable[[th.Tensor, th.Tensor], th.Tensor],
    parameters_stdp : List = [],
    stdp_learners : List = [],
    learning_rate : int = 1e-4,
    device : str = 'cuda'
        ) -> Tuple[float, float] :
    net.train()
    total_acc, total_loss = 0, 0
    length = 0
    optimizer_stdp = th.optim.SGD(parameters_stdp, lr = learning_rate)
    for i, (data, target) in tqdm(enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device) 
        optimizer_stdp.zero_grad()
        optimizer.zero_grad()
        y_hat = net(data).mean(0)
        loss = loss_fn(y_hat, target)
        loss.backward()
        for i in range(stdp_learners.__len__()):
            with th.no_grad():
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
    return loss, acc

def train_evaluate_stdp(
    data_set, lr, num_epochs, train_loader, test_loader, Loss_function, epsilon, attack, save, device
    ) -> List:
    instances_stdp = (
                      layer.Linear,
                      )
    tau_pre = 2.
    tau_post = 10.
    def f_weight(x):
        return th.clamp(x, -1, 1.)
    step_mode = 'm'
    stdp_learners = []
    net = SNN(T = 20).to(device)
    for i, layers in enumerate(net.modules()):
        if isinstance(layers, nn.Sequential):
            for j, layer_in in enumerate(layers):
                if isinstance(layer_in, neuron.BaseNode):
                    synapse = layers[j-1]
                    sn_layer = layer_in
                    stdp_learners.append(
                        learning.STDPLearner(
                            step_mode = step_mode,
                            synapse = synapse,
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
            
    if data_set == 'MNIST':
            net = SNN(T = 20).to(device)
    else:
        net = spiking_vgg.spiking_vgg16(
            num_classes = 10, spiking_neuron = neuron.IFNode
            ).to(device)
    optimizer = th.optim.Adam(net.parameters(), lr = lr)
    loss, acc = [], []
    clean_loss, clean_acc, adv_loss, adv_acc = [], [], [], []
    for epoch in range(num_epochs):
        epoch_loss, epoch_acc = train_STDP(
            net = net,
            optimizer = optimizer,
            data_loader = train_loader,
            loss_fn = Loss_function,
            parameters_stdp = parameters_stdp,
            stdp_learners = stdp_learners,
            learning_rate = lr,
            device = device
                )
        loss.append(epoch_loss)
        acc.append(epoch_acc)
        print(f'{epoch} epoch\'s of Loss : {epoch_loss}, accuracy rate : {epoch_acc}')
        if save:
            th.save(net.state_dict(), f"./saved/STDP_{data_set}.pt")
        if (epoch + 1) % 10 == 0:
            if attack :
                epoch_adv_loss, epoch_adv_acc = test_SNN(
                    net = net, 
                    data_loader=test_loader,  
                    loss_fn = Loss_function,
                    data_set= data_set,
                    attack = True,
                    epsilon = epsilon,
                    device = device,
                    name = 'snn'
                    )
                print(f'adv acc of {epoch+1} : {epoch_adv_acc}, and adv loss of {epoch+1} : {epoch_adv_loss}')
                adv_acc.append(epoch_adv_acc)
                adv_loss.append(epoch_adv_loss)
            epoch_clean_loss, epoch_clean_acc = test_SNN(
                net = net, 
                data_loader= test_loader,
                loss_fn = Loss_function,
                data_set= data_set,
                attack = False,
                device = device,
                name = 'snn'
                )
            clean_acc.append(epoch_clean_acc)
            clean_loss.append(epoch_clean_loss)
            print(f'clean acc of {epoch+1} : {epoch_clean_acc}, and clean loss of {epoch+1} : {epoch_clean_loss}')
    total = [loss, acc, clean_loss, clean_acc, adv_loss, adv_acc]
    return total