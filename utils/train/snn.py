import torch.nn as nn
import torch as th
from torch.utils.data import DataLoader
from typing import Callable, Tuple, List
from tqdm.auto import tqdm
from utils.model.snn import SNN, SNN_VGG
from utils.adversarial.adversial_image import *
from ..spikingjelly.spikingjelly.activation_based import functional, neuron
from ..spikingjelly.spikingjelly.activation_based.model import spiking_vgg


def train_SNN(
    net: nn.Module,
    optimizer,
    data_loader: DataLoader[tuple[th.Tensor, th.Tensor]],
    loss_fn: Callable[[th.Tensor, th.Tensor], th.Tensor],
    device: str = "cuda",
) -> Tuple[float, float]:
    net.train()
    total_loss, total_acc = 0, 0
    length = 0
    for i, (data, target) in tqdm(enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        data = data.unsqueeze(0).repeat(10, 1, 1, 1, 1)

        y_hat = net(data).mean(0)

        loss = loss_fn(y_hat, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        pred_target = y_hat.argmax(1)
        total_acc += (pred_target == target).sum().item()
        length += len(target)
        functional.reset_net(net)
    loss = total_loss / length
    acc = (total_acc / length) * 100
    return loss, acc


def test_SNN(
    net: nn.Module,
    data_loader: DataLoader[tuple[th.Tensor, th.Tensor]],
    loss_fn: Callable[[th.Tensor, th.Tensor], th.Tensor],
    data_set: Callable[[str], None],
    attack: bool = False,
    epsilon: float = 0.05,
    device: str = "cuda",
    name: str = "snn",
) -> Tuple[float, float]:
    net.eval()
    total_acc, total_loss = 0, 0
    length = 0
    l2, linf, conf_orig, conf_adv, asr = 0, 0, 0, 0, 0
    for i, (data, target) in tqdm(enumerate(iter(data_loader))):
        data, target = data.to(device), target.to(device)
        if attack:
            was_training = net.training
            net.train()
            with th.enable_grad():
                adv_imgs = generate_adversial_image(net, data, target, epsilon=epsilon)
            save_image(data, adv_imgs, f"./images/comparison_image_{name}_{data_set}.png", target)
            # conf_orig = conf_orig + compute_confidence(net, data, target) * len(target)
            # conf_adv = conf_adv + compute_confidence(net,adv_imgs, target) * len(target)
            # l2, linf = l2 + compute_norm_differences(data, adv_imgs)[0] * len(target), linf + compute_norm_differences(data, adv_imgs)[1] * len(target)
            # asr = asr + compute_attack_success_rate(net, adv_imgs, target) * len(target)
            data = adv_imgs
            if not was_training:
                net.eval()
        y_hat = net(data)
        loss = loss_fn(y_hat, target)
        total_loss += loss.item()
        pred_target = y_hat.argmax(1)
        total_acc += (pred_target == target).sum().item()
        functional.reset_net(net)
        length += len(target)
    # conf_orig /= length
    # conf_adv /= length
    # l2 /= length
    # linf /= length
    # asr /= length
    total_loss /= length
    total_acc = total_acc / length * 100
    # print(f"FGSM (epsilon = {epsilon}) evaluate result")
    # print(f"  - avg L2 norm difference   : {l2:.4f}")
    # print(f"  - adv Linf norm difference   : {linf:.4f}")
    # print(f"  - adv success(ASR)    : {asr*100:.2f}%")
    # print(f"  - ori avg confidence : {conf_orig:.4f}")
    # print(f"  - adv avg confidence : {conf_adv:.4f}")
    return total_loss, total_acc


def train_evaluate_snn(
    data_set, learning_rate, num_epochs, train_loader, test_loader, Loss_function, epsilon, attack, save, device
) -> List:
    if data_set == "MNIST":
        net = SNN(T=20).to(device)
    else:
        net = spiking_vgg.spiking_vgg16(num_classes=10, spiking_neuron=neuron.IFNode).to(device)

    optimizer = th.optim.Adam(net.parameters(), lr=learning_rate)
    loss, acc = [], []
    clean_loss, clean_acc, adv_loss, adv_acc = [], [], [], []
    for epoch in range(num_epochs):
        epoch_loss, epoch_acc = train_SNN(
            net=net, data_loader=train_loader, loss_fn=Loss_function, device=device, optimizer=optimizer
        )
        loss.append(epoch_loss)
        acc.append(epoch_acc)
        print(f"{epoch} epoch's of Loss : {epoch_loss}, accuracy rate : {epoch_acc}")
        if save:
            th.save(net.state_dict(), f"./saved/snn_{data_set}.pt")
        if (epoch + 1) % 10 == 0:
            if attack:
                epoch_adv_loss, epoch_adv_acc = test_SNN(
                    net=net,
                    data_loader=test_loader,
                    loss_fn=Loss_function,
                    data_set=data_set,
                    attack=True,
                    epsilon=epsilon,
                    device=device,
                    name="snn",
                )
                print(f"adv acc of {epoch+1} : {epoch_adv_acc}, and adv loss of {epoch+1} : {epoch_adv_loss}")
                adv_acc.append(epoch_adv_acc)
                adv_loss.append(epoch_adv_loss)
            epoch_clean_loss, epoch_clean_acc = test_SNN(
                net=net,
                data_loader=test_loader,
                loss_fn=Loss_function,
                data_set=data_set,
                attack=False,
                device=device,
                name="snn",
            )
            clean_acc.append(epoch_clean_acc)
            clean_loss.append(epoch_clean_loss)
            print(f"clean acc of {epoch+1} : {epoch_clean_acc}, and clean loss of {epoch+1} : {epoch_clean_loss}")
    total = [loss, acc, clean_loss, clean_acc, adv_loss, adv_acc]
    return total
