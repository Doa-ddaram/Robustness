import torch as th
import torch.nn as nn
from torchvision.models import vgg16
from dataclasses import replace
from .spikingjelly.spikingjelly.activation_based import functional, learning, neuron
from .spikingjelly.spikingjelly.activation_based.model import spiking_vgg
from typing import Tuple
from tqdm.auto import tqdm
from .model import CNN, SNN
from .adversarial_image import generate_adversial_image_fgsm, save_image
from .config import Config


# Training function
def train_model(config: Config, mode=None) -> Tuple[float, float]:
    net = config.network
    net.train()
    total_acc, total_loss, length = 0, 0, 0

    for i, (data, target) in tqdm(enumerate(iter(config.train_loader))):
        data, target = data.to(config.device), target.to(config.device)

        config.optimizer.zero_grad()

        if config.parameters_stdp:
            config.optimizer_stdp.zero_grad()

        y_hat = net(data).mean(0) if config.method != "CNN" else net(data)
        loss = config.loss_fn(y_hat, target)

        if mode == "stdp":
            if config.parameters_stdp:
                for learner in config.stdp_learners:
                    with th.no_grad():
                        learner.step(on_grad=True)
                        learner.synapse.weight.data.clamp_(0.0, 1.0)

        if mode == "gd":
            loss.backward()
            config.optimizer.step()

        # if mode == "stdp":
        #    if config.parameters_stdp:
        #        config.optimizer_stdp.step()

        functional.reset_net(net)

        if config.parameters_gd:
            for learner in config.stdp_learners:
                learner.reset()

        total_loss += loss.item()
        total_acc += (y_hat.argmax(1) == target).sum().item()
        length += len(target)

        del loss, y_hat, target, data
        th.cuda.empty_cache()

    return total_loss / length, (total_acc / length) * 100


def evaluate_model(config: Config) -> Tuple[float, float]:
    net = config.network
    # net.eval()
    if config.load:
        net.load_state_dict(th.load(f"./saved/{config.method.lower()}_{config.data_set}.pt"))

    net.eval()

    total_loss, total_acc = 0, 0
    length = 0

    if config.stdp_learners:
        for i in config.stdp_learners:
            i.disable()

    for i, (data, target) in tqdm(enumerate(iter(config.test_loader))):
        data, target = data.to(config.device), target.to(config.device)
        if config.attack:
            net.train()

            adv_imgs = generate_adversial_image_fgsm(net, data, target, config.epsilon)
            save_image(
                data, adv_imgs, f"./images/comparison_image_{config.method.lower()}_{config.data_set}.png", target
            )
            data = adv_imgs
            net.eval()

        with th.no_grad():
            y_hat = net(data).mean(0) if config.method != "CNN" else net(data)
            loss = config.loss_fn(y_hat, target)

        total_loss += loss.item()
        pred_target = y_hat.argmax(1)
        total_acc += (pred_target == target).sum().item()
        if config.method != "CNN":
            functional.reset_net(net)
        length += len(target)

    if config.stdp_learners:
        for i in config.stdp_learners:
            i.enable()

    return total_loss / length, (total_acc / length) * 100


# Unified training function and evaluate function

def train_evaluate(config: Config) -> None:
    if config.method == "CNN":
        net = CNN().to(config.device) if config.data_set == "MNIST" else vgg16().to(config.device)
    else:
        net = (
            SNN(T=20).to(config.device)
            if config.data_set == "MNIST"
            else spiking_vgg.spiking_vgg16(num_classes=10, spiking_neuron=neuron.IFNode).to(config.device)
        )
    stdp_learners, parameters_stdp, parameters_gd = [], [], []

    if config.method == "STDP":
        added = False
        for layers in net.modules():
            if isinstance(layers, nn.Sequential):
                for i, layer_in in enumerate(layers):
                    if isinstance(layer_in, nn.Conv2d) and not added:
                        print(f"Adding STDP learner from {layers[i]} to {layers[i + 1]}")
                        stdp_learners.append(
                            learning.STDPLearner(
                                step_mode="m",
                                synapse=layers[i],
                                sn=layers[i + 1],
                                tau_pre=2.0,
                                tau_post=100.0,
                                f_pre=lambda x: th.clamp(x, -1, 1.0),
                                f_post=lambda x: th.clamp(x, -1, 1.0),
                            )
                        )
                        added = True

        added = False

        for module in net.modules():
            if isinstance(module, nn.Conv2d) and not added:
                print(f"Adding STDP learner to {module}")
                added = True
                parameters_stdp.extend(module.parameters())
            else:
                print(f"Adding GD Optimizer to {module}")
                parameters_gd.extend(module.parameters())
    else:
        parameters_gd = net.parameters()

    all_parameters = list(net.parameters())
    parameters_stdp_set = set(parameters_stdp)
    parameters_gd = [p for p in all_parameters if p not in parameters_stdp_set]

    config.optimizer = th.optim.Adam(all_parameters, lr=config.lr)
    
    config.network = net

    config.optimizer_stdp = (
        th.optim.SGD(config.parameters_stdp, lr=config.lr * 0.1, momentum=0.0) if config.parameters_stdp else None
    )

    attack = config.attack
    if attack:
        adv_config = replace(config, attack=True)
        config = replace(config, attack=False)

    layer_num = 0
    layer_num_2 = 3

    weight = config.network.layer[layer_num].weight.clone()
    weight2 = config.network.layer[layer_num_2].weight.clone()

    print("Weight stats after STDP:")
    print("min:", weight.data.min().item())
    print("max:", weight.data.max().item())
    print("mean:", weight.data.mean().item())

    for epoch in range(config.num_epochs):
        if config.load:
            if attack:
                epoch_adv_loss, epoch_adv_acc = evaluate_model(adv_config)
                print(f"{epoch + 1} epoch - adv_loss: {epoch_adv_loss:.4f}, adv_acc: {epoch_adv_acc:.2f}%")
                # wandb.log({
                #          "attack loss" : adv_loss,
                #          "attack acc" : adv_acc
                #      },
                #          step = epoch + 1
                #      )
            epoch_clean_loss, epoch_clean_acc = evaluate_model(config)
            print(f"epoch - clean_loss: {epoch_clean_loss:.4f}, clean_acc: {epoch_clean_acc:.2f}%")
            # wandb.log({
            #          "clean loss" : clean_loss,
            #          "clean acc" : clean_acc
            #      },
            #          step = epoch
            #          )
        else:
            if epoch % 2 == 1:
                epoch_loss, epoch_acc = train_model(config, "stdp")

                print("After STDP:", (weight - config.network.layer[layer_num].weight).abs().sum())
                print("After STDP:", (weight2 - config.network.layer[layer_num_2].weight).abs().sum())

                weight = config.network.layer[layer_num].weight.clone()
                weight2 = config.network.layer[layer_num_2].weight.clone()

                print("Weight stats after STDP:")
                print("min:", weight.data.min().item())
                print("max:", weight.data.max().item())
                print("mean:", weight.data.mean().item())

            else:
                epoch_loss, epoch_acc = train_model(config, "gd")

                print("After GD:", (weight - config.network.layer[layer_num].weight).abs().sum())
                print("After GD:", (weight2 - config.network.layer[layer_num_2].weight).abs().sum())

                weight = config.network.layer[layer_num].weight.clone()
                weight2 = config.network.layer[layer_num_2].weight.clone()

            print(f"{epoch + 1} epoch - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            # wandb.log({
            #              "attack loss" : epoch_loss,
            #              "attack acc" : epoch_acc
            #          },
            #              step = epoch + 1
            #          )
            if True:
                th.save(net.state_dict(), f"./saved/{config.method.lower()}_{config.data_set}.pt")
            if attack:
                epoch_adv_loss, epoch_adv_acc = evaluate_model(adv_config)
                # wandb.log({
                #          "attack loss" : adv_loss,
                #          "attack acc" : adv_acc
                #      },
                #          step = epoch + 1
                #      )
                print(f"adv_loss: {epoch_adv_loss:.4f}, adv_acc: {epoch_adv_acc:.2f}%")
                config.attack = False
            epoch_clean_loss, epoch_clean_acc = evaluate_model(config)
            print(f"clean_loss: {epoch_clean_loss:.4f}, clean_acc: {epoch_clean_acc:.2f}%")
            # wandb.log({
            #          "clean loss" : clean_loss,
            #          "clean acc" : clean_acc
            #      },
            #          step = epoch
            #          )