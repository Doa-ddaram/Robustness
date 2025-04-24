import torch as th
import torch.nn as nn
from torchvision.models import vgg16
from dataclasses import replace
from .spikingjelly.spikingjelly.activation_based import functional, learning
from typing import Tuple
from tqdm.auto import tqdm
from .model import CNN, SNN, SNN_CIFAR10
from .adversarial_image import generate_adversial_image_fgsm, save_image
from .config import Config
import wandb

# Training function
def train_model(config: Config, mode=None) -> Tuple[float, float]:
    net = config.network
    net.train()
    total_acc, total_loss, length = 0, 0, 0

    for i, (data, target) in tqdm(enumerate(config.train_loader), desc="Training", total=len(config.train_loader)):
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

        if mode == "gd":
            loss.backward()
            config.optimizer.step()

        if mode == "stdp":
            if config.parameters_stdp:
                config.optimizer_stdp.step()
                with th.no_grad():
                    for learner in config.stdp_learners:
                        learner.synapse.weight.data.clamp_(-10, 5.0)

        functional.reset_net(net)
        if config.parameters_stdp:
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
    if config.load:
        net.load_state_dict(th.load(f"./saved/{config.method.lower()}_{config.data_set}.pt"))

    net.eval()
    total_loss, total_acc = 0, 0
    length = 0

    for i in config.stdp_learners:
        i.disable()

    for i, (data, target) in tqdm(enumerate(config.test_loader), desc="Evaluation", total=len(config.test_loader)):
        data, target = data.to(config.device), target.to(config.device)
        if config.attack:
            net.train()
            adv_imgs = generate_adversial_image_fgsm(net, data, target, config.epsilon)
            save_image(data, adv_imgs, f"./images/comparison_image_{config.method.lower()}_{config.data_set}.png", target)
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

    for i in config.stdp_learners:
        i.enable()

    return total_loss / length, (total_acc / length) * 100

def train_evaluate(config: Config) -> None:
    if config.method == "CNN":
        net = CNN().to(config.device) if config.data_set == "MNIST" else vgg16().to(config.device)
    else:
        net = (
            SNN(T=config.timestep).to(config.device)
            if config.data_set == "MNIST"
            else SNN_CIFAR10(T=config.timestep).to(config.device)
        )

    stdp_learners, parameters_stdp, parameters_gd = [], [], []

    if config.method == "STDP":
        added = 0
        for layers in net.modules():
            if isinstance(layers, nn.Sequential):
                for i, layer_in in enumerate(layers):
                    if isinstance(layer_in, nn.Conv2d) and added < 1:
                        print(f"Adding STDP learner from {layers[i]} to {layers[i + 1]}")
                        stdp_learners.append(
                            learning.STDPLearner(
                                step_mode="m",
                                synapse=layers[i],
                                sn=layers[i + 1],
                                tau_pre=5.0,
                                tau_post=10.0,
                                f_pre = lambda x: 0.001 * x,
                                f_post = lambda x: -0.001 * x
                                                        )
                                                    )
                        added += 1

        for module in net.modules():
            if isinstance(module, nn.Conv2d) and added > 0:
                parameters_stdp.extend(module.parameters())
    else:
        parameters_gd = net.parameters()

    all_parameters = list(net.parameters())
    parameters_stdp_set = set(parameters_stdp)
    parameters_gd = [p for p in all_parameters if p not in parameters_stdp_set]

    config.optimizer = th.optim.Adam(list(parameters_stdp) + list(parameters_gd), lr=config.lr, weight_decay=0)
    config.network = net
    config.parameters_stdp = parameters_stdp
    config.stdp_learners = stdp_learners
    config.optimizer_stdp = (
        th.optim.SGD(config.parameters_stdp, lr=config.lr * 0.001, momentum=0.0) if config.parameters_stdp else None
    )

    attack = config.attack
    if attack:
        adv_config = replace(config, attack=True)
        config = replace(config, attack=False)
        
    Config = {
        'dataset' : config.data_set,
        'batch_size' : config.batch_size,
        'num_epochs' : config.num_epochs,
        'learning_rate' : config.lr,
        'seed' : config.seed,
        'epsilon' : config.epsilon
    }

    wandb.init(project = config.data_set,
            group = config.method,
            config = Config,
            name = config.data_set + '_' + config.method)
    
    # layer_num = 0
    # layer_num_2 = 3
    
    # weight = config.network.layer[layer_num].weight.clone()
    # weight2 = config.network.layer[layer_num_2].weight.clone()

    print("Weight stats after STDP:")
    print("min:", weight.data.min().item())
    print("max:", weight.data.max().item())
    print("mean:", weight.data.mean().item())
            
    for epoch in range(config.num_epochs):
        if config.method == "SNN":
            mode = "gd"
        elif config.method == "STDP":
            if epoch % 5 == 0:
                mode = "stdp"
            else:
                mode = "gd"

        if config.load:
            if attack:
                epoch_adv_loss, epoch_adv_acc = evaluate_model(adv_config)
                print(f"{epoch + 1} Epoch - adv_loss: {epoch_adv_loss:.4f}, adv_acc: {epoch_adv_acc:.2f}%")
                wandb.log({
                      "adv loss" : epoch_adv_loss,
                      "adv acc" : epoch_adv_acc
                  },
                      step = epoch
                      )
            epoch_clean_loss, epoch_clean_acc = evaluate_model(config)
            print(f"Epoch - clean_loss: {epoch_clean_loss:.4f}, clean_acc: {epoch_clean_acc:.2f}%")
            wandb.log({
                      "clean loss" : epoch_clean_loss,
                      "clean acc" : epoch_clean_acc
                  },
                      step = epoch
                      )
        else:
            epoch_loss, epoch_acc = train_model(config, mode)

            # print(f"After {mode.upper()}:", (weight - config.network.layer[layer_num].weight).abs().sum().item())
            # print(f"After {mode.upper()}:", (weight2 - config.network.layer[layer_num_2].weight).abs().sum().item())

            # weight = config.network.layer[layer_num].weight.clone()
            # weight2 = config.network.layer[layer_num_2].weight.clone()

            # print(f"Weight stats after {mode.upper()}:")
            # print("min:", weight.data.min().item())
            # print("max:", weight.data.max().item())
            # print("mean:", weight.data.mean().item())

            print(f"{epoch + 1} Epoch - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            wandb.log({
                      "train loss" : epoch_loss,
                      "train acc" : epoch_acc
                  },
                      step = epoch
                      )
            if True:
                th.save(net.state_dict(), f"./saved/{config.method.lower()}_{config.data_set}.pt")

            if mode == "gd":
                if attack:
                    epoch_adv_loss, epoch_adv_acc = evaluate_model(adv_config)
                    print(f"adv_loss: {epoch_adv_loss:.4f}, adv_acc: {epoch_adv_acc:.2f}%")
                    wandb.log({
                      "adv loss" : epoch_adv_loss,
                      "adv acc" : epoch_adv_acc
                  },
                      step = epoch
                      )
                epoch_clean_loss, epoch_clean_acc = evaluate_model(config)
                print(f"clean_loss: {epoch_clean_loss:.4f}, clean_acc: {epoch_clean_acc:.2f}%")
                wandb.log({
                      "clean loss" : epoch_clean_loss,
                      "clean acc" : epoch_clean_acc
                  },
                      step = epoch
                      )
