import torch as th
import torch.nn as nn
from torchvision.models import vgg16
from dataclasses import replace
from .spikingjelly.spikingjelly.activation_based import functional, learning
from .spikingjelly.spikingjelly.activation_based.model import spiking_vgg
from typing import Tuple
from tqdm.auto import tqdm
from .model import CNN, SNN, SNN_CIFAR10, SpikingResNet18
from .adversarial_image import generate_adversial_image_fgsm, save_image
from .config import Config
import wandb

# Training function
# Function to train the model
def train_model(config: Config, mode=None) -> Tuple[float, float]:
    net = config.network
    net.train()  # Set the network to training mode
    total_acc, total_loss, length = 0, 0, 0  # Initialize metrics

    # Iterate over the training data
    for i, (data, target) in tqdm(enumerate(config.train_loader), desc="Training", total=len(config.train_loader)):
        data, target = data.to(config.device), target.to(config.device)  # Move data to the configured device

        config.optimizer.zero_grad()  # Reset gradients for the optimizer
        if config.parameters_stdp:
            config.optimizer_stdp.zero_grad()  # Reset gradients for STDP optimizer if applicable

        # Forward pass through the network
        y_hat = net(data).mean(0) if config.method != "CNN" else net(data)
        loss = config.loss_fn(y_hat, target)  # Compute the loss

        # STDP training mode
        if mode == "stdp":
            if config.parameters_stdp:
                for learner in config.stdp_learners:
                    with th.no_grad():
                        learner.step(on_grad=True)  # Perform STDP updates

        # Gradient descent training mode
        if mode == "gd":
            loss.backward()  # Backpropagate the loss
            config.optimizer.step()  # Update weights using the optimizer

        if mode == "stdp":
            if config.parameters_stdp:
                config.optimizer_stdp.step()  # Update STDP parameters
                with th.no_grad():
                    for learner in config.stdp_learners:
                        learner.synapse.weight.data.clamp_(-10, 5.0)  # Clamp weights to a range

        functional.reset_net(net)  # Reset the network's state
        if config.parameters_stdp:
            for learner in config.stdp_learners:
                learner.reset()  # Reset STDP learners

        # Update metrics
        total_loss += loss.item()
        total_acc += (y_hat.argmax(1) == target).sum().item()
        length += len(target)

        # Free up memory
        del loss, y_hat, target, data
        th.cuda.empty_cache()

    # Return average loss and accuracy
    return total_loss / length, (total_acc / length) * 100

# Function to evaluate the model
def evaluate_model(config: Config) -> Tuple[float, float, (float|None)]:
    net = config.network  # Get the network from the configuration
    if config.load:
        # Load pre-trained model weights if specified
        net.load_state_dict(th.load(f"./saved/{config.method.lower()}_{config.data_set}.pt"))

    net.eval()  # Set the network to evaluation mode
    total_loss, total_acc = 0, 0  # Initialize metrics
    length = 0
    attack_successes = 0  # Track adversarial attack success rate
    for i in config.stdp_learners:
        i.disable()  # Disable STDP learners during evaluation

    # Iterate over the test data
    for i, (data, target) in tqdm(enumerate(config.test_loader), desc="Evaluation", total=len(config.test_loader)):
        data, target = data.to(config.device), target.to(config.device)  # Move data to the configured device
        
        if config.attack:
            # Generate adversarial examples if attack mode is enabled
            net.train()
            clean_pred = net(data).mean(0).argmax(1) if config.method != "CNN" else net(data).argmax(1)
            adv_imgs = generate_adversial_image_fgsm(net, data, target, config.epsilon)
            save_image(data, adv_imgs, f"./images/image_comparison/comparison_image_{config.method.lower()}_{config.data_set}.png", target)
            data = adv_imgs
            net.eval()

        with th.no_grad():
            # Forward pass through the network
            y_hat = net(data).mean(0) if config.method != "CNN" else net(data)
            loss = config.loss_fn(y_hat, target)  # Compute the loss

        # Update metrics
        total_loss += loss.item()
        pred_target = y_hat.argmax(1)
        total_acc += (pred_target == target).sum().item()
        
        if config.attack:
            # Track adversarial attack success rate
            attack_successes += (pred_target != clean_pred).sum().item()

        if config.method != "CNN":
            functional.reset_net(net)  # Reset the SNN's state
        length += len(target)
    for i in config.stdp_learners:
        i.enable()  # Re-enable STDP learners after evaluation
    attack_success_rate = (attack_successes / length) * 100 if config.attack else None
    # Return average loss, accuracy, and attack success rate
    return total_loss / length, (total_acc / length) * 100, attack_success_rate

# Function to train and evaluate the model
def train_evaluate(config: Config) -> None:
    # Initialize the network based on the method and dataset
    if config.method == "CNN":
        net = CNN().to(config.device) if config.data_set == "MNIST" else vgg16().to(config.device)
    else:
        net = (
            SNN(T=config.timestep).to(config.device)
            if config.data_set == "MNIST"
            else SpikingResNet18(T=config.timestep).to(config.device)
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
        # Create separate configurations for clean and adversarial evaluation
        adv_config = replace(config, attack=True)
        config = replace(config, attack=False)

    if config.load:
        # Load pre-trained model weights
        print(f"Loading {config.method.lower()}_{config.data_set}.pt")
        
        # Evaluate the model if pre-trained weights are loaded
        if attack:
            adv_loss, adv_acc, attack_success_rate = evaluate_model(adv_config)
            print(f"adv_loss: {adv_loss:.4f}, adv_acc: {adv_acc:.2f}%, attack_success_rate: {attack_success_rate:.2f}%")
        clean_loss, clean_acc, attack_success_rate = evaluate_model(config)
        print(f"clean_loss: {clean_loss:.4f}, clean_acc: {clean_acc:.2f}%")
        print(f'finishing evaluation {config.method.lower()}')
    else:
        # if training, Log configuration to Weights & Biases
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
        
        for epoch in range(config.num_epochs):
            if config.method == "SNN":
                mode = "gd"
            elif config.method == "STDP":
                if epoch % 5 == 0:
                    mode = "stdp"
                else:
                    mode = "gd"

            epoch_loss, epoch_acc = train_model(config, mode)
            print(f"{epoch + 1} Epoch - Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.2f}%")
            wandb.log({
                        "train loss" : epoch_loss,
                        "train acc" : epoch_acc
                    },
                        step = epoch
                        )
            if config.save:
                th.save(net.state_dict(), f"./saved/{config.method.lower()}_{config.data_set}.pt")

            if mode == "gd":
                if attack:
                    epoch_adv_loss, epoch_adv_acc, epoch_attack_success_rate = evaluate_model(adv_config)
                    print(f"adv_loss: {epoch_adv_loss:.4f}, adv_acc: {epoch_adv_acc:.2f}%")
                    wandb.log({
                        "adv loss" : epoch_adv_loss,
                        "adv acc" : epoch_adv_acc,
                        "attack_success_rate" : epoch_attack_success_rate
                    },
                        step = epoch
                        )
                epoch_clean_loss, epoch_clean_acc, epoch_attack_success_rate = evaluate_model(config)
                print(f"clean_loss: {epoch_clean_loss:.4f}, clean_acc: {epoch_clean_acc:.2f}%")
                wandb.log({
                        "clean loss" : epoch_clean_loss,
                        "clean acc" : epoch_clean_acc
                    },
                        step = epoch
                        )
