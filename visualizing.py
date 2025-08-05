from utils.model import SNN, SpikingVGG16
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt

import numpy as np
def draw_weight_map(fig : plt.Figure, axe : plt.Axes, weight_map : th.Tensor):
    """
    Draw the weight map on the given axes.

    Args:
        fig (plt.Figure): The figure object to draw on.
        axe (plt.Axes): The axes object to draw on.
        weight_map (torch.Tensor): The weight map tensor to visualize.
    """
    # Turn off the axis for the given axes object to make the visualization cleaner
    axe.axis("off")
    axe.imshow(weight_map, cmap="gray")

def extract_first_conv_weights(model):
    """ Extract the weights of the first Conv2d layer in the model."""
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            abs_weight = th.abs(module.weight.detach())
            return abs_weight.cpu().numpy().flatten()
    raise ValueError("No Conv2d layer found in the model.")

def plot_first_conv_comparison(model1, model2, name1 : str ="SNN-GD", name2 : str ="SNN-GD+STDP"):
    w1 = extract_first_conv_weights(model1)
    w2 = extract_first_conv_weights(model2)

    min_val = 0
    max_val = max(w1.max(), w2.max())
    bin_count = 40
    bins = np.linspace(min_val, max_val, bin_count + 1)
    
    fig, axes = plt.subplots(2, 1, figsize=(6, 4))

    axes[0].hist(w1, bins=bins, color='red', edgecolor='black')
    axes[0].legend([name1])
    axes[0].set_ylim(0, 120)
    axes[0].set_xlim(left = 0, right = 2.5)
    axes[0].tick_params(axis='x', labelsize=14)
    axes[0].tick_params(axis='y', labelsize=14)
    
    axes[1].hist(w2, bins=bins, color='blue', edgecolor='black')
    axes[1].legend([name2])
    axes[1].set_ylim(0, 120)
    axes[1].set_xlim(left = 0, right = 2.5)
    axes[1].tick_params(axis='x', labelsize=14)
    axes[1].tick_params(axis='y', labelsize=14)
    
    plt.tight_layout()
    fig.savefig("./images/weight_map/weight_histogram.pdf")
    plt.close()

if __name__ == "__main__":
    # Example usage of the plot_first_conv_comparison function on MNIST dataset
    net_snn_mnist = SNN().to(th.device("cuda:0"))
    net_snn_mnist.load_state_dict(th.load("./saved/snn_MNIST.pt"))
    net_stdp_mnist = SNN().to(th.device("cuda:0"))
    net_stdp_mnist.load_state_dict(th.load("./saved/stdp_MNIST.pt"))
    plot_first_conv_comparison(net_snn_mnist, net_stdp_mnist)

    # Example usage of the draw_weight_map function on CIFAR10 dataset
    # net_snn_cifar10 = SpikingVGG16(num_classes=10).to(th.device("cuda:0"))
    # net_snn_cifar10.load_state_dict(th.load("./saved/snn_CIFAR10.pt"))
    # net_stdp_cifar10 = SpikingVGG16(num_classes=10).to(th.device("cuda:0"))
    # net_stdp_cifar10.load_state_dict(th.load("./saved/stdp_CIFAR10.pt"))
    # plot_first_conv_comparison(net_snn_cifar10, net_stdp_cifar10)
    