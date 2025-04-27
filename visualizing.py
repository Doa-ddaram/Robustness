from utils.model import SNN, CNN
import torch as th
import torch.nn as nn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def draw_weight_map(fig, axe, weight_map):
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
    
    
def initialize_low_variance(conv_layer: nn.Conv2d, threthold : float = 0.1) -> th.Tensor:
    """
    Calculate the variance of the weights of a convolutional layer.
    """
    weight = conv_layer.weight.data
    num_filters = weight.shape[0]
    for i in range(num_filters):
        var = weight[i].var().item()
        if var < threthold :
            weight[i].zero_()
            print(f"Filter {i} has low variance, setting to zero.")
    return weight

if __name__ == "__main__":
    # Create an instance of the SNN model with T=10 and move it to the GPU
    method = "snn"
    net = SNN().to(th.device("cuda:0"))
    net.load_state_dict(th.load(f"./saved/{method}_MNIST.pt"))
    
    # Initialize an empty list to store Conv2D modules
    conv_list = []
    
    #Iterate through the modules of the network and collect Conv2D layers
    for module in net.modules():
        if isinstance(module, nn.Conv2d): 
            conv_list.append(module)
    
    # Create a figure and two subplots for visualizing weight maps
    fig, axes = plt.subplots(len(conv_list), 1)
    fig.suptitle(f"Weight Map of {method}")
    for idx, (axe, module) in enumerate(zip(axes, conv_list)):
        if isinstance(module, nn.Conv2d):
            # Extract the weights of the Conv2D layer and clone them to CPU
            weight = module.weight.detach().cpu().clone()[:,:3]        
            weight = th.abs(weight)
            
            # If the weight shape is not compatible, adjust it
            if weight[1].shape != 3: 
                weight = weight[:, 0, :, :].unsqueeze(dim=1)
            else: 
                weight = weight.view(weight.shape[0] * weight.shape[1], -1, weight.shape[2], weight.shape[3])
                    
            # Create a grid of weights for visualization
            grid = make_grid(weight, nrow=8, normalize=True, padding=1)
            grid = grid.permute(1, 2, 0)
            
            # Set the title of the axes to indicate the content being visualized
            axe.set_title(f"{idx + 1}th convolution Weight Map shape")
            
            # Draw the weight map on the current axes
            draw_weight_map(fig, axe, grid)
            
    # Save the figure containing the weight maps to a file
    fig.savefig(f"./images/weight_map/weight_map_{method}.png")
    print(f"saved weight map to ./images/weight_map/weight_map_{method}.png")
    
    fig, axes = plt.subplots(len(conv_list), 1)

    for idx, (axe, module) in enumerate(zip(axes, conv_list)):
        if isinstance(module, nn.Conv2d):
            
            module.data = initialize_low_variance(module)
            
            # Extract the weights of the Conv2D layer and clone them to CPU
            weight = module.weight.detach().cpu().clone()[:,:3]        
            weight = th.abs(weight)
            
            # If the weight shape is not compatible, adjust it
            if weight[1].shape != 3: 
                weight = weight[:, 0, :, :].unsqueeze(dim=1)
            else: 
                weight = weight.view(weight.shape[0] * weight.shape[1], -1, weight.shape[2], weight.shape[3])
                    
            # Create a grid of weights for visualization
            grid = make_grid(weight, nrow=8, normalize=True, padding=1)
            grid = grid.permute(1, 2, 0)
            
            # Set the title of the axes to indicate the content being visualized
            axe.set_title(f"{idx + 1}th convolution Weight Map shape")
            
            # Draw the weight map on the current axes
            draw_weight_map(fig, axe, grid)
            
    # Save the figure containing the weight maps to a file
    fig.savefig(f"./images/weight_map/weight_map_{method}_scailing.png")
    print(f"saved weight map to ./images/weight_map/weight_map_{method}_scailing.png")