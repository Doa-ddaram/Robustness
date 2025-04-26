from utils.model import SNN, CNN
import torch as th
import torch.nn as nn

def variance(conv_layer : nn.Conv2d) :
    with th.no_grad():
        weights = conv_layer.weight.data
        num_filters = weights.shape[0]
        for i in range(num_filters):
            var = weights[i].var().item()
            print(var)
        print(weights.shape)
        # mean = weight.mean(dim=1, keepdim=True)
        # variance = ((weight - mean) ** 2).mean(dim=1, keepdim=True)
        # return mean,variance
    
if __name__ == "__main__":
    net = SNN(T=10).to(th.device("cuda:0"))
    net.load_state_dict(th.load(f"./saved/stdp_MNIST.pt"))
    for module in net.modules():
        if isinstance(module, nn.Conv2d):
            # mean, variance = variance(module)
            # print(mean, variance)
            variance(module)