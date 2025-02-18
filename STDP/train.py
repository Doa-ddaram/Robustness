import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam
from spikingjelly.activation_based import learning, layer, neuron, functional

T = 8
N = 32
C = 3
H = 24
W = 24
lr = 0.1
tau_pre = 2.
tau_post = 10.
step_mode = 'm'

def f_weight(x):
    return torch.clamp(x, -1, 1.)

net = nn.Sequential(
    layer.Conv2d(1, 32, kernel_size = 3, stride = 1, padding = 1),
    neuron.LIFNode(tau = 2., surrogate_function = surrogate.ATan()),
    layer.MaxPool2d(kernel_size = 2, stride = 2),
    layer.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1),
    neuron.LIFNode(tau = 2., surrogate_function = surrogate.ATan()),
    layer.MaxPool2d(kernel_size = 2, stride = 2),
    layer.Flatten()
    layer.
)

stdp_learners = []

for i in range(net.__len__()):
    if isinstance(net[i], instances_stdp):
        stdp_learners.append(
            learning.STDPLearner(
                step_mode = 'm', synapse = net[i], sn = net[i+1],
                tau_pre = tau_pre, tau_post = tau_post, f_pre = f_weight, f_post = f_weight)
        )

parameters_stdp = []
for m in net.modules():
    if isinstance(m, instances_stdp):
        for p in m.parameters():
            params_stdp.append(p)
            
parameters_stdp_set = set(parameters_stdp)
parameters_gd = []
for p in net.parameters():
    if p not in parameters_stdp_set:
        parameters_gd.append(p)

optimizer_gd.zero_grad()
optimizer_stdp.zero_grad()
y_hat = net(data).mean(0)
Loss = F.cross_entropy(y_hat, target)
loss.backward()

optimizer_stdp.zero_grad()
for i in range(stdp_learners.__len__()):
    stdp_learners[i].step(on_grad = False)
    
    
optimizer_gd.step()
optimizer_stdp.step()

functional.reset_net(net)

for i in range(stdp_learners.__len__()):
    stdp_learners[i].reset()