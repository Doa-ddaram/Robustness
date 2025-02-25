import torch as th
def nes(net : th.nn.Module,
        image : th.Tensor,
        epsilon : float = 0.3,
        alpha : float = 0.01,
        iteration : int = 10,
        sigma : float = 0.01,
        sample_size : int = 20
):
    image = image.clone().detach()
    ori_image = image.clone().detach()
    
    for _ in range(iteration):
        gradient_prob = th.zeros_like(image)
        
        for _ in range(sample_size):
            noise = th.randn_like(image) * sigma
            loss_p = net(image + noise).max(1)[0].sum()
            loss_m = net(image - noise).max(1)[0].sum()
            gradient_prob  += (loss_p - loss_m) * noise
            
        gradient_prob /= (2 * sigma * sample_size)
        
        image = image + alpha * gradient_prob.sign()
        image = th.clamp(image, 0, 1)
    return image

def nes_stdp(net,
        image : th.Tensor,
        epsilon : float = 0.3,
        alpha : float = 0.01,
        iteration : int = 10,
        sigma : float = 0.01,
        sample_size : int = 20
):
    image = image.to(th.device("cuda:0"))
    image = image.clone().detach().to(th.device("cuda:0"))
    ori_image = image.clone().detach()
    
    for _ in range(iteration):
        gradient_prob = th.zeros_like(image)
        
        for _ in range(sample_size):
            noise = th.randn_like(image, device = th.device("cuda:0")) * sigma
            loss_p = net(image + noise).mean(0).max(1)[0].sum()
            loss_m = net(image - noise).mean(0).max(1)[0].sum()
            gradient_prob  += (loss_p - loss_m) * noise
            
        gradient_prob /= (2 * sigma * sample_size)
        
        image = image + alpha * gradient_prob.sign()
        image = th.clamp(image, 0, 1)
    return image

