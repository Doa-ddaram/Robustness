import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
from .spikingjelly.spikingjelly.activation_based import functional

def generate_adversial_image_fgsm(model, image, target, epsilon=0.05):
    """
    Method : FGSM
    param model : trained model
    param image : original image (shape : [batch_size, 1, 28, 28])
    param target : target of original image
    param epsilon : adversial intensity
    """
    image = image.clone().detach().to(th.device("cuda:0"))
    target = target.clone().detach().to(th.device("cuda:0"))

    # model = SNN(T=10).cuda()
    # model.load_state_dict(th.load(f"./saved/snn_MNIST.pt"))

    image.requires_grad = True
    model.zero_grad()
    y_hat = model(image)

    if y_hat.dim() == 3:
        y_hat = y_hat.mean(0)
    loss = F.cross_entropy(y_hat, target)

    # model.zero_grad()
    # loss.backward()

    # grad_sign = image.grad.sign()

    grad_sign = th.autograd.grad(loss, image, retain_graph=False, create_graph=False)[0].sign()

    adversarial_image = image + epsilon * grad_sign  # image + pertubation
    adversarial_image = th.clamp(adversarial_image, 0, 1).detach()

    return adversarial_image


def generate_adv_image_pgd(model, image, target, epsilon=0.05):
    """
    Method : PGD
    param model : trained model
    param image : original image (shape : [batch_size, 1, 28, 28])
    param target : target of original image
    param epsilon : adversial intensity
    """
    image = image.clone().detach().to(th.device("cuda:0"))
    target = target.clone().detach().to(th.device("cuda:0"))

    ori_image = image.data

    iters = 20

    loss = F.cross_entropy

    alpha = 0.05

    for i in range(iters):
        image.requires_grad_()

        outputs = model(image).mean(0)

        # print("image.requires_grad:", image.requires_grad)
        # print("outputs.requires_grad:", outputs.requires_grad)

        model.zero_grad()
        cost = loss(outputs, target).to(th.device("cuda:0"))
        cost.backward()

        outputs = outputs.detach()

        # print("cost.requires_grad:", cost.requires_grad)
        # print("cost.grad_fn:", cost.grad_fn)

        adv_images = image + alpha * image.grad.sign()
        eta = th.clamp(adv_images - ori_image, min=-epsilon, max=epsilon)
        image = th.clamp(ori_image + eta, min=0, max=1).detach().clone().requires_grad_(True)

        functional.reset_net(model)

    return adv_images


def save_image(original_image, adversarial_image, filename, target):
    """
    param original_image : original_image (shape : [batch_size, 1, 28, 28] or [batch_size, 3, 32, 32])
    param adversarial_image : adversarial_image (shape : [batch_size, 1, 28, 28] or [batch_size, 3, 32, 32])
    param filename : saved file name
    param target : target of the image
    """
    import random

    original_image = original_image.mean(1)
    adversarial_image = adversarial_image.mean(1)
    
    r = random.randint(0, len(original_image) - 1)
    channel = original_image[r].shape[0]
    if channel == 1:
        original_image = original_image[r].squeeze(0).detach().cpu().numpy()
        adversarial_image = adversarial_image[r].squeeze(0).detach().cpu().numpy()
        c = "gray"
        label = list(range(0, 10))
    elif channel == 28:
        original_image = original_image[r].detach().cpu().numpy()
        adversarial_image = adversarial_image[r].detach().cpu().numpy()
        c = "gray"
        label = list(range(0, 10))
    elif channel == 32:
        original_image = original_image * 0.5 + 0.5
        original_image = original_image[r].permute(1, 2, 0).detach().cpu().numpy()
        adversarial_image = adversarial_image * 0.5 + 0.5
        adversarial_image = adversarial_image[r].permute(1, 2, 0).detach().cpu().numpy()
        c = None
        label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    else:
        original_image = original_image * 0.5 + 0.5
        original_image = original_image[r].permute(1, 2, 0).detach().cpu().numpy()
        adversarial_image = adversarial_image * 0.5 + 0.5
        adversarial_image = adversarial_image[r].permute(1, 2, 0).detach().cpu().numpy()
        c = None
        label = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(original_image, cmap=c)
    axes[0].set_title(f"Original\nLabel: {label[target[r].item()]}")
    axes[0].axis("off")

    axes[1].imshow(adversarial_image, cmap=c)
    axes[1].set_title(f"adversarial\nLabel: {label[target[r].item()]}")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()