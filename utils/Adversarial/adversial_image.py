import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn

def generate_adversial_image(model, image, target, epsilon = 0.05):
    '''
    Method : FGSM
    param model : trained model
    param image : original image (shape : [batch_size, 1, 28, 28])
    param target : target of original image
    param epsilon : adversial intensity
    '''
    image = image.clone().detach().to(th.device("cuda:0"))
    image.requires_grad = True
    y_hat = model(image).mean(0)
    loss = F.cross_entropy(y_hat, target)
    model.zero_grad()
    loss.backward()
    grad_sign = image.grad.sign()
    adversarial_image = image + epsilon * grad_sign # image + pertubation
    adversarial_image = th.clamp(adversarial_image, 0, 1).detach()
    return adversarial_image
    
def save_image(original_image, adversarial_image, filename, target):
    '''
    param original_image : original_image (shape : [batch_size, 1 or 3, 28, 28])
    param adversarial_image : adversarial_image (shape : [batch_size, 1 or 3, 28, 28])
    param filename : saved file name
    param target : target of the image
    '''
    import random
    r = random.randint(0, len(original_image) - 1)
    
    channel = original_image[r].shape[0]
    if channel == 1 :
        original_image = original_image[r].squeeze(0).detach().cpu().numpy()
        adversarial_image = adversarial_image[r].squeeze(0).detach().cpu().numpy()
        c = 'gray'
    else:
        original_image = original_image * 0.5 + 0.5
        original_image = original_image[r].permute(1, 2, 0).detach().cpu().numpy()
        adversarial_image = adversarial_image * 0.5 + 0.5
        adversarial_image = adversarial_image[r].permute(1, 2, 0).detach().cpu().numpy()
        c = None
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    
    axes[0].imshow(original_image, cmap = c)
    axes[0].set_title(f"Original\nLabel: {target[r].item()}")
    axes[0].axis('off')
    
    axes[1].imshow(adversarial_image, cmap = c)
    axes[1].set_title(f"adversarial\nLabel: {target[r].item()}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
def compute_norm_differences(orig: th.Tensor, adv: th.Tensor):
    '''
    :param orig: origin image (B, C, H, W)
    :param adv:  adv example (B, C, H, W)
    :return: (mean_l2, mean_linf)
    '''
    # norm calculate
    diff = adv - orig
    # L2 norm (each sample)
    l2 = diff.view(diff.size(0), -1).norm(p=2, dim=1)  # (B,)
    # Linf norm (each sample)
    linf = diff.view(diff.size(0), -1).norm(p=float('inf'), dim=1)  # (B,)

    # batch all avg
    return l2.mean().item(), linf.mean().item()

def compute_confidence(model, images: th.Tensor, labels: th.Tensor):
    """
    :param model: net
    :param images: iknput image (B, C, H, W)
    :param labels: label (B,)
    :return: avg confidence
    """
    with th.no_grad():
        logits = model(images)
        probs = F.softmax(logits, dim=1)
        conf = probs[th.arange(len(labels)), labels]
    return conf.mean().item()

def compute_attack_success_rate(model, adv: th.Tensor, labels: th.Tensor):
    """
    :param model: net
    :param adv: adv example (B, C, H, W)
    :param labels: origin label (B,)
    :return: adv success (0.0 ~ 1.0)
    """
    with th.no_grad():
        preds = model(adv).argmax(dim=1)
    # 
    incorrect = (preds != labels).sum().item()
    return incorrect / len(labels)

def evaluate_adversarial(model, data_loader, device, epsilon=0.1, loss_fn=None):
    """
    :param model: net
    :param data_loader: data_loader
    :param device: 'cuda'
    :param epsilon: FGSM intensity
    :param loss_fn: Loss function
    """
    if loss_fn is None:
        loss_fn = th.nn.CrossEntropyLoss()

    model.eval()

    total_l2, total_linf = 0.0, 0.0
    total_attack_success = 0.0
    total_samples = 0

    total_conf_orig, total_conf_adv = 0.0, 0.0

    for data, target in data_loader:
        data, target = data.to(device), target.to(device)

        # 1)
        conf_orig = compute_confidence(model, data, target)

        # 2) 
        data.requires_grad = True
        output = model(data)
        loss = loss_fn(output, target)
        model.zero_grad()
        loss.backward()

        data_grad = data.grad.data
        adv_data = data + epsilon * data_grad.sign()
        adv_data = th.clamp(adv_data, 0, 1)

        # 3) 
        conf_adv = compute_confidence(model, adv_data, target)

        # 4)
        l2, linf = compute_norm_differences(data, adv_data)

        # 5) 
        asr = compute_attack_success_rate(model, adv_data, target)

        batch_size = data.size(0)
        total_l2 += l2 * batch_size
        total_linf += linf * batch_size
        total_conf_orig += conf_orig * batch_size
        total_conf_adv += conf_adv * batch_size
        total_attack_success += asr * batch_size
        total_samples += batch_size

    # 
    mean_l2 = total_l2 / total_samples
    mean_linf = total_linf / total_samples
    mean_asr = total_attack_success / total_samples
    mean_conf_orig = total_conf_orig / total_samples
    mean_conf_adv = total_conf_adv / total_samples

    print(f"FGSM (epsilon = {epsilon}) evaluate result")
    print(f"  - avg L2 norm difference   : {mean_l2:.4f}")
    print(f"  - adv Linf norm difference   : {mean_linf:.4f}")
    print(f"  - adv success(ASR)    : {mean_asr*100:.2f}%")
    print(f"  - ori avg confidence : {mean_conf_orig:.4f}")
    print(f"  - adv avg confidence : {mean_conf_adv:.4f}")