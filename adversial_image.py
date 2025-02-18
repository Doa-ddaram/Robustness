import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt

def generate_adversial_image(model, image, target, epsilon = 0.1):
    '''
    Method : FGSM
    param model : trained model
    param image : original image (shape : [1, 28, 28])
    param target : target of original image
    param epsilon : adversial intensity
    '''
    image = image.unsqueeze(0)
    image = image.requires_grad_(True)
    
    y_hat = model(image)
    Loss = F.cross_entropy(y_hat, target.unsqueeze(0))
    
    model.zero_grad()
    Loss.backward()
    
    perturbation = epsilon * image.grad.sign()
    adversarial_image = image + perturbation
    adversarial_image = th.clamp(adversarial_image, 0, 1.)
    return adversarial_image.squeeze(0).detach()

def display(image, target, adversarial = False):
    '''
    param image : image (shape : [1, 28, 28])
    param target : target of the image
    param adversarial : image whether adversial
    '''
    image = image.squeeze(0)
    plt.imshow(image.cpu().numpy(), cmap = 'gray')
    plt.title(f"target:{target.item()} ({'Adversarial' if adversarial else 'Original'})")
    plt.axis('off')
    plt.show()
    
def save(original_image, adversarial_image, filename, target):
    '''
    param original_image : original_image (shape : [1, 28, 28])
    param adversarial_image : adversarial_image (shape : [1, 28, 28])
    param filename : saved file name
    param target : target of the image
    '''
    
    original_image = original_image.squeeze(0)
    adversarial_image = adversarial_image.squeeze(0)
    
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    
    axes[0].imshow(original_image.cpu().numpy(), cmap = 'gray')
    axes[0].set_title(f"Original\nLabel: {target.item()}")
    axes[0].axis('off')
    
    axes[1].imshow(adversarial_image.cpu().numpy(), cmap = 'gray')
    axes[1].set_title(f"adversarial\nLabel: {target.item()}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()