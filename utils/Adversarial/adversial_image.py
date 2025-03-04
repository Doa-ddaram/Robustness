import torch as th
import torch.nn.functional as F
import matplotlib.pyplot as plt
def generate_adversial_image(model, image, target, epsilon = 0.05):
    '''
    Method : FGSM
    param model : trained model
    param image : original image (shape : [1, 28, 28])
    param target : target of original image
    param epsilon : adversial intensity
    '''
    image.requires_grad = True
    y_hat = model(image)
    loss = F.cross_entropy(y_hat, target)
    model.zero_grad()
    loss.backward()
    grad_sign = image.grad.sign()
    if image.grad is None:
        raise ValueError("Not calculate Gradient")
    adversarial_image = image + epsilon * grad_sign # image + pertubation
    adversarial_image = th.clamp(adversarial_image, 0, 1).detach()
    return adversarial_image

def generate_adversial_image_stdp(model, image, target, epsilon = 0.05):
    '''
    Method : FGSM
    param model : trained model
    param image : original image (shape : [time, 1, 28, 28])
    param target : target of original image
    param epsilon : adversial intensity
    '''
    image = image.to(th.device("cuda:0"))
    image = image.clone().detach().to(th.device("cuda:0"))
    image.requires_grad = True
    y_hat = model(image).mean(0)
    loss = F.cross_entropy(y_hat, target)
    model.zero_grad()
    loss.backward()
    grad_sign = image.grad.sign()
    if image.grad is None:
        raise ValueError("Not calculate Gradient")
    adversarial_image = image + epsilon * grad_sign # image + pertubation
    adversarial_image = th.clamp(adversarial_image, 0, 1).detach()
    return adversarial_image
    
def save_image(original_image, adversarial_image, filename, target):
    '''
    param original_image : original_image (shape : [1, 28, 28])
    param adversarial_image : adversarial_image (shape : [1, 28, 28])
    param filename : saved file name
    param target : target of the image
    '''
    channel = original_image[0].shape[0]
    if channel == 1 :
        original_image = original_image[0].squeeze(0).detach().cpu().numpy()
        adversarial_image = adversarial_image[0].squeeze(0).detach().cpu().numpy()
        c = 'gray'
    else:
        original_image = original_image * 0.5 + 0.5
        original_image = original_image[0].permute(1, 2, 0).detach().cpu().numpy()
        adversarial_image = adversarial_image * 0.5 + 0.5
        adversarial_image = adversarial_image[0].permute(1, 2, 0).detach().cpu().numpy()
        c = None
    fig, axes = plt.subplots(1, 2, figsize = (10, 5))
    
    axes[0].imshow(original_image, cmap = c)
    axes[0].set_title(f"Original\nLabel: {target[0].item()}")
    axes[0].axis('off')
    
    axes[1].imshow(adversarial_image, cmap = c)
    axes[1].set_title(f"adversarial\nLabel: {target[0].item()}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    
'''
Not yet, implementling hardly under this codes.
'''

def BIM(model, images, labels, epsilon, alpha, num_steps):
    images = images.clone().detach().requires_grad_(True)

    for _ in range(num_steps):
        # Forward pass
        outputs = model(images)
        model.zero_grad()
        
        # Calculate loss
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), labels)
        
        # Backward pass
        loss.backward()
        
        # Apply gradient sign update
        with torch.no_grad():
            images = images + alpha * images.grad.sign()
            images = torch.clamp(images, 0, 1)  # Ensure pixel values are within [0, 1]
            images = images.detach().requires_grad_(True)

    return images

def PGD(model, images, labels, epsilon, alpha, num_steps):
    images = images.clone().detach().requires_grad_(True)
    
    # Random initialization of perturbations within epsilon bounds
    delta = torch.zeros_like(images).uniform_(-epsilon, epsilon)
    images = images + delta
    
    for _ in range(num_steps):
        outputs = model(images)
        model.zero_grad()
        
        loss = F.nll_loss(F.log_softmax(outputs, dim=1), labels)
        loss.backward()
        
        with torch.no_grad():
            # Apply the gradient-based update
            delta = alpha * images.grad.sign()
            images = images + delta
            # Project the perturbations back to the epsilon-ball around the original image
            images = torch.clamp(images, 0, 1)
            images = torch.min(torch.max(images, images - epsilon), images + epsilon)
            images = images.detach().requires_grad_(True)
    
    return images

def CW(model, images, labels, epsilon=0.1, c=1e-4, num_iterations=1000):
    images = images.clone().detach().requires_grad_(True)
    
    # Initialize the perturbations
    perturbation = torch.zeros_like(images, requires_grad=True)
    
    # Set optimizer for the perturbation
    optimizer = torch.optim.Adam([perturbation], lr=0.01)
    
    for i in range(num_iterations):
        optimizer.zero_grad()
        
        # Apply perturbation to the image
        perturbed_images = torch.clamp(images + perturbation, 0, 1)
        
        # Get model predictions
        outputs = model(perturbed_images)
        
        # Define the loss as the difference between the true label and the model prediction
        loss = F.cross_entropy(outputs, labels)
        
        # Regularization term to make perturbation small
        loss += c * torch.sum(perturbation ** 2)
        
        # Backpropagate the loss and update perturbation
        loss.backward()
        optimizer.step()
        
        # Ensure perturbations remain within a valid range
        with torch.no_grad():
            perturbation.data = torch.clamp(perturbation.data, -epsilon, epsilon)
    
    return torch.clamp(images + perturbation, 0, 1)


def deepfool(model, image, label, max_iter=50, overshoot=0.02):
    image = image.clone().detach().requires_grad_(True)
    original_image = image.clone().detach()
    
    output = model(image)
    _, predicted = torch.max(output.data, 1)
    
    # If the model already misclassifies the image, return it directly
    if predicted != label:
        return image
    
    # Loop until we either exceed max iterations or successfully fool the model
    for _ in range(max_iter):
        output = model(image)
        _, predicted = torch.max(output.data, 1)
        
        if predicted != label:
            break
        
        # Calculate the gradients
        output[0, predicted].backward(retain_graph=True)
        image_grad = image.grad.data
        
        # Calculate the perturbation
        perturbation = -output[0, predicted].grad * image_grad
        perturbation = perturbation / torch.norm(perturbation, p=2)
        
        # Apply the perturbation and adjust the image
        image = image + (1 + overshoot) * perturbation
        
        # Ensure pixel values stay in the valid range
        image = torch.clamp(image, 0, 1)
        
    return image

def SimBA(model, dataset, image_size):
    batch_size = 32
    dataset_image_size = 28 * 28
    assert image_size == dataset_image_size
    