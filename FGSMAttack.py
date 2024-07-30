import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.autograd import Variable
from PIL import Image

# Load a pre-trained model
model = models.resnet50(pretrained=True)
model.eval()

# Image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(img_path):
    img = cv2.imread(cv2.samples.findFile(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = Image.fromarray(img)
    img = preprocess(img)
    img = img.unsqueeze(0)
    return img

def fgsm_attack(image, epsilon, data_grad):
    # Collect the sign of the gradients
    sign_data_grad = data_grad.sign()
    # Create the perturbed image by adjusting each pixel of the input image
    perturbed_image = image + epsilon * sign_data_grad
    # Adding clipping to maintain [0,1] range
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    return perturbed_image

def generate_adversarial_example(model, img_path, epsilon):
    # Load and preprocess the image
    img = load_image(img_path)
    input_img = Variable(img, requires_grad=True)

    # Set the model in evaluation mode and disable gradients computation
    model.eval()
    output = model(input_img)
    _, init_pred = torch.max(output, 1)

    # Calculate the loss
    loss = F.nll_loss(F.log_softmax(output, dim=1), init_pred)

    # Zero all existing gradients
    model.zero_grad()

    # Backward pass to calculate gradients
    loss.backward()

    # Collect the gradients of the input image
    data_grad = input_img.grad.data

    # Call FGSM attack
    perturbed_image = fgsm_attack(input_img, epsilon, data_grad)

    return perturbed_image

def display_images(original_img_path, adversarial_img):
    # Load the original image
    original_img = cv2.imread(cv2.samples.findFile(original_img_path))
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    original_img = cv2.resize(original_img, (224, 224))

    # Convert the adversarial image from tensor to numpy array
    adversarial_img = adversarial_img.squeeze().detach().cpu().numpy()
    adversarial_img = np.transpose(adversarial_img, (1, 2, 0))
    adversarial_img = np.clip(adversarial_img, 0, 1)

    # Plot the images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(adversarial_img)
    ax[1].set_title('Adversarial Image')
    ax[1].axis('off')

    plt.show()

# Parameters
img_path = 'path/to/your/image.jpg'  # Update the image path
epsilon = 0.03  # Perturbation magnitude

# Generate the adversarial example
adversarial_example = generate_adversarial_example(model, img_path, epsilon)

# Display the original and adversarial images
display_images(img_path, adversarial_example)
