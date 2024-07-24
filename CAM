import torch
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

# Load an image and preprocess
img_path = 'path/to/your/image.jpg'  # Update the image path
img = cv2.imread(cv2.samples.findFile(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))  # Ensure the image is resized correctly

# Convert from OpenCV to PIL image
img = Image.fromarray(img)

# Apply the preprocessing transformations
img = preprocess(img)
img = img.unsqueeze(0)

# Convert image to Variable
input_img = Variable(img, requires_grad=True)

class HeatmapsCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.Heatmaps_information = None
        self.activations = None
        self.hook_layers()

    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        target_layer = dict(self.model.named_modules())[self.target_layer]
        target_layer.register_forward_hook(forward_hook)
        target_layer.register_backward_hook(backward_hook)

    def compute_Heatmaps_information(self):
        self.Heatmaps_information = self.gradients.pow(2)

    def generate_cam(self, input_img, class_idx=None):
        output = self.model(input_img)

        if class_idx is None:
            class_idx = torch.argmax(output)

        # Compute log-probabilities
        log_probs = F.log_softmax(output, dim=1)
        target_log_prob = log_probs[0, class_idx]

        self.model.zero_grad()
        target_log_prob.backward(retain_graph=True)

        self.compute_Heatmaps_information()

        activations = self.activations.detach().cpu().numpy()[0]
        Heatmaps_info = self.Heatmaps_information.detach().cpu().numpy()[0]

        weights = np.mean(Heatmaps_info, axis=(1, 2))  # Average Heatmaps Information over spatial dimensions

        cam = np.zeros(activations.shape[1:], dtype=np.float32)

        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        cam -= np.min(cam)
        cam /= np.max(cam)
        return cam

# Initialize HeatmapsCAM
Heatmaps_cam = HeatmapsCAM(model, 'layer4.2')

# Generate CAM
cam = Heatmaps_cam.generate_cam(input_img)

# Visualize the result
img = cv2.imread(cv2.samples.findFile(img_path))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (224, 224))  # Ensure the image is resized correctly
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = np.float32(heatmap) / 255
cam_img = heatmap + np.float32(img) / 255
cam_img = cam_img / np.max(cam_img)

plt.imshow(cam_img)
plt.show()
