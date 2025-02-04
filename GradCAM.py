import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def generate_gradcam(model, input_tensor, class_index):
    model.eval()
    
    # Forward pass
    output = model(input_tensor)

    # Zero gradients
    model.zero_grad()

    # Backward pass for the specific class
    output[0, class_index].backward()

    # Capture gradients and activations
    gradients = model.gradients.data.cpu().numpy()[0]  # [C, H, W]
    activations = model.activations.data.cpu().numpy()[0]  # [C, H, W]

    # Compute the weights (global average pooling)
    weights = np.mean(gradients, axis=(1, 2))  # Average gradients over spatial dimensions

    # Create Grad-CAM by combining activations and weights
    cam = np.zeros(activations.shape[1:], dtype=np.float32)  # Initialize a blank heatmap
    for i, w in enumerate(weights):
        cam += w * activations[i, :, :]

    # Normalize the heatmap
    cam = np.maximum(cam, 0)  # ReLU
    cam = cam / np.max(cam)  # Normalize to [0, 1]

    return cam

def visualize_gradcam(original_image, cam):
    # Resize the heatmap to the original image size
    cam = cv2.resize(cam, (original_image.shape[1], original_image.shape[0]))

    # Normalize original image (if it’s in tensor form)
    if isinstance(original_image, torch.Tensor):
        original_image = original_image.permute(1, 2, 0).numpy()  # Convert CHW -> HWC
        original_image = (original_image - original_image.min()) / (original_image.max() - original_image.min())

    # Convert to color
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_VIRIDIS)
    heatmap = np.float32(heatmap) / 255

    original_image = np.float32(original_image) / 255

    # Superimpose the heatmap on the original image
    superimposed_img = heatmap + np.float32(original_image)
    superimposed_img = superimposed_img / np.max(superimposed_img)  # Normalize

    return superimposed_img


