import random
import numpy as np
import torch
import cv2
from torchvision import transforms
from PIL import Image, ImageFilter

class CutOut(object):
    def __init__(self, mask_size, p=0.5):
        self.mask_size = mask_size
        self.p = p

    def __call__(self, img):
        if np.random.rand() > self.p:
            return img

        # Ensure the image is a tensor
        if not isinstance(img, torch.Tensor):
            raise TypeError('Input image must be a torch.Tensor')

        # Get height and width of the image
        h, w = img.shape[1], img.shape[2]
        mask_size_half = self.mask_size // 2
        offset = 1 if self.mask_size % 2 == 0 else 0

        cx = np.random.randint(mask_size_half, w + offset - mask_size_half)
        cy = np.random.randint(mask_size_half, h + offset - mask_size_half)

        xmin, xmax = cx - mask_size_half, cx + mask_size_half + offset
        ymin, ymax = cy - mask_size_half, cy + mask_size_half + offset
        xmin, xmax = max(0, xmin), min(w, xmax)
        ymin, ymax = max(0, ymin), min(h, ymax)

        img[:, ymin:ymax, xmin:xmax] = 0
        return img


class SLORandomPad:
    def __init__(self, size):
        self.size = size

    def __call__(self, img):
        pad_width = max(0, self.size[0] - img.width)
        pad_height = max(0, self.size[1] - img.height)
        pad_left = random.randint(0, pad_width)
        pad_top = random.randint(0, pad_height)
        pad_right = pad_width - pad_left
        pad_bottom = pad_height - pad_top
        return transforms.functional.pad(img, (pad_left, pad_top, pad_right, pad_bottom))


class FundRandomRotate:
    def __init__(self, prob, degree):
        self.prob = prob
        self.degree = degree

    def __call__(self, img):
        if random.random() < self.prob:
            angle = random.uniform(-self.degree, self.degree)
            return transforms.functional.rotate(img, angle)
        return img
    

# Defining Augmentation Functions here for later use in Part (d)
# --------------------------------------------------------------
# Ben Graham's 
# CLAHE
# Sharpening
# Gaussian Blur

# Custom transformation for CLAHE
class CLAHETransform(object):
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size

    def __call__(self, img):
        img = np.array(img)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        img = clahe.apply(img)
        img = Image.fromarray(img)
        return img

# Circular Crop
class CircleCropTransform(object):
    def __call__(self, img):
        img = np.array(img)
        height, width, _ = img.shape
        center = (width // 2, height // 2)
        radius = min(width, height) // 2
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        img[mask == 0] = 0
        img = Image.fromarray(img)
        return img
    
# Ben Graham's 
class BenGrahamTransform(object):
    def __call__(self, img):
        img = np.array(img)  # Convert PIL Image to NumPy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert from RGB to HSV

        v_channel = img[:, :, 2]
        v_channel = cv2.normalize(v_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img[:, :, 2] = v_channel

        # Convert back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        # Convert back to PIL Image
        img = Image.fromarray(img)
        return img

# This version of Ben Graham's tranformation uses CLAHE to improve contrast and sharpness. This essentially
# increases the visibility of blood_vessels, spots, and other features in the Fundus Image. 
class EnhancedBenGrahamTransform:
    def __call__(self, img):
        img = np.array(img)  # Convert PIL Image to NumPy array
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) 

        # Normalize the V channel
        v_channel = img[:, :, 2]
        v_channel = cv2.normalize(v_channel, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        img[:, :, 2] = v_channel

        # Apply CLAHE to the V channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img[:, :, 2] = clahe.apply(img[:, :, 2])

        # Convert back to RGB
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)

        # Sharpen the image
        img = Image.fromarray(img)
        img = img.filter(ImageFilter.SHARPEN)

        return img
