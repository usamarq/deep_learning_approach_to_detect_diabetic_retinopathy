import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np
import torch
from PIL import Image
import random 
from GradCAM import generate_gradcam, visualize_gradcam
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plotTrainingValCurves(train_losses, val_kappas, val_losses, train_accuracies, val_accuracies):
    # Plot training loss
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='magenta' )
    plt.title('Training & Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(train_accuracies, label='Training Accuracy', color='red')
    plt.plot(val_accuracies, label='Validation Accuracy', color='green' )
    plt.title('Training & Validatation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot validation kappa
    plt.subplot(1, 3, 3)
    plt.plot(val_kappas, label='Validation Kappa', color='orange')
    plt.title('Validation Kappa Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Kappa Score')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plotPatientEyeImgsWithLabels(df, rnd_patient, image_dir, transform):
    patient_images = df[df['patient_id'] == rnd_patient]   
    patient_dr_level = patient_images['patient_DR_Level'].iloc[0]

    plt.figure(figsize=(15, 6))
    plt.suptitle(f"Patient ID: {rnd_patient} | Patient_DR_level: {patient_dr_level}")

    for i, (_, row) in enumerate(patient_images.iterrows()):
        img_path = os.path.join(image_dir, row['img_path'])  # Construct image path
        img = Image.open(img_path).convert('RGB')  # Open image

        # Apply transformation
        img_transformed = transform(img)
        img_name = row['img_path'].split('/')[-1].split('.')[0]  # Extract image name

        # Determine label based on eye side
        if '_l' in img_name:  # Left eye
            label = row['left_eye_DR_Level']
            eye = 'Left Eye DR Level'
        elif '_r' in img_name:  # Right eye
            label = row['right_eye_DR_Level']
            eye = 'Right Eye DR Level'
        else:
            label = 'Unknown'
            eye = 'Unknown'

        # Plot the original image
        plt.subplot(2, len(patient_images), i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{eye}: {label}')

        # Plot the transformed image
        plt.subplot(2, len(patient_images), i + 1 + len(patient_images))
        plt.imshow(img_transformed.permute(1, 2, 0).clamp(0, 1))  # Transform image for plotting
        plt.axis('off')
        plt.title(f'Transformed Image')

    plt.tight_layout()
    plt.show()

def plotRandomPatientEyeImagesWithGRADCam(df, rnd_patient, image_dir, transform, model):
    
    patient_images = df[df['patient_id'] == rnd_patient]
    patient_dr_level = patient_images['patient_DR_Level'].iloc[0]

    # Set up the plot with two rows (original and Grad-CAM)
    plt.figure(figsize=(15, 10))
    plt.suptitle(f"Patient ID: {rnd_patient} | Actual Patient_DR_level: {patient_dr_level}")
    num_images = len(patient_images)

    for i in range(num_images):
        row = patient_images.iloc[i]  # Access the row using iloc
        # Load the original image
        img_path = os.path.join(image_dir, row['img_path'])
        img = Image.open(img_path).convert('RGB')
        img_name = row['img_path'].split('/')[-1].split('.')[0] 

        # Determine label based on eye side
        if '_l' in img_name:  # Left eye
            label = row['left_eye_DR_Level']
            eye = 'Actual Left Eye DR Level'
        elif '_r' in img_name:  # Right eye
            label = row['right_eye_DR_Level']
            eye = 'Actual Right Eye DR Level'
        else:
            label = 'Unknown'
            eye = 'Unknown'
        
        # Transform the image for model input
        img_transformed = transform(img).unsqueeze(0).to('cuda') 
        img_name = os.path.basename(row['img_path'])
        
        # Get model prediction
        model.eval()
        with torch.no_grad():
            output = model(img_transformed)
            predicted_class = output.argmax(dim=1).item()
        
        # Generate Grad-CAM
        cam = generate_gradcam(model, img_transformed, predicted_class)

        # Plot the original image (1st row)
        plt.subplot(2, num_images, i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'{eye}: {label}')

        # Plot the Grad-CAM heatmap (2nd row)
        plt.subplot(2, num_images, i + 1 + num_images)
        plt.imshow(visualize_gradcam(np.array(img), cam)) 
        plt.title(f'Predicted DR Level: {predicted_class}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(true_labels, predicted_labels, class_names):
    cm = confusion_matrix(true_labels, predicted_labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)

    plt.figure(figsize=(6, 5))
    disp.plot(cmap='viridis', xticks_rotation='vertical')
    plt.title('Confusion Matrix')
    plt.show()