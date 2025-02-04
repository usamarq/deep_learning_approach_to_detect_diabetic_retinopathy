
## Overview
This project is focused on developing a deep learning model for detecting diabetic retinopathy using various neural network architectures and techniques.

## File Descriptions

- **TrainModel.py**: Contains the implementation of the training and evaluation processes for the model, including metrics computation and model checkpointing.

- **MyModels.py**: Defines various neural network architectures based on ResNet and VGG models, including single and dual input configurations, as well as models with attention mechanisms.

- **Visualizations.py**: Provides functions for visualizing training and validation curves, confusion matrices, and images with Grad-CAM overlays to interpret model predictions.

- **RetinoPathyDatasetClass.py**: Implements custom dataset classes for loading and processing images and annotations for diabetic retinopathy, supporting both single and dual image modes.

- **pretrained_DR_resize/how_to_use.py**: Demonstrates how to load pretrained models and their state dictionaries for transfer learning purposes.

- **GradCAM.py**: Contains functions for generating and visualizing Grad-CAM heatmaps, which help in understanding the regions of the input images that the model focuses on for its predictions.

- **Augmentations.py**: Implements various image augmentation techniques to enhance the training dataset, including CutOut, CLAHE, and custom transformations.

- **AttentionMechanisms.py**: Defines attention mechanisms such as spatial, channel, and self-attention, which can be integrated into neural network architectures to improve performance.

- **Ensembles.py**: Implements ensemble methods for combining predictions from multiple models, including max voting and weighted average techniques, along with a function to save predictions to a CSV file for submission.

- **evaluationnotebook.ipynb**: A Jupyter notebook that provides an interactive environment for evaluating the trained models on validation and test datasets, visualizing results, and analyzing model performance through various metrics and plots.

- **!readme.txt**: A placeholder file that may contain additional project-related information or notes.
