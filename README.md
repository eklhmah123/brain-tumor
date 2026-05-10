# 🧠 Brain Tumor Detection using Vision Transformer (ViT)

This project implements a **Vision Transformer (ViT)** model for the classification of brain tumors from MRI images. The model is trained to identify four different classes: glioma, meningioma, pituitary tumor, and no tumor.

## 📋 Overview

Brain tumor detection is a critical medical imaging task where early and accurate diagnosis can significantly impact patient outcomes. This project leverages state-of-the-art **Vision Transformer (ViT)** architecture to automatically classify brain MRI scans into four categories:

- **Glioma** - A type of tumor that occurs in the brain and spinal cord
- **Meningioma** - A tumor that arises from the meninges, the membranes surrounding the brain and spinal cord
- **Pituitary tumor** - Tumors that develop in the pituitary gland
- **No tumor** - Healthy brain scans

## 🏗️ Model Architecture

The project uses a Vision Transformer (ViT) implementation with the following configuration:

| Parameter | Value |
|-----------|-------|
| Image Size | 224 × 224 pixels |
| Patch Size | 32 × 32 pixels |
| Embedding Dimension | 256 |
| Number of Transformer Layers | 4 |
| Number of Attention Heads | 8 |
| MLP Hidden Dimension | 512 |
| Dropout Rate | 0.1 |
| Output Classes | 4 |

## 📊 Dataset

The model is trained on a brain tumor MRI dataset with the following distribution:

| Class | Training Samples | Testing Samples |
|-------|-----------------|-----------------|
| Glioma | ~1,400 | ~400 |
| Meningioma | ~1,400 | ~400 |
| Pituitary tumor | ~1,400 | ~400 |
| No tumor | ~1,400 | ~400 |
| **Total** | **5,600** | **1,600** |


# Brain Tumor Detection using Deep Learning

## Overview

This project implements a deep learning pipeline for detecting brain tumors from MRI images using transfer learning with the VGG16 architecture. The system classifies MRI scans into four categories: Glioma Tumor, Meningioma Tumor, Pituitary Tumor, and No Tumor.

---

## Dataset

- **Total Images**: 3,264 MRI scans
- **Training Set**: 2,872 images
- **Testing Set**: 1,311 images
- **Classes**: 4 (Glioma, Meningioma, Pituitary, No Tumor)
- **Source**: Kaggle - Brain MRI Images for Brain Tumor Detection

---

## Technical Note on the Commit Issue

After committing changes, the text in the README became corrupted. This is a common issue when working with Jupyter Notebooks (`.ipynb` files) and version control.

**Why it happened:**
- Jupyter notebooks are stored as complex JSON files
- Git can misinterpret large diffs, especially when notebook metadata or outputs change significantly
- Merge conflicts or incorrect commit commands can corrupt plain-text rendering
- README content within notebook cells can get scrambled during the commit process

**Resolution:**
This dedicated Markdown file contains the final, corrected content in a clean format, separate from the `.ipynb` file to prevent future corruption.

---

## Notebook Metadata

- **File**: `brain_tumour_detection_using_deep_learning.ipynb`
- **Framework**: TensorFlow/Keras with VGG16 transfer learning
- **Purpose**: Multi-class classification of brain MRI images for tumor detection

---

## Project Pipeline Steps

### Step 1: Mount Google Drive

The notebook mounts Google Drive to access the dataset stored in cloud storage, enabling persistent storage of MRI images without local downloads.

### Step 2: Import Required Libraries

Key libraries imported include:
- `os` - File system operations and directory traversal
- `numpy` - Numerical operations and image array manipulation
- `random` - Generating random values for data augmentation
- `PIL` (Pillow) - Image processing and enhancement
- `tensorflow.keras` - Model building and training
- `sklearn.utils` - Data shuffling

### Step 3: Load Datasets

The data loading process involves:
- Defining paths to Training and Testing directories
- Iterating through subdirectories (tumor types)
- Creating lists of image paths and corresponding labels
- Shuffling the data to prevent order bias during training

### Step 4: Data Visualization

Random sample images from the training set are displayed in a 2×5 grid with their class labels. This confirms proper data loading and provides visual inspection of the MRI scans.

### Step 5: Image Preprocessing and Augmentation

**Augmentation techniques applied:**
- Random brightness adjustments (80-120% of original)
- Random contrast adjustments (80-120% of original)

**Preprocessing steps:**
- Resizing all images to 128×128 pixels
- Normalizing pixel values to the range [0, 1]

### Step 6: Label Encoding

String class labels are converted to integers (0-3) by creating a mapping between class names and numerical labels.

### Step 7: Data Generator

A batch generator is implemented for efficient training that:
- Processes images in batches of 20
- Applies augmentation on-the-fly during training
- Yields batches of images and encoded labels

### Step 8: Model Architecture

**Base Model**: VGG16 pre-trained on ImageNet
- Input shape: 128×128×3
- All base layers frozen initially
- Top 3 convolutional layers unfrozen for fine-tuning

**Added Layers:**
- Flatten layer to convert CNN output to 1D
- Dropout (0.3) for regularization
- Dense layer (128 units, ReLU activation)
- Dropout (0.2) for additional regularization
- Output layer (4 units, Softmax activation)

### Step 9: Model Compilation

- **Optimizer**: Adam with learning rate 0.0001
- **Loss Function**: Sparse categorical crossentropy
- **Metrics**: Sparse categorical accuracy

### Step 10: Training

- **Batch Size**: 20
- **Epochs**: 5
- **Training Time**: Approximately 1.5 hours on GPU

**Training Progress:**
- Epoch 1: Loss decreased to 0.6350, Accuracy reached 73.6%
- Epoch 2: Loss dropped to 0.2420, Accuracy improved to 90.3%
- Epoch 3: Loss reached 0.1813, Accuracy increased to 93.2%
- Epoch 4: Loss reduced to 0.1137, Accuracy achieved 95.7%
- Epoch 5: Final loss at 0.0820, Final accuracy at 97.2%

### Step 11: Model Evaluation

**Classification Report:**

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glioma Tumor | 0.97 | 0.98 | 0.98 |
| Meningioma Tumor | 0.93 | 0.90 | 0.91 |
| Pituitary Tumor | 0.95 | 1.00 | 0.97 |
| No Tumor | 0.93 | 0.91 | 0.92 |

**Overall Accuracy**: 95%

**Confusion Matrix Results:**
- Glioma: 294 correct, 6 misclassified
- Meningioma: 269 correct, 31 misclassified
- Pituitary: 404 correct, 1 misclassified
- No Tumor: 277 correct, 29 misclassified

### Step 12: Results Visualization

Training history plots showing accuracy and loss curves over 5 epochs are generated to visualize model learning progress.

---

## Key Findings

1. The model achieved 97.2% training accuracy and 95% test accuracy
2. Pituitary tumors showed perfect recall (100%)
3. Some confusion exists between Meningioma and No Tumor classes
4. Transfer learning with VGG16 proved effective for medical image classification

---

## How to Run

1. **Clone the repository**
2. **Install dependencies**: TensorFlow, NumPy, Matplotlib, scikit-learn, Pillow
3. **Download the dataset** from Kaggle
4. **Place Training and Testing folders** in the project directory
5. **Run the Jupyter Notebook** cell by cell

---

## Future Improvements

- Hyperparameter tuning (learning rate, batch size, dropout rates)
- Experiment with other architectures (ResNet50, InceptionV3)
- Train on larger, more diverse datasets
- Deploy model as TensorFlow Lite for mobile devices

---

# 🧠 Brain Tumor Detection using Transfer Learning

This project implements multiple deep learning models for detecting brain tumors from MRI images. It leverages pre‑trained architectures with **transfer learning** and **fine‑tuning** techniques.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Models Used](#models-used)
- [Project Structure](#project-structure)
- [Installation & Requirements](#installation--requirements)
- [Usage](#usage)
- [Results](#results)
- [Model Saving & Loading](#model-saving--loading)
- [Prediction on New Images](#prediction-on-new-images)
- [Future Work](#future-work)
- [License](#license)

---

## 📖 Overview

Brain tumor detection is a critical task in medical imaging. This project uses **Convolutional Neural Networks (CNNs)** with transfer learning to classify MRI images as either **having a tumor** or **not having a tumor**.

The notebook:
- Loads and preprocesses MRI images
- Applies data augmentation (random flips, rotations)
- Builds four models using popular pre‑trained architectures
- Compares their performance
- Saves the trained models for later use

---

## 📊 Dataset

The dataset used is a **brain tumor MRI dataset** containing two classes:

| Class | Description |
|-------|-------------|
| `yes` | Images with brain tumor |
| `no`  | Images without brain tumor |

**Dataset statistics:**
- Total tumor images: **155**
- Total non‑tumor images: **98**

> **Note:** The dataset is expected to be in a Google Drive folder. Update the `path` variable in the notebook to point to your dataset location.

---

## 🧠 Models Used

Four pre‑trained models are fine‑tuned on the brain tumor dataset:

| Model | Input Size | Preprocessing Function |
|-------|------------|------------------------|
| VGG16 | 224×224×3  | `vgg16.preprocess_input` |
| MobileNetV3 (Small) | 224×224×3 | `mobilenet_v3.preprocess_input` |
| ResNet50 | 224×224×3 | `resnet50.preprocess_input` |
| InceptionV3 | 299×299×3 | `inception_v3.preprocess_input` |

### Architecture Customization

For each base model:
 The top classification layer is removed (`include_top=False`)
- Early layers are **frozen** (not trainable)
 A custom head is added:
  - GlobalAveragePooling2D or `Flatten
  - `Dense(64, activation='relu')
  - `Dropout(0.5)
  - `Dense(8, activation='relu')
  - `Dense(1, activation='sigmoid') (binary classification)

--
---

## ⚙️ Installation & Requirements

### Libraries Required

``bash
pip install numpy pandas tensorflow opencv-python matplotlib imutils scikit-learn pillow


🔧 Future Work
Increase dataset size for better generalization

Implement cross‑validation

Add ensemble methods (voting / stacking)

Deploy the best model as a web app using Flask or Streamlit

Experiment with other architectures (EfficientNet, DenseNet)

# Vision Transformer for Brain Tumor Classification

A comprehensive implementation of Vision Transformer (ViT) for brain tumor classification from MRI images. This notebook classifies brain tumors into four categories: glioma, meningioma, no tumor, and pituitary tumor.

---

## Overview

This project implements a Vision Transformer architecture for medical image classification to detect and classify brain tumors from MRI scans. The model achieves high accuracy in distinguishing between different types of brain tumors.

---

## Dataset Classes

| Class | Description |
|-------|-------------|
| **Glioma** | A type of tumor that occurs in the brain and spinal cord |
| **Meningioma** | A tumor that arises from the meninges, the membranes surrounding the brain |
| **No Tumor** | Healthy brain MRI scans |
| **Pituitary** | Tumor affecting the pituitary gland |

---

## Dataset Information

- **Total training images:** 5,600
- **Total testing images:** 1,600
- **Image size:** 224×224 pixels
- **Dataset source:** Brain Tumor MRI Dataset

---

## Model Architecture

### Vision Transformer (ViT) Configuration

| Parameter | Value |
|-----------|-------|
| Image Size | 224×224 |
| Patch Size | 32×32 |
| Dimension | 256 |
| Depth | 4 |
| Heads | 8 |
| MLP Dimension | 512 |
| Dropout | 0.1 |
| Embedding Dropout | 0.1 |

---

## Data Preprocessing

### Training Transforms

| Transformation | Parameters |
|----------------|------------|
| Resize | 224×224 |
| Random Horizontal Flip | Probability = 0.5 |
| Random Rotation | ±10 degrees |
| To Tensor | Convert to PyTorch tensor |
| Normalization | Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225] |

### Testing Transforms

| Transformation | Parameters |
|----------------|------------|
| Resize | 224×224 |
| To Tensor | Convert to PyTorch tensor |
| Normalization | Mean: [0.485, 0.456, 0.406], Std: [0.229, 0.224, 0.225] |

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Number of Workers | 2 |
| Device | CUDA (GPU) |

---

## Requirements

bash
pip install kagglehub vit-pytorch prettytable arrow seaborn

Future Improvements
Implement cross-validation

Experiment with different patch sizes

Add more aggressive augmentation strategies

Implement learning rate scheduling

Use ensemble methods

Deploy as web application for clinical use
