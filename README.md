# Implementing a Convolutional Neural Network from Scratch for MNIST Classification

**Authors:** Ria Singh, Cooper Hawley, Daryon Roshanzaer, Anjali Dev  

[**Full Project Report (PDF)**](cnn_research_paper.pdf)

## Overview
This project implements a **LeNet-style Convolutional Neural Network (CNN) entirely from scratch** to classify handwritten digits from the MNIST dataset.  
By building each component manually — from convolution and pooling layers to the fully connected layers and backpropagation — we gained a deep understanding of CNN internals and how data augmentation impacts performance.

Unlike most modern implementations that rely on high-level frameworks like PyTorch or TensorFlow, our approach eliminates all automated layers, ensuring transparency and full control over the training pipeline.

---

## Project Highlights
- **Custom CNN Architecture:** Implemented convolution, pooling, activation functions, dense layers, and softmax manually.
- **Data Augmentation:** Pixel jitter, label corruption, symmetry computation, and intensity metrics.
- **Pure Python Implementation:** Fully replicated LeNet-5 architecture without high-level libraries for forward and backward passes.
- **Performance:** Achieved 98.17% accuracy with the pure-Python CNN and 98.36% with the PyTorch baseline.
- **Robustness Analysis:** Evaluated effects of noise and label corruption on accuracy, loss, and learned feature spaces.
- **Visualization:**  
  - Learned convolutional kernels  
  - Gradient attribution maps  
  - Accuracy/loss trends under various corruptions

---

## Results
| Model Variant         | Test Accuracy | Notes |
|-----------------------|--------------:|-------|
| PyTorch Baseline      | 98.36%        | Standard LeNet implementation |
| Pure-Python CNN       | 98.17%        | All layers manually implemented |
| 20% Pixel Jitter      | 97%+          | Maintained strong accuracy despite noise |
| 20% Label Corruption  | 97%+          | Higher loss due to confidence calibration issues |

---

## Methods
### Dataset & Preprocessing
- MNIST dataset (60,000 train / 10,000 test images, 28×28 grayscale).
- Normalization and custom feature engineering (symmetry, intensity).
- Data augmentation for robustness testing.

### Model Architecture
- Conv1 → ReLU → MaxPool  
- Conv2 → ReLU → MaxPool  
- Flatten → FC1 → ReLU → FC2 → ReLU → FC3 → Softmax  
- Implemented He initialization, SGD with momentum, and early stopping.

### Validation & Testing
- Compared custom layer outputs against PyTorch equivalents.
- Used reference JSON snapshots to debug off-by-one and gradient sign errors.
- Measured feature-space deviation under different corruption levels.
