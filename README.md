# AI_vs_Real_images
A high-performance binary classifier developed in PyTorch to distinguish between real photographs and AI-generated synthetic images.
# 🛡️ CIFAKE: AI-Generated Image Detection using ResNet-18

## 📌 Project Overview
This project implements a **ResNet-18** deep learning architecture fine-tuned on the **CIFAKE dataset** (100,000 images) to classify images as "REAL" or "FAKE" (AI-generated).

By leveraging **Transfer Learning** and **Frequency Domain Analysis**, the model achieves a validation accuracy of **95.64%**.

---

## 🔬 Technical Approach

### 1. Data Analysis (Spatial vs. Frequency Domain)
Before training, I implemented a **Fast Fourier Transform (FFT)** analysis to visualize the "spectral fingerprints" of AI images.
* **Spatial Domain:** Standard RGB images resized to $224 \times 224$.
* **Frequency Domain:** Log-scaled magnitude spectrums revealed that AI-generated images often contain high-frequency artifacts (grid patterns) not present in natural photography.



### 2. Model Architecture
* **Backbone:** Pre-trained ResNet-18 (`IMAGENET1K_V1`).
* **Head:** Replaced the 1000-class fully connected layer with a **Binary Classifier** (2 output features).
* **Freezing Strategy:** * Layers 1 and 2 were frozen to preserve general feature extraction (edges, textures).
  * **Layers 3, 4, and the FC head** were unfrozen for fine-tuning to detect specific synthetic artifacts.

### 3. Training Dynamics
* **Optimizer:** Adam with filtered parameters (only updating unfrozen weights).
* **Loss Function:** CrossEntropyLoss.
* **Hardware:** Accelerated using NVIDIA Tesla T4 GPU (Google Colab).

---

## 📊 Results

| Metric | Value |
| :--- | :--- |
| **Training Samples** | 80,000 |
| **Testing Samples** | 20,000 |
| **Epochs** | 6 (on the full 100k set) |
| **Final Accuracy** | **95.64%** |

