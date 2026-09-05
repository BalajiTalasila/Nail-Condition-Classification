# 💅 Nail Condition Classification System

An AI-powered deep learning system for automated classification of nail conditions from images using multiple convolutional neural network architectures.

The project compares three deep learning models:

- ConvNeXtV2-Tiny
- DenseNet121
- EfficientNet-B0

After extensive training and evaluation, **EfficientNet-B0 achieved the best overall performance** and was selected as the proposed model. The system also includes **Grad-CAM explainability** and an interactive **Streamlit web application** for real-time predictions.

---

# 📌 Project Overview

Nail abnormalities can provide important visual indicators of various medical conditions. However, manual identification and classification can be challenging and may require professional expertise.

This project applies deep learning techniques to classify nail images into six different categories. Multiple state-of-the-art convolutional neural network architectures were trained and evaluated to identify the most effective model.

The final system uses **EfficientNet-B0** as the proposed model because it achieved the best combination of:

- Classification accuracy
- Macro F1-score
- Matthews Correlation Coefficient (MCC)
- Inference speed

The application also integrates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations for model predictions.

> ⚠️ **Disclaimer:** This project is developed for educational and research purposes only. It should not be used as a replacement for professional medical diagnosis.

---

# 🎯 Objectives

The primary objectives of this project are:

- Develop an automated nail condition classification system.
- Compare multiple deep learning architectures.
- Evaluate model performance using multiple metrics.
- Identify the best-performing model.
- Analyze class-wise performance.
- Implement Grad-CAM for explainable AI.
- Develop a user-friendly web application for image classification.

---

# 🩺 Supported Nail Conditions

The system classifies nail images into the following six categories:

| Class | Condition |
|---|---|
| 1 | Acral Lentiginous Melanoma |
| 2 | Healthy Nail |
| 3 | Onychogryphosis |
| 4 | Blue Finger |
| 5 | Clubbing |
| 6 | Pitting |

---

# 🧠 Models Used

Three deep learning architectures were trained and evaluated.

## 1. ConvNeXtV2-Tiny

ConvNeXtV2-Tiny is a modern convolutional neural network architecture designed for efficient and high-performance image classification.

## 2. DenseNet121

DenseNet121 uses dense connections between layers, enabling efficient feature reuse and improved gradient flow.

## 3. EfficientNet-B0

EfficientNet-B0 uses compound scaling to balance network depth, width, and image resolution efficiently.

Based on the experimental results, **EfficientNet-B0 was selected as the proposed model**.

---

# 🏗️ Project Workflow

The complete workflow of the project is shown below:

```text
Dataset Collection
        │
        ▼
Data Cleaning
        │
        ▼
Duplicate Detection
        │
        ▼
Dataset Preparation
        │
        ▼
Train / Validation / Test Split
        │
        ▼
Model Training
        │
 ┌──────┼───────────────┐
 ▼      ▼               ▼
ConvNeXt DenseNet121 EfficientNet-B0
 ▼      ▼               ▼
 └──────┼───────────────┘
        ▼
Model Evaluation
        │
        ▼
Performance Comparison
        │
        ▼
Best Model Selection
        │
        ▼
Grad-CAM Explainability
        │
        ▼
Streamlit Web Application