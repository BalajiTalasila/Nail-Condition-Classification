# 💅 Nail Condition Classification System

An AI-powered deep learning system for the classification of nail conditions from images using convolutional neural networks and transfer learning.

The proposed system uses **EfficientNet-B0** as the final model and incorporates **Grad-CAM (Gradient-weighted Class Activation Mapping)** to provide visual explanations for model predictions.

---

## 📌 Project Overview

Nail-related conditions can exhibit visual characteristics that may be identified from images. This project investigates the use of deep learning models for automatic classification of six nail-related conditions.

Three deep learning architectures were trained and evaluated:

- EfficientNet-B0
- ConvNeXtV2-Tiny
- DenseNet121

The models were compared using multiple evaluation metrics, including:

- Accuracy
- Precision
- Recall
- Macro F1-Score
- Weighted F1-Score
- Matthews Correlation Coefficient (MCC)
- Inference Time

Based on the experimental results, **EfficientNet-B0 achieved the best overall performance**.

---

# 🎯 Supported Classes

The system classifies images into the following six categories:

1. Acral Lentiginous Melanoma
2. Healthy Nail
3. Onychogryphosis
4. Blue Finger
5. Clubbing
6. Pitting

---

# 🗂️ Dataset

The dataset used in this project was obtained from Kaggle:

**Nail Disease Detection Dataset**

Dataset source: Nikhil Gurav on Kaggle

> Note: The dataset is not included in this repository because of its size. Please download the dataset separately from Kaggle and organize it according to the project directory structure.

---

# 🧠 Models Evaluated

Three transfer learning models were evaluated in this project.

| Model | Accuracy | Macro F1-Score | MCC | Average Inference Time |
|---|---:|---:|---:|---:|
| ConvNeXtV2-Tiny | 98.26% | 98.23% | 97.90% | 7.88 ms |
| DenseNet121 | 97.57% | 97.81% | 97.06% | 4.07 ms |
| **EfficientNet-B0** | **98.44%** | **98.36%** | **98.11%** | **2.14 ms** |

---

# 🏆 Best Model

## EfficientNet-B0

EfficientNet-B0 was selected as the proposed model because it achieved the best overall experimental performance.

### Performance

- **Test Accuracy:** 98.44%
- **Macro F1-Score:** 98.36%
- **Matthews Correlation Coefficient:** 98.11%
- **Average Inference Time:** 2.14 ms per image

Among the evaluated models, EfficientNet-B0 achieved:

- The highest accuracy
- The highest Macro F1-Score
- The highest MCC
- The fastest inference time

Therefore, EfficientNet-B0 provided the best combination of classification performance and computational efficiency.

---

# 📊 Per-Class Performance

The best-performing model varied slightly across individual classes.

| Class | Best Model | F1-Score |
|---|---|---:|
| Acral Lentiginous Melanoma | EfficientNet-B0 | 99.56% |
| Healthy Nail | DenseNet121 | 100.00% |
| Onychogryphosis | EfficientNet-B0 | 99.03% |
| Blue Finger | DenseNet121 | 98.40% |
| Clubbing | ConvNeXtV2-Tiny | 98.73% |
| Pitting | ConvNeXtV2-Tiny | 97.96% |

Although different models achieved the highest F1-score for certain individual classes, EfficientNet-B0 demonstrated the best overall performance across the complete test dataset.

---

# 🔬 Methodology

The project follows the following workflow:

```text
Dataset Collection
        │
        ▼
Data Preprocessing
        │
        ▼
Dataset Splitting
        │
        ▼
Model Training
        │
        ├── ConvNeXtV2-Tiny
        │
        ├── DenseNet121
        │
        └── EfficientNet-B0
                │
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
        Streamlit Deployment