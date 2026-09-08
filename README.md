# Nail Condition Classification Using Deep Learning

An artificial intelligence-based nail condition classification system developed using deep learning and computer vision techniques. The project evaluates multiple convolutional neural network architectures for the classification of six nail-related conditions and provides explainable predictions using Grad-CAM.

---

## Project Overview

This project investigates the effectiveness of deep learning models for automated classification of nail conditions from images.

Four deep learning models were evaluated:

- EfficientNet-B0
- DenseNet121
- ConvNeXtV2-Tiny
- Proposed Attention-EfficientNet-B0

The final system uses EfficientNet-B0 because it achieved the strongest overall combination of classification performance and computational efficiency.

The application provides:

- Nail image classification
- Prediction confidence
- Class probability distribution
- Grad-CAM visual explanations
- Interactive web-based interface using Streamlit

---

# Supported Classes

The classification system supports six classes:

1. Acral Lentiginous Melanoma
2. Healthy Nail
3. Onychogryphosis
4. Blue Finger
5. Clubbing
6. Pitting

---

# Model Performance

The models were evaluated using a test dataset containing 576 images.

| Model | Accuracy | Macro F1-Score | Inference Time | Parameters |
|---|---:|---:|---:|---:|
| **EfficientNet-B0** | **98.44%** | **98.36%** | **2.14 ms/image** | **4.02 M** |
| ConvNeXtV2-Tiny | 98.26% | 98.23% | 7.88 ms/image | 27.87 M |
| Proposed Attention-EfficientNet-B0 | 98.09% | 98.06% | 3.51 ms/image | 4.22 M |
| DenseNet121 | 97.57% | 97.81% | 4.07 ms/image | 6.96 M |

EfficientNet-B0 achieved:

- Highest overall accuracy
- Highest Macro F1-score
- Fastest inference time
- Smallest model size among the evaluated architectures

---

# Statistical Comparison

A McNemar's statistical test was performed to compare EfficientNet-B0 with the proposed Attention-EfficientNet-B0 model.

| Metric | Result |
|---|---:|
| Test Images | 576 |
| EfficientNet-B0 Accuracy | 98.44% |
| Proposed Model Accuracy | 98.09% |
| Accuracy Difference | 0.35 percentage points |
| McNemar's Test P-Value | 0.6875 |
| Statistical Significance | Not Significant |

The statistical analysis indicates that the difference in performance between EfficientNet-B0 and the proposed model was not statistically significant at a significance level of 0.05.

---

# Calibration Analysis

Model calibration was evaluated using Expected Calibration Error (ECE) and Multiclass Brier Score.

| Model | Expected Calibration Error | Brier Score |
|---|---:|---:|
| EfficientNet-B0 | 0.065050 | 0.004695 |
| Proposed Attention-EfficientNet-B0 | **0.008949** | 0.004805 |

The proposed model demonstrated better Expected Calibration Error, indicating improved alignment between prediction confidence and observed accuracy.

---

# Unseen Source Evaluation

An additional evaluation was performed using 71 images from previously unseen image sources.

| Model | Accuracy |
|---|---:|
| DenseNet121 | **95.77%** |
| Proposed Attention-EfficientNet-B0 | **95.77%** |
| ConvNeXtV2-Tiny | 94.37% |
| EfficientNet-B0 | 92.96% |

This evaluation provides additional insight into model generalization across previously unseen image sources.

---

# Explainable AI Using Grad-CAM

The system integrates Gradient-weighted Class Activation Mapping (Grad-CAM) to visualize image regions that contributed to the model's prediction.

The Grad-CAM visualization helps analyze:

- Regions influencing model predictions
- Attention patterns
- Potential reliance on contextual image features
- High-confidence misclassifications

Grad-CAM analysis was also performed on selected high-confidence misclassification cases.

---

# Project Structure

```text
Nail-Condition-Classification/
│
├── data/
│   ├── grouped_split/
│   └── ...
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── models/
│
├── src/
│   ├── train_efficientnet_b0.py
│   ├── train_densenet121.py
│   ├── train_convnextv2.py
│   ├── train_proposed_model.py
│   │
│   ├── evaluate_efficientnet_b0.py
│   ├── evaluate_densenet121.py
│   ├── evaluate_convnextv2.py
│   ├── evaluate_proposed_model.py
│   │
│   ├── predict_efficientnet_b0.py
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md