## Final Model Comparison

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | MCC | Inference Time | Parameters | Model Size |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DenseNet121 | 97.57% | 97.79% | 97.85% | 97.81% | 97.56% | 0.9706 | 4.07 ms/image | 6.96 M | 26.87 MB |
| ConvNeXtV2-Tiny | 98.26% | 98.45% | 98.04% | 98.23% | 98.26% | 0.9790 | 7.88 ms/image | 27.87 M | 106.32 MB |
| EfficientNet-B0 | 98.44% | 98.54% | 98.22% | 98.36% | 98.44% | 0.9811 | 2.14 ms/image | 4.02 M | 15.48 MB |
| **Proposed Attention-EfficientNet-B0** | **98.91%** | **99.03%** | **98.78%** | **98.90%** | **98.91%** | **0.9868** | **3.51 ms/image** | **4.22 M** | **16.28 MB** |

### Key Result

The proposed Attention-EfficientNet-B0 achieves the highest performance across the reported classification metrics, reaching **98.91% accuracy**, **98.90% Macro F1**, and **0.9868 MCC**, while maintaining a relatively small model size of **16.28 MB**.
