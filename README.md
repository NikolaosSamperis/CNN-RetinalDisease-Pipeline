[Notebook (Google Colab)](https://colab.research.google.com/drive/1-rAbH-aQBs9hCh4e7p0crPhQbWJvsZlw) • 
[EfficientNet (Paper)](https://arxiv.org/abs/1905.11946) • 
[DenseNet-121 (Paper)](https://arxiv.org/abs/1608.06993) • 
[CLAHE Overview](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization) • 
[TensorFlow](https://www.tensorflow.org/) • 
[Keras](https://keras.io/) • 
[OpenCV](https://opencv.org/) • 
[Albumentations](https://albumentations.ai/) • 
[scikit-learn](https://scikit-learn.org/)

# 🧠 CNN-RetinalDisease-Pipeline

**Automated Cataract and Glaucoma Detection via Fundus Image Classification using Deep Learning**  
*A comparative study of EfficientNetB0 and DenseNet-121 with transfer learning, data augmentation, and contrast enhancement.*

---

## 📌 Project Overview

This project explores the use of **deep convolutional neural networks (CNNs)** for classifying retinal fundus images into three categories: **Normal**, **Cataract**, and **Glaucoma**. It investigates two popular architectures—EfficientNetB0 and DenseNet-121—through a series of experiments involving transfer learning, preprocessing (CLAHE), dataset balancing, and custom classification heads.

---

## 📊 Key Features

- ✅ Multi-class classification of retinal diseases
- 🔁 Transfer learning with pre-trained CNNs
- 🧪 Fine-tuning strategies and loss function experiments
- ⚖️ Dataset balancing and class weighting
- 🎨 Contrast enhancement using CLAHE
- 📈 Evaluation metrics: Accuracy, Precision, Recall, F1-score, AUC-ROC

---

## 🧪 Model Comparison

| Model          | Best Accuracy | Glaucoma F1-score | AUC-ROC (avg) |
|----------------|---------------|--------------------|----------------|
| **EfficientNetB0** | 83%          | 71%               | ~0.92          |
| **DenseNet-121**   | 81%          | 67%               | ~0.91          |

---

## 📁 Repository Structure

```
.
├── notebooks/
│   └── Pipeline.ipynb              # Core Jupyter Notebook
├── requirements.txt               # Python dependencies
├── README.md                      # Project overview
└── (Optional) data/               # Raw and processed images (if uploaded)
```

---

## 📦 Installation

Set up a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/NikolaosSamperis/CNN-RetinalDisease-Pipeline.git
   cd CNN-RetinalDisease-Pipeline
   ```

2. Open the notebook:
   - Use the [Colab link](https://colab.research.google.com/drive/1-rAbH-aQBs9hCh4e7p0crPhQbWJvsZlw)
   - Or run it locally with Jupyter

3. Follow the workflow in the notebook to preprocess images, train models, and evaluate results.

---

## 📊 Results Summary

- **Experiment 8 (EfficientNetB0)** achieved the highest accuracy (83%) and best glaucoma classification performance.
- CLAHE preprocessing enhanced visual clarity, especially for low-contrast features.
- Dataset balancing and class weighting significantly improved model robustness.
- Focal loss required careful tuning to avoid instability.

---

## ⚠️ Ethical Considerations

- Emphasize human-in-the-loop validation for AI-assisted diagnosis.
- Avoid potential bias due to small or non-diverse datasets.
- Ensure compliance with GDPR and HIPAA when handling medical data.
- Prioritize explainability and interpretability for clinical use.

---

## 📄 License

This project is released under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).

---

## 🙌 Acknowledgments

Thanks to the developers and contributors of:
- [EfficientNet (Google AI)](https://github.com/google/automl)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [CLAHE](https://doi.org/10.1023/B:JVLC.0000021715.57353.38)





