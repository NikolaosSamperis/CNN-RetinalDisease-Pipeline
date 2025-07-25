[![Colab](https://img.shields.io/badge/Notebook-Google%20Colab-orange?logo=googlecolab&logoColor=white)](https://colab.research.google.com/)
[![EfficientNet](https://img.shields.io/badge/EfficientNet-Paper-blue)](https://arxiv.org/abs/1905.11946)
[![DenseNet](https://img.shields.io/badge/DenseNet121-Paper-blue)](https://arxiv.org/abs/1608.06993)
[![CLAHE](https://img.shields.io/badge/CLAHE-Contrast_Enhancement-lightgrey)](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep_Learning-red?logo=keras)](https://keras.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-ML-blue?logo=scikit-learn)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# 🧠 CNN Retinal Disease Pipeline

**Automated Cataract and Glaucoma Detection via Fundus Image Classification using Deep Learning**  
*A comparative study of EfficientNetB0 and DenseNet-121 with transfer learning, data augmentation, and contrast enhancement.*

---

## 📌 Project Overview

This project explores the use of **deep convolutional neural networks (CNNs)** for classifying retinal fundus images into three categories: **Normal**, **Cataract**, and **Glaucoma**. It investigates two popular architectures—EfficientNetB0 and DenseNet-121—through a series of experiments involving transfer learning, preprocessing (CLAHE), dataset balancing, and a custom classification head.

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

| Model (Experiment)      | Accuracy | Normal AUC | Cataract AUC | Glaucoma AUC | Normal F1 | Cataract F1 | Glaucoma F1 |
|-------------------------|----------|------------|--------------|--------------|-----------|-------------|-------------|
| EfficientNetB0 (Exp. 8) | **83%**  | 0.90       | 0.94         | **0.91**     | 0.88      | 0.82        | **0.71**    |
| DenseNet-121 (Exp. 3)   | 81%      | 0.87       | **0.98**     | 0.88         | 0.85      | 0.85        | 0.67        |

---

## 🖼️ Image Preprocessing Pipeline

The figure below illustrates the image preprocessing pipeline applied to retinal fundus images across all three classes (Normal, Cataract, Glaucoma). The steps include:

- Raw image capture  
- CLAHE contrast enhancement  
- Resizing to 224×224 pixels  
- Data augmentation (e.g., rotation, flipping)

![Image Preprocessing Pipeline](./assets/image_preprocessing_pipeline.png)

---

## 📁 Repository Structure

```
.
├── assets/
│   └── image_preprocessing_pipeline.png  # Visual summary of preprocessing pipeline
├── notebooks/
│   └── Pipeline.ipynb                    # Core Jupyter Notebook
├── AI_final_report.pdf                   # Final report summarizing project findings
├── LICENSE.txt                           # MIT License file
├── README.md                             # Project documentation
├── requirements.txt                      # Python dependencies
└── (external) Google Drive link          # Dataset: raw, processed, augmented images

```

## 📁 Dataset Access

The raw and processed fundus image datasets used in this project are available on Google Drive:

🔗 [Access the dataset here](https://drive.google.com/drive/folders/18HJRbyhWfaJok8r070PRf5G1ZxfyCb4W?usp=sharing)

This includes:
- Raw images
- CLAHE-enhanced images
- Augmented training sets
- Train/validation/test splits

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

- **Experiment 8 (EfficientNetB0)** achieved the highest classification performance, with AUC scores of 0.90 for Normal, 0.94 for Cataract, and 0.91 for Glaucoma, indicating strong generalization across all classes and particularly robust glaucoma detection.
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

This project was created for educational purposes as part of coursework.

Feel free to use, modify, and share the code **with proper attribution**. Please note that this project is provided as-is, without any warranty or guarantee of functionality.

If you use this code, kindly credit the original author by linking back to this repository.

---

## 🙌 Acknowledgments

Thanks to the developers and contributors of:
- [EfficientNet (Google AI)](https://github.com/google/automl)
- [DenseNet](https://arxiv.org/abs/1608.06993)
- [CLAHE](https://doi.org/10.1023/B:JVLC.0000021715.57353.38)





