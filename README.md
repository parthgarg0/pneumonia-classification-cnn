# Chest X-Ray Pneumonia Detection using CNN

A deep learning project for binary classification of chest X-ray images into **Normal** and **Pneumonia** using a Convolutional Neural Network (CNN) built with **TensorFlow** and **Keras**.

---

## 📌 Project Overview

Pneumonia is an infection that inflames the air sacs in one or both lungs, which may fill with fluid or pus. Early and accurate detection of pneumonia from chest radiographs is critical for timely medical intervention.

This project implements an end-to-end computer vision pipeline that:
1. Downloads the chest X-ray dataset automatically via `kagglehub`.
2. Preprocesses and resizes grayscale images to $224 \times 224$.
3. Applies on-the-fly data augmentation (random flips, rotations, and zooms) to prevent overfitting.
4. Trains a deep Convolutional Neural Network (CNN) with 3 convolutional blocks and a multi-layer dense classification head.
5. Evaluates the model on test data with standard classification metrics (Accuracy, Precision, Recall, F1-Score, Confusion Matrix).
6. Generates loss and accuracy progression curves across training epochs.
7. Saves the trained model artifact to `pneumonia_cnn_model.keras`.

---

## 📂 Dataset Information

The model uses the [Chest X-Ray Images (Normal and Pneumonia)](https://www.kaggle.com/datasets/ghost5612/chest-x-ray-images-normal-and-pneumonia) dataset.
- **Classes**:
  - `0`: NORMAL
  - `1`: PNEUMONIA
- The dataset is structured into standard partitions:
  - `train/`
  - `val/`
  - `test/`

The dataset is downloaded seamlessly at runtime via `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("ghost5612/chest-x-ray-images-normal-and-pneumonia")
```

---

## 🏗️ Model Architecture

```text
Input (224 x 224 x 1, Grayscale)
│
├── Rescaling (1.0 / 255.0)
│
├── Data Augmentation (RandomFlip, RandomRotation, RandomZoom)
│
├── [Conv Block 1] Conv2D(32 filters, 3x3, ReLU, Same) -> MaxPooling2D(2x2)
│
├── [Conv Block 2] Conv2D(64 filters, 3x3, ReLU, Same) -> MaxPooling2D(2x2)
│
├── [Conv Block 3] Conv2D(128 filters, 3x3, ReLU, Same) -> MaxPooling2D(2x2)
│
├── Flatten
│
├── Dense(64, ReLU)
├── Dense(32, ReLU)
├── Dense(16, ReLU)
│
└── Output Layer: Dense(1, Sigmoid) -> Binary Classification
```

### Hyperparameters
- **Input Dimensions**: $224 \times 224 \times 1$
- **Batch Size**: 32
- **Epochs**: 15
- **Optimizer**: Adam
- **Loss Function**: Binary Crossentropy
- **Evaluation Metric**: Accuracy

---

## 📁 Repository Structure

```text
├── pneumonia_classification.py  # Main training, evaluation, and visualization script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Ignored files and directories
├── README.md                    # Project documentation
├── confusion_matrix.png         # Generated confusion matrix visualization
├── training_graphs.png          # Generated training & validation accuracy/loss plots
└── pneumonia_cnn_model.keras    # Saved trained Keras model artifact
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Git

### 2. Clone the Repository
```bash
git clone https://github.com/parthgarg0/pneumonia-classification-cnn.git
cd pneumonia-classification-cnn
```

### 3. Create and Activate Virtual Environment
On Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

On Linux / macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Install Dependencies
```bash
pip install -r requirements.txt
```

### 5. Run Training and Evaluation
```bash
python pneumonia_classification.py
```

---

## 📊 Experimental Results & Evaluation

The model was evaluated on the independent test dataset of **624 chest X-ray images** (234 Normal, 390 Pneumonia).

### Performance Metrics Summary

| Metric | Score | Note |
|---|---|---|
| **Test Accuracy** | **81.41%** | Overall correct diagnoses on test set |
| **Recall (Sensitivity)** | **98.46%** | Crucial clinical metric: near-zero missed pneumonia cases |
| **Precision** | **77.73%** | Reliable positive predictions |
| **F1-Score** | **86.88%** | Harmonic mean of precision and recall |
| **Test Loss** | **0.5005** | Binary crossentropy loss |

### Detailed Classification Report

```text
              precision    recall  f1-score   support

      NORMAL       0.95      0.53      0.68       234
   PNEUMONIA       0.78      0.98      0.87       390

    accuracy                           0.81       624
   macro avg       0.87      0.76      0.78       624
weighted avg       0.84      0.81      0.80       624
```

### Visualizations

#### Training Progression (Accuracy & Loss)
![Training & Validation Curves](training_graphs.png)

#### Confusion Matrix
![Confusion Matrix](confusion_matrix.png)

---

## 💾 Model Export & Inference

Once training completes, the model is serialized to `pneumonia_cnn_model.keras`.

To load and perform inference on a new chest X-ray image:
```python
import tensorflow as tf
import numpy as np

# Load trained model
model = tf.keras.models.load_model("pneumonia_cnn_model.keras")

# Load and preprocess image
img = tf.keras.utils.load_img("path/to/xray.jpeg", color_mode="grayscale", target_size=(224, 224))
img_array = tf.keras.utils.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)[0][0]
label = "Pneumonia" if prediction >= 0.5 else "Normal"
confidence = prediction if prediction >= 0.5 else 1.0 - prediction

print(f"Prediction: {label} ({confidence * 100:.2f}% confidence)")
```

---

## 📜 License

This project is open-source and available under the [MIT License](LICENSE).
