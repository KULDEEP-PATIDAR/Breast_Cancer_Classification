# 🩺 Breast Cancer Classification using Artificial Neural Network (ANN)

This project implements a **binary classification system** to predict the type of breast cancer — **Malignant (M)** or **Benign (B)** — using an **Artificial Neural Network (ANN)**.  
The model is trained and evaluated on a real-world medical dataset and achieves **high predictive accuracy** with proper regularization and hyperparameter tuning.

---

## 📌 Problem Statement
Early detection of breast cancer plays a critical role in improving survival rates.  
The goal of this project is to build a reliable **classification model** that can accurately identify malignant tumors based on diagnostic features.

---

## 📊 Dataset
- **Name**: Breast Cancer Dataset  
- **Source**: Kaggle  
- **Link**: https://www.kaggle.com/datasets/yasserh/breast-cancer-dataset/data  
- **Type**: Binary Classification  
- **Target Variable**: `diagnosis`
  - `M` → Malignant (1)
  - `B` → Benign (0)
- **Number of Features**: 30 numerical features extracted from cell nuclei

---

## ⚙️ Technologies Used
- Python
- Google Colab
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Keras Tuner
- Matplotlib

---

## 🧠 Model Architecture
- **Model Type**: Artificial Neural Network (ANN)
- **Input Layer**: 30 features
- **Hidden Layers**:
  - Dense layers with ReLU / Tanh / Sigmoid activations
  - Dropout layers for regularization
- **Output Layer**:
  - Sigmoid activation for binary classification

---

## 🔧 Data Preprocessing
- Converted categorical target labels:
  - `M → 1`, `B → 0`
- Applied **StandardScaler** for feature scaling
- Split dataset into training and testing sets

---

## 🔍 Hyperparameter Tuning
- Used **Keras Tuner – Random Search**
- Tuned parameters:
  - Number of hidden layers
  - Number of neurons per layer
  - Activation functions
  - Dropout rates
  - Optimizer selection
- Applied **EarlyStopping** to prevent overfitting and restore best weights

---

## 📈 Model Performance
- **Test Accuracy**: **97.37%**
- Training and validation curves were analyzed to ensure:
  - Proper convergence
  - Minimal overfitting
  - Stable validation performance

---

## 🧪 Evaluation Techniques
- Accuracy score
- Training vs Validation Loss and Accuracy plots
- Performanc
