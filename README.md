# Multimodal White Blood Cell Classification and Hematological Prediction System

## 1. Project Description

### What is the problem?
In hematology, analyzing white blood cells (WBCs) is essential for understanding a patient’s immune response. Traditionally, this is done in two separate ways:

1. **Numerical analysis** — doctors examine blood test results such as total White Blood Cell count (WBC), Neutrophils percentage, and Lymphocytes percentage.
2. **Morphological analysis** — experts look at stained microscopic images of individual white blood cells under a microscope to identify their type.

Both approaches require significant time, experience, and human expertise. Manual interpretation can be slow, subjective, and prone to human error, especially in busy laboratories or resource-limited settings. 

Our project addresses this challenge by developing an intelligent web-based application that can **automatically**:
- Predict the patient’s condition (Bacterial Infection, Normal, or Possible Immunodeficiency/Severe Condition) using blood biomarkers.
- Classify the exact type of a single white blood cell (Neutrophil, Lymphocyte, Monocyte, Eosinophil, or Basophil) from a microscopic image.

By integrating both numerical and image-based AI models into one platform, the system provides a fast, consistent, and educational tool for WBC analysis.

### Why is it important?
White blood cells are the main components of the body’s immune system. Changes in their count or type can signal serious conditions such as bacterial or viral infections, allergic reactions, chronic inflammation, leukemia, or immunodeficiency disorders. Early and accurate detection helps guide proper treatment.

An AI-assisted tool like this is valuable because it can:
- Help medical students and researchers quickly explore relationships between lab values and cell morphology.
- Serve as a prototype for future clinical decision support systems in hematology.
- Reduce interpretation variability and provide probabilistic outputs with explanations.
- Support education by showing both numerical predictions and visual classifications with scientific insights.

### What type of data did we use?
We worked with two complementary types of data:

- **Tabular (Numerical) Data**: Blood biomarker values including WBC count (×10³/µL), Neutrophils (%), and Lymphocytes (%). The target variable consists of three classes: Bacterial Infection, Normal, and Possible Immunodeficiency / Severe Condition.
- **Image Data**: High-resolution microscopic images of peripheral blood smears stained to highlight white blood cells. Each image focuses on individual cells belonging to one of five major WBC types.

---

## 2. The Data

### Where did we get the data?
- **Biomarkers Dataset**: We used a curated medical dataset containing real-world-like patient records with white blood cell differential counts and corresponding diagnostic labels (Bacterial Infection, Normal, Severe Condition). The data reflects typical clinical patterns observed in hematology laboratories.
- **Image Dataset**: The **White Blood Cells Dataset** by Masoud Nickparvar, publicly available on Kaggle. This dataset contains thousands of labeled microscopic images covering five classes: Neutrophil, Lymphocyte, Monocyte, Eosinophil, and Basophil. The images were acquired from stained blood smears and represent real variations in cell morphology, staining intensity, and background.

### How did we prepare the data?

**For the Biomarkers Model:**
- **Feature Engineering**:
  - Applied natural logarithm transformation on WBC count (`WBC_log`) to reduce skewness and handle extreme high values (up to ~1500 ×10³/µL).
  - Created the **Neutrophil-to-Lymphocyte Ratio (NLR)** = Neutrophils / Lymphocytes. This ratio is a well-known clinical marker of systemic inflammation and infection.
  - Combined the original features with the newly engineered ones.
- **Preprocessing**:
  - Handled missing values and outliers.
  - Applied **StandardScaler** to bring all features to a similar scale (mean = 0, standard deviation = 1). This is important because Random Forest, while robust, benefits from scaled features when combined with other steps.
- **Class Distribution**: We observed imbalance, especially in the "Severe Condition" class, which had significantly fewer samples.

**For the CNN Image Model:**
- **Image Preprocessing**:
  - Resized all images to a fixed input size of **128 × 128 pixels** to ensure consistent input shape for the neural network.
  - Normalized pixel values by dividing by 255, scaling them to the range [0, 1].
- **Data Augmentation** (applied only during training):
  - Random rotation up to 15 degrees
  - Width and height shifts (10%)
  - Zoom range of 10%
  - Horizontal flipping
  These techniques artificially increase dataset diversity, improve generalization, and help prevent overfitting.
- **Splitting**: The dataset was divided into training (80%), validation (20%), and a completely separate test set for final evaluation.

---

## 3. Methods and Techniques

### Biomarkers Model (Traditional Machine Learning)
We implemented a **Random Forest Classifier** from scikit-learn.

**Why Random Forest?**
- Excellent at capturing complex, non-linear relationships between biomarkers.
- Naturally handles mixed feature types and provides built-in feature importance.
- Robust to noisy medical data and less prone to overfitting compared to single decision trees.
- Can be easily interpreted through feature importance scores.

**Technical Pipeline**:
1. Feature engineering (`WBC_log` + NLR)
2. Standard scaling using `StandardScaler`
3. Training a Random Forest model with multiple decision trees
4. Hyperparameter tuning and cross-validation (where possible)
5. Evaluation using accuracy, precision, recall, F1-score, and confusion matrix

**Feature Importance Results**:
- `WBC_log`: ~65% — the most dominant feature
- Neutrophils (%): ~22%
- NLR: ~7%
- Lymphocytes (%): ~4%

This ranking strongly aligns with clinical knowledge in hematology.

### CNN Image Model (Deep Learning)
We designed and trained a custom **Convolutional Neural Network (CNN)** using TensorFlow and Keras.

**Model Architecture**:
```python
Conv2D(32, (3,3), padding='same') → BatchNormalization → MaxPooling2D
Conv2D(64, (3,3), padding='same') → BatchNormalization → MaxPooling2D
Conv2D(128, (3,3), padding='same') → BatchNormalization → MaxPooling2D
Conv2D(256, (3,3), padding='same') → BatchNormalization → MaxPooling2D
GlobalAveragePooling2D()
Dense(256, activation='relu') → Dropout(0.5)
Dense(5, activation='softmax')
```

**Key Design Choices**:
- **Increasing filter sizes** (32→256) allow the network to learn hierarchical features — from simple edges to complex cell structures.
- **Batch Normalization** stabilizes and accelerates training by normalizing activations.
- **Global Average Pooling** reduces the number of parameters and helps prevent overfitting compared to Flatten + Dense layers.
- **Dropout (0.5)** randomly deactivates neurons during training to improve generalization.
- **Softmax activation** produces probability distributions over the five WBC classes.

**Training Configuration**:
- Optimizer: Adam with learning rate = 0.0005
- Loss: Categorical Crossentropy
- Epochs: up to 30
- Callbacks:
  - ModelCheckpoint (save best model based on validation accuracy)
  - EarlyStopping (patience = 10, restore best weights)
  - ReduceLROnPlateau (factor = 0.2, patience = 3)

**Evaluation Metrics**:
- Test accuracy
- Classification report (per-class precision, recall, F1-score)
- Confusion matrix
- Training and validation accuracy/loss curves

---

## 4. Web Application

The frontend is a modern, responsive single-page application built with HTML, CSS, and vanilla JavaScript. It features:

- Clean two-column grid layout (Biomarkers + CNN Image)
- Real-time image preview with controlled size (`max-height: 260px`, `object-fit: contain`)
- Beautiful result visualization:
  - Prominent predicted class with confidence bar
  - Color-coded horizontal probability bars for all classes
  - Scientific description and clinical insight for each predicted class
- Soft purple, blue, and green gradient design for a calm, professional medical appearance
- Fully functional fetch calls to backend endpoints (`/predict-biomarkers` and `/predict-image`)

The backend is built with Flask and serves both models:
- Loads the saved Random Forest and CNN (`.keras`) models
- Applies the exact same preprocessing steps used during training
- Returns consistent JSON responses with prediction and full probability list

---

## 5. Results and Evaluation

### Biomarkers Model
- **Test Accuracy**: **94.4%**

The model performed strongly on the test set. Feature importance analysis confirmed that `WBC_log` dominates predictions, which is consistent with clinical practice. However, the severe imbalance in the "Possible Immunodeficiency / Severe Condition" class limited performance on that category.

We thoroughly evaluated the model using:
- Overall accuracy
- Per-class precision, recall, and F1-score
- Confusion matrix to visualize misclassifications

### CNN Image Model
The CNN successfully learned to distinguish the five WBC types based on morphological features such as nucleus shape, cytoplasm color, and granule presence. Data augmentation significantly improved robustness to variations in staining and imaging conditions.

Evaluation was performed on a held-out test set using:
- Test accuracy and loss
- Detailed classification report
- Confusion matrix
- Learning curves (accuracy and loss over epochs)

The combination of Batch Normalization, Dropout, and augmentation helped the model generalize well despite the complexity of microscopic images.

---

## 6. How This Project Can Be Used in Real Work

This application serves multiple valuable purposes:

- **Educational Tool**: Medical students can experiment with different biomarker combinations and immediately see how they influence predictions. They can also upload cell images and compare the model’s classification with their own observations.
- **Research Support**: Researchers can use the system to quickly test hypotheses about biomarker patterns or cell morphology. The probability outputs and explanations help in understanding model reasoning.
- **Prototype Development**: It demonstrates a complete end-to-end pipeline (data → preprocessing → modeling → web deployment) that can be extended into more advanced clinical support tools.
- **Interactive Learning**: The rich result display (confidence bars, descriptions, and insights) makes it excellent for teaching hematology concepts.

---


## 7. Technical Summary

This project successfully integrates **traditional machine learning** and **deep learning** into a unified web platform. On the numerical side, we applied careful feature engineering (log transformation + NLR) and used a Random Forest classifier. On the image side, we designed a custom CNN with modern techniques including Batch Normalization, Global Average Pooling, Dropout, and aggressive data augmentation.

The result is a complete, functional system that connects laboratory numbers with visual cell morphology, providing both predictions and educational explanations.


---

# 🔬 WBC Classifier — White Blood Cell Analysis System

An AI-powered web application that analyzes white blood cells through two complementary approaches: **numerical biomarker prediction** and **microscopic image classification**, with an experimental **U-Net segmentation** module.

---

## 📁 Folder Structure

```
fullWBC2/
├── app.py                          ← Unified Flask backend (run this)
├── model.pkl                       ← Trained Random Forest model
├── scaler.pkl                      ← StandardScaler for biomarker preprocessing
├── label_encoder.pkl               ← Label encoder for biomarker classes
├── wbc_classification_model.keras  ← Trained CNN model for image classification
├── unet_wbc.pth                    ← Trained UNet model for segmentation (optional)
├── logs.txt                        ← Auto-generated prediction logs
│
├── templates/
│   └── index.html                  ← Unified frontend (both pages)
│
├── saved_code/
│   ├── PredictFromCBC2(2).ipynb   ← Biomarkers ML model notebook
│   ├── projectBio.ipynb            ← CNN image classifier notebook
│   └── Segmentation(2).ipynb      ← UNet segmentation notebook
│
└── white-blood-cells-dataset/      ← Image training data (Kaggle)
```

---

## ⚙️ Requirements

Make sure you have **Python 3.11** installed, then install all dependencies:

```bash
pip install flask numpy>=1.26 joblib pillow scikit-learn tensorflow opencv-python torch torchvision
```

> ⚠️ **Important version note:** TensorFlow 2.21 requires `numpy>=1.26,<2.0`. If opencv upgrades numpy to 2.x, fix it with:
> ```bash
> pip install "numpy>=1.26,<2.0" --force-reinstall
> ```

---

## 🚀 How to Run

1. Open a terminal inside the project folder:

```bash
cd C:\Users\user\Desktop\fullWBC2
```

2. Start the app:

```bash
python app.py
```

3. Open your browser and go to:

```
http://127.0.0.1:5000
```

That's it. No environment setup, no framework, no build step.

---

## 🌐 Pages

### Page 1 — Classifier
Two side-by-side tools:

| Tool | Input | Output |
|------|-------|--------|
| **Biomarkers Classifier** | WBC count, Neutrophils %, Lymphocytes % | Bacterial Infection / Normal / Possible Immunodeficiency |
| **CNN Image Classifier** | Microscopic cell image (.jpg/.png) | Neutrophil / Lymphocyte / Monocyte / Eosinophil / Basophil |

### Page 2 — Segmentation *(Prototype)*
Upload a blood smear image and receive:
- Original image
- Probability mask
- Binary mask (detected WBC region)

Requires `unet_wbc.pth` to be present in the root folder.

---

## 🤖 Models

### Biomarkers Model — Random Forest
- **Input features:** WBC count (log-transformed), Neutrophils %, Lymphocytes %, Neutrophil-to-Lymphocyte Ratio (NLR)
- **Classes:** Bacterial Infection · Normal · Possible Immunodeficiency / Severe Condition
- **Test Accuracy:** 94.4%
- **Feature importance:** WBC_log (~65%) › Neutrophils (~22%) › NLR (~7%) › Lymphocytes (~4%)

### CNN Image Classifier — Custom CNN (TensorFlow/Keras)
- **Input:** 128×128 RGB microscopic cell image
- **Architecture:** 4× Conv2D blocks (32→64→128→256 filters) + BatchNorm + GlobalAvgPool + Dropout(0.5)
- **Classes:** Neutrophil · Lymphocyte · Monocyte · Eosinophil · Basophil
- **Training:** Adam (lr=0.0005), EarlyStopping, ReduceLROnPlateau, data augmentation

### Segmentation Model — U-Net (PyTorch)
- **Input:** 256×256 RGB blood smear image
- **Output:** Binary segmentation mask isolating WBC regions
- **Status:** Prototype / future plan

---

## Common Issues & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `TemplateNotFound: index.html` | `index.html` not in `templates/` folder | Move `index.html` into `fullWBC2/templates/` |
| `numpy has no attribute 'dtypes'` | NumPy version mismatch with TensorFlow | `pip install "numpy>=1.26,<2.0" --force-reinstall` |
| `InconsistentVersionWarning` (sklearn) | Models saved with sklearn 1.6.1, older version installed | `pip install "scikit-learn>=1.6"` — harmless warning otherwise |
| `unet_wbc.pth not found` | Segmentation model file missing | Re-run `Segmentation(2).ipynb` in Colab and download the `.pth` file |
| `PyTorch and OpenCV are required` | Missing dependencies for segmentation | `pip install opencv-python torch torchvision` |

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serves the main web UI |
| `POST` | `/predict-biomarkers` | JSON body: `{WBC, Neutrophils, Lymphocytes}` |
| `POST` | `/predict-image` | Form data: `image` file |
| `POST` | `/predict-segmentation` | Form data: `image` file |
