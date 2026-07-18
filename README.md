 BREAST CANCER DETECTION USING DEEP LEARNING (ANN)

A Deep Learning binary classification model, built with an Artificial Neural Network (ANN), that predicts whether a breast tumor is Malignant (cancerous) or Benign (non-cancerous), based on measurements of cell nuclei from a digitized image of a breast mass.

OVERVIEW

Early and accurate detection of breast cancer significantly improves treatment outcomes. This project builds an ANN-based binary classifier using the well-known Breast Cancer Wisconsin (Diagnostic) dataset, achieving 99% test accuracy.


DATASET


Source: Breast Cancer Wisconsin (Diagnostic) Dataset — built into scikit-learn (sklearn.datasets.load_breast_cancer)
Samples: 569 patients
Features: 30 numeric features describing cell nuclei (radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension — each as mean, standard error, and worst value)
Target: Malignant (0) / Benign (1) — 212 Malignant, 357 Benign


MODEL ARCHITECTURE

Input Layer (30 features)
    ↓
Dense(32, ReLU) → Dropout(0.2)
    ↓
Dense(16, ReLU) → Dropout(0.2)
    ↓
Dense(1, Sigmoid) → Malignant / Benign

Optimizer: Adam
Loss Function: Binary Crossentropy
Metrics: Accuracy, Precision, Recall

METHODOLOGY


Data Check: Verified zero missing values (pre-cleaned dataset); checked class balance (63:37 — mild imbalance)
Preprocessing: Stratified 80-20 train-test split, followed by StandardScaler normalization
Training: 50 epochs, batch size 16, with a validation split to monitor overfitting
Evaluation: Precision, Recall, F1-score, and Confusion Matrix — prioritizing Recall for the Malignant class, since missing a cancer case is far more critical than a false alarm

 RESULTS

MetricScoreTest Accuracy   - 99%
Malignant — Precision / Recall   -  100% / 98% 
Benign — Precision / Recall    - 99% / 100%

Only 1 misclassification out of 114 test samples.

 FEATURE IMPORTANCE

A Random Forest classifier was used purely as an analysis tool (not the production model) to rank feature importance and identify the most predictive measurements.

Key Insight: worst area, worst concave points, and mean concave points emerged as the strongest predictors — consistent with the clinical understanding that larger, more irregularly-shaped cell nuclei are associated with malignancy.


 Prediction Demo

Two prediction interfaces were explored:


Full 30-feature lookup — verifies the model against known dataset samples by index (for testing/validation only)
Simplified 6-feature version — takes only the top 6 most important measurements and fills the rest with dataset averages, trading a small amount of confidence for a much more practical, user-friendly interface


TECH STACK 


Python, Pandas, NumPy
TensorFlow / Keras
Scikit-learn
Matplotlib, Seaborn






