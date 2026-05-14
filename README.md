# Multimodal Explainable AI Medical Diagnostic Agent  
## X-ray Images + Blood Laboratory Data

A graduation project developed by:

- **Waleed Etawi**
- **Aya Al-Majali**
- **Sofian Twissi**

Supervised by:

- **Dr. Motaz Al-Hami**

Princess Sumaya University for Technology  
Department of Data Science  
2025/2026

---

## Overview

This project presents a **Multimodal Explainable AI Medical Diagnostic Agent** that combines **X-ray images** and **blood laboratory data** to support medical diagnosis.

In real medical practice, clinicians rarely depend on one source of evidence. They usually compare radiographic findings with laboratory results and patient context before forming a conclusion. This project follows the same idea by developing an AI system that can process multiple medical modalities and provide interpretable outputs.

The system supports:

- Chest X-ray analysis for thoracic findings
- Bone X-ray abnormality detection
- Blood laboratory prediction
- Multimodal fusion using chest X-ray images and lab data
- Explainability using Grad-CAM and Integrated Gradients
- Downloadable diagnostic reports

The system is intended as a **research and educational prototype**, not as a clinically validated diagnostic tool.

---

## Key Features

### Image-Based Diagnosis

The system uses **DenseNet-121** for medical image classification.

Supported image branches:

- **Chest X-ray model** for multi-label thoracic disease prediction
- **Bone X-ray model** for normal/abnormal classification using the MURA dataset

### Laboratory-Based Diagnosis

The laboratory model uses **32 selected blood tests** covering:

- Hematology
- Kidney function
- Electrolytes
- Liver function
- Blood gas analysis

Each lab value is converted into clinically meaningful engineered features:

- Original numerical value
- Low indicator
- Normal indicator
- High indicator
- Low-distance abnormality feature
- High-distance abnormality feature

This allows the model to understand both the **direction** and **severity** of abnormal laboratory values.

### Cross-Attention Fusion

The main contribution of this project is the use of a **cross-attention fusion model**.

Instead of simply combining final prediction scores, the fusion model allows laboratory features to interact directly with spatial chest X-ray features before making the final prediction.

This makes the model more powerful because it learns relationships between:

- Visual abnormalities in chest X-rays
- Biological signals from blood laboratory tests

### Explainable AI

The project includes explainability methods to make model outputs easier to understand.

| Modality | Explainability Method | Purpose |
|---|---|---|
| X-ray Images | Grad-CAM / Grad-CAM++ | Highlights important image regions |
| Laboratory Data | Integrated Gradients | Shows influential lab features |

Explainability is important because medical AI should not only provide predictions, but also show why a prediction was made.

---

## Thoracic Target Classes

The chest X-ray and fusion models predict six thoracic findings:

1. Atelectasis
2. Cardiomegaly
3. Edema
4. Lung Opacity
5. No Finding
6. Pleural Effusion

---

## System Architecture

The project contains four major branches:

```text
Input Layer
│
├── Chest X-ray Image
├── Bone X-ray Image
└── Blood Laboratory Data
        │
        ▼
Preprocessing
│
├── Image resizing to 224 × 224
├── Image normalization
├── Lab cleaning
├── Missing-value handling
├── Scaling
└── Clinical feature engineering
        │
        ▼
Models
│
├── DenseNet-121 Chest X-ray Model
├── DenseNet-121 Bone X-ray Model
├── Laboratory Neural Network
└── Cross-Attention Fusion Model
        │
        ▼
Explainability
│
├── Grad-CAM for X-ray images
└── Integrated Gradients for laboratory data
        │
        ▼
Outputs
│
├── Disease probabilities
├── Heatmaps
├── Lab feature importance
└── Downloadable diagnostic report
