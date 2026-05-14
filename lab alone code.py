"""
Strong Lab-Only Disease Classifier
----------------------------------
Input: 32 lab test values
Extra engineered input: each test is converted into LOW / NORMAL / HIGH flags using clinical reference ranges.
Output: probability for each illness class.

What this script saves:
1) best_lab_model.pth
2) blood_scaler.joblib
3) blood_label_encoder.joblib
4) lab_feature_columns.joblib
5) lab_reference_ranges.joblib
6) lab_model_metadata.joblib

Expected CSV:
- LABS_TRAIN_SYNCED.csv
- must contain the 32 BLOOD_COLS below
- must contain a diagnosis column
"""

import os
import copy
import random
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

# =========================
# CONFIG
# =========================
DATA_PATH = "LABS_TRAIN_SYNCED.csv"
SAVE_DIR = "flask/models/lab alone"
WEIGHTS_SAVE_PATH = os.path.join(SAVE_DIR, "best_lab_model.pth")
SCALER_SAVE_PATH = os.path.join(SAVE_DIR, "blood_scaler.joblib")
IMPUTER_SAVE_PATH = os.path.join(SAVE_DIR, "blood_imputer.joblib")
ENCODER_SAVE_PATH = os.path.join(SAVE_DIR, "blood_label_encoder.joblib")
FEATURE_COLS_SAVE_PATH = os.path.join(SAVE_DIR, "lab_feature_columns.joblib")
RANGES_SAVE_PATH = os.path.join(SAVE_DIR, "lab_reference_ranges.joblib")
METADATA_SAVE_PATH = os.path.join(SAVE_DIR, "lab_model_metadata.joblib")

os.makedirs(SAVE_DIR, exist_ok=True)

SEED = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.10
BATCH_SIZE = 64
EPOCHS = 250
PATIENCE = 35
LR = 7e-4
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.04
GRAD_CLIP = 2.0

# Your 32 lab tests
BLOOD_COLS = [
    "Hematocrit", "Platelet Count", "Creatinine", "Potassium", "Hemoglobin",
    "White Blood Cells", "MCHC", "Red Blood Cells", "MCV", "MCH", "RDW",
    "Urea Nitrogen", "Sodium", "Bicarbonate", "Anion Gap", "Glucose",
    "Magnesium", "Calcium, Total", "Neutrophils", "Monocytes", "Eosinophils",
    "Lymphocytes", "Alanine Aminotransferase (ALT)", "Asparate Aminotransferase (AST)",
    "Lactate", "Alkaline Phosphatase", "Bilirubin, Total", "pH", "Albumin",
    "Base Excess", "pO2", "pCO2",
]

# Approximate adult reference ranges.
# IMPORTANT: update these to match your dataset units if needed.
REFERENCE_RANGES = {
    "Hematocrit": (0.36, 0.50),
    "Platelet Count": (150, 450),
    "Creatinine": (0.6, 1.3),
    "Potassium": (3.5, 5.1),
    "Hemoglobin": (120, 170),
    "White Blood Cells": (4.0, 11.0),
    "MCHC": (320, 360),
    "Red Blood Cells": (4.2, 5.9),
    "MCV": (80, 100),
    "MCH": (27, 33),
    "RDW": (11.5, 14.5),
    "Urea Nitrogen": (7, 20),
    "Sodium": (135, 145),
    "Bicarbonate": (22, 29),
    "Anion Gap": (8, 16),
    "Glucose": (70, 140),
    "Magnesium": (1.7, 2.4),
    "Calcium, Total": (8.5, 10.5),
    "Neutrophils": (40, 75),
    "Monocytes": (2, 10),
    "Eosinophils": (0, 6),
    "Lymphocytes": (20, 45),
    "Alanine Aminotransferase (ALT)": (7, 56),
    "Asparate Aminotransferase (AST)": (10, 40),
    "Lactate": (0.5, 2.2),
    "Alkaline Phosphatase": (44, 147),
    "Bilirubin, Total": (0.1, 1.2),
    "pH": (7.35, 7.45),
    "Albumin": (3.5, 5.0),
    "Base Excess": (-2, 2),
    "pO2": (75, 100),
    "pCO2": (35, 45),
}


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_blood_work(df: pd.DataFrame) -> pd.DataFrame:
    """Fix common unit mistakes and impossible values before modeling."""
    df = df.copy()

    # Convert columns to numeric safely.
    for col in BLOOD_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Hematocrit: if written as percent, convert to ratio.
    if "Hematocrit" in df.columns:
        mask = df["Hematocrit"] > 1.0
        df.loc[mask, "Hematocrit"] = df.loc[mask, "Hematocrit"] / 100.0

    # Hemoglobin and MCHC: if written as g/dL, convert to g/L.
    for col in ["Hemoglobin", "MCHC"]:
        if col in df.columns:
            mask = df[col] < 50.0
            df.loc[mask, col] = df.loc[mask, col] * 10.0

    # pH impossible values.
    if "pH" in df.columns:
        mask = (df["pH"] < 6.8) | (df["pH"] > 7.8)
        df.loc[mask, "pH"] = np.nan

    # Remove impossible negative values for tests that cannot be negative.
    non_negative = [c for c in BLOOD_COLS if c not in ["Base Excess"]]
    for col in non_negative:
        if col in df.columns:
            df.loc[df[col] < 0, col] = np.nan

    return df


def add_low_normal_high_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinical category flags for each lab: low, normal, high."""
    out = df.copy()
    for col in BLOOD_COLS:
        low, high = REFERENCE_RANGES[col]
        out[f"{col}__low"] = (out[col] < low).astype(np.float32)
        out[f"{col}__normal"] = ((out[col] >= low) & (out[col] <= high)).astype(np.float32)
        out[f"{col}__high"] = (out[col] > high).astype(np.float32)

        # How far below/above the range, normalized by range width.
        width = max(high - low, 1e-6)
        out[f"{col}__low_distance"] = np.maximum((low - out[col]) / width, 0).astype(np.float32)
        out[f"{col}__high_distance"] = np.maximum((out[col] - high) / width, 0).astype(np.float32)

    return out


def get_feature_columns() -> list[str]:
    feature_cols = []
    feature_cols.extend(BLOOD_COLS)
    for col in BLOOD_COLS:
        feature_cols.extend([
            f"{col}__low",
            f"{col}__normal",
            f"{col}__high",
            f"{col}__low_distance",
            f"{col}__high_distance",
        ])
    return feature_cols


def load_data():
    print(f"Loading data from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    missing = [c for c in BLOOD_COLS + ["diagnosis"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["diagnosis"]).copy()
    df["primary_diagnosis"] = df["diagnosis"].astype(str).apply(lambda x: x.split(";")[0].strip())
    df = normalize_blood_work(df)

    X_raw = df[BLOOD_COLS].copy()

    # Median imputation is fitted only on train later, but for engineered flags we need temporary medians.
    # This is only to avoid NaNs in flag calculation; final imputer still fits correctly on train split.
    temp_medians = X_raw.median(numeric_only=True)
    X_temp = X_raw.fillna(temp_medians)
    X_feat = add_low_normal_high_features(X_temp)
    feature_cols = get_feature_columns()

    y_raw = df["primary_diagnosis"].values
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw).astype(np.int64)

    print("Classes:")
    for cls, count in Counter(y_raw).items():
        print(f"  {cls}: {count}")

    return X_feat[feature_cols], y, label_encoder, feature_cols


class FocalLoss(nn.Module):
    """Focal loss helps the model focus more on hard and minority-class examples."""
    def __init__(self, alpha=None, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = nn.functional.cross_entropy(
            logits,
            targets,
            weight=self.alpha,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        pt = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.block(x))


class StrongLabDiagnosisNet(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.20),
        )
        self.res1 = ResidualBlock(256, 0.20)
        self.down1 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(0.18),
        )
        self.res2 = ResidualBlock(128, 0.18)
        self.down2 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.output = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.input(x)
        x = self.res1(x)
        x = self.down1(x)
        x = self.res2(x)
        x = self.down2(x)
        return self.output(x)


def make_weighted_sampler(y_train: np.ndarray) -> WeightedRandomSampler:
    counts = np.bincount(y_train)
    weights_per_class = 1.0 / np.maximum(counts, 1)
    sample_weights = weights_per_class[y_train]
    sample_weights = torch.DoubleTensor(sample_weights)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def find_best_temperature(logits: torch.Tensor, y_true: torch.Tensor) -> float:
    """Simple temperature calibration for better probability confidence."""
    best_temp = 1.0
    best_nll = float("inf")
    temps = torch.linspace(0.5, 3.0, 51, device=logits.device)
    for t in temps:
        loss = nn.functional.cross_entropy(logits / t, y_true).item()
        if loss < best_nll:
            best_nll = loss
            best_temp = float(t.item())
    return best_temp


def train():
    seed_everything(SEED)
    X_df, y, label_encoder, feature_cols = load_data()
    num_classes = len(label_encoder.classes_)

    # Stratified split. If a class is very tiny, stratify can fail; the dataset should ideally have more samples.
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X_df, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=SEED, stratify=y_trainval
    )

    imputer = SimpleImputer(strategy="median")
    scaler = RobustScaler()

    X_train_imp = imputer.fit_transform(X_train)
    X_val_imp = imputer.transform(X_val)
    X_test_imp = imputer.transform(X_test)

    X_train_sc = scaler.fit_transform(X_train_imp).astype(np.float32)
    X_val_sc = scaler.transform(X_val_imp).astype(np.float32)
    X_test_sc = scaler.transform(X_test_imp).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Input features after engineering: {X_train_sc.shape[1]}")

    X_train_t = torch.tensor(X_train_sc, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val_sc, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)
    X_test_t = torch.tensor(X_test_sc, dtype=torch.float32).to(device)
    y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)

    sampler = make_weighted_sampler(y_train)
    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=False,
    )

    class_counts = np.bincount(y_train, minlength=num_classes)
    # Balanced class weights, softened so tiny classes do not dominate too aggressively.
    class_weights = len(y_train) / (num_classes * np.maximum(class_counts, 1))
    class_weights = np.sqrt(class_weights)
    class_weights = class_weights / class_weights.mean()
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32).to(device)

    model = StrongLabDiagnosisNet(input_dim=X_train_sc.shape[1], num_classes=num_classes).to(device)
    criterion = FocalLoss(alpha=class_weights_t, gamma=1.5, label_smoothing=LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2)

    best_macro_f1 = -1.0
    best_val_loss = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    epochs_no_improve = 0

    print("\nStarting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            train_loss += loss.item() * xb.size(0)

        scheduler.step(epoch)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = criterion(val_logits, y_val_t).item()
            val_preds = torch.argmax(val_logits, dim=1).cpu().numpy()
            val_macro_f1 = f1_score(y_val, val_preds, average="macro", zero_division=0)
            val_weighted_f1 = f1_score(y_val, val_preds, average="weighted", zero_division=0)

        improved = (val_macro_f1 > best_macro_f1) or (
            np.isclose(val_macro_f1, best_macro_f1) and val_loss < best_val_loss
        )
        if improved:
            best_macro_f1 = val_macro_f1
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"Epoch {epoch:03d} | "
                f"Train Loss: {train_loss / len(X_train_sc):.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Macro-F1: {val_macro_f1:.4f} | "
                f"Val Weighted-F1: {val_weighted_f1:.4f}"
            )

        if epochs_no_improve >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"\nLoading best model. Best Val Macro-F1: {best_macro_f1:.4f}")
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_logits = model(X_val_t)
        temperature = find_best_temperature(val_logits, y_val_t)
        test_logits = model(X_test_t) / temperature
        test_probs = torch.softmax(test_logits, dim=1).cpu().numpy()
        test_preds = np.argmax(test_probs, axis=1)

    acc = accuracy_score(y_test, test_preds)
    macro_f1 = f1_score(y_test, test_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, test_preds, average="weighted", zero_division=0)

    print("\nTraining Complete.")
    print(f"Accuracy: {acc:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")
    print(f"Weighted-F1: {weighted_f1:.4f}")
    print(f"Temperature: {temperature:.3f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_preds, target_names=label_encoder.classes_, zero_division=0))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, test_preds))

    print(f"\nSaving model/assets to: {SAVE_DIR}")
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": X_train_sc.shape[1],
        "num_classes": num_classes,
        "temperature": temperature,
        "blood_cols": BLOOD_COLS,
        "feature_cols": feature_cols,
        "class_names": list(label_encoder.classes_),
    }, WEIGHTS_SAVE_PATH)
    joblib.dump(imputer, IMPUTER_SAVE_PATH)
    joblib.dump(scaler, SCALER_SAVE_PATH)
    joblib.dump(label_encoder, ENCODER_SAVE_PATH)
    joblib.dump(feature_cols, FEATURE_COLS_SAVE_PATH)
    joblib.dump(REFERENCE_RANGES, RANGES_SAVE_PATH)
    joblib.dump({
        "accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "temperature": temperature,
        "class_counts_train": class_counts.tolist(),
        "classes": list(label_encoder.classes_),
    }, METADATA_SAVE_PATH)


def classify_single_lab(values: dict, model_path=WEIGHTS_SAVE_PATH):
    """
    Example inference function.
    values must be a dict with the 32 BLOOD_COLS as keys.
    Returns sorted illness probabilities and LOW/NORMAL/HIGH state for every lab.
    """
    checkpoint = torch.load(model_path, map_location="cpu")
    feature_cols = checkpoint["feature_cols"]
    class_names = checkpoint["class_names"]
    temperature = checkpoint.get("temperature", 1.0)

    imputer = joblib.load(IMPUTER_SAVE_PATH)
    scaler = joblib.load(SCALER_SAVE_PATH)
    ranges = joblib.load(RANGES_SAVE_PATH)

    raw = pd.DataFrame([{col: values.get(col, np.nan) for col in BLOOD_COLS}])
    raw = normalize_blood_work(raw)

    # Need imputed raw labs before high/normal/low engineering.
    raw_imp = pd.DataFrame(imputer.transform(add_low_normal_high_features(raw.fillna(raw.median(numeric_only=True)))[feature_cols]), columns=feature_cols)
    X = scaler.transform(raw_imp).astype(np.float32)

    model = StrongLabDiagnosisNet(checkpoint["input_dim"], checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        logits = model(torch.tensor(X, dtype=torch.float32)) / temperature
        probs = torch.softmax(logits, dim=1).numpy()[0]

    illness_probs = sorted(
        [{"illness": cls, "probability": float(prob)} for cls, prob in zip(class_names, probs)],
        key=lambda x: x["probability"],
        reverse=True,
    )

    lab_states = {}
    for col in BLOOD_COLS:
        val = raw.iloc[0][col]
        low, high = ranges[col]
        if pd.isna(val):
            state = "missing"
        elif val < low:
            state = "low"
        elif val > high:
            state = "high"
        else:
            state = "normal"
        lab_states[col] = {"value": None if pd.isna(val) else float(val), "state": state, "range": [low, high]}

    return {"illness_probabilities": illness_probs, "lab_states": lab_states}


if __name__ == "__main__":
    train()
