# ==========================================
# SECTION 1: SETUP XAI ENVIRONMENT & IMPORTS
# ==========================================
# Install required packages
import subprocess
import sys

# Install XAI libraries
packages = ['captum', 'shap', 'matplotlib']
for pkg in packages:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', pkg])

print(" XAI packages installed successfully!")

# Import all required libraries
import os
import cv2
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import glob

# XAI Libraries - Using custom implementation for GradCAM
# Integrated Gradients for neural network attribution
from captum.attr import IntegratedGradients

# SHAP for tabular data (using Kernel SHAP as alternative to Tree SHAP for neural networks)
import shap

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print(" All XAI libraries imported successfully!")

# ==========================================
# LOAD TRAINED MODELS & DATA
# ==========================================
# Configuration - UPDATE THESE PATHS TO MATCH YOUR ENVIRONMENT
DATA_DIR = r"E:\grad project 1\medical agent\data sets\symile-mimic-a-multimodal\gp2\Final Final model\final fusion code"
IMAGE_FOLDER = os.path.join(DATA_DIR, "images")
LABS_TRAIN_PATH = os.path.join(DATA_DIR, "LABS_TRAIN_SYNCED.csv")
LABS_TEST_PATH = os.path.join(DATA_DIR, "LABS_TEST_SYNCED.csv")

# Model paths
BLOOD_MODEL_PATH = r"E:\grad project 1\medical agent\data sets\symile-mimic-a-multimodal\gp2\Final Final model\final fusion code\weights\blood_net_FULL_best.pth"
IMAGE_MODEL_PATH = r"E:\grad project 1\medical agent\data sets\symile-mimic-a-multimodal\gp2\Final Final model\final fusion code\weights\densenet_multi_label_FULL_best.pth"
FUSION_MODEL_PATH = r"E:\grad project 1\medical agent\data sets\symile-mimic-a-multimodal\gp2\Final Final model\final fusion code\weights\cross_attention_fusion_best.pth"

# Target columns
target_cols = ["Atelectasis", "Cardiomegaly", "Edema", "Lung Opacity", "No Finding", "Pleural Effusion"]
num_classes = len(target_cols)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ==========================================
# DEFINE MODEL ARCHITECTURES
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.drop = nn.Dropout(dropout)
        self.gelu = nn.GELU()

    def forward(self, x):
        residual = x
        out = self.gelu(self.bn1(self.fc1(x)))
        out = self.drop(out)
        out = self.bn2(self.fc2(out))
        return self.gelu(out + residual)

class AdvancedBloodNet(nn.Module):
    def __init__(self, input_dim, num_targets):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.3)
        )
        self.res1 = ResidualBlock(256, dropout=0.3)
        self.res2 = ResidualBlock(256, dropout=0.3)
        self.compression = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(128, num_targets)
        )

    def forward(self, x):
        return self.compression(self.res2(self.res1(self.entry(x))))

def get_image_model(num_classes):
    model = models.densenet121(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model

class CrossAttentionFusion(nn.Module):
    def __init__(self, image_model, tabular_model, num_classes=6, embed_dim=256, num_heads=4):
        super().__init__()
        self.image_model = image_model
        self.tabular_model = tabular_model
        self.img_proj = nn.Linear(1024, embed_dim)
        self.tab_proj = nn.Linear(128, embed_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=0.3
        )
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.fusion_head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.BatchNorm1d(128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, image, tabular):
        img_feats = F.relu(self.image_model.features(image), inplace=True)
        B, C, H, W = img_feats.shape
        img_seq = img_feats.view(B, C, H * W).permute(0, 2, 1)
        img_seq_proj = self.img_proj(img_seq)

        tab_feats = self.tabular_model.entry(tabular)
        tab_feats = self.tabular_model.res1(tab_feats)
        tab_feats = self.tabular_model.res2(tab_feats)
        for layer in self.tabular_model.compression[:-1]:
            tab_feats = layer(tab_feats)

        tab_proj = self.tab_proj(tab_feats).unsqueeze(1)
        attn_output, _ = self.cross_attention(
            query=tab_proj, key=img_seq_proj, value=img_seq_proj
        )

        return self.fusion_head(self.layer_norm(attn_output.squeeze(1)))

print(" Model architectures defined!")

# ==========================================
# DATA LOADING & PREPROCESSING
# ==========================================
print("Loading and preprocessing data...")

# Load lab data
train_df = pd.read_csv(LABS_TRAIN_PATH)
test_df = pd.read_csv(LABS_TEST_PATH)
train_df['is_train'] = 1
test_df['is_train'] = 0
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Process tabular features
label_cols = [c for c in combined_df.columns if c.endswith("_label")]
z_cols = [c for c in combined_df.columns if c.endswith("_z")]

def clean_lab_label(value):
    MISSING_TOKENS = {"", " ", "nan", "na", "n/a", "none", "null", "missing"}
    if pd.isna(value): return "Missing"
    s = str(value).strip().lower()
    if s in MISSING_TOKENS: return "Missing"
    s = s.title()
    if s in {"High", "Normal", "Low", "Missing"}: return s
    return "Missing"

X_labels = combined_df[label_cols].copy()
for col in label_cols:
    X_labels[col] = X_labels[col].apply(clean_lab_label)

X_labels_oh = pd.get_dummies(X_labels, columns=label_cols, dtype=np.float32)
X_z_filled = combined_df[z_cols].astype(np.float32).fillna(0.0)
X_z_missing = combined_df[z_cols].astype(np.float32).isna().astype(np.float32)
X_z_missing.columns = [f"{c}_missing" for c in z_cols]

all_tabular_features = list(X_labels_oh.columns) + list(X_z_filled.columns) + list(X_z_missing.columns)

# Create master dataframe
y_df = combined_df[target_cols].copy().fillna(0)
for col in target_cols:
    y_df[col] = pd.to_numeric(y_df[col], errors="coerce").fillna(0).clip(0, 1)

master_df = pd.concat([
    combined_df[['hadm_id', 'is_train']], X_labels_oh, X_z_filled, X_z_missing, y_df
], axis=1)

# Map images
all_image_paths = glob.glob(os.path.join(IMAGE_FOLDER, '**', '*.*'), recursive=True)
image_path_dict = {
    os.path.basename(p).split('.')[0]: p 
    for p in all_image_paths if p.lower().endswith(('.png', '.jpg', '.jpeg'))
}

master_df['image_exists'] = master_df['hadm_id'].apply(
    lambda x: str(x).split('.')[0] in image_path_dict
)

# Get test samples with images
test_fusion_df = master_df[
    (master_df['image_exists'] == True) & 
    (master_df['is_train'] == 0)
].copy().reset_index(drop=True)

print(f" Loaded {len(test_fusion_df)} test samples with images")
print(f"   Tabular features: {len(all_tabular_features)}")
print(f"   Target classes: {len(target_cols)}")

# ==========================================
# INITIALIZE MODELS & LOAD WEIGHTS
# ==========================================
print("Loading trained model weights...")

# Initialize models
tabular_model = AdvancedBloodNet(len(all_tabular_features), num_classes).to(device)
image_model = get_image_model(num_classes).to(device)
fusion_model = CrossAttentionFusion(image_model, tabular_model, num_classes=num_classes).to(device)

# Load weights
tabular_model.load_state_dict(torch.load(BLOOD_MODEL_PATH, map_location=device))
image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
fusion_model.load_state_dict(torch.load(FUSION_MODEL_PATH, map_location=device))

# Set to evaluation mode
tabular_model.eval()
image_model.eval()
fusion_model.eval()

print(" All models loaded successfully!")
print(f"   - Blood Model (Tabular): {sum(p.numel() for p in tabular_model.parameters()):,} parameters")
print(f"   - Image Model (DenseNet): {sum(p.numel() for p in image_model.parameters()):,} parameters")
print(f"   - Fusion Model (Cross-Attention): {sum(p.numel() for p in fusion_model.parameters()):,} parameters")

# ==========================================
# GRADCAM++ FOR X-RAY IMAGES
# ==========================================
class ApplyCLAHE(object):
    """Apply CLAHE preprocessing to images"""
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    
    def __call__(self, img):
        img_np = np.array(img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        clahe_img = self.clahe.apply(gray)
        return Image.fromarray(cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2RGB))

# Define transforms for XAI
xai_transform = transforms.Compose([
    ApplyCLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Image denormalization for visualization
def denormalize_image(tensor):
    """Reverse the normalization for visualization"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return tensor * std + mean

class XRayGradCAM:
    """Custom GradCAM++ implementation for X-ray images"""
    
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        
        # Use the last convolutional layer of DenseNet
        if target_layer is None:
            target_layer = list(model.features.children())[-1]
        self.target_layer = target_layer
    
    def explain(self, image_tensor, target_class=None, show_heatmap=True):
        """
        Generate GradCAM++ heatmap for an image
        
        Args:
            image_tensor: Preprocessed image tensor (1, 3, 224, 224)
            target_class: Class index to explain (None = use prediction)
            show_heatmap: Whether to overlay heatmap on image
            
        Returns:
            heatmap: numpy array of shape (H, W) with importance scores
        """
        image_tensor = image_tensor.to(device)
        image_tensor.requires_grad = True
        
        # Get prediction if target not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(image_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Custom GradCAM implementation
        features = []
        gradients_list = []
        
        def forward_hook(module, input, output):
            features.append(output)
            output.register_hook(lambda grad: gradients_list.append(grad))
            
        # Register hook to get feature maps and gradients
        handle = self.target_layer.register_forward_hook(forward_hook)
        
        # Forward pass
        self.model.zero_grad()
        output = self.model(image_tensor)
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        handle.remove()
        
        feature_maps = features[0]  # Shape: [B, C, H, W]
        gradients = gradients_list[0]
        
        # Global average pooling of gradients to get weights
        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # Shape: [B, C, 1, 1]
        
        # Weighted sum of feature maps
        heatmap = torch.sum(feature_maps * weights, dim=1, keepdim=True)
        heatmap = F.relu(heatmap)
        
        # Normalize to 0-1
        heatmap = heatmap[0, 0].cpu().detach().numpy()
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap, target_class
    
    def visualize(self, image_tensor, heatmap, title="GradCAM++ Explanation", 
                  original_image=None, save_path=None):
        """Visualize the GradCAM++ heatmap"""
        
        # Get original image for overlay
        if original_image is None:
            original_image = denormalize_image(image_tensor.cpu().detach())
            original_image = original_image.permute(1, 2, 0).numpy()
            original_image = np.clip(original_image, 0, 1)
        
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(original_image)
        axes[0].set_title("Original X-Ray", fontsize=12, fontweight='bold')
        axes[0].axis('off')
        
        # Heatmap
        im1 = axes[1].imshow(heatmap, cmap='jet')
        axes[1].set_title("GradCAM++ Heatmap", fontsize=12, fontweight='bold')
        axes[1].axis('off')
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
        
        # Overlay
        axes[2].imshow(original_image)
        axes[2].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[2].set_title("Overlay", fontsize=12, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Saved to: {save_path}")
        
        plt.show()
        
        return fig

# Initialize GradCAM explainer for image model
image_gradcam = XRayGradCAM(image_model)
print(" GradCAM++ initialized for X-ray images!")

# ==========================================
# INTEGRATED GRADIENTS FOR LAB REPORTS
# ==========================================
class LabReportExplainer:
    """Integrated Gradients explainer for lab report tabular data"""
    
    def __init__(self, model, feature_names, n_steps=50):
        self.model = model
        self.feature_names = feature_names
        self.n_steps = n_steps
        self.model.eval()
        
        # Initialize Integrated Gradients
        self.ig = IntegratedGradients(model)
        
        # Create baseline (all zeros - representing "normal" or "missing" values)
        self.baseline = torch.zeros(1, len(feature_names)).to(device)
    
    def explain(self, tabular_tensor, target_class=None, show_plot=True):
        """
        Generate Integrated Gradients attribution for lab report
        
        Args:
            tabular_tensor: Tabular feature tensor (1, num_features)
            target_class: Class index to explain (None = use prediction)
            show_plot: Whether to show waterfall plot
            
        Returns:
            attribution: numpy array of feature importances
        """
        tabular_tensor = tabular_tensor.to(device).float()
        tabular_tensor.requires_grad = True
        
        # Get prediction if target not specified
        if target_class is None:
            with torch.no_grad():
                output = self.model(tabular_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Generate Integrated Gradients attribution
        attribution, delta = self.ig.attribute(
            tabular_tensor,
            baselines=self.baseline,
            target=int(target_class),
            n_steps=self.n_steps,
            return_convergence_delta=True
        )
        
        # Convert to numpy
        attribution = attribution[0].cpu().detach().numpy()
        
        return attribution, target_class
    
    def visualize(self, attribution, title="Integrated Gradients Explanation",
                  top_n=20, save_path=None):
        """Visualize feature importance as waterfall plot"""
        
        # Get feature importance
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': attribution
        })
        importance_df['Abs_Importance'] = np.abs(importance_df['Importance'])
        importance_df = importance_df.sort_values('Abs_Importance', ascending=True)
        
        # Take top N features
        if len(importance_df) > top_n:
            importance_df = importance_df.tail(top_n)
        
        # Create waterfall plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        colors = ['green' if x > 0 else 'red' for x in importance_df['Importance']]
        bars = ax.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
        
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Feature Importance (Integrated Gradients)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(axis='x', linestyle='--', alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Positive (increases risk)'),
            Patch(facecolor='red', label='Negative (decreases risk)')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Saved to: {save_path}")
        
        plt.show()
        
        return importance_df
    
    def get_top_features(self, attribution, top_k=10):
        """Get top K most important features"""
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': attribution
        })
        importance_df['Abs_Importance'] = np.abs(importance_df['Importance'])
        importance_df = importance_df.sort_values('Abs_Importance', ascending=False)
        
        return importance_df.head(top_k)

# Initialize Integrated Gradients explainer for tabular model
lab_explainer = LabReportExplainer(tabular_model, all_tabular_features)
print(" Integrated Gradients initialized for lab reports!")

# ==========================================
# UNIFIED XAI VISUALIZATION CLASS
# ==========================================
class UnifiedXAIVisualizer:
    """Unified XAI visualization for multimodal fusion model"""
    
    def __init__(self, image_model, tabular_model, fusion_model,
                 image_gradcam, lab_explainer, target_cols):
        self.image_model = image_model
        self.tabular_model = tabular_model
        self.fusion_model = fusion_model
        self.image_gradcam = image_gradcam
        self.lab_explainer = lab_explainer
        self.target_cols = target_cols
    
    def get_prediction(self, image_tensor, tabular_tensor):
        """Get predictions from all models"""
        image_tensor = image_tensor.to(device)
        tabular_tensor = tabular_tensor.to(device)
        
        with torch.no_grad():
            img_pred = torch.sigmoid(image_model(image_tensor)).cpu().numpy()[0]
            tab_pred = torch.sigmoid(tabular_model(tabular_tensor)).cpu().numpy()[0]
            fus_pred = torch.sigmoid(fusion_model(image_tensor, tabular_tensor)).cpu().numpy()[0]
        
        return img_pred, tab_pred, fus_pred
    
    def explain_sample(self, sample_idx, save_dir=None):
        """Generate comprehensive XAI explanation for a single sample"""
        
        # Get sample data
        row = test_fusion_df.iloc[sample_idx]
        patient_id = str(row['hadm_id']).split('.')[0]
        
        # Load image
        img_path = image_path_dict[patient_id]
        original_img = Image.open(img_path).convert('RGB')
        image_tensor = xai_transform(original_img).unsqueeze(0).to(device)
        
        # Get tabular data
        tabular_data = row[all_tabular_features].values.astype(np.float32)
        tabular_tensor = torch.tensor(tabular_data).unsqueeze(0).to(device)
        
        # Get predictions
        img_pred, tab_pred, fus_pred = self.get_prediction(image_tensor, tabular_tensor)
        
        # Create figure
        fig = plt.figure(figsize=(20, 16))
        
        # ===== ROW 1: Original Image =====
        ax1 = fig.add_subplot(3, 4, 1)
        ax1.imshow(original_img)
        ax1.set_title(f"Patient ID: {patient_id}\nOriginal X-Ray", fontsize=11, fontweight='bold')
        ax1.axis('off')
        
        # ===== ROW 2: Image Model Explanation =====
        # Get GradCAM for image model
        heatmap, img_class = image_gradcam.explain(image_tensor)
        
        # Original denormalized
        img_denorm = denormalize_image(image_tensor.cpu().detach()[0])
        img_denorm = img_denorm.permute(1, 2, 0).numpy()
        img_denorm = np.clip(img_denorm, 0, 1)
        
        ax2 = fig.add_subplot(3, 4, 2)
        ax2.imshow(img_denorm)
        ax2.set_title("Image Model: GradCAM++", fontsize=11, fontweight='bold')
        ax2.axis('off')
        
        ax3 = fig.add_subplot(3, 4, 3)
        im3 = ax3.imshow(heatmap, cmap='jet')
        ax3.set_title("Attention Heatmap", fontsize=11, fontweight='bold')
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
        
        ax4 = fig.add_subplot(3, 4, 4)
        heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

        cam_blend = img_denorm + heatmap_c

        cam_blend = cam_blend / np.max(cam_blend)

        ax4.imshow(cam_blend)

        ax4.set_title("Image Overlay", fontsize=11, fontweight='bold')
        ax4.axis('off')
        
        # ===== ROW 3: Tabular Model Explanation =====
        # Get Integrated Gradients for tabular model
        ig_attr, tab_class = lab_explainer.explain(tabular_tensor)
        
        # Get top features
        top_features = lab_explainer.get_top_features(ig_attr, top_k=15)
        
        ax5 = fig.add_subplot(3, 4, 5)
        ax5.axis('off')
        ax5.text(0.1, 0.9, "Tabular Model: Integrated Gradients", 
                fontsize=11, fontweight='bold', transform=ax5.transAxes)
        
        # Feature importance bar chart
        ax6 = fig.add_subplot(3, 4, 6)
        colors = ['green' if x > 0 else 'red' for x in top_features['Importance']]
        ax6.barh(top_features['Feature'].values, top_features['Importance'].values, color=colors)
        ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax6.set_xlabel('Importance')
        ax6.set_title("Top 15 Lab Features", fontsize=11, fontweight='bold')
        
        # ===== ROW 4: Predictions Comparison =====
        ax7 = fig.add_subplot(3, 4, 7)
        x_pos = np.arange(len(self.target_cols))
        width = 0.25
        ax7.bar(x_pos - width, img_pred, width, label='Image Model', color='#1f77b4', alpha=0.8)
        ax7.bar(x_pos, tab_pred, width, label='Tabular Model', color='#d62728', alpha=0.8)
        ax7.bar(x_pos + width, fus_pred, width, label='Fusion Model', color='#2ca02c', alpha=0.8)
        ax7.set_xticks(x_pos)
        ax7.set_xticklabels(self.target_cols, rotation=45, ha='right', fontsize=9)
        ax7.set_ylabel('Probability')
        ax7.set_title("Prediction Comparison", fontsize=11, fontweight='bold')
        ax7.legend(fontsize=8)
        ax7.set_ylim(0, 1)
        
        # Fusion model explanation (combined)
        ax8 = fig.add_subplot(3, 4, 8)
        ax8.axis('off')
        ax8.text(0.1, 0.9, "Fusion Model: Combined Explanation", 
                fontsize=11, fontweight='bold', transform=ax8.transAxes)
        ax8.text(0.1, 0.7, f"• Image contributes: {img_pred.max():.2%} max prob", 
                fontsize=10, transform=ax8.transAxes)
        ax8.text(0.1, 0.5, f"• Tabular contributes: {tab_pred.max():.2%} max prob", 
                fontsize=10, transform=ax8.transAxes)
        ax8.text(0.1, 0.3, f"• Fusion prediction: {fus_pred.max():.2%} max prob", 
                fontsize=10, transform=ax8.transAxes)
        
        plt.suptitle(f"XAI Explanation for Sample {sample_idx} (Patient: {patient_id})", 
                    fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        if save_dir:
            save_path = os.path.join(save_dir, f"xai_sample_{sample_idx}.png")
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f" Saved to: {save_path}")
        
        plt.show()
        
        return {
            'patient_id': patient_id,
            'image_prediction': img_pred,
            'tabular_prediction': tab_pred,
            'fusion_prediction': fus_pred,
            'image_attention': heatmap,
            'tabular_importance': ig_attr
        }

# Initialize unified visualizer
xai_visualizer = UnifiedXAIVisualizer(
    image_model, tabular_model, fusion_model,
    image_gradcam, lab_explainer, target_cols
)
print(" Unified XAI Visualizer initialized!")

# ==========================================
# GRADCAM++ ON SAMPLE X-RAY PREDICTIONS
# ==========================================
print("=" * 60)
print("GRADCAM++ EXPLANATIONS FOR X-RAY IMAGES")
print("=" * 60)

# Select sample indices to explain
sample_indices = [0, 5, 10, 15]  # Change these to analyze different samples

# Create output directory
output_dir = "./xai_outputs/gradcam"
os.makedirs(output_dir, exist_ok=True)

for idx in sample_indices:
    print(f"\n--- Processing Sample {idx} ---")
    
    # Get sample data
    row = test_fusion_df.iloc[idx]
    patient_id = str(row['hadm_id']).split('.')[0]
    
    true_labels = [col for col in target_cols if row[col] == 1.0]
    true_label_str = ', '.join(true_labels) if true_labels else 'None'
    
    # Load and preprocess image
    img_path = image_path_dict[patient_id]
    original_img = Image.open(img_path).convert('RGB')
    image_tensor = xai_transform(original_img).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        pred = torch.sigmoid(image_model(image_tensor)).cpu().numpy()[0]
        pred_class = pred.argmax()
    
    # Generate GradCAM++ heatmap
    heatmap, _ = image_gradcam.explain(image_tensor, target_class=pred_class)
    
    # Get original denormalized image
    img_denorm = denormalize_image(image_tensor.cpu().detach()[0])
    img_denorm = img_denorm.permute(1, 2, 0).numpy()
    img_denorm = np.clip(img_denorm, 0, 1)
    
    # Create visualization
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Original image
    axes[0].imshow(original_img)
    axes[0].set_title(f"Original X-Ray\nPatient: {patient_id}", fontsize=11, fontweight='bold')
    axes[0].axis('off')
    
    # Preprocessed image
    axes[1].imshow(img_denorm)
    axes[1].set_title("Preprocessed Input", fontsize=11, fontweight='bold')
    axes[1].axis('off')
    
    # GradCAM heatmap
    im2 = axes[2].imshow(heatmap, cmap='jet')
    axes[2].set_title(f"GradCAM++ Heatmap\nTarget: {target_cols[pred_class]}", fontsize=11, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    
    # Overlay
    axes[3].imshow(img_denorm)
    axes[3].imshow(heatmap, cmap='jet', alpha=0.5)
    axes[3].set_title("Overlay", fontsize=11, fontweight='bold')
    axes[3].axis('off')
    
    plt.suptitle(f"GradCAM++ Explanation - Sample {idx}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, f"gradcam_sample_{idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    
    # Print prediction details
    print(f"   Predictions: {dict(zip(target_cols, pred.round(3)))}")
    print(f"   Top prediction: {target_cols[pred_class]} ({pred[pred_class]:.3f})")
    
    plt.show()

print(f"\n GradCAM++ explanations saved to: {output_dir}")

# ==========================================
# INTEGRATED GRADIENTS ON SAMPLE LAB REPORTS
# ==========================================
print("=" * 60)
print("INTEGRATED GRADIENTS EXPLANATIONS FOR LAB REPORTS")
print("=" * 60)

# Select sample indices to explain
sample_indices = [0, 5, 10, 15]  # Change these to analyze different samples

# Create output directory
output_dir = "./xai_outputs/integrated_gradients"
os.makedirs(output_dir, exist_ok=True)

for idx in sample_indices:
    print(f"\n--- Processing Sample {idx} ---")
    
    # Get sample data
    row = test_fusion_df.iloc[idx]
    patient_id = str(row['hadm_id']).split('.')[0]
    
    true_labels = [col for col in target_cols if row[col] == 1.0]
    true_label_str = ', '.join(true_labels) if true_labels else 'None'
    
    # Get tabular data
    tabular_data = row[all_tabular_features].values.astype(np.float32)
    tabular_tensor = torch.tensor(tabular_data).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        pred = torch.sigmoid(tabular_model(tabular_tensor)).cpu().numpy()[0]
        pred_class = pred.argmax()
    
    # Generate Integrated Gradients attribution
    ig_attr, _ = lab_explainer.explain(tabular_tensor, target_class=pred_class)
    
    # Get top features
    top_features = lab_explainer.get_top_features(ig_attr, top_k=20)
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Feature importance bar chart
    ax1 = axes[0]
    colors = ['green' if x > 0 else 'red' for x in top_features['Importance']]
    bars = ax1.barh(top_features['Feature'].values, top_features['Importance'].values, color=colors)
    ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax1.set_xlabel('Feature Importance (Integrated Gradients)', fontsize=12)
    ax1.set_title(f"Top 20 Lab Features\nTarget: {target_cols[pred_class]}", fontsize=12, fontweight='bold')
    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='green', label='Positive (increases risk)'),
        Patch(facecolor='red', label='Negative (decreases risk)')
    ]
    ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)
    
    # Prediction bar chart
    ax2 = axes[1]
    x_pos = np.arange(len(target_cols))
    colors2 = ['#2ca02c' if p > 0.5 else '#d62728' for p in pred]
    ax2.bar(x_pos, pred, color=colors2, alpha=0.8)
    ax2.axhline(y=0.5, color='black', linestyle='--', linewidth=1, label='Threshold (0.5)')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(target_cols, rotation=45, ha='right', fontsize=10)
    ax2.set_ylabel('Probability', fontsize=12)
    ax2.set_title(f"Prediction Probabilities\nPatient: {patient_id}", fontsize=12, fontweight='bold')
    ax2.set_ylim(0, 1)
    ax2.legend()
    
    plt.suptitle(f"Integrated Gradients Explanation - Sample {idx}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, f"ig_sample_{idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    
    # Print top features
    print(f"   Top 5 positive features:")
    pos_features = top_features[top_features['Importance'] > 0].head(5)
    for _, row_feat in pos_features.iterrows():
        print(f"      - {row_feat['Feature']}: {row_feat['Importance']:.4f}")
    
    print(f"   Top 5 negative features:")
    neg_features = top_features[top_features['Importance'] < 0].tail(5)
    for _, row_feat in neg_features.iterrows():
        print(f"      - {row_feat['Feature']}: {row_feat['Importance']:.4f}")
    
    plt.show()

print(f"\n Integrated Gradients explanations saved to: {output_dir}")

# ==========================================
# COMBINED FUSION EXPLANATIONS
# ==========================================
print("=" * 60)
print("COMBINED FUSION MODEL EXPLANATIONS")
print("=" * 60)

# Select sample indices to explain
sample_indices = [0, 3, 6, 9]  # Change these to analyze different samples

# Create output directory
output_dir = "./xai_outputs/fusion"
os.makedirs(output_dir, exist_ok=True)

for idx in sample_indices:
    print(f"\n--- Processing Sample {idx} ---")
    
    # Get sample data
    row = test_fusion_df.iloc[idx]
    patient_id = str(row['hadm_id']).split('.')[0]
    
    true_labels = [col for col in target_cols if row[col] == 1.0]
    true_label_str = ', '.join(true_labels) if true_labels else 'None'
    
    # Load image
    img_path = image_path_dict[patient_id]
    original_img = Image.open(img_path).convert('RGB')
    image_tensor = xai_transform(original_img).unsqueeze(0).to(device)
    
    # Get tabular data
    tabular_data = row[all_tabular_features].values.astype(np.float32)
    tabular_tensor = torch.tensor(tabular_data).unsqueeze(0).to(device)
    
    # Get predictions from all models
    with torch.no_grad():
        img_pred = torch.sigmoid(image_model(image_tensor)).cpu().numpy()[0]
        tab_pred = torch.sigmoid(tabular_model(tabular_tensor)).cpu().numpy()[0]
        fus_pred = torch.sigmoid(fusion_model(image_tensor, tabular_tensor)).cpu().numpy()[0]
    
    # Get GradCAM for image
    heatmap, img_class = image_gradcam.explain(image_tensor)
    
    # Get Integrated Gradients for tabular
    ig_attr, tab_class = lab_explainer.explain(tabular_tensor)
    top_features = lab_explainer.get_top_features(ig_attr, top_k=10)
    
    # Get original denormalized image
    img_denorm = denormalize_image(image_tensor.cpu().detach()[0])
    img_denorm = img_denorm.permute(1, 2, 0).numpy()
    img_denorm = np.clip(img_denorm, 0, 1)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 12))
    
    # ===== Column 1: Original Image =====
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.imshow(original_img)
    ax1.set_title(f"Original X-Ray\nPatient: {patient_id}", fontsize=11, fontweight='bold')
    ax1.axis('off')
    
    # ===== Column 2: Image Model GradCAM =====
    ax2 = fig.add_subplot(2, 4, 2)
    heatmap_c = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_c = cv2.cvtColor(heatmap_c, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    cam_blend = img_denorm + heatmap_c

    cam_blend = cam_blend / np.max(cam_blend)

    ax2.imshow(cam_blend)

    ax2.set_title("Image Model: GradCAM++", fontsize=11, fontweight='bold')
    ax2.axis('off')
    
    # ===== Column 3: Tabular Model Feature Importance =====
    ax3 = fig.add_subplot(2, 4, 3)
    colors = ['green' if x > 0 else 'red' for x in top_features['Importance']]
    ax3.barh(top_features['Feature'].values, top_features['Importance'].values, color=colors)
    ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Importance')
    ax3.set_title("Tabular: Top 10 Features", fontsize=11, fontweight='bold')
    ax3.grid(axis='x', linestyle='--', alpha=0.3)
    
    # ===== Column 4: Prediction Comparison =====
    ax4 = fig.add_subplot(2, 4, 4)
    x_pos = np.arange(len(target_cols))
    width = 0.25
    ax4.bar(x_pos - width, img_pred, width, label='Image Model', color='#1f77b4', alpha=0.8)
    ax4.bar(x_pos, tab_pred, width, label='Tabular Model', color='#d62728', alpha=0.8)
    ax4.bar(x_pos + width, fus_pred, width, label='Fusion Model', color='#2ca02c', alpha=0.8)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(target_cols, rotation=45, ha='right', fontsize=9)
    ax4.set_ylabel('Probability')
    ax4.set_title("Prediction Comparison", fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.set_ylim(0, 1)
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=0.5)
    
    # ===== Row 2: Detailed Analysis =====
    
    # Image attention heatmap
    ax5 = fig.add_subplot(2, 4, 5)
    im5 = ax5.imshow(heatmap, cmap='jet')
    ax5.set_title("Image Attention Map", fontsize=11, fontweight='bold')
    ax5.axis('off')
    plt.colorbar(im5, ax=ax5, fraction=0.046, pad=0.04)
    
    # Fusion model explanation
    ax6 = fig.add_subplot(2, 4, 6)
    ax6.axis('off')
    
    # Calculate modality contributions
    img_contribution = img_pred.max()
    tab_contribution = tab_pred.max()
    fus_contribution = fus_pred.max()
    
    explanation_text = f"""
    FUSION MODEL ANALYSIS
    =====================
    
    Image Model:
    • Max probability: {img_contribution:.2%}
    • Top prediction: {target_cols[img_pred.argmax()]}
    
    Tabular Model:
    • Max probability: {tab_contribution:.2%}
    • Top prediction: {target_cols[tab_pred.argmax()]}
    
    Fusion Model:
    • Max probability: {fus_contribution:.2%}
    • Top prediction: {target_cols[fus_pred.argmax()]}
    
    Modality Agreement:
    • Image → {target_cols[img_pred.argmax()]}
    • Tabular → {target_cols[tab_pred.argmax()]}
    • Fusion → {target_cols[fus_pred.argmax()]}
    """
    ax6.text(0.05, 0.95, explanation_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Top positive features
    ax7 = fig.add_subplot(2, 4, 7)
    pos_features = top_features[top_features['Importance'] > 0].head(5)
    ax7.barh(pos_features['Feature'].values, pos_features['Importance'].values, color='green')
    ax7.set_xlabel('Positive Importance')
    ax7.set_title("Top 5 Positive Lab Features", fontsize=11, fontweight='bold')
    
    # Top negative features
    ax8 = fig.add_subplot(2, 4, 8)
    neg_features = top_features[top_features['Importance'] < 0].tail(5)
    ax8.barh(neg_features['Feature'].values, neg_features['Importance'].values, color='red')
    ax8.set_xlabel('Negative Importance')
    ax8.set_title("Top 5 Negative Lab Features", fontsize=11, fontweight='bold')
    
    plt.suptitle(f"Combined Fusion Explanation - Sample {idx} (Patient: {patient_id})\nTrue Labels: {true_label_str}", 
                fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    # Save figure
    save_path = os.path.join(output_dir, f"fusion_sample_{idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    
    plt.show()

print(f"\n Combined fusion explanations saved to: {output_dir}")

# ==========================================
# GLOBAL FEATURE IMPORTANCE DASHBOARD
# ==========================================
print("=" * 60)
print("GLOBAL FEATURE IMPORTANCE DASHBOARD")
print("=" * 60)

# Process multiple samples for global importance
n_samples = min(50, len(test_fusion_df))  # Process 50 samples
print(f"Processing {n_samples} samples for global importance...")

all_attributions = []

for idx in range(n_samples):
    if idx % 10 == 0:
        print(f"   Processing sample {idx}/{n_samples}...")
    
    # Get sample data
    row = test_fusion_df.iloc[idx]
    tabular_data = row[all_tabular_features].values.astype(np.float32)
    tabular_tensor = torch.tensor(tabular_data).unsqueeze(0).to(device)
    
    # Get prediction
    with torch.no_grad():
        pred = torch.sigmoid(tabular_model(tabular_tensor)).cpu().numpy()[0]
        pred_class = pred.argmax()
    
    # Generate Integrated Gradients attribution
    ig_attr, _ = lab_explainer.explain(tabular_tensor, target_class=pred_class)
    all_attributions.append(ig_attr)

# Convert to numpy array
all_attributions = np.array(all_attributions)

# Calculate global statistics
mean_importance = np.mean(all_attributions, axis=0)
std_importance = np.std(all_attributions, axis=0)
abs_mean_importance = np.abs(mean_importance)

# Create feature importance dataframe
importance_df = pd.DataFrame({
    'Feature': all_tabular_features,
    'Mean_Importance': mean_importance,
    'Std_Importance': std_importance,
    'Abs_Mean_Importance': abs_mean_importance
})
importance_df = importance_df.sort_values('Abs_Mean_Importance', ascending=False)

# Create output directory
output_dir = "./xai_outputs/global"
os.makedirs(output_dir, exist_ok=True)

# Create global dashboard
fig = plt.figure(figsize=(20, 16))

# ===== Plot 1: Top 30 Features by Mean Absolute Importance =====
ax1 = fig.add_subplot(2, 2, 1)
top_30 = importance_df.head(30)
colors = ['green' if x > 0 else 'red' for x in top_30['Mean_Importance']]
ax1.barh(top_30['Feature'].values, top_30['Mean_Importance'].values, color=colors, alpha=0.8)
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
ax1.set_xlabel('Mean Feature Importance', fontsize=12)
ax1.set_title('Top 30 Most Important Lab Features\n(Global Average)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', linestyle='--', alpha=0.3)

# ===== Plot 2: Feature Importance Distribution =====
ax2 = fig.add_subplot(2, 2, 2)
ax2.hist(mean_importance, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero')
ax2.axvline(x=np.mean(mean_importance), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(mean_importance):.4f}')
ax2.set_xlabel('Mean Feature Importance', fontsize=12)
ax2.set_ylabel('Number of Features', fontsize=12)
ax2.set_title('Distribution of Feature Importances', fontsize=12, fontweight='bold')
ax2.legend()

# ===== Plot 3: Top 15 Positive Features =====
ax3 = fig.add_subplot(2, 2, 3)
pos_features = importance_df[importance_df['Mean_Importance'] > 0].head(15)
ax3.barh(pos_features['Feature'].values, pos_features['Mean_Importance'].values, 
         xerr=pos_features['Std_Importance'].values, color='green', alpha=0.7, capsize=3)
ax3.set_xlabel('Mean Importance (+ Std)', fontsize=12)
ax3.set_title('Top 15 Positive Features\n(Increase Disease Risk)', fontsize=12, fontweight='bold')

# ===== Plot 4: Top 15 Negative Features =====
ax4 = fig.add_subplot(2, 2, 4)
neg_features = importance_df[importance_df['Mean_Importance'] < 0].tail(15)
ax4.barh(neg_features['Feature'].values, neg_features['Mean_Importance'].values,
         xerr=neg_features['Std_Importance'].values, color='red', alpha=0.7, capsize=3)
ax4.set_xlabel('Mean Importance (- Std)', fontsize=12)
ax4.set_title('Top 15 Negative Features\n(Decrease Disease Risk)', fontsize=12, fontweight='bold')

plt.suptitle('Global Feature Importance Dashboard\n(Aggregated across {} samples)'.format(n_samples), 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()

# Save figure
save_path = os.path.join(output_dir, "global_feature_importance.png")
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n Saved to: {save_path}")

plt.show()

# Save feature importance to CSV
csv_path = os.path.join(output_dir, "global_feature_importance.csv")
importance_df.to_csv(csv_path, index=False)
print(f" Feature importance data saved to: {csv_path}")

# Print summary
print("\n" + "=" * 60)
print("TOP 10 MOST IMPORTANT FEATURES (GLOBAL)")
print("=" * 60)
for i, row in importance_df.head(10).iterrows():
    print(f"   {row['Feature']:40s} | {row['Mean_Importance']:+.4f} ± {row['Std_Importance']:.4f}")

# ==========================================
# COMPARE MODEL VS. FUSION ATTENTION MAPS
# ==========================================
print("=" * 60)
print("MODEL VS. FUSION ATTENTION COMPARISON")
print("=" * 60)

# Create GradCAM for fusion model
class FusionGradCAM:
    """GradCAM for fusion model"""
    
    def __init__(self, fusion_model, image_model):
        self.fusion_model = fusion_model
        self.image_model = image_model
        self.fusion_model.eval()
        self.image_model.eval()
    
    def explain(self, image_tensor, tabular_tensor, target_class=None):
        """Generate GradCAM for fusion model"""
        image_tensor = image_tensor.to(device)
        tabular_tensor = tabular_tensor.to(device)
        image_tensor.requires_grad = True
        
        # Get prediction if target not specified
        if target_class is None:
            with torch.no_grad():
                output = self.fusion_model(image_tensor, tabular_tensor)
                target_class = output.argmax(dim=1).item()
        
        # Forward pass
        output = self.fusion_model(image_tensor, tabular_tensor)
        
        # Backward pass
        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Get gradients
        gradients = image_tensor.grad.cpu().detach().numpy()
        
        # Get activations
        with torch.no_grad():
            activations = self.fusion_model.image_model.features(image_tensor)
        
        # Global average pooling of gradients
        weights = np.mean(gradients[0], axis=(1, 2))
        
        # Apply weights to activations
        activations = activations.cpu().detach().numpy()[0]
        heatmap = np.zeros((activations.shape[1], activations.shape[2]))
        
        for i, w in enumerate(weights):
            heatmap += w * activations[i]
        
        # ReLU and normalize
        heatmap = np.maximum(heatmap, 0)
        heatmap = cv2.resize(heatmap, (224, 224))
        heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
        
        return heatmap, target_class

# Initialize fusion GradCAM
fusion_gradcam = FusionGradCAM(fusion_model, image_model)

# Select sample indices
sample_indices = [0, 5, 10]

# Create output directory
output_dir = "./xai_outputs/attention_comparison"
os.makedirs(output_dir, exist_ok=True)

for idx in sample_indices:
    print(f"\n--- Processing Sample {idx} ---")
    
    # Get sample data
    row = test_fusion_df.iloc[idx]
    patient_id = str(row['hadm_id']).split('.')[0]
    
    true_labels = [col for col in target_cols if row[col] == 1.0]
    true_label_str = ', '.join(true_labels) if true_labels else 'None'
    
    # Load image
    img_path = image_path_dict[patient_id]
    original_img = Image.open(img_path).convert('RGB')
    image_tensor = xai_transform(original_img).unsqueeze(0).to(device)
    
    # Get tabular data
    tabular_data = row[all_tabular_features].values.astype(np.float32)
    tabular_tensor = torch.tensor(tabular_data).unsqueeze(0).to(device)
    
    # Get predictions
    with torch.no_grad():
        img_pred = torch.sigmoid(image_model(image_tensor)).cpu().numpy()[0]
        fus_pred = torch.sigmoid(fusion_model(image_tensor, tabular_tensor)).cpu().numpy()[0]
    
    # Get GradCAM for standalone image model
    standalone_heatmap, _ = image_gradcam.explain(image_tensor)
    
    # Get GradCAM for fusion model
    fusion_heatmap, _ = fusion_gradcam.explain(image_tensor, tabular_tensor)
    
    # Get original denormalized image
    img_denorm = denormalize_image(image_tensor.cpu().detach()[0])
    img_denorm = img_denorm.permute(1, 2, 0).numpy()
    img_denorm = np.clip(img_denorm, 0, 1)
    
    # Create comparison visualization
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Standalone Image Model
    axes[0, 0].imshow(original_img)
    axes[0, 0].set_title(f"Original X-Ray\nPatient: {patient_id}", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(img_denorm)
    axes[0, 1].set_title("Preprocessed Input", fontsize=11, fontweight='bold')
    axes[0, 0].axis('off')
    
    im2 = axes[0, 2].imshow(standalone_heatmap, cmap='jet')
    axes[0, 2].set_title("Standalone Image Model\nGradCAM++", fontsize=11, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)
    
    axes[0, 3].imshow(img_denorm)
    axes[0, 3].imshow(standalone_heatmap, cmap='jet', alpha=0.5)
    axes[0, 3].set_title("Standalone Overlay", fontsize=11, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 2: Fusion Model
    axes[1, 0].imshow(original_img)
    axes[1, 0].set_title("Original X-Ray", fontsize=11, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Show key tabular features
    top_tab_features = pd.to_numeric(row[all_tabular_features], errors='coerce').nlargest(3)
    tab_text = "Top 3 Lab Values:\n"
    for feat, val in top_tab_features.items():
        tab_text += f"• {feat[:20]}: {val:.2f}\n"
    axes[1, 1].text(0.1, 0.5, tab_text, transform=axes[1, 1].transAxes, fontsize=10,
                   verticalalignment='center', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    axes[1, 1].set_title("Key Lab Features", fontsize=11, fontweight='bold')
    axes[1, 1].axis('off')
    
    im3 = axes[1, 2].imshow(fusion_heatmap, cmap='jet')
    axes[1, 2].set_title("Fusion Model\nGradCAM++", fontsize=11, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im3, ax=axes[1, 2], fraction=0.046, pad=0.04)
    
    axes[1, 3].imshow(img_denorm)
    axes[1, 3].imshow(fusion_heatmap, cmap='jet', alpha=0.5)
    axes[1, 3].set_title("Fusion Overlay", fontsize=11, fontweight='bold')
    axes[1, 3].axis('off')
    
    # Add prediction comparison
    fig.text(0.5, 0.02, 
             f"Standalone Image: {target_cols[img_pred.argmax()]} ({img_pred.max():.2f}) | "
             f"Fusion Model: {target_cols[fus_pred.argmax()]} ({fus_pred.max():.2f})",
             ha='center', fontsize=12, fontweight='bold')
    
    plt.suptitle(f"Attention Map Comparison - Sample {idx}\nTrue Labels: {true_label_str}", fontsize=14, fontweight='bold', y=1.05)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure
    save_path = os.path.join(output_dir, f"attention_comparison_{idx}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"   Saved: {save_path}")
    
    # Calculate attention difference
    diff = fusion_heatmap - standalone_heatmap
    print(f"   Attention difference (mean): {diff.mean():.4f}")
    print(f"   Attention difference (std): {diff.std():.4f}")
    
    plt.show()

print(f"\n Attention comparison saved to: {output_dir}")