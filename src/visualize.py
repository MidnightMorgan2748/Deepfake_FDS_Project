"""
Visualization suite for DeepFake Detection:
- Real vs Deepfake sample image previews
- Laplacian noise residual maps
- Fractal Dimension & Texture Complexity histograms
- 2D Fourier power spectrum comparisons
- 5-Fold Cross-Validated ROC & Precision-Recall curves
- Confusion matrix heatmaps
- Performance comparison benchmark charts
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless / script execution
import matplotlib.pyplot as plt

# Styling configuration
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
FONT_TITLE = {'fontsize': 13, 'fontweight': 'bold'}
FONT_LABEL = {'fontsize': 11, 'fontweight': 'medium'}

def plot_sample_comparison(real_img_paths, fake_img_paths, output_path, n=4):
    """
    Plots a side-by-side grid comparing real photographs and deepfake images.
    """
    n = min(n, len(real_img_paths), len(fake_img_paths))
    fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7))
    
    for i in range(n):
        # Real images (top row)
        img_r = cv2.imread(real_img_paths[i])
        if img_r is not None:
            img_r = cv2.cvtColor(img_r, cv2.COLOR_BGR2RGB)
            axes[0, i].imshow(img_r)
        axes[0, i].set_title(f"Real Image #{i+1}", **FONT_TITLE)
        axes[0, i].axis("off")
        
        # Fake images (bottom row)
        img_f = cv2.imread(fake_img_paths[i])
        if img_f is not None:
            img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2RGB)
            axes[1, i].imshow(img_f)
        axes[1, i].set_title(f"AI-Generated #{i+1}", **FONT_TITLE)
        axes[1, i].axis("off")
        
    plt.suptitle("Visual Comparison: Real Photographs vs. AI-Generated Deepfakes", fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def plot_residual_analysis(real_img_path, fake_img_path, output_path):
    """
    Generates a 4-panel forensic analysis figure matching Figure 1 of IEEE ICASSP 2025:
    - Original Real image & its Laplacian noise residual
    - Original AI image & its Laplacian noise residual
    """
    from src.features import compute_laplacian_residual
    
    img_r = cv2.imread(real_img_path, cv2.IMREAD_GRAYSCALE)
    img_f = cv2.imread(fake_img_path, cv2.IMREAD_GRAYSCALE)
    if img_r.shape != (512, 512):
        img_r = cv2.resize(img_r, (512, 512))
    if img_f.shape != (512, 512):
        img_f = cv2.resize(img_f, (512, 512))
        
    res_r = compute_laplacian_residual(img_r)
    res_f = compute_laplacian_residual(img_f)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    # Real Image & Residual
    axes[0, 0].imshow(img_r, cmap='gray')
    axes[0, 0].set_title("Real Photograph (Luminance)", **FONT_TITLE)
    axes[0, 0].axis("off")
    
    im_rr = axes[0, 1].imshow(res_r, cmap='seismic', vmin=-30, vmax=30)
    axes[0, 1].set_title("Real: Laplacian Noise Residual", **FONT_TITLE)
    axes[0, 1].axis("off")
    fig.colorbar(im_rr, ax=axes[0, 1], fraction=0.046, pad=0.04)
    
    # Fake Image & Residual
    axes[1, 0].imshow(img_f, cmap='gray')
    axes[1, 0].set_title("AI Deepfake (Luminance)", **FONT_TITLE)
    axes[1, 0].axis("off")
    
    im_rf = axes[1, 1].imshow(res_f, cmap='seismic', vmin=-30, vmax=30)
    axes[1, 1].set_title("AI Deepfake: Laplacian Residual", **FONT_TITLE)
    axes[1, 1].axis("off")
    fig.colorbar(im_rf, ax=axes[1, 1], fraction=0.046, pad=0.04)
    
    plt.suptitle("Forensics Analysis of Residual Noise Texture (ICASSP 2025)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def plot_feature_distributions(fd_real, fd_fake, tc_real, tc_fake, output_path):
    """
    Plots distribution histograms of Fractal Dimension (FD) and Texture Complexity (TC)
    demonstrating the separable forensic boundaries between real and AI images.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    
    # FD distribution
    axes[0].hist(fd_real, bins=25, alpha=0.65, label='Real Photos', color='#2b5c8f', edgecolor='black')
    axes[0].hist(fd_fake, bins=25, alpha=0.65, label='AI Deepfakes', color='#d95f02', edgecolor='black')
    axes[0].set_title("Fractal Dimension (FD) Distribution", **FONT_TITLE)
    axes[0].set_xlabel("Estimated Fractal Dimension (Slope)", **FONT_LABEL)
    axes[0].set_ylabel("Frequency", **FONT_LABEL)
    axes[0].legend(frameon=True)
    axes[0].grid(True, linestyle='--', alpha=0.6)
    
    # TC distribution
    axes[1].hist(tc_real, bins=25, alpha=0.65, label='Real Photos', color='#2b5c8f', edgecolor='black')
    axes[1].hist(tc_fake, bins=25, alpha=0.65, label='AI Deepfakes', color='#d95f02', edgecolor='black')
    axes[1].set_title("Texture Complexity (TC) Distribution", **FONT_TITLE)
    axes[1].set_xlabel("Mean Block Texture Complexity (TC)", **FONT_LABEL)
    axes[1].set_ylabel("Frequency", **FONT_LABEL)
    axes[1].legend(frameon=True)
    axes[1].grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle("Exploratory Data Analysis: Forensic Feature Distributions", fontsize=14, fontweight='bold')
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def plot_spectral_comparison(real_residuals_dir, fake_residuals_dir, output_path, n_samples=20):
    """
    Plots the average 2D Fourier power spectrum and 1D Azimuthal radial profile
    highlighting the novel spectral frequency forensic anomalies.
    """
    from src.features import compute_spectral_residual_features
    
    real_files = [os.path.join(real_residuals_dir, f) for f in os.listdir(real_residuals_dir) if f.endswith('.npy')][:n_samples]
    fake_files = [os.path.join(fake_residuals_dir, f) for f in os.listdir(fake_residuals_dir) if f.endswith('.npy')][:n_samples]
    
    real_specs = [compute_spectral_residual_features(np.load(f)) for f in real_files]
    fake_specs = [compute_spectral_residual_features(np.load(f)) for f in fake_files]
    
    avg_real_profile = np.mean([s[:32] for s in real_specs], axis=0)
    avg_fake_profile = np.mean([s[:32] for s in fake_specs], axis=0)
    
    plt.figure(figsize=(9, 5))
    x_bins = np.arange(len(avg_real_profile))
    plt.plot(x_bins, avg_real_profile, label="Real Images (Natural Noise)", color='#2b5c8f', lw=2.5, marker='o')
    plt.plot(x_bins, avg_fake_profile, label="AI Deepfakes (Neural Noise)", color='#d95f02', lw=2.5, marker='s')
    
    plt.title("Novel Forensic Contribution: Radial Spectral Residual Profile", **FONT_TITLE)
    plt.xlabel("Spatial Frequency Radial Bins (Low -> High Frequency)", **FONT_LABEL)
    plt.ylabel("Mean Log Spectral Power", **FONT_LABEL)
    plt.legend(frameon=True)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def plot_roc_curves(benchmark_results, output_path):
    """
    Plots cross-validated ROC curves for all evaluated models, matching Figure 2 of the paper.
    """
    plt.figure(figsize=(8, 7))
    plt.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Chance (AUC = 0.50)')
    
    colors = ['#2b5c8f', '#7570b3', '#1b9e77', '#e7298a', '#d95f02']
    
    for idx, (name, res) in enumerate(benchmark_results.items()):
        roc = res.get("roc")
        if roc is not None:
            auc = res["summary"]["roc_auc"]
            color = colors[idx % len(colors)]
            plt.plot(roc["fpr"], roc["tpr"], lw=2.5, color=color, label=f"{name} (AUC = {auc:.3f})")
            
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.05])
    plt.xlabel("False Positive Rate (FPR)", **FONT_LABEL)
    plt.ylabel("True Positive Rate (TPR / Recall)", **FONT_LABEL)
    plt.title("5-Fold Cross-Validated ROC Performance", **FONT_TITLE)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def plot_confusion_matrix(cm, model_name, output_path):
    """
    Plots a formatted confusion matrix heatmap.
    """
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix: {model_name}", **FONT_TITLE)
    plt.colorbar(fraction=0.046, pad=0.04)
    
    classes = ['Real (0)', 'Fake (1)']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=10)
    plt.yticks(tick_marks, classes, fontsize=10)
    
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            text = f"{val:.1f}" if isinstance(val, float) else f"{val}"
            plt.text(j, i, text, horizontalalignment="center",
                     color="white" if val > thresh else "black",
                     fontsize=12, fontweight='bold')
                     
    plt.ylabel("Actual Label", **FONT_LABEL)
    plt.xlabel("Predicted Label", **FONT_LABEL)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path

def plot_benchmark_bar_chart(benchmark_results, output_path):
    """
    Plots a multi-metric bar chart comparing Balanced Accuracy, ROC-AUC, Real Recall, and Fake Recall.
    """
    models = list(benchmark_results.keys())
    b_accs = [benchmark_results[m]["summary"]["balanced_accuracy"] * 100 for m in models]
    aucs = [benchmark_results[m]["summary"]["roc_auc"] * 100 for m in models]
    f1s = [benchmark_results[m]["summary"]["macro_f1"] * 100 for m in models]
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(11, 6))
    rects1 = ax.bar(x - width, b_accs, width, label='Balanced Accuracy (%)', color='#2b5c8f')
    rects2 = ax.bar(x, aucs, width, label='ROC-AUC (x100)', color='#1b9e77')
    rects3 = ax.bar(x + width, f1s, width, label='Macro F1 (%)', color='#d95f02')
    
    ax.set_ylabel('Score (%)', **FONT_LABEL)
    ax.set_title('Cross-Validation Performance Comparison Across Architectures', **FONT_TITLE)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right', fontsize=10)
    ax.legend(frameon=True)
    ax.set_ylim([0, 110])
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}%',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8, fontweight='bold')
                        
    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    return output_path
