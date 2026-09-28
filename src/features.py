"""
Feature extraction engine for DeepFake Detection based on:
1. Laplacian Noise Residuals (IEEE ICASSP 2025)
2. Multi-scale Fractal Dimension (Box-Counting)
3. Local Block-wise Texture Complexity (TC)
4. Novelty 1: High-Frequency Residual Spectral Power (FFT Azimuthal Profile)
5. Novelty 2: Spatial Co-occurrence & Statistical Texture Moments
6. Baseline: Scaled Grayscale Flattened Vectors
"""

import os
import cv2
import numpy as np
from tqdm import tqdm

# Discrete 3x3 Laplacian differential operator kernel from Equation (1) of the paper
LAPLACIAN_KERNEL = np.array([
    [0,  1,  0],
    [1, -4,  1],
    [0,  1,  0]
], dtype=np.float32)

def compute_laplacian_residual(image):
    """
    Extracts high-frequency noise residual by convolving the image with the discrete Laplacian kernel.
    Input: Grayscale image (uint8 or float32, typically 512x512).
    Output: Residual image R (float32).
    """
    img_float = image.astype(np.float32)
    residual = cv2.filter2D(img_float, -1, LAPLACIAN_KERNEL)
    return residual

def extract_and_save_residuals(src_folder, dst_folder):
    """
    Computes Laplacian residuals for all images in src_folder and saves them as .npy files in dst_folder.
    """
    os.makedirs(dst_folder, exist_ok=True)
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    files = [f for f in os.listdir(src_folder) if f.lower().endswith(valid_exts)]
    
    saved_count = 0
    for filename in tqdm(files, desc=f"Residuals for {os.path.basename(src_folder)}"):
        base_name, _ = os.path.splitext(filename)
        out_path = os.path.join(dst_folder, f"{base_name}.npy")
        
        if os.path.exists(out_path):
            saved_count += 1
            continue
            
        img_path = os.path.join(src_folder, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        residual = compute_laplacian_residual(img)
        np.save(out_path, residual)
        saved_count += 1
        
    return saved_count

def compute_box_count(residual, threshold=0.01, num_scales=44):
    """
    Box-counting algorithm to estimate Fractal Dimension (FD) from noise residuals.
    Implements Equation (2) from the IEEE ICASSP 2025 paper:
    FD = lim_{D -> 0} -log N(D) / log(D)
    
    Uses 44 logarithmic scales between 2 and 112 as detailed in Section III-B of the paper.
    Returns:
    - fd_slope: Estimated scalar fractal dimension.
    - box_counts: 44-dimensional vector of log-counts (used for Nyström / direct features).
    """
    Z = np.abs(residual)
    h, w = Z.shape
    
    # Paper specification: 44 values of D with 2 < D < 112 on logarithmic scale
    # To tile neatly or resize, we choose scales D
    D_values = np.unique(np.round(np.logspace(np.log10(3), np.log10(110), num=num_scales)).astype(int))
    
    counts = []
    log_inv_D = []
    
    for D in D_values:
        # Resize to grid size (h // D, w // D) using max-pooling or nearest interpolation
        grid_h, grid_w = max(1, h // D), max(1, w // D)
        # Using block-wise maximum over tiles of size D
        # A box contains intensity > threshold if its max exceeds threshold
        sub_sampled = cv2.resize(Z, (grid_w, grid_h), interpolation=cv2.INTER_AREA)
        count = np.sum(sub_sampled > threshold)
        counts.append(max(1, count))
        log_inv_D.append(np.log(1.0 / D))
        
    counts = np.array(counts, dtype=np.float64)
    log_inv_D = np.array(log_inv_D, dtype=np.float64)
    log_counts = np.log(counts)
    
    # Linear regression slope: log(N(D)) vs log(1/D)
    coeffs = np.polyfit(-np.log(D_values), log_counts, 1)
    fd_slope = float(coeffs[0])
    
    # Pad or interpolate counts to ensure exactly num_scales features
    if len(counts) != num_scales:
        interp_counts = np.interp(
            np.linspace(0, 1, num_scales),
            np.linspace(0, 1, len(log_counts)),
            log_counts
        )
    else:
        interp_counts = log_counts
        
    return fd_slope, interp_counts

def compute_texture_complexity(residual, D=16):
    """
    Computes local Texture Complexity (TC) from Equations (3) and (4) in Section III-B:
    t(R) = 1 - sum_{m,n} ( 2^{-r_{m,n}} * r_{m,n} / D^2 )
    TC = log( t(R) / (1 - t(R)) ) + 4
    
    For a 512x512 image and D=16, yields 32x32 = 1024 local block complexity values.
    Returns:
    - tc_map: (32, 32) spatial complexity map.
    - tc_vector: (1024,) flattened spatial feature vector.
    - tc_stats: Statistical summary (mean, std, median, skewness, min, max, 25th, 75th percentiles).
    """
    R = np.abs(residual).astype(np.float64)
    h, w = R.shape
    
    blocks_h = h // D
    blocks_w = w // D
    
    tc_map = np.zeros((blocks_h, blocks_w), dtype=np.float64)
    
    D2 = float(D * D)
    for bi in range(blocks_h):
        for bj in range(blocks_w):
            block = R[bi * D : (bi + 1) * D, bj * D : (bj + 1) * D]
            # Equation (4): t(R) = 1 - sum(2^(-r) * r / D^2)
            # Safe computation: r >= 0
            val = (2.0 ** (-block)) * block
            t = 1.0 - (np.sum(val) / D2)
            
            # Clamp t to prevent numerical overflow in log(t / (1 - t))
            t = np.clip(t, 1e-6, 1.0 - 1e-6)
            # Equation (3): TC = log(t / (1 - t)) + 4
            tc = np.log(t / (1.0 - t)) + 4.0
            tc_map[bi, bj] = tc
            
    tc_vector = tc_map.flatten()
    
    # Statistical moments
    mean_val = float(np.mean(tc_vector))
    std_val = float(np.std(tc_vector))
    p25 = float(np.percentile(tc_vector, 25))
    p50 = float(np.median(tc_vector))
    p75 = float(np.percentile(tc_vector, 75))
    min_val = float(np.min(tc_vector))
    max_val = float(np.max(tc_vector))
    skew_val = float(np.mean(((tc_vector - mean_val) / (std_val + 1e-8)) ** 3))
    
    tc_stats = np.array([mean_val, std_val, p25, p50, p75, min_val, max_val, skew_val], dtype=np.float64)
    return tc_map, tc_vector, tc_stats

def compute_spectral_residual_features(residual, num_radial_bins=32):
    """
    Novel Contribution 1: Residual Frequency-Domain Forensics.
    Computes 2D Fourier power spectrum of Laplacian residual and extracts azimuthal radial power profile.
    Generative AI synthesis engines (Diffusion/GANs) introduce periodic grid harmonics and anomalous
    high-frequency spectral energy roll-off absent in physical sensor noise.
    """
    R = residual.astype(np.float64)
    h, w = R.shape
    
    # 2D FFT and center shift
    F = np.fft.fft2(R)
    Fshift = np.fft.fftshift(F)
    magnitude_spectrum = np.abs(Fshift) ** 2
    log_spectrum = np.log1p(magnitude_spectrum)
    
    # Radial Azimuthal Average
    center_y, center_x = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - center_x, y - center_y)
    max_radius = min(center_y, center_x)
    
    # Bin radial frequencies
    bin_edges = np.linspace(0, max_radius, num_radial_bins + 1)
    radial_profile = []
    
    for i in range(num_radial_bins):
        mask = (r >= bin_edges[i]) & (r < bin_edges[i + 1])
        if np.any(mask):
            radial_profile.append(float(np.mean(log_spectrum[mask])))
        else:
            radial_profile.append(0.0)
            
    radial_profile = np.array(radial_profile, dtype=np.float64)
    
    # High-to-low frequency ratio & spectral energy entropy
    low_freq_energy = np.mean(radial_profile[:num_radial_bins // 4]) + 1e-8
    high_freq_energy = np.mean(radial_profile[3 * num_radial_bins // 4:]) + 1e-8
    hf_lf_ratio = high_freq_energy / low_freq_energy
    
    spectral_features = np.hstack([radial_profile, [hf_lf_ratio]])
    return spectral_features

def compute_residual_texture_stats(residual):
    """
    Novel Contribution 2: Higher-order statistical and spatial moments on Laplacian residual:
    Variance, Kurtosis, Skewness, Mean Absolute Deviation, and Gradient Energy.
    """
    R = residual.astype(np.float64)
    abs_R = np.abs(R)
    
    mean = np.mean(abs_R)
    var = np.var(abs_R)
    std = np.std(abs_R) + 1e-8
    skew = np.mean(((abs_R - mean) / std) ** 3)
    kurt = np.mean(((abs_R - mean) / std) ** 4) - 3.0  # Excess kurtosis
    mad = np.mean(np.abs(abs_R - mean))
    
    # Horizontal and vertical differential gradients
    diff_h = np.mean(np.abs(R[:, 1:] - R[:, :-1]))
    diff_v = np.mean(np.abs(R[1:, :] - R[:-1, :]))
    
    return np.array([mean, var, std, skew, kurt, mad, diff_h, diff_v], dtype=np.float64)

def extract_all_features_for_image(image_path_or_residual):
    """
    Extracts all feature sets for a single sample:
    1. Paper Features: FD (slope + 44 scales) + TC (1024 vector + 8 stats)
    2. Novel Features: Spectral Profile (33) + Residual Moments (8)
    """
    if isinstance(image_path_or_residual, str):
        if image_path_or_residual.endswith('.npy'):
            res = np.load(image_path_or_residual)
        else:
            img = cv2.imread(image_path_or_residual, cv2.IMREAD_GRAYSCALE)
            if img.shape != (512, 512):
                img = cv2.resize(img, (512, 512))
            res = compute_laplacian_residual(img)
    else:
        res = image_path_or_residual
        
    fd_slope, fd_counts = compute_box_count(res)
    tc_map, tc_vector, tc_stats = compute_texture_complexity(res)
    spectral = compute_spectral_residual_features(res)
    moments = compute_residual_texture_stats(res)
    
    # Paper-style compact feature: FD slope (1) + TC mean (1)
    paper_compact = np.array([fd_slope, tc_stats[0]], dtype=np.float64)
    # Paper-style full feature: FD counts (44) + TC vector (1024)
    paper_full = np.hstack([fd_counts, tc_vector])
    # Paper-style hybrid: FD counts (44) + TC stats (8)
    paper_stats = np.hstack([fd_counts, tc_stats])
    # Novel ensemble feature: Paper stats (52) + Spectral (33) + Moments (8) = 93 features
    novel_feature = np.hstack([paper_stats, spectral, moments])
    
    return {
        "fd_slope": fd_slope,
        "fd_counts": fd_counts,
        "tc_vector": tc_vector,
        "tc_stats": tc_stats,
        "spectral": spectral,
        "moments": moments,
        "paper_compact": paper_compact,
        "paper_full": paper_full,
        "paper_stats": paper_stats,
        "novel_feature": novel_feature
    }

def build_feature_matrices(residuals_real_dir, residuals_fake_dir, max_samples=None):
    """
    Builds structured feature matrices X and labels y for all classes:
    Returns:
    - datasets: dictionary containing feature matrices for:
      - 'paper_compact': (N, 2)
      - 'paper_full': (N, 1068)
      - 'paper_stats': (N, 52)
      - 'novel_feature': (N, 93)
    - y: binary labels (0 = Real, 1 = Fake)
    """
    real_files = sorted([os.path.join(residuals_real_dir, f) for f in os.listdir(residuals_real_dir) if f.endswith('.npy')])
    fake_files = sorted([os.path.join(residuals_fake_dir, f) for f in os.listdir(residuals_fake_dir) if f.endswith('.npy')])
    
    if max_samples:
        real_files = real_files[:max_samples]
        fake_files = fake_files[:max_samples]
        
    all_files = real_files + fake_files
    y = np.array([0] * len(real_files) + [1] * len(fake_files), dtype=int)
    
    p_compact, p_full, p_stats, n_feat = [], [], [], []
    
    for path in tqdm(all_files, desc="Extracting Feature Matrices"):
        res = np.load(path)
        feats = extract_all_features_for_image(res)
        p_compact.append(feats["paper_compact"])
        p_full.append(feats["paper_full"])
        p_stats.append(feats["paper_stats"])
        n_feat.append(feats["novel_feature"])
        
    return {
        "paper_compact": np.array(p_compact),
        "paper_full": np.array(p_full),
        "paper_stats": np.array(p_stats),
        "novel_feature": np.array(n_feat)
    }, y
