"""
Dataset loading, verification, preprocessing, and download utilities for DeepFake Detection.
"""

import os
import glob
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image

def get_image_paths(folder_path):
    """Retrieve all image file paths from a directory."""
    if not os.path.exists(folder_path):
        return []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    paths = []
    for root, _, files in os.walk(folder_path):
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                paths.append(os.path.join(root, f))
    return sorted(paths)

def preprocess_image(image_path, target_size=(512, 512)):
    """
    Load an image, extract the luminance/grayscale channel,
    and resize to the target dimensions (512x512).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Fallback to PIL in case OpenCV fails on certain encodings
        try:
            with Image.open(image_path) as pil_img:
                img = np.array(pil_img.convert("L"))
        except Exception:
            return None
    if img.shape != target_size:
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA if (img.shape[0] > target_size[0]) else cv2.INTER_CUBIC)
    return img

def preprocess_and_save_folder(src_folder, dst_folder, target_size=(512, 512), max_images=None):
    """
    Preprocess all images in src_folder and save resized 512x512 grayscale images to dst_folder.
    """
    os.makedirs(dst_folder, exist_ok=True)
    image_paths = get_image_paths(src_folder)
    if max_images:
        image_paths = image_paths[:max_images]
    
    saved_count = 0
    for path in tqdm(image_paths, desc=f"Preprocessing {os.path.basename(src_folder)}"):
        filename = os.path.basename(path)
        base_name, _ = os.path.splitext(filename)
        out_filename = f"{base_name}.png"
        out_path = os.path.join(dst_folder, out_filename)
        
        # Skip if already exists
        if os.path.exists(out_path):
            saved_count += 1
            continue
            
        img = preprocess_image(path, target_size=target_size)
        if img is not None:
            cv2.imwrite(out_path, img)
            saved_count += 1
            
    return saved_count

def check_dataset_status(base_data_dir):
    """
    Returns a dictionary of image and residual counts across all pipeline stages.
    """
    raw_real = len(get_image_paths(os.path.join(base_data_dir, "raw", "real")))
    raw_fake = len(get_image_paths(os.path.join(base_data_dir, "raw", "fake")))
    preproc_real = len(get_image_paths(os.path.join(base_data_dir, "preprocessed", "real")))
    preproc_fake = len(get_image_paths(os.path.join(base_data_dir, "preprocessed", "fake")))
    
    res_real = len(glob.glob(os.path.join(base_data_dir, "residuals", "real", "*.npy")))
    res_fake = len(glob.glob(os.path.join(base_data_dir, "residuals", "fake", "*.npy")))
    
    status = {
        "raw_real": raw_real,
        "raw_fake": raw_fake,
        "raw_total": raw_real + raw_fake,
        "preprocessed_real": preproc_real,
        "preprocessed_fake": preproc_fake,
        "residuals_real": res_real,
        "residuals_fake": res_fake,
    }
    return status

def generate_benchmark_synthetic_samples(real_dir, fake_dir, count=50, size=(512, 512)):
    """
    Generates synthetic benchmark samples adhering to forensic properties described in IEEE ICASSP 2025:
    - Real images exhibit high-entropy sensor noise (Poisson-Gaussian noise, PRNU fingerprint, natural textures).
    - AI images exhibit characteristic latent-space denoising artifacts: oversmoothed homogeneous regions,
      checkerboard upsampling patterns, and low high-frequency residual entropy.
    Used for unit testing, offline development, and smoke validation.
    """
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    np.random.seed(42)
    h, w = size
    
    for i in range(count):
        real_file = os.path.join(real_dir, f"benchmark_real_{i:04d}.png")
        fake_file = os.path.join(fake_dir, f"benchmark_fake_{i:04d}.png")
        
        # 1. Generate realistic camera photograph simulation:
        # Base natural gradient / illumination + natural textures + camera sensor noise
        x = np.linspace(0, 4 * np.pi, w)
        y = np.linspace(0, 4 * np.pi, h)
        xx, yy = np.meshgrid(x, y)
        base_signal = (np.sin(xx * (0.5 + 0.1 * i)) * np.cos(yy * (0.4 + 0.05 * i)) + 1.0) * 100.0
        # Add high-frequency sensor noise (Poisson-Gaussian)
        sensor_noise = np.random.normal(loc=0, scale=12.0 + (i % 5), size=(h, w))
        prnu_pattern = np.random.uniform(-3, 3, size=(h, w))
        real_img = np.clip(base_signal + sensor_noise + prnu_pattern, 0, 255).astype(np.uint8)
        cv2.imwrite(real_file, real_img)
        
        # 2. Generate AI-generated image simulation:
        # Generative diffusion models smooth homogeneous areas and introduce periodic latent upsampling patterns
        # Base signal with latent smoothing
        smoothed_signal = cv2.GaussianBlur(base_signal, (15, 15), 5.0)
        # Add subtle periodic checkerboard artifact (2x2 / 4x4 upsampling deconvolution grid)
        grid_x, grid_y = np.indices((h, w))
        checkerboard = ((grid_x // 4 + grid_y // 4) % 2) * 4.0 - 2.0
        # Reduced random sensor noise due to neural synthesis
        ai_noise = np.random.normal(loc=0, scale=2.5, size=(h, w))
        fake_img = np.clip(smoothed_signal + checkerboard + ai_noise, 0, 255).astype(np.uint8)
        cv2.imwrite(fake_file, fake_img)
        
    return count
