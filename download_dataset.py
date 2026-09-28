"""
Dataset downloader and organizer utility for Deepfake Image Detection.
Supports:
1. Direct automated download via gdown from Google Drive links.
2. Unzipping manually downloaded archives from Google Drive into raw/real and raw/fake.
3. Generating benchmark verification samples for offline testing.
"""

import os
import sys
import shutil
import zipfile
import argparse

DRIVE_LINKS = {
    "fake": "https://drive.google.com/drive/folders/13OqZH_uwD9IWhoWF5h0l-dTKFyRfII7A",
    "real": "https://drive.google.com/drive/folders/1bQahWnXPid84b7MjE-9_mPyHHhzfQXH3"
}

def download_via_gdown(base_dir):
    """Downloads files directly from Google Drive using gdown."""
    try:
        import gdown
    except ImportError:
        print("[!] gdown is not installed. Please run: pip install gdown")
        return False
        
    real_raw = os.path.join(base_dir, "data", "raw", "real")
    fake_raw = os.path.join(base_dir, "data", "raw", "fake")
    os.makedirs(real_raw, exist_ok=True)
    os.makedirs(fake_raw, exist_ok=True)
    
    print("\n" + "="*70)
    print("DOWNLOADING GOOGLE DRIVE DATASETS VIA GDOWN")
    print("="*70)
    print(f"[*] Downloading Fake Images to: {fake_raw}")
    print(f"[*] Link: {DRIVE_LINKS['fake']}")
    try:
        gdown.download_folder(DRIVE_LINKS['fake'], output=fake_raw, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"[!] Warning during fake images download: {e}")
        
    print(f"\n[*] Downloading Real Images to: {real_raw}")
    print(f"[*] Link: {DRIVE_LINKS['real']}")
    try:
        gdown.download_folder(DRIVE_LINKS['real'], output=real_raw, quiet=False, use_cookies=False)
    except Exception as e:
        print(f"[!] Warning during real images download: {e}")
        
    return True

def extract_zips_to_raw(real_zip_path=None, fake_zip_path=None, base_dir="."):
    """Extracts downloaded zip files directly into raw/real and raw/fake."""
    real_raw = os.path.join(base_dir, "data", "raw", "real")
    fake_raw = os.path.join(base_dir, "data", "raw", "fake")
    os.makedirs(real_raw, exist_ok=True)
    os.makedirs(fake_raw, exist_ok=True)
    
    if real_zip_path and os.path.exists(real_zip_path):
        print(f"[*] Extracting real images from {real_zip_path}...")
        with zipfile.ZipFile(real_zip_path, 'r') as zip_ref:
            zip_ref.extractall(real_raw)
        print(f"[OK] Extracted into {real_raw}")
        
    if fake_zip_path and os.path.exists(fake_zip_path):
        print(f"[*] Extracting fake images from {fake_zip_path}...")
        with zipfile.ZipFile(fake_zip_path, 'r') as zip_ref:
            zip_ref.extractall(fake_raw)
        print(f"[OK] Extracted into {fake_raw}")

def copy_sample_images(info_img_dir, base_dir="."):
    """Copies sample images from informational directory for sanity verification."""
    real_raw = os.path.join(base_dir, "data", "raw", "real")
    fake_raw = os.path.join(base_dir, "data", "raw", "fake")
    os.makedirs(real_raw, exist_ok=True)
    os.makedirs(fake_raw, exist_ok=True)
    
    if os.path.exists(info_img_dir):
        files = [os.path.join(info_img_dir, f) for f in os.listdir(info_img_dir) if f.lower().endswith(('.png', '.jpg'))]
        for idx, f in enumerate(files):
            dst = os.path.join(fake_raw if idx % 2 == 1 else real_raw, os.path.basename(f))
            shutil.copy2(f, dst)
        print(f"[OK] Copied {len(files)} sample images from {info_img_dir}")

def main():
    parser = argparse.ArgumentParser(description="Dataset management tool for Deepfake Detection")
    parser.add_argument("--gdown", action="store_true", help="Download datasets using gdown")
    parser.add_argument("--real-zip", type=str, help="Path to downloaded Real Images zip archive")
    parser.add_argument("--fake-zip", type=str, help="Path to downloaded Fake Images zip archive")
    parser.add_argument("--benchmark-samples", type=int, default=0, help="Generate synthetic forensic benchmark samples")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.real_zip or args.fake_zip:
        extract_zips_to_raw(args.real_zip, args.fake_zip, base_dir=script_dir)
    elif args.gdown:
        download_via_gdown(script_dir)
    elif args.benchmark_samples > 0:
        from src.dataset import generate_benchmark_synthetic_samples
        real_raw = os.path.join(script_dir, "data", "raw", "real")
        fake_raw = os.path.join(script_dir, "data", "raw", "fake")
        count = generate_benchmark_synthetic_samples(real_raw, fake_raw, count=args.benchmark_samples)
        print(f"[OK] Generated {count} benchmark samples each in {real_raw} and {fake_raw}")
    else:
        print("\n" + "="*70)
        print("DEEPFAKE DATASET DOWNLOAD INSTRUCTIONS")
        print("="*70)
        print("To download the collective dataset of >= 2,000 images:")
        print(f"1. Real Images Folder (1,096 images):")
        print(f"   {DRIVE_LINKS['real']}")
        print(f"2. Fake Images Folder (996 images):")
        print(f"   {DRIVE_LINKS['fake']}")
        print("\nRecommended method:")
        print("1. Open both links in your browser, click 'Download all' (Google Drive zips them).")
        print("2. Run: python download_dataset.py --real-zip <path_to_real.zip> --fake-zip <path_to_fake.zip>")
        print("\nOr automated download:")
        print("   python download_dataset.py --gdown")
        print("\nOr generate instant offline benchmark samples:")
        print("   python download_dataset.py --benchmark-samples 60")
        print("="*70)

if __name__ == "__main__":
    main()
