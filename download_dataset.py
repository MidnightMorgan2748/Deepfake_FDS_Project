"""
High-speed multithreaded dataset downloader and organizer utility for Deepfake Detection.
Downloads the real dataset of 1,000+ Real and 1,000+ Fake images directly from Google Drive
using concurrent threads and pre-extracted file IDs, completely bypassing Google Drive's 50-file folder limit.
"""

import os
import sys
import json
import urllib.request
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

MANIFEST_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "manifest.json")

def download_single_file(item, output_dir, timeout=15, retries=3):
    """Downloads an individual image file using Google Drive direct export URL."""
    file_id = item["id"]
    filename = item.get("name", f"{file_id}.jpg")
    # Clean filename
    safe_name = os.path.basename(filename)
    if not safe_name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
        safe_name = f"{safe_name}.jpg"
        
    out_path = os.path.join(output_dir, safe_name)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 1000:
        return safe_name, True
        
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status == 200:
                    data = resp.read()
                    if len(data) > 1000:  # Valid image payload
                        with open(out_path, "wb") as f:
                            f.write(data)
                        return safe_name, True
        except Exception:
            continue
            
    # Fallback to Google CDN thumbnail if export endpoint rate-limits
    cdn_url = f"https://lh3.googleusercontent.com/d/{file_id}=s512"
    try:
        cdn_req = urllib.request.Request(cdn_url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(cdn_req, timeout=timeout) as resp:
            if resp.status == 200:
                data = resp.read()
                if len(data) > 1000:
                    with open(out_path, "wb") as f:
                        f.write(data)
                    return safe_name, True
    except Exception:
        pass
        
    return safe_name, False

def download_manifest_dataset(manifest_file, base_dir=".", max_per_class=1000, num_workers=8):
    """
    Downloads Real and Fake images concurrently from manifest.json.
    """
    if not os.path.exists(manifest_file):
        raise FileNotFoundError(f"Manifest file not found at {manifest_file}. Please run create_manifest.py first.")
        
    with open(manifest_file, "r", encoding="utf-8") as f:
        manifest = json.load(f)
        
    real_items = manifest.get("real", [])
    fake_items = manifest.get("fake", [])
    
    if max_per_class:
        real_items = real_items[:max_per_class]
        fake_items = fake_items[:max_per_class]
        
    real_out = os.path.join(base_dir, "data", "raw", "real")
    fake_out = os.path.join(base_dir, "data", "raw", "fake")
    os.makedirs(real_out, exist_ok=True)
    os.makedirs(fake_out, exist_ok=True)
    
    print("="*70)
    print(f"DOWNLOADING COMPLETE DATASET ({len(real_items)} Real, {len(fake_items)} Fake)")
    print(f"Bypassing Google Drive 50-file limit via {num_workers} parallel workers...")
    print("="*70)
    
    # 1. Download Real images
    print(f"\n[*] Downloading {len(real_items)} Real Images to: {real_out}")
    real_success = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_single_file, item, real_out): item for item in real_items}
        for future in tqdm(as_completed(futures), total=len(real_items), desc="Downloading Real"):
            _, success = future.result()
            if success:
                real_success += 1
                
    # 2. Download Fake images
    print(f"\n[*] Downloading {len(fake_items)} Fake Images to: {fake_out}")
    fake_success = 0
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(download_single_file, item, fake_out): item for item in fake_items}
        for future in tqdm(as_completed(futures), total=len(fake_items), desc="Downloading Fake"):
            _, success = future.result()
            if success:
                fake_success += 1
                
    print("\n" + "="*70)
    print(f"[OK] Download Summary: {real_success}/{len(real_items)} Real, {fake_success}/{len(fake_items)} Fake downloaded successfully!")
    print("="*70)
    return real_success, fake_success

def main():
    parser = argparse.ArgumentParser(description="Multithreaded Dataset Downloader for DeepFake Detection")
    parser.add_argument("--count", type=int, default=1000, help="Number of images to download per class (default: 1000)")
    parser.add_argument("--workers", type=int, default=8, help="Number of parallel worker threads (default: 8)")
    parser.add_argument("--manifest", type=str, default=MANIFEST_PATH, help="Path to manifest.json")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    download_manifest_dataset(args.manifest, base_dir=script_dir, max_per_class=args.count, num_workers=args.workers)

if __name__ == "__main__":
    main()
