# DeepFake Image Detection: Forensic Residual Noise Texture Analysis
### Foundations of Data Science (FDS) — Assignment 2

An end-to-end Data Science and Machine Learning framework for detecting AI-generated images (deepfakes) using explainable, physics-grounded forensic features derived from camera sensor noise residuals and neural synthesis artifacts.

This repository replicates and extends the state-of-the-art methodology published at **IEEE ICASSP 2025**:
> **"Forensics Analysis of Residual Noise Texture in Digital Images for Detection of Deepfake"**  
> *Arthur Méreur, Antoine Mallet, Rémi Cogranne, Minoru Kuribayashi*  
> [IEEE Xplore: https://doi.org/10.1109/ICASSP49660.2025.10887712](https://doi.org/10.1109/ICASSP49660.2025.10887712)

---

## Key Features & Highlights

1. **Replication of Research Paper Methodology**:
   - **Discrete Laplacian Residual Filter**: 2D convolution with 3×3 differential kernel isolating high-frequency noise.
   - **Multi-Scale Fractal Dimension (FD)**: Box-counting over 44 logarithmic scales ($2 < D < 112$) evaluating scale-invariant self-similarity.
   - **Block-wise Texture Complexity (TC)**: Local non-overlapping $16 \times 16$ block entropy and non-linear log-odds transformation ($32 \times 32 = 1,024$ spatial features).
   - **Linear Support Vector Machine (SVM)**: Reproducing Table I and Figure 2 of the IEEE paper.

2. **Novel Scientific Contributions**:
   - **Radial Spectral Fourier Profiling (FFT)**: Azimuthal average power spectrum exposing periodic deconvolution grid harmonics and high-frequency attenuation characteristic of neural generators.
   - **Multi-Domain Calibrated Hybrid Ensemble**: Solves the feature dimension mismatch identified in the paper by combining Histogram Gradient Boosting, Extra-Trees, and RBF SVM via soft probability voting.
   - **Spatial Texture Moments**: Skewness, excess kurtosis, mean absolute deviation, and directional differential gradients.

3. **Rigorous Experimental Validation**:
   - 5-Fold Stratified Cross-Validation on all pipelines.
   - Comprehensive metrics: Balanced Accuracy, ROC-AUC, Average Precision (PR-AUC), Precision, Recall, and Macro F1.
   - Publication-quality visualizations: Residual heatmaps, feature distribution histograms, ROC curves, confusion matrices, and benchmark comparisons.

4. **Complete Report Generation (Assignment Step 3)**:
   - Full academic report in [REPORT.md](REPORT.md).
   - Automated compilation into a submission-ready PDF: [reports/Deepfake_Detection_Report.pdf](reports/Deepfake_Detection_Report.pdf).

---

## Directory Structure

```
Actual_Project/
├── DeepFake_Detection_FDS.ipynb  # Interactive self-contained Jupyter Notebook (Local + Colab)
├── README.md                     # Project documentation and setup guide
├── REPORT.md                     # Comprehensive academic report (Assignment Step 3)
├── requirements.txt              # Verified project dependencies
├── main.py                       # Modular CLI orchestrator (preprocess -> extract -> train -> plot)
├── download_dataset.py           # Dataset downloader and archive organizer
├── generate_pdf_report.py        # Compiles submission PDF from benchmark metrics and figures
├── src/                          # Core Python library
│   ├── __init__.py
│   ├── dataset.py                # Preprocessing (512x512 grayscale) and image loaders
│   ├── features.py               # Laplacian residuals, FD box-counting, TC, and FFT spectral profiling
│   ├── models.py                 # Paper Linear SVM, Baselines (RF, RBF, LogReg), and Novel Ensemble
│   ├── evaluate.py               # 5-fold Stratified Cross-Validation engine
│   └── visualize.py              # Publication-ready figure plotting suite
├── data/                         # Dataset storage
│   ├── raw/                      # Raw Real and Fake images (>= 2,000 images)
│   ├── preprocessed/             # Standardized 512x512 grayscale images
│   ├── residuals/                # Extracted Laplacian .npy residual maps
│   └── features/                 # Cached feature matrices (X_compact, X_stats, X_novel, y)
└── reports/                      # Submission artifacts
    ├── Deepfake_Detection_Report.pdf  # Compiled submission-ready PDF report
    ├── benchmark_metrics.json         # Raw 5-fold cross-validation results
    └── figures/                       # High-resolution experimental figures (300 DPI)
```

---

## Getting Started

### 1. Installation
Ensure Python 3.10+ is installed. Clone or navigate to the directory and install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Dataset Management
The assignment requires a collective dataset exceeding 2,000 images (Real: 1,096 images, Fake: 996 images).
Links provided in `Informational_no_project/Drive lInks.docx`:
- **Real Images (1,096 images)**: `https://drive.google.com/drive/folders/1bQahWnXPid84b7MjE-9_mPyHHhzfQXH3`
- **Fake Images (996 images)**: `https://drive.google.com/drive/folders/13OqZH_uwD9IWhoWF5h0l-dTKFyRfII7A`

#### Option A: Direct Browser Download (Recommended & Fastest)
1. Open both links in a browser and click **Download all** (Google Drive zips them instantly).
2. Extract the downloaded archives:
```bash
python download_dataset.py --real-zip path/to/real.zip --fake-zip path/to/fake.zip
```

#### Option B: Automated Download via gdown
```bash
python download_dataset.py --gdown
```

#### Option C: Offline Benchmark Smoke Testing
To test and verify the entire pipeline immediately without waiting for large downloads:
```bash
python download_dataset.py --benchmark-samples 60
```

---

## Running the Pipeline

### Command Line Interface (`main.py`)
Run the entire end-to-end pipeline with one command:
```bash
python main.py --step all
```

Or execute individual stages:
```bash
# Check dataset pipeline status
python main.py --step status

# Preprocess raw images (512x512 grayscale)
python main.py --step preprocess

# Extract Laplacian residuals, FD, TC, and spectral features
python main.py --step extract

# Run 5-fold Stratified Cross-Validation on all models
python main.py --step train

# Generate all figures and plots
python main.py --step visualize
```

### Generating the Final PDF Report
```bash
python generate_pdf_report.py
```
This generates `reports/Deepfake_Detection_Report.pdf` with all figures, equations, architecture descriptions, and benchmark tables formatted to academic standards.

### Running the Interactive Jupyter Notebook
Launch Jupyter or open in Google Colab / VS Code:
```bash
jupyter notebook DeepFake_Detection_FDS.ipynb
```
The notebook features interactive previews, residual heatmaps, feature histograms, cross-validation benchmarking, and a custom inference function to test arbitrary unseen images.

---

## Benchmark Results (5-Fold Stratified Cross-Validation)

| Pipeline / Model | Feature Set | Balanced Acc (%) | ROC-AUC | Real Recall | Fake Recall | Macro F1 |
|:---|:---|:---:|:---:|:---:|:---:|:---:|
| **Paper Linear SVM (FD + TC)** | FD Slope + TC Mean (Paper) | 50.00 ± 0.00 | 0.4716 ± 0.0371 | 0.00% | 100.00% | 0.3420 |
| **Baseline Random Forest** | FD + TC Statistical Descriptors | 62.52 ± 0.41 | 0.6824 ± 0.0200 | 53.71% | 71.34% | 0.6234 |
| **Baseline RBF SVM** | FD + TC Statistical Descriptors | 55.49 ± 1.55 | 0.6008 ± 0.0151 | 31.24% | 79.75% | 0.5298 |
| **Baseline Logistic Regression** | FD + TC Statistical Descriptors | 57.93 ± 1.28 | 0.6218 ± 0.0242 | 34.04% | 81.83% | 0.5579 |
| **Novel Hybrid Ensemble** | FD + TC + Spectral + Texture Moments | **69.72 ± 2.57** | **0.7714 ± 0.0282** | **64.16%** | **75.29%** | **0.6968** |

---

## Citation & References

```bibtex
@inproceedings{mereur2025forensics,
  title={Forensics Analysis of Residual Noise Texture in Digital Images for Detection of Deepfake},
  author={M{\'e}reur, Arthur and Mallet, Antoine and Cogranne, R{\'e}mi and Kuribayashi, Minoru},
  booktitle={ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  year={2025},
  organization={IEEE},
  doi={10.1109/ICASSP49660.2025.10887712}
}
```
