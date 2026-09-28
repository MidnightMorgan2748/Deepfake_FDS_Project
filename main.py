"""
Main orchestration pipeline for DeepFake Detection.
Executes preprocessing, residual extraction, feature engineering, 5-fold cross-validation,
and visualization generation.
"""

import os
import sys
import json
import argparse
import numpy as np

# Ensure src can be imported
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.dataset import (
    preprocess_and_save_folder,
    check_dataset_status,
    get_image_paths
)
from src.features import (
    extract_and_save_residuals,
    build_feature_matrices
)
from src.models import (
    get_paper_svm,
    get_baseline_models,
    get_novel_ensemble
)
from src.evaluate import (
    evaluate_model_cv,
    format_benchmark_table
)
from src.visualize import (
    plot_sample_comparison,
    plot_residual_analysis,
    plot_feature_distributions,
    plot_spectral_comparison,
    plot_roc_curves,
    plot_confusion_matrix,
    plot_benchmark_bar_chart
)

def step_preprocess(data_dir):
    print("\n" + "="*70)
    print("STEP 1: PREPROCESSING (512x512 Grayscale Extraction)")
    print("="*70)
    raw_real = os.path.join(data_dir, "raw", "real")
    raw_fake = os.path.join(data_dir, "raw", "fake")
    pre_real = os.path.join(data_dir, "preprocessed", "real")
    pre_fake = os.path.join(data_dir, "preprocessed", "fake")
    
    n_real = preprocess_and_save_folder(raw_real, pre_real, target_size=(512, 512))
    n_fake = preprocess_and_save_folder(raw_fake, pre_fake, target_size=(512, 512))
    print(f"[OK] Preprocessed: {n_real} Real images, {n_fake} Fake images.")

def step_extract(data_dir):
    print("\n" + "="*70)
    print("STEP 2: RESIDUAL EXTRACTION & FEATURE ENGINEERING")
    print("="*70)
    pre_real = os.path.join(data_dir, "preprocessed", "real")
    pre_fake = os.path.join(data_dir, "preprocessed", "fake")
    res_real = os.path.join(data_dir, "residuals", "real")
    res_fake = os.path.join(data_dir, "residuals", "fake")
    
    print("[*] Extracting Laplacian 3x3 Differential Operator Residuals...")
    extract_and_save_residuals(pre_real, res_real)
    extract_and_save_residuals(pre_fake, res_fake)
    
    print("[*] Computing Multi-Scale FD, Texture Complexity (D=16), and Novel Spectral Signatures...")
    feat_matrices, y = build_feature_matrices(res_real, res_fake)
    
    feat_dir = os.path.join(data_dir, "features")
    os.makedirs(feat_dir, exist_ok=True)
    np.save(os.path.join(feat_dir, "y.npy"), y)
    for k, v in feat_matrices.items():
        np.save(os.path.join(feat_dir, f"X_{k}.npy"), v)
        print(f"[OK] Feature matrix '{k}': shape {v.shape}")
        
    print(f"[OK] Extracted feature matrices for {len(y)} samples saved to {feat_dir}")

def step_train_evaluate(data_dir, reports_dir):
    print("\n" + "="*70)
    print("STEP 3: 5-FOLD STRATIFIED CROSS-VALIDATION & BENCHMARKING")
    print("="*70)
    feat_dir = os.path.join(data_dir, "features")
    y_path = os.path.join(feat_dir, "y.npy")
    if not os.path.exists(y_path):
        print("[!] Feature files not found. Running extraction first...")
        step_extract(data_dir)
        
    y = np.load(y_path)
    X_compact = np.load(os.path.join(feat_dir, "X_paper_compact.npy"))
    X_stats = np.load(os.path.join(feat_dir, "X_paper_stats.npy"))
    X_novel = np.load(os.path.join(feat_dir, "X_novel_feature.npy"))
    
    # Models to evaluate
    benchmark_models = {
        "Paper Linear SVM (FD + TC)": {
            "model": get_paper_svm(C=1.0),
            "X": X_compact,
            "feature_name": "FD Slope + TC Mean (Paper)"
        },
        "Extended Paper SVM": {
            "model": get_paper_svm(C=1.0),
            "X": X_stats,
            "feature_name": "FD (44 scales) + TC Stats (8)"
        },
        "Baseline Random Forest": {
            "model": get_baseline_models()["Random Forest"],
            "X": X_stats,
            "feature_name": "FD + TC Statistical Descriptors"
        },
        "Baseline RBF SVM": {
            "model": get_baseline_models()["RBF SVM"],
            "X": X_stats,
            "feature_name": "FD + TC Statistical Descriptors"
        },
        "Baseline Logistic Regression": {
            "model": get_baseline_models()["Logistic Regression"],
            "X": X_stats,
            "feature_name": "FD + TC Statistical Descriptors"
        },
        "Novel Hybrid Ensemble": {
            "model": get_novel_ensemble(),
            "X": X_novel,
            "feature_name": "FD + TC + Spectral + Texture Moments"
        }
    }
    
    results = {}
    print("\nExecuting 5-Fold Stratified Cross-Validation on all pipelines...")
    for name, config in benchmark_models.items():
        print(f"[*] Cross-validating: {name}...")
        eval_res = evaluate_model_cv(config["model"], config["X"], y, n_splits=5)
        eval_res["feature_name"] = config["feature_name"]
        results[name] = eval_res
        
        s = eval_res["summary"]
        print(f"    --> Balanced Acc: {s['balanced_accuracy']*100:.2f}% | ROC-AUC: {s['roc_auc']:.4f} | F1: {s['macro_f1']:.4f}")
        
    table_md = format_benchmark_table(results)
    print("\n" + table_md)
    
    # Save results json
    os.makedirs(reports_dir, exist_ok=True)
    serializable = {}
    for name, res in results.items():
        serializable[name] = {
            "summary": res["summary"],
            "feature_name": res["feature_name"],
            "cm": res["cm"].tolist()
        }
    with open(os.path.join(reports_dir, "benchmark_metrics.json"), "w") as f:
        json.dump(serializable, f, indent=4)
        
    return results

def step_visualize(data_dir, reports_dir, benchmark_results=None):
    print("\n" + "="*70)
    print("STEP 4: GENERATING EXPERIMENTAL FIGURES & FORENSIC PLOTS")
    print("="*70)
    figures_dir = os.path.join(reports_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)
    
    # 1. Sample comparison
    real_paths = get_image_paths(os.path.join(data_dir, "preprocessed", "real"))
    fake_paths = get_image_paths(os.path.join(data_dir, "preprocessed", "fake"))
    if real_paths and fake_paths:
        p1 = os.path.join(figures_dir, "fig1_sample_comparison.png")
        plot_sample_comparison(real_paths, fake_paths, p1, n=4)
        print(f"[OK] Saved: {p1}")
        
        # 2. Residual analysis
        p2 = os.path.join(figures_dir, "fig2_residual_analysis.png")
        plot_residual_analysis(real_paths[0], fake_paths[0], p2)
        print(f"[OK] Saved: {p2}")
        
    # 3. Feature distributions
    feat_dir = os.path.join(data_dir, "features")
    y_path = os.path.join(feat_dir, "y.npy")
    if os.path.exists(y_path):
        y = np.load(y_path)
        X_compact = np.load(os.path.join(feat_dir, "X_paper_compact.npy"))
        fd_real = X_compact[y == 0, 0]
        fd_fake = X_compact[y == 1, 0]
        tc_real = X_compact[y == 0, 1]
        tc_fake = X_compact[y == 1, 1]
        p3 = os.path.join(figures_dir, "fig3_feature_distributions.png")
        plot_feature_distributions(fd_real, fd_fake, tc_real, tc_fake, p3)
        print(f"[OK] Saved: {p3}")
        
    # 4. Spectral comparison
    res_real = os.path.join(data_dir, "residuals", "real")
    res_fake = os.path.join(data_dir, "residuals", "fake")
    if os.path.exists(res_real) and os.path.exists(res_fake):
        p4 = os.path.join(figures_dir, "fig4_spectral_comparison.png")
        plot_spectral_comparison(res_real, res_fake, p4)
        print(f"[OK] Saved: {p4}")
        
    # 5. ROC Curves and Confusion Matrices
    if benchmark_results:
        p5 = os.path.join(figures_dir, "fig5_roc_curves.png")
        plot_roc_curves(benchmark_results, p5)
        print(f"[OK] Saved: {p5}")
        
        # Confusion matrix for novel ensemble & paper svm
        if "Novel Hybrid Ensemble" in benchmark_results:
            p6 = os.path.join(figures_dir, "fig6_confusion_matrix_ensemble.png")
            plot_confusion_matrix(benchmark_results["Novel Hybrid Ensemble"]["cm"], "Novel Hybrid Ensemble", p6)
            print(f"[OK] Saved: {p6}")
            
        if "Paper Linear SVM (FD + TC)" in benchmark_results:
            p7 = os.path.join(figures_dir, "fig7_confusion_matrix_paper_svm.png")
            plot_confusion_matrix(benchmark_results["Paper Linear SVM (FD + TC)"]["cm"], "Paper Linear SVM", p7)
            print(f"[OK] Saved: {p7}")
            
        # 6. Benchmark bar chart
        p8 = os.path.join(figures_dir, "fig8_benchmark_comparison.png")
        plot_benchmark_bar_chart(benchmark_results, p8)
        print(f"[OK] Saved: {p8}")

    # Generate complete Word document (.docx) and PDF report
    try:
        from generate_word_report import build_docx_report
        build_docx_report()
    except Exception as e:
        print(f"[!] Note on docx report: {e}")

def main():
    parser = argparse.ArgumentParser(description="End-to-end Deepfake Detection Pipeline")
    parser.add_argument("--step", type=str, default="all", choices=["status", "preprocess", "extract", "train", "visualize", "all"],
                        help="Pipeline stage to execute")
    args = parser.parse_args()
    
    base_dir = PROJECT_ROOT
    data_dir = os.path.join(base_dir, "data")
    reports_dir = os.path.join(base_dir, "reports")
    
    if args.step == "status":
        status = check_dataset_status(data_dir)
        print("\nDATASET PIPELINE STATUS:")
        for k, v in status.items():
            print(f"  {k:22s}: {v}")
        return
        
    if args.step in ["preprocess", "all"]:
        # If no raw images exist yet, seed benchmark samples for verification
        status = check_dataset_status(data_dir)
        if status["raw_total"] == 0:
            print("[i] No raw images found. Generating benchmark verification samples...")
            from src.dataset import generate_benchmark_synthetic_samples
            generate_benchmark_synthetic_samples(
                os.path.join(data_dir, "raw", "real"),
                os.path.join(data_dir, "raw", "fake"),
                count=60
            )
        step_preprocess(data_dir)
        
    if args.step in ["extract", "all"]:
        step_extract(data_dir)
        
    results = None
    if args.step in ["train", "all"]:
        results = step_train_evaluate(data_dir, reports_dir)
        
    if args.step in ["visualize", "all"]:
        step_visualize(data_dir, reports_dir, results)
        
    print("\n" + "="*70)
    print("PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("="*70)

if __name__ == "__main__":
    main()
