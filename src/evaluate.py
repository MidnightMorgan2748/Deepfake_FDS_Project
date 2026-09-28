"""
Evaluation engine implementing 5-fold Stratified Cross-Validation and comprehensive metrics
as required by Assignment 2 and IEEE ICASSP 2025.
"""

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix
)

def evaluate_model_cv(model, X, y, n_splits=5, random_state=42):
    """
    Evaluates a model using n-fold Stratified Cross-Validation.
    Returns:
    - metrics: dictionary with mean and standard deviation for key metrics:
      - balanced_accuracy
      - accuracy
      - roc_auc
      - pr_auc
      - precision_real, recall_real, f1_real
      - precision_fake, recall_fake, f1_fake
      - macro_f1
    - oof_predictions: out-of-fold predicted classes and probabilities
    - mean_confusion_matrix: average confusion matrix across folds
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    oof_y_pred = np.zeros(len(y), dtype=int)
    oof_y_prob = np.zeros(len(y), dtype=np.float64)
    
    fold_metrics = {
        "balanced_accuracy": [],
        "accuracy": [],
        "roc_auc": [],
        "pr_auc": [],
        "precision_real": [],
        "recall_real": [],
        "f1_real": [],
        "precision_fake": [],
        "recall_fake": [],
        "f1_fake": [],
        "macro_f1": []
    }
    
    cm_sum = np.zeros((2, 2), dtype=np.float64)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Fit model on training fold
        model.fit(X_train, y_train)
        
        # Out-of-fold inference
        y_pred = model.predict(X_val)
        oof_y_pred[val_idx] = y_pred
        
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_val)[:, 1]
        elif hasattr(model, "decision_function"):
            # Sigmoid scaling for decision function
            df = model.decision_function(X_val)
            y_prob = 1.0 / (1.0 + np.exp(-df))
        else:
            y_prob = y_pred.astype(float)
            
        oof_y_prob[val_idx] = y_prob
        
        # Compute fold metrics
        b_acc = balanced_accuracy_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)
        try:
            auc = roc_auc_score(y_val, y_prob)
            pr_auc = average_precision_score(y_val, y_prob)
        except Exception:
            auc, pr_auc = 0.5, 0.5
            
        prec, rec, f1, _ = precision_recall_fscore_support(y_val, y_pred, labels=[0, 1], zero_division=0)
        macro_f1 = np.mean(f1)
        
        fold_metrics["balanced_accuracy"].append(b_acc)
        fold_metrics["accuracy"].append(acc)
        fold_metrics["roc_auc"].append(auc)
        fold_metrics["pr_auc"].append(pr_auc)
        fold_metrics["precision_real"].append(prec[0])
        fold_metrics["recall_real"].append(rec[0])
        fold_metrics["f1_real"].append(f1[0])
        fold_metrics["precision_fake"].append(prec[1])
        fold_metrics["recall_fake"].append(rec[1])
        fold_metrics["f1_fake"].append(f1[1])
        fold_metrics["macro_f1"].append(macro_f1)
        
        cm_sum += confusion_matrix(y_val, y_pred, labels=[0, 1])
        
    # Aggregate metrics
    summary = {}
    for k, v in fold_metrics.items():
        summary[k] = float(np.mean(v))
        summary[f"{k}_std"] = float(np.std(v))
        
    cm_avg = cm_sum / n_splits
    
    # Compute overall ROC and PR curves
    fpr, tpr, roc_thresh = roc_curve(y, oof_y_prob)
    pr_prec, pr_rec, pr_thresh = precision_recall_curve(y, oof_y_prob)
    
    results = {
        "summary": summary,
        "oof_pred": oof_y_pred,
        "oof_prob": oof_y_prob,
        "cm": cm_avg,
        "roc": {"fpr": fpr, "tpr": tpr},
        "pr": {"precision": pr_prec, "recall": pr_rec}
    }
    
    return results

def format_benchmark_table(benchmark_results):
    """
    Formats multi-model benchmark results into a clean markdown table.
    """
    header = "| Pipeline / Model | Feature Set | Balanced Acc (%) | ROC-AUC | Real Recall | Fake Recall | Macro F1 |\n"
    separator = "|:---|:---|:---:|:---:|:---:|:---:|:---:|\n"
    lines = [header, separator]
    
    for name, res in benchmark_results.items():
        s = res["summary"]
        b_acc = f"{s['balanced_accuracy'] * 100:.2f} ± {s['balanced_accuracy_std'] * 100:.2f}"
        auc = f"{s['roc_auc']:.4f} ± {s['roc_auc_std']:.4f}"
        rec_real = f"{s['recall_real'] * 100:.2f}%"
        rec_fake = f"{s['recall_fake'] * 100:.2f}%"
        macro_f1 = f"{s['macro_f1']:.4f}"
        feat_name = res.get("feature_name", "Handcrafted")
        
        lines.append(f"| {name} | {feat_name} | {b_acc} | {auc} | {rec_real} | {rec_fake} | {macro_f1} |\n")
        
    return "".join(lines)
