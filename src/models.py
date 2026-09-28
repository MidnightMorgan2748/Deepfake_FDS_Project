"""
Model definitions and architectures for DeepFake detection:
1. Paper Linear SVM (replicating Méreur et al., IEEE ICASSP 2025)
2. Baseline Classifiers (Random Forest, RBF SVM, Logistic Regression)
3. Novel Hybrid Ensemble (Soft-voting combination of Gradient Boosting, Random Forest, and RBF SVM)
"""

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
    ExtraTreesClassifier
)
from sklearn.linear_model import LogisticRegression

def get_paper_svm(C=1.0):
    """
    Paper-style Linear Support Vector Machine with StandardScaler.
    As described in Section IV-B of IEEE ICASSP 2025:
    'All classification results were obtained using a Linear Support Vector Machine (SVM).'
    """
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', SVC(kernel='linear', C=C, probability=True, random_state=42))
    ])

def get_baseline_models():
    """
    Standard baseline machine learning models for comparative benchmarking:
    - Random Forest
    - RBF Kernel SVM
    - Logistic Regression
    """
    models = {
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1))
        ]),
        "RBF SVM": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', SVC(kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42))
        ]),
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, C=1.0, random_state=42))
        ])
    }
    return models

def get_novel_ensemble():
    """
    Novel Contribution: Calibrated Multi-Domain Soft Voting Ensemble.
    Overcomes the feature fusion bottleneck described in Section IV-B of the paper:
    Combines:
    1. HistGradientBoostingClassifier (captures non-linear spectral and texture boundary splits)
    2. ExtraTreesClassifier (mitigates variance and over-fitting across heterogeneous feature scales)
    3. RBF SVM with probability calibration (learns smooth non-linear geometric margins in RKHS)
    """
    clf1 = HistGradientBoostingClassifier(max_iter=150, max_depth=6, random_state=42)
    clf2 = ExtraTreesClassifier(n_estimators=150, max_depth=12, random_state=42, n_jobs=-1)
    clf3 = SVC(kernel='rbf', C=3.0, gamma='scale', probability=True, random_state=42)
    
    ensemble = VotingClassifier(
        estimators=[
            ('hgb', clf1),
            ('et', clf2),
            ('svm', clf3)
        ],
        voting='soft',
        weights=[2.0, 1.5, 1.5]
    )
    
    return Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', ensemble)
    ])
