"""
train_real_models.py
====================
Train Random Forest models on REAL dataset (from MATLAB/Simulink) for three tasks:
1. Fault Classification (fault type: AG, BC, BCG, ABC, NONE)
2. Zone Detection (which zone: 0, 1, 2, 3)
3. Operating Time Regression (t_R1, t_R2 - IDMT trip times)

Uses cross-validation due to small dataset size.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             mean_absolute_error, r2_score,
                             classification_report)
from sklearn.pipeline import make_pipeline

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, 'data')
RESULTS_DIR = os.path.join(ROOT, 'results_real')
os.makedirs(RESULTS_DIR, exist_ok=True)

# Features used in training (must match convert script output)
FEATURES = [
    'R1_Ia', 'R1_Ib', 'R1_Ic', 'R1_I0', 'R1_I1', 'R1_I2',
    'R1_Ia_ang', 'R1_Ib_ang', 'R1_Ic_ang',
    'R2_Ia', 'R2_Ib', 'R2_Ic', 'R2_I0', 'R2_I1', 'R2_I2',
    'R2_Ia_ang', 'R2_Ib_ang', 'R2_Ic_ang',
    'VA_a', 'VA_b', 'VA_c', 'VA_V0', 'VA_V2',
    'VB_a', 'VB_b', 'VB_c', 'VB_V0', 'VB_V2',
]


def load_dataset(csv_path):
    """Load and prepare dataset."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows x {len(df.columns)} cols")

    # Check features
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"WARNING: Missing features: {missing}")

    X = df[FEATURES].values
    return df, X


def run_classification_cv(name, X, y, cv_folds=5):
    """Run classification with cross-validation."""
    print(f"\n=== {name} (CV={cv_folds}) ===")

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1, class_weight='balanced')
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"Individual folds: {cv_scores}")

    # Fit on full data for final model
    model.fit(X, y)

    # Get predictions for confusion matrix (using CV predictions)
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return model, {'cv_acc_mean': cv_scores.mean(), 'cv_acc_std': cv_scores.std(),
                   'cv_scores': cv_scores.tolist()}


def run_regression_cv(name, X, y, cv_folds=5):
    """Run regression with cross-validation."""
    print(f"\n=== {name} (CV={cv_folds}) ===")

    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(n_estimators=500, random_state=42, n_jobs=-1)
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Cross-validation for MAE and R2
    cv_mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)

    print(f"CV MAE: {cv_mae.mean():.4f} (+/- {cv_mae.std()*2:.4f})")
    print(f"CV R²:  {cv_r2.mean():.4f} (+/- {cv_r2.std()*2:.4f})")

    # Fit on full data
    model.fit(X, y)

    # CV predictions for analysis
    from sklearn.model_selection import cross_val_predict
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    return model, {'cv_mae_mean': cv_mae.mean(), 'cv_mae_std': cv_mae.std(),
                   'cv_r2_mean': cv_r2.mean(), 'cv_r2_std': cv_r2.std(),
                   'cv_mae': cv_mae.tolist(), 'cv_r2': cv_r2.tolist()}


def main():
    print("=" * 60)
    print("Training Random Forest on REAL Dataset")
    print("=" * 60)

    # Load real dataset
    real_csv = os.path.join(DATA_DIR, 'fault_dataset_real.csv')
    if not os.path.exists(real_csv):
        print(f"ERROR: {real_csv} not found. Run convert_matlab_dataset.py first.")
        return

    df_real, X_real = load_dataset(real_csv)

    # Also load simulated dataset for comparison/augmentation
    sim_csv = os.path.join(DATA_DIR, 'fault_dataset.csv')
    if os.path.exists(sim_csv):
        df_sim, X_sim = load_dataset(sim_csv)
        print(f"Simulated dataset: {len(df_sim)} rows")

    # ============================================================
    # TASK 1: Fault Classification (11 classes including NONE)
    # ============================================================
    le_type = LabelEncoder()
    y_type = le_type.fit_transform(df_real['fault_type'])

    model_type, metrics_type = run_classification_cv(
        'T1_Fault_Classification', X_real, y_type, cv_folds=5
    )

    # ============================================================
    # TASK 2: Zone Detection (4 classes: 0, 1, 2, 3)
    # ============================================================
    y_zone = df_real['zone'].values

    model_zone, metrics_zone = run_classification_cv(
        'T2_Zone_Detection', X_real, y_zone, cv_folds=5
    )

    # ============================================================
    # TASK 3: Operating Time Regression (only for fault cases)
    # ============================================================
    fault_mask = df_real['fault'] == 1
    X_fault = X_real[fault_mask]

    # T3a: R1 trip time
    y_tR1 = df_real.loc[fault_mask, 't_R1'].values
    # Only use cases where R1 operates (t_R1 > 0)
    r1_mask = y_tR1 > 0
    if r1_mask.sum() > 5:
        model_tR1, metrics_tR1 = run_regression_cv(
            'T3a_R1_Trip_Time', X_fault[r1_mask], y_tR1[r1_mask], cv_folds=5
        )
    else:
        print("\n=== T3a_R1_Trip_Time ===")
        print("Insufficient R1 operating cases")
        model_tR1, metrics_tR1 = None, {}

    # T3b: R2 trip time
    y_tR2 = df_real.loc[fault_mask, 't_R2'].values
    r2_mask = y_tR2 > 0
    if r2_mask.sum() > 5:
        model_tR2, metrics_tR2 = run_regression_cv(
            'T3b_R2_Trip_Time', X_fault[r2_mask], y_tR2[r2_mask], cv_folds=5
        )
    else:
        print("\n=== T3b_R2_Trip_Time ===")
        print("Insufficient R2 operating cases")
        model_tR2, metrics_tR2 = None, {}

    # ============================================================
    # Save models and metrics
    # ============================================================
    models = {
        'T1_fault_type': (model_type, le_type),
        'T2_zone': (model_zone, None),
        'T3a_t_R1': (model_tR1, None),
        'T3b_t_R2': (model_tR2, None),
    }

    # Save models
    joblib.dump({k: v[0] for k, v in models.items() if v[0] is not None},
                os.path.join(RESULTS_DIR, 'models_real.joblib'))

    # Save label encoders
    joblib.dump({'fault_type': le_type},
                os.path.join(RESULTS_DIR, 'encoders_real.joblib'))

    # Save metrics
    metrics = {
        'T1_fault_classification': metrics_type,
        'T2_zone_detection': metrics_zone,
        'T3a_R1_trip_time': metrics_tR1,
        'T3b_R2_trip_time': metrics_tR2,
    }

    with open(os.path.join(RESULTS_DIR, 'metrics_real.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Models and metrics saved to {RESULTS_DIR}")
    print(f"{'='*60}")

    # Print summary
    print("\nSUMMARY:")
    print(f"  T1 Fault Classification: CV Acc = {metrics_type.get('cv_acc_mean', 0):.3f}")
    print(f"  T2 Zone Detection:       CV Acc = {metrics_zone.get('cv_acc_mean', 0):.3f}")
    if metrics_tR1:
        print(f"  T3a R1 Trip Time:        CV MAE = {metrics_tR1.get('cv_mae_mean', 0):.4f}s, R² = {metrics_tR1.get('cv_r2_mean', 0):.4f}")
    if metrics_tR2:
        print(f"  T3b R2 Trip Time:        CV MAE = {metrics_tR2.get('cv_mae_mean', 0):.4f}s, R² = {metrics_tR2.get('cv_r2_mean', 0):.4f}")


if __name__ == '__main__':
    main()