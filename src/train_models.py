"""
train_models.py
================
Trains and evaluates the five ML protection tasks on the simulated dataset:

  M1  Fault DETECTION        (binary)       - "is there a fault?"
  M2  Fault CLASSIFICATION   (11 classes)   - "which type? AG/BC/ABC/..."
  M3  RELAY SELECTION        (3 classes)    - "which relay must operate?"
                                              <- the prof's key ask: the model
                                                 decides R1 / R2 / no-trip
  M4  FAULT LOCATION         (regression)   - distance from Bus A in km
  M5  IDMT TRIP TIME         (regression)   - learn t = TMS*k/((I/Is)^a - 1)
                                              i.e. the IDMT characteristic
                                              embedded in an ML model

Models compared on each classification task: Decision Tree, Random Forest,
SVM (RBF), MLP neural network. Regression: Random Forest + MLP.
Evaluation: stratified 80/20 split + 5-fold cross-validation.

All fitted models are saved to results/models.joblib for the demo script.
Run:  python src/train_models.py
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix,
                             mean_absolute_error, r2_score,
                             classification_report)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS = os.path.join(ROOT, 'results')
os.makedirs(RESULTS, exist_ok=True)

# ---------------------------------------------------------------- features --
# Only quantities a real numerical relay measures: phase & sequence current
# magnitudes and angles at both relay points + bus voltages.
FEATURES = [
    'R1_Ia', 'R1_Ib', 'R1_Ic', 'R1_I0', 'R1_I1', 'R1_I2',
    'R1_Ia_ang', 'R1_Ib_ang', 'R1_Ic_ang',
    'R2_Ia', 'R2_Ib', 'R2_Ic', 'R2_I0', 'R2_I1', 'R2_I2',
    'R2_Ia_ang', 'R2_Ib_ang', 'R2_Ic_ang',
    'VA_a', 'VA_b', 'VA_c', 'VA_V0', 'VA_V2',
    'VB_a', 'VB_b', 'VB_c', 'VB_V0', 'VB_V2',
]


def classifiers():
    return {
        'DecisionTree': DecisionTreeClassifier(max_depth=12, random_state=0),
        'RandomForest': RandomForestClassifier(n_estimators=200,
                                               random_state=0, n_jobs=-1),
        'SVM-RBF': make_pipeline(StandardScaler(),
                                 SVC(kernel='rbf', C=10, gamma='scale')),
        'MLP': make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=(64, 32),
                                           max_iter=2000, random_state=0)),
    }


def run_classification(name, X, y, report):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2,
                                          random_state=0, stratify=y)
    best_acc, best_name, best_model = -1, None, None
    report[name] = {}
    print(f"\n=== {name} ===")
    for mname, model in classifiers().items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        acc = accuracy_score(yte, pred)
        f1 = f1_score(yte, pred, average='macro')
        cv = cross_val_score(model, X, y, cv=5, n_jobs=-1).mean()
        report[name][mname] = {'test_acc': round(acc, 4),
                               'macro_f1': round(f1, 4),
                               'cv5_acc': round(cv, 4)}
        print(f"  {mname:<13} acc={acc:.4f}  macroF1={f1:.4f}  cv5={cv:.4f}")
        if acc > best_acc:
            best_acc, best_name, best_model = acc, mname, model
    # keep test split + best model for the plots script
    pred = best_model.predict(Xte)
    report[name]['best'] = best_name
    print(f"  -> best: {best_name}")
    print(classification_report(yte, pred, zero_division=0))
    return best_model, (yte, pred)


def run_regression(name, X, y, report):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=0)
    models = {
        'RandomForest': RandomForestRegressor(n_estimators=300,
                                              random_state=0, n_jobs=-1),
        'MLP': make_pipeline(StandardScaler(),
                             MLPRegressor(hidden_layer_sizes=(128, 64),
                                          max_iter=4000, random_state=0)),
    }
    best_r2, best_model, best_name = -np.inf, None, None
    report[name] = {}
    print(f"\n=== {name} ===")
    for mname, model in models.items():
        model.fit(Xtr, ytr)
        pred = model.predict(Xte)
        mae = mean_absolute_error(yte, pred)
        r2 = r2_score(yte, pred)
        report[name][mname] = {'MAE': round(mae, 4), 'R2': round(r2, 5)}
        print(f"  {mname:<13} MAE={mae:.4f}  R2={r2:.5f}")
        if r2 > best_r2:
            best_r2, best_model, best_name = r2, model, mname
    report[name]['best'] = best_name
    pred = best_model.predict(Xte)
    return best_model, (yte, pred)


def main():
    df = pd.read_csv(os.path.join(ROOT, 'data', 'fault_dataset.csv'))
    X = df[FEATURES].values
    report, saved = {}, {}

    # M1 detection
    m, split = run_classification('M1_detection', X, df['fault'].values, report)
    saved['M1'] = (m, split)

    # M2 classification (11-class)
    m, split = run_classification('M2_fault_type', X,
                                  df['fault_type'].values, report)
    saved['M2'] = (m, split)

    # M3 relay selection  <- key deliverable
    m, split = run_classification('M3_relay_selection', X,
                                  df['correct_relay'].values, report)
    saved['M3'] = (m, split)

    # M4 fault location (faulted rows only)
    fmask = df['fault'] == 1
    m, split = run_regression('M4_location_km', X[fmask],
                              df.loc[fmask, 'distance_km'].values, report)
    saved['M4'] = (m, split)

    # M5 IDMT trip-time emulation: predict R1 trip time where R1 operates
    tmask = fmask & (df['t_R1'] > 0)
    m, split = run_regression('M5_idmt_trip_time', X[tmask],
                              df.loc[tmask, 't_R1'].values, report)
    saved['M5'] = (m, split)

    # persist
    joblib.dump({k: v[0] for k, v in saved.items()},
                os.path.join(RESULTS, 'models.joblib'))
    joblib.dump({k: v[1] for k, v in saved.items()},
                os.path.join(RESULTS, 'test_splits.joblib'))
    with open(os.path.join(RESULTS, 'metrics.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\nmodels + metrics saved to {RESULTS}")


if __name__ == '__main__':
    main()
