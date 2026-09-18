"""
real_data_pipeline.py
=====================
Complete ML pipeline for real MATLAB/Simulink dataset (~4.4M time-domain samples).

INPUT: 44 CSV files (5 fault types × 3 zones × 3 positions)
       Each file: 100,001 time samples × 31 channels
       Channels: time + 5 current points × 3 phases + 5 voltage points × 3 phases

OUTPUT: Three protection models:
  1. Fault Classification (LG, LL, LLG, LLL, LLLG)
  2. Zone Detection (Z1, Z2, Z3)
  3. Operating Time Regression (IDMT trip time)
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal as scipy_signal
from scipy.stats import skew, kurtosis

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\Copy of Fault_Dataset_All\Fault_Dataset_All\data set')
RESULTS_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\results_real_full')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Time windows (seconds)
PRE_FAULT_START, PRE_FAULT_END = 0.02, 0.04
FAULT_START, FAULT_END = 0.12, 0.14
TRANSIENT_START, TRANSIENT_END = 0.08, 0.12

# Fundamental frequency
F0 = 50.0  # Hz

# Measurement points
CURRENT_POINTS = ['R1', 'R2', 'R3', 'RE', 'SE']
VOLTAGE_POINTS = ['R1', 'R2', 'R3', 'RE', 'SE']
PHASES = ['A', 'B', 'C']

# Fault type mapping
FAULT_MAP = {
    'LG': 'AG',
    'LL': 'BC',
    'LLG': 'BCG',
    'LLL': 'ABC',
    'LLLG': 'ABC',
}

# Zone mapping
ZONE_MAP = {'Z1': 1, 'Z2': 2, 'Z3': 3}


# ============================================================
# FEATURE EXTRACTION
# ============================================================
def compute_phasor(signal_data, fs, f0=F0):
    """Compute phasor magnitude and angle using DFT at fundamental frequency."""
    N = len(signal_data)
    t = np.arange(N) / fs
    cos = np.cos(2 * np.pi * f0 * t)
    sin = np.sin(2 * np.pi * f0 * t)
    real = 2.0 / N * np.sum(signal_data * cos)
    imag = -2.0 / N * np.sum(signal_data * sin)
    mag = np.sqrt(real**2 + imag**2)
    ang = np.angle(real + 1j * imag, deg=True)
    return mag, ang


def compute_sequence_components(Ia, Ib, Ic):
    """Compute symmetrical components (I0, I1, I2) from phase phasors."""
    a = np.exp(1j * 2 * np.pi / 3)
    A = np.array([[1, 1, 1], [1, a**2, a], [1, a, a**2]], dtype=complex)
    Iabc = np.array([Ia, Ib, Ic])
    I012 = np.linalg.solve(A, Iabc)
    return I012[0], I012[1], I012[2]


def extract_window_features(signal_data, t, fs, window_name):
    """Extract comprehensive features from a signal window."""
    features = {}

    # 1. Phasor features
    mag, ang = compute_phasor(signal_data, fs)
    features[f'{window_name}_mag'] = mag
    features[f'{window_name}_ang'] = ang

    # 2. Statistical features
    features[f'{window_name}_mean'] = np.mean(signal_data)
    features[f'{window_name}_std'] = np.std(signal_data)
    features[f'{window_name}_rms'] = np.sqrt(np.mean(signal_data**2))
    features[f'{window_name}_kurt'] = kurtosis(signal_data) if len(signal_data) > 3 else 0
    features[f'{window_name}_skew'] = skew(signal_data) if len(signal_data) > 2 else 0

    # 3. Peak features
    features[f'{window_name}_peak_mag'] = np.max(np.abs(signal_data))
    features[f'{window_name}_peak_pos'] = t[np.argmax(np.abs(signal_data))]
    features[f'{window_name}_crest_factor'] = features[f'{window_name}_peak_mag'] / features[f'{window_name}_rms'] if features[f'{window_name}_rms'] > 0 else 0

    # 4. Zero-crossing rate
    zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
    features[f'{window_name}_zcr'] = zero_crossings / (len(signal_data) * fs)

    # 5. FFT features (fundamental + harmonics)
    if len(signal_data) > 10:
        fft_vals = np.abs(np.fft.rfft(signal_data))
        fft_freq = np.fft.rfftfreq(len(signal_data), 1/fs)
        fund_mask = (fft_freq >= 49) & (fft_freq <= 51)
        harm2_mask = (fft_freq >= 99) & (fft_freq <= 101)
        harm3_mask = (fft_freq >= 149) & (fft_freq <= 151)
        features[f'{window_name}_energy_fund'] = np.sum(fft_vals[fund_mask]**2)
        features[f'{window_name}_energy_harm2'] = np.sum(fft_vals[harm2_mask]**2)
        features[f'{window_name}_energy_harm3'] = np.sum(fft_vals[harm3_mask]**2)
        features[f'{window_name}_harmonic_ratio'] = features[f'{window_name}_energy_harm2'] / features[f'{window_name}_energy_fund'] if features[f'{window_name}_energy_fund'] > 0 else 0

    return features


def extract_file_features(file_path):
    """Extract features from a single CSV file."""
    # Load data
    df = pd.read_csv(file_path)

    # Sampling frequency
    fs = 1 / (df['time'].iloc[1] - df['time'].iloc[0])
    t = df['time'].values

    # Define windows
    windows = {
        'pre': (PRE_FAULT_START, PRE_FAULT_END),
        'trans': (TRANSIENT_START, TRANSIENT_END),
        'fault': (FAULT_START, FAULT_END),
    }

    features = {}

    # Extract features for each measurement point and phase
    for point in CURRENT_POINTS:
        for phase in PHASES:
            col_name = f'I_{point}_{phase}'
            if col_name not in df.columns:
                continue

            signal_data = df[col_name].values

            for window_name, (w_start, w_end) in windows.items():
                mask = (t >= w_start) & (t <= w_end)
                if np.sum(mask) < 10:
                    continue

                w_features = extract_window_features(signal_data[mask], t[mask], fs, window_name)

                # Add to features dict with structured naming
                for feat_name, value in w_features.items():
                    features[f'{point}_{phase}_{feat_name}'] = value

    # Voltage features (only fault window + pre-fault for now to keep size manageable)
    for point in VOLTAGE_POINTS:
        for phase in PHASES:
            col_name = f'V_{point}_{phase}'
            if col_name not in df.columns:
                continue

            signal_data = df[col_name].values

            for window_name in ['pre', 'fault']:
                w_start, w_end = windows[window_name]
                mask = (t >= w_start) & (t <= w_end)
                if np.sum(mask) < 10:
                    continue

                w_features = extract_window_features(signal_data[mask], t[mask], fs, window_name)
                for feat_name, value in w_features.items():
                    features[f'V_{point}_{phase}_{feat_name}'] = value

    # Sequence components for each measurement point (fault window)
    for point in CURRENT_POINTS:
        for window_name in ['pre', 'fault']:
            w_start, w_end = windows[window_name]
            mask = (t >= w_start) & (t <= w_end)
            if np.sum(mask) < 10:
                continue

            seq_features = {}
            for phase in PHASES:
                col_name = f'I_{point}_{phase}'
                if col_name not in df.columns:
                    continue
                mag, ang = compute_phasor(df.loc[mask, col_name].values, fs)
                seq_features[phase] = mag * np.exp(1j * np.deg2rad(ang))

            if all(p in seq_features for p in PHASES):
                I0, I1, I2 = compute_sequence_components(
                    seq_features['A'], seq_features['B'], seq_features['C']
                )
                features[f'{point}_{window_name}_I0'] = np.abs(I0)
                features[f'{point}_{window_name}_I1'] = np.abs(I1)
                features[f'{point}_{window_name}_I2'] = np.abs(I2)

    return features


def parse_filename(filename):
    """Parse metadata from filename."""
    name = filename.replace('_Data.csv', '').replace('_merged.csv', '').replace('_merged', '')
    parts = name.split('_')
    zone = ZONE_MAP.get(parts[0], 0)
    position = int(parts[1]) if len(parts) > 1 else 0
    fault_raw = parts[2] if len(parts) > 2 else 'UNKNOWN'
    fault_type = FAULT_MAP.get(fault_raw, fault_raw)
    return {'zone': zone, 'position': position, 'fault_type': fault_type, 'fault_raw': fault_raw}


# ============================================================
# DATASET CREATION
# ============================================================
def create_dataset():
    """Create ML-ready dataset from all CSV files."""
    print("=" * 70)
    print("CREATING ML-READY DATASET")
    print("=" * 70)

    csv_files = sorted([f for f in DATA_DIR.glob('*.csv')])
    print(f"Found {len(csv_files)} CSV files")

    rows = []
    for i, file_path in enumerate(csv_files):
        print(f"  [{i+1}/{len(csv_files)}] {file_path.name}")

        try:
            features = extract_file_features(file_path)
            metadata = parse_filename(file_path.name)
            row = {**features, **metadata}
            rows.append(row)
        except Exception as e:
            print(f"    ERROR: {e}")

    df = pd.DataFrame(rows)

    # Save raw feature dataset
    out_path = RESULTS_DIR / 'real_features_raw.csv'
    df.to_csv(out_path, index=False)
    print(f"\nSaved raw features: {out_path}")
    print(f"  Shape: {df.shape}")

    # Print class distribution
    print("\nFault type distribution:")
    print(df['fault_type'].value_counts().to_string())
    print("\nZone distribution:")
    print(df['zone'].value_counts().sort_index().to_string())
    print("\nPosition distribution:")
    print(df['position'].value_counts().sort_index().to_string())

    return df


# ============================================================
# MODEL TRAINING
# ============================================================
def select_ml_features(df):
    """Select optimal feature set for ML."""
    # Primary features: phasor magnitudes + angles + sequence components
    # Focus on R1, R2 (protection relays) + RE, SE (line ends)
    feature_cols = []

    for point in ['R1', 'R2', 'R3', 'RE', 'SE']:
        for phase in PHASES:
            # Current features (all windows)
            for window in ['pre', 'trans', 'fault']:
                feature_cols.extend([
                    f'{point}_{phase}_{window}_mag',
                    f'{point}_{phase}_{window}_ang',
                ])
            # Sequence components
            for window in ['pre', 'fault']:
                feature_cols.extend([
                    f'{point}_{window}_I0',
                    f'{point}_{window}_I1',
                    f'{point}_{window}_I2',
                ])

    # Voltage features (fault window only)
    for point in ['R1', 'R2', 'R3', 'RE', 'SE']:
        for phase in PHASES:
            feature_cols.extend([
                f'V_{point}_{phase}_fault_mag',
                f'V_{point}_{phase}_fault_ang',
            ])

    # Filter to available columns
    available = [c for c in feature_cols if c in df.columns]
    print(f"Selected {len(available)} ML features")

    return available


def prepare_targets(df):
    """Prepare target variables."""
    y_fault = df['fault_type'].values
    y_zone = df['zone'].values

    # Operating time targets (physics-based approximation)
    # For real dataset, we compute trip times using IDMT formula
    # based on fault current magnitude and relay settings
    y_t_R1, y_t_R2 = compute_operating_times(df)

    return y_fault, y_zone, y_t_R1, y_t_R2


def compute_operating_times(df):
    """Compute IDMT operating times using physics-based relay model."""
    # IEC Standard Inverse: t = TMS * k / (M^alpha - 1)
    # k=0.14, alpha=0.02 for SI curve
    k, alpha = 0.14, 0.02

    # Relay settings (from coordination)
    TMS_R1, TMS_R2 = 0.161, 0.05
    Is_R1, Is_R2 = 234.3, 203.7  # pickup currents (A)

    t_R1 = []
    t_R2 = []

    for _, row in df.iterrows():
        # Use fault window current at R1 and R2
        # Take max of three phases
        try:
            I_R1 = max(row[f'R1_{p}_fault_mag'] for p in PHASES)
            I_R2 = max(row[f'R2_{p}_fault_mag'] for p in PHASES)
        except KeyError:
            t_R1.append(-1.0)
            t_R2.append(-1.0)
            continue

        # Compute trip times
        def idmt_time(I, Is, TMS):
            M = I / Is
            if M <= 1.0:
                return -1.0
            t = TMS * k / (M**alpha - 1.0)
            return max(0.04, t)

        t1 = idmt_time(I_R1, Is_R1, TMS_R1)
        t2 = idmt_time(I_R2, Is_R2, TMS_R2)

        t_R1.append(t1)
        t_R2.append(t2)

    return np.array(t_R1), np.array(t_R2)


def run_classification_cv(name, X, y, cv_folds=5):
    """Run classification with cross-validation."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import classification_report, confusion_matrix

    print(f"\n=== {name} (CV={cv_folds}) ===")

    model = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=1000,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=None,
            min_samples_split=3
        )
    )

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"Individual folds: {cv_scores}")

    # Fit on full data
    model.fit(X, y)

    # Get CV predictions for analysis
    y_pred = cross_val_predict(model, X, y, cv=cv, n_jobs=-1)

    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, zero_division=0))

    cm = confusion_matrix(y, y_pred)
    print(f"Confusion Matrix:\n{cm}")

    return model, {
        'cv_acc_mean': cv_scores.mean(),
        'cv_acc_std': cv_scores.std(),
        'cv_scores': cv_scores.tolist(),
    }


def run_regression_cv(name, X, y, cv_folds=5):
    """Run regression with cross-validation."""
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.pipeline import make_pipeline

    print(f"\n=== {name} (CV={cv_folds}) ===")

    model = make_pipeline(
        StandardScaler(),
        RandomForestRegressor(
            n_estimators=1000,
            random_state=42,
            n_jobs=-1,
            max_depth=None,
            min_samples_split=3
        )
    )

    cv = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_mae = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
    cv_r2 = cross_val_score(model, X, y, cv=cv, scoring='r2', n_jobs=-1)

    print(f"CV MAE: {cv_mae.mean():.4f} (+/- {cv_mae.std()*2:.4f})")
    print(f"CV R²:  {cv_r2.mean():.4f} (+/- {cv_r2.std()*2:.4f})")

    # Fit on full data
    model.fit(X, y)

    return model, {
        'cv_mae_mean': cv_mae.mean(),
        'cv_mae_std': cv_mae.std(),
        'cv_r2_mean': cv_r2.mean(),
        'cv_r2_std': cv_r2.std(),
        'cv_mae': cv_mae.tolist(),
        'cv_r2': cv_r2.tolist(),
    }


def train_models(df):
    """Train all three protection models."""
    from sklearn.preprocessing import LabelEncoder
    import joblib

    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    # Select features
    feature_cols = select_ml_features(df)
    X = df[feature_cols].values

    # Prepare targets
    y_fault, y_zone, y_t_R1, y_t_R2 = prepare_targets(df)

    # Encode categorical targets
    le_fault = LabelEncoder()
    y_fault_enc = le_fault.fit_transform(y_fault)

    le_zone = LabelEncoder()
    y_zone_enc = le_zone.fit_transform(y_zone)

    print(f"\nFeature matrix: {X.shape}")
    print(f"Fault classes: {len(np.unique(y_fault_enc))}")
    print(f"Zone classes: {len(np.unique(y_zone_enc))}")

    models = {}
    metrics = {}

    # ============================================================
    # TASK 1: FAULT CLASSIFICATION
    # ============================================================
    model_fault, metrics_fault = run_classification_cv(
        'T1_Fault_Classification', X, y_fault_enc, cv_folds=5
    )
    models['fault_type'] = (model_fault, le_fault)
    metrics['fault_classification'] = metrics_fault

    # ============================================================
    # TASK 2: ZONE DETECTION
    # ============================================================
    model_zone, metrics_zone = run_classification_cv(
        'T2_Zone_Detection', X, y_zone_enc, cv_folds=5
    )
    models['zone'] = (model_zone, None)
    metrics['zone_detection'] = metrics_zone

    # ============================================================
    # TASK 3: OPERATING TIME REGRESSION
    # ============================================================
    r1_mask = y_t_R1 > 0
    r2_mask = y_t_R2 > 0

    if np.sum(r1_mask) > 5:
        model_t_R1, metrics_t_R1 = run_regression_cv(
            'T3a_R1_Trip_Time', X[r1_mask], y_t_R1[r1_mask], cv_folds=5
        )
        models['t_R1'] = (model_t_R1, None)
        metrics['R1_trip_time'] = metrics_t_R1
    else:
        print("\nInsufficient R1 operating cases for regression")
        models['t_R1'] = (None, None)
        metrics['R1_trip_time'] = {}

    if np.sum(r2_mask) > 5:
        model_t_R2, metrics_t_R2 = run_regression_cv(        'T3b_R2_Trip_Time', X[r2_mask], y_t_R2[r2_mask], cv_folds=5
        )
        models['t_R2'] = (model_t_R2, None)
        metrics['R2_trip_time'] = metrics_t_R2
    else:
        print("\nInsufficient R2 operating cases for regression")
        models['t_R2'] = (None, None)
        metrics['R2_trip_time'] = {}

    # ============================================================
    # SAVE MODELS
    # ============================================================
    print(f"\nSaving models to {RESULTS_DIR}...")

    valid_models = {k: v[0] for k, v in models.items() if v[0] is not None}
    joblib.dump(valid_models, RESULTS_DIR / 'models_real.joblib')
    joblib.dump({'fault_type': le_fault}, RESULTS_DIR / 'encoders_real.joblib')

    with open(RESULTS_DIR / 'metrics_real.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    # Save feature list
    with open(RESULTS_DIR / 'features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    print(f"  Saved {len(valid_models)} models")

    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)

    print(f"\nDataset: {len(df)} fault cases x {X.shape[1]} features")
    print(f"Fault type classes: {list(le_fault.classes_)}")
    print(f"Zone classes: {list(np.unique(y_zone))}")

    print(f"\nT1 Fault Classification:")
    print(f"  CV Accuracy: {metrics_fault['cv_acc_mean']:.3f} (+/- {metrics_fault['cv_acc_std']:.3f})")

    print(f"\nT2 Zone Detection:")
    print(f"  CV Accuracy: {metrics_zone['cv_acc_mean']:.3f} (+/- {metrics_zone['cv_acc_std']:.3f})")

    if metrics.get('R1_trip_time'):
        r1m = metrics['R1_trip_time']
        print(f"\nT3a R1 Trip Time:")
        print(f"  CV MAE: {r1m['cv_mae_mean']:.4f}s (+/- {r1m['cv_mae_std']:.4f}s)")
        print(f"  CV R²:  {r1m['cv_r2_mean']:.3f} (+/- {r1m['cv_r2_std']:.3f})")

    if metrics.get('R2_trip_time'):
        r2m = metrics['R2_trip_time']
        print(f"\nT3b R2 Trip Time:")
        print(f"  CV MAE: {r2m['cv_mae_mean']:.4f}s (+/- {r2m['cv_mae_std']:.4f}s)")
        print(f"  CV R²:  {r2m['cv_r2_mean']:.3f} (+/- {r2m['cv_r2_std']:.3f})")

    print(f"\nModels saved to: {RESULTS_DIR}")
    print("=" * 70)

    return models, metrics, feature_cols


def main():
    start_time = time.time()

    # Step 1: Create dataset
    df = create_dataset()

    # Step 2: Train models
    train_models(df)

    elapsed = time.time() - start_time
    print(f"\nTotal pipeline time: {elapsed:.1f}s")


if __name__ == '__main__':
    main()