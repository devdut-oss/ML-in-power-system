"""
final_protection_pipeline.py
============================
FINAL working ML pipeline for real MATLAB/Simulink dataset.

DATA: 44 CSV files, each = ONE fault scenario with 100,000 time-domain samples
      (3 zones x 3 positions x 5 fault types = 45 scenarios, 44 files)

TASKS:
1. Fault Classification (AG, BC, ABC)
2. Zone Detection (Z1, Z2, Z3)
3. Operating Time Regression (IDMT trip time)

APPROACH:
- Extract phasor features (magnitude + angle) from steady-state fault window
- Compute symmetrical components (I0, I1, I2)
- Compute IDMT trip times from physics
- Train Random Forest models
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score, classification_report
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\Copy of Fault_Dataset_All\Fault_Dataset_All\data set')
RESULTS_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\results_final_protection')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Sampling frequency (from data inspection)
FS = 500000  # Hz
F0 = 50.0    # Fundamental frequency

# Time windows
PRE_FAULT_START, PRE_FAULT_END = 0.02, 0.04
FAULT_START, FAULT_END = 0.12, 0.14

# ============================================================
# DATA LOADING
# ============================================================
def load_csv_file(file_path):
    """Load CSV file, handling various formats."""
    strategies = [
        lambda: pd.read_csv(file_path),
        lambda: pd.read_csv(file_path, comment='#'),
        lambda: pd.read_csv(file_path, skiprows=1),
        lambda: pd.read_csv(file_path, header=None),
    ]

    for strategy in strategies:
        try:
            df = strategy()
            if df is not None and len(df) > 10:
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 5:
                    return df
        except Exception:
            continue
    return None


def parse_filename(filename):
    """Parse metadata from filename."""
    name = filename.replace('_Data.csv', '').replace('_merged.csv', '').replace('_merged', '')
    parts = name.split('_')
    if len(parts) < 2:
        return None

    zone_part = parts[0]
    pos_part = parts[1]

    zone = int(zone_part[1:]) if zone_part.startswith('Z') and len(zone_part) > 1 else 0
    position = int(pos_part) if pos_part.isdigit() else 0

    fault_raw = '_'.join(parts[2:]) if len(parts) > 2 else 'UNKNOWN'

    # Map to standard fault types
    fault_map = {
        'LG': 'AG',
        'LL': 'BC',
        'LLG': 'ABC',
        'LLL': 'ABC',
        'LLLG': 'ABC',
    }
    fault_type = fault_map.get(fault_raw, fault_raw)
    if fault_type not in ['AG', 'BC', 'ABC']:
        fault_type = 'OTHER'

    return {
        'zone': zone,
        'position': position,
        'fault_type': fault_type,
        'fault_raw': fault_raw
    }


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
    """Compute symmetrical components (I0, I1, I2)."""
    a = np.exp(1j * 2 * np.pi / 3)
    A = np.array([[1, 1, 1], [1, a**2, a], [1, a, a**2]], dtype=complex)
    Iabc = np.array([Ia, Ib, Ic])
    I012 = np.linalg.solve(A, Iabc)
    return I012[0], I012[1], I012[2]


def extract_features_from_dataframe(df):
    """Extract phasor features from a loaded dataframe."""
    if df is None or len(df) < 10:
        return None

    cols = df.columns.tolist()

    # Identify current and voltage columns
    current_cols = [c for c in cols if any(p in c.upper() for p in ['I_R1', 'I_R2', 'I_R3', 'I_RE', 'I_SE'])]
    voltage_cols = [c for c in cols if any(p in c.upper() for p in ['V_R1', 'V_R2', 'V_R3', 'V_RE', 'V_SE'])]

    # If we can't find expected columns, use positional approach
    if len(current_cols) < 3 and len(voltage_cols) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'time' in numeric_cols:
            numeric_cols.remove('time')
        if len(numeric_cols) >= 30:
            current_cols = numeric_cols[:15]
            voltage_cols = numeric_cols[15:30]
        else:
            return None

    # Identify time column
    time_col = 'time' if 'time' in cols else None
    if time_col is None:
        # Look for time-like column
        for c in cols:
            if 'time' in c.lower():
                time_col = c
                break

    # Get time values
    if time_col is not None:
        time_vals = df[time_col].values
        fault_mask = (time_vals >= FAULT_START) & (time_vals <= FAULT_END)
        if np.sum(fault_mask) < 10:
            fault_mask = (time_vals >= np.percentile(time_vals, 80))
    else:
        fault_mask = slice(None)

    features = {}

    # Extract phasor features for currents
    for point_prefix in ['R1', 'R2', 'R3', 'RE', 'SE']:
        for phase in ['A', 'B', 'C']:
            col_name = None
            for col in current_cols:
                if point_prefix in col and phase in col:
                    col_name = col
                    break

            if col_name is None:
                continue

            signal = df[col_name].values
            if len(signal) == 0:
                continue

            if isinstance(fault_mask, np.ndarray):
                signal_window = signal[fault_mask] if len(signal) == len(fault_mask) else signal
            else:
                signal_window = signal

            if len(signal_window) < 5:
                continue

            # Compute phasor
            mag, ang = compute_phasor(signal_window, FS)

            features[f'{point_prefix}_{phase}_fault_mag'] = mag
            features[f'{point_prefix}_{phase}_fault_ang'] = ang

            # Statistical features
            features[f'{point_prefix}_{phase}_fault_rms'] = np.sqrt(np.mean(signal_window**2))
            features[f'{point_prefix}_{phase}_fault_peak'] = np.max(np.abs(signal_window))
            features[f'{point_prefix}_{phase}_fault_crest'] = features[f'{point_prefix}_{phase}_fault_peak'] / features[f'{point_prefix}_{phase}_fault_rms'] if features[f'{point_prefix}_{phase}_fault_rms'] > 0 else 0

    # Extract phasor features for voltages
    for point_prefix in ['R1', 'R2', 'R3', 'RE', 'SE']:
        for phase in ['A', 'B', 'C']:
            col_name = None
            for col in voltage_cols:
                if point_prefix in col and phase in col:
                    col_name = col
                    break

            if col_name is None:
                continue

            signal = df[col_name].values
            if len(signal) == 0:
                continue

            if isinstance(fault_mask, np.ndarray):
                signal_window = signal[fault_mask] if len(signal) == len(fault_mask) else signal
            else:
                signal_window = signal

            if len(signal_window) < 5:
                continue

            mag, ang = compute_phasor(signal_window, FS)

            features[f'V_{point_prefix}_{phase}_fault_mag'] = mag
            features[f'V_{point_prefix}_{phase}_fault_ang'] = ang

    # Compute sequence components for R1 and R2 currents
    for point_prefix in ['R1', 'R2']:
        for phase in ['A', 'B', 'C']:
            if f'{point_prefix}_{phase}_fault_mag' not in features:
                continue

            # Convert phasors to complex
            phasors = {}
            for ph in ['A', 'B', 'C']:
                mag_key = f'{point_prefix}_{ph}_fault_mag'
                ang_key = f'{point_prefix}_{ph}_fault_ang'
                if mag_key in features and ang_key in features:
                    phasors[ph] = features[mag_key] * np.exp(1j * np.deg2rad(features[ang_key]))

            if all(ph in phasors for ph in ['A', 'B', 'C']):
                I0, I1, I2 = compute_sequence_components(phasors['A'], phasors['B'], phasors['C'])
                features[f'{point_prefix}_fault_I0'] = np.abs(I0)
                features[f'{point_prefix}_fault_I1'] = np.abs(I1)
                features[f'{point_prefix}_fault_I2'] = np.abs(I2)

    return features


def create_dataset():
    """Create ML-ready dataset from all CSV files."""
    print("=" * 60)
    print("CREATING ML-READY DATASET")
    print("=" * 60)

    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} CSV files")

    rows = []
    skipped = 0

    for i, filename in enumerate(csv_files):
        if i % 5 == 0:
            print(f"  Processed {i}/{len(csv_files)} files...")

        file_path = DATA_DIR / filename
        metadata = parse_filename(filename)
        if metadata is None:
            skipped += 1
            continue

        df = load_csv_file(file_path)
        if df is None:
            skipped += 1
            continue

        features = extract_features_from_dataframe(df)
        if features is None:
            skipped += 1
            continue

        row = {**metadata, **features}
        rows.append(row)

    print(f"  Successfully processed: {len(rows)} files")
    print(f"  Skipped: {skipped} files")

    if len(rows) == 0:
        raise ValueError("No valid data files processed!")

    df = pd.DataFrame(rows)
    df = df.fillna(0)

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    print("\nFault type distribution:")
    print(df['fault_type'].value_counts().to_string())
    print("\nZone distribution:")
    print(df['zone'].value_counts().sort_index().to_string())
    print("\nPosition distribution:")
    print(df['position'].value_counts().sort_index().head().to_string())

    return df


def compute_idmt_trip_times(df):
    """Compute IDMT trip times using IEC standard formula."""
    # IEC Standard Inverse: t = TMS * k / ((I/Is)^alpha - 1)
    k, alpha = 0.14, 0.02  # SI curve
    TMS_R1, TMS_R2 = 0.161, 0.05
    Is_R1, Is_R2 = 234.3, 203.7

    t_R1 = []
    t_R2 = []

    for _, row in df.iterrows():
        # Get fault current magnitudes at R1 and R2
        try:
            I_R1 = max(row.get(f'R1_{p}_fault_mag', 0) for p in ['A', 'B', 'C'])
            I_R2 = max(row.get(f'R2_{p}_fault_mag', 0) for p in ['A', 'B', 'C'])
        except:
            t_R1.append(-1.0)
            t_R2.append(-1.0)
            continue

        def trip_time(I, Is, TMS):
            if I <= Is:
                return -1.0
            M = I / Is
            t = TMS * k / (M**alpha - 1.0)
            return max(0.04, t)

        t1 = trip_time(I_R1, Is_R1, TMS_R1)
        t2 = trip_time(I_R2, Is_R2, TMS_R2)

        t_R1.append(t1)
        t_R2.append(t2)

    return pd.Series(t_R1), pd.Series(t_R2)


def train_models(df):
    """Train ML models for the three tasks."""
    print("\n" + "=" * 60)
    print("TRAINING ML MODELS")
    print("=" * 60)

    # Prepare feature matrix
    exclude_cols = ['zone', 'position', 'fault_type', 'fault_raw']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values

    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # Prepare targets
    y_fault = df['fault_type'].values
    y_zone = df['zone'].values

    le_fault = LabelEncoder()
    y_fault_enc = le_fault.fit_transform(y_fault)

    le_zone = LabelEncoder()
    y_zone_enc = le_zone.fit_transform(y_zone)

    print(f"\nTarget classes:")
    print(f"  Fault types: {list(le_fault.classes_)}")
    print(f"  Zones: {list(np.unique(y_zone))}")

    models = {}
    metrics = {}

    # ============================================================
    # TASK 1: FAULT CLASSIFICATION
    # ============================================================
    print("\n--- TASK 1: FAULT CLASSIFICATION ---")

    rf_fault = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=None,
            min_samples_split=3
        )
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_fault, X, y_fault_enc, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"Folds: {['{:.3f}'.format(s) for s in cv_scores]}")

    rf_fault.fit(X, y_fault_enc)
    models['fault_classification'] = rf_fault
    metrics['fault_classification'] = {
        'cv_acc_mean': cv_scores.mean(),
        'cv_acc_std': cv_scores.std(),
        'cv_scores': cv_scores.tolist()
    }

    # ============================================================
    # TASK 2: ZONE DETECTION
    # ============================================================
    print("\n--- TASK 2: ZONE DETECTION ---")

    rf_zone = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=500,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=None,
            min_samples_split=3
        )
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_zone, X, y_zone_enc, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    rf_zone.fit(X, y_zone_enc)
    models['zone_detection'] = rf_zone
    metrics['zone_detection'] = {
        'cv_acc_mean': cv_scores.mean(),
        'cv_acc_std': cv_scores.std(),
        'cv_scores': cv_scores.tolist()
    }

    # ============================================================
    # TASK 3: OPERATING TIME REGRESSION
    # ============================================================
    print("\n--- TASK 3: OPERATING TIME REGRESSION ---")
    print("  Computing IDMT trip times from fault currents...")

    y_t_R1, y_t_R2 = compute_idmt_trip_times(df)

    r1_mask = y_t_R1 > 0
    r2_mask = y_t_R2 > 0

    print(f"  R1 operating cases: {r1_mask.sum()}")
    print(f"  R2 operating cases: {r2_mask.sum()}")

    if r1_mask.sum() > 5:
        rf_t_R1 = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=3
            )
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(rf_t_R1, X[r1_mask], y_t_R1[r1_mask], cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_r2 = cross_val_score(rf_t_R1, X[r1_mask], y_t_R1[r1_mask], cv=cv, scoring='r2', n_jobs=-1)

        print(f"  R1 Trip Time: MAE={cv_mae.mean():.4f}s, R²={cv_r2.mean():.3f}")

        rf_t_R1.fit(X[r1_mask], y_t_R1[r1_mask])
        models['R1_trip_time'] = rf_t_R1
        metrics['R1_trip_time'] = {
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std()
        }
    else:
        print("  Insufficient R1 operating cases")
        metrics['R1_trip_time'] = {}

    if r2_mask.sum() > 5:
        rf_t_R2 = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=500,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=3
            )
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(rf_t_R2, X[r2_mask], y_t_R2[r2_mask], cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_r2 = cross_val_score(rf_t_R2, X[r2_mask], y_t_R2[r2_mask], cv=cv, scoring='r2', n_jobs=-1)

        print(f"  R2 Trip Time: MAE={cv_mae.mean():.4f}s, R²={cv_r2.mean():.3f}")

        rf_t_R2.fit(X[r2_mask], y_t_R2[r2_mask])
        models['R2_trip_time'] = rf_t_R2
        metrics['R2_trip_time'] = {
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std()
        }
    else:
        print("  Insufficient R2 operating cases")
        metrics['R2_trip_time'] = {}

    # ============================================================
    # SAVE MODELS
    # ============================================================
    print(f"\nSaving models to {RESULTS_DIR}...")

    import joblib
    joblib.dump(models, RESULTS_DIR / 'models.joblib')
    joblib.dump({'fault': le_fault, 'zone': le_zone}, RESULTS_DIR / 'encoders.joblib')

    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved {len(models)} models")

    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nDataset: {df.shape[0]} samples x {df.shape[1]} features")
    print(f"Features used: {X.shape[1]}")

    fault_m = metrics['fault_classification']
    zone_m = metrics['zone_detection']

    print(f"\nT1 Fault Classification:")
    print(f"  Accuracy: {fault_m['cv_acc_mean']:.3f} +/- {fault_m['cv_acc_std']:.3f}")

    print(f"\nT2 Zone Detection:")
    print(f"  Accuracy: {zone_m['cv_acc_mean']:.3f} +/- {zone_m['cv_acc_std']:.3f}")

    if 'R1_trip_time' in metrics:
        r1m = metrics['R1_trip_time']
        print(f"\nT3a R1 Trip Time:")
        print(f"  MAE: {r1m['cv_mae_mean']:.4f}s")
        print(f"  R²:  {r1m['cv_r2_mean']:.3f}")

    if 'R2_trip_time' in metrics:
        r2m = metrics['R2_trip_time']
        print(f"\nT3b R2 Trip Time:")
        print(f"  MAE: {r2m['cv_mae_mean']:.4f}s")
        print(f"  R²:  {r2m['cv_r2_mean']:.3f}")

    print(f"\nModels saved to: {RESULTS_DIR}")
    print("=" * 60)

    return models, metrics


def main():
    print("FINAL PROTECTION ML PIPELINE")
    print("=" * 60)
    print("Processing real MATLAB/Simulink CSV data")
    print("Tasks: Fault Classification | Zone Detection | IDMT Trip Time")
    print()

    df = create_dataset()
    train_models(df)

    print("\nPipeline completed successfully!")


if __name__ == '__main__':
    main()