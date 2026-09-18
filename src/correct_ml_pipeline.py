"""
correct_ml_pipeline.py
========================
Correct ML pipeline for real MATLAB/Simulink CSV data.

KEY INSIGHT: Each CSV file = ONE fault scenario (44 total scenarios).
Each file has 100,000 time-domain samples of the SAME fault.

APPROACH:
1. Extract phasor features from steady-state fault window (0.12-0.14s)
2. Compute symmetrical components (I0, I1, I2)
3. Compute IDMT trip times using physics
4. Train models with proper validation (leave-one-out for small dataset)

Tasks:
1. Fault Classification (AG, BC, ABC)
2. Zone Detection (Z1, Z2, Z3)
3. Operating Time Regression (IDMT trip time)
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
from sklearn.model_selection import LeaveOneOut, cross_val_score, KFold
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\Copy of Fault_Dataset_All\Fault_Dataset_All\data set')
RESULTS_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\results_correct')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FS = 500000  # Hz
F0 = 50.0    # Fundamental frequency

# Fault type mapping
FAULT_MAP = {'LG': 'AG', 'LL': 'BC', 'LLG': 'ABC', 'LLL': 'ABC', 'LLLG': 'ABC'}
ZONE_MAP = {'Z1': 1, 'Z2': 2, 'Z3': 3}

# IDMT parameters (IEC SI curve)
TMS_R1, TMS_R2 = 0.161, 0.05
Is_R1, Is_R2 = 234.3, 203.7  # Pickup currents (A)
K_SI, ALPHA_SI = 0.14, 0.02


# ============================================================
# DATA LOADING
# ============================================================
def load_csv_file(file_path):
    """Load CSV file, handling various formats."""
    strategies = [
        lambda: pd.read_csv(file_path),
        lambda: pd.read_csv(file_path, comment='#'),
        lambda: pd.read_csv(file_path, skiprows=1),
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

    zone = ZONE_MAP.get(parts[0], 0)
    position = int(parts[1]) if parts[1].isdigit() else 0
    fault_raw = '_'.join(parts[2:]) if len(parts) > 2 else 'UNKNOWN'
    fault_type = FAULT_MAP.get(fault_raw, fault_raw)

    return {'zone': zone, 'position': position, 'fault_type': fault_type, 'fault_raw': fault_raw}


# ============================================================
# FEATURE EXTRACTION - CORRECTED
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
    """Compute symmetrical components (I0, I1, I2)."""
    a = np.exp(1j * 2 * np.pi / 3)
    A = np.array([[1, 1, 1], [1, a**2, a], [1, a, a**2]], dtype=complex)
    Iabc = np.array([Ia, Ib, Ic])
    I012 = np.linalg.solve(A, Iabc)
    return I012[0], I012[1], I012[2]


def extract_features_from_dataframe(df):
    """Extract phasor features from a loaded dataframe - CORRECT VERSION."""
    if df is None or len(df) < 10:
        return None

    cols = df.columns.tolist()

    # Identify current and voltage columns
    current_cols = []
    voltage_cols = []

    # Check if this is a Z3-style file (ref_ prefix)
    has_ref_cols = any(col.startswith('ref_') for col in cols)

    if has_ref_cols:
        # Z3 format: ref_2c_A, ref_Bb_A, ref_Fc_A, ref_Jd_A, ref_nd_A, etc.
        # Map to standard names
        z3_current_map = {
            'ref_2c': 'R1', 'ref_Bb': 'R2', 'ref_Fc': 'R3',
            'ref_Jd': 'SE', 'ref_nd': 'RE', 'ref_T': 'R2',
            'ref_Xb': 'RE', 'ref_fb': 'R2', 'ref_jc': 'R3', 'ref_x': 'RE'
        }
        z3_voltage_map = {
            'ref_T': 'R1', 'ref_Xb': 'R2', 'ref_fb': 'R3',
            'ref_jc': 'RE', 'ref_x': 'SE'
        }
        for col in cols:
            upper = col.upper()
            # Extract the base reference name (first segment before underscore)
            parts = col.split('_')
            base = parts[1] if len(parts) > 1 else ''  # e.g., '2c', 'Bb', 'Fc'
            if base in z3_current_map:
                current_cols.append(col)
            elif base in z3_voltage_map:
                voltage_cols.append(col)
    else:
        # Standard format: I_R1_A, I_R2_A, V_R1_A, V_R2_A
        for col in cols:
            upper = col.upper()
            if any(f'I_{p}_' in upper or f'I_{p}_' in col for p in ['R1', 'R2', 'R3', 'RE', 'SE']):
                current_cols.append(col)
            elif any(f'V_{p}_' in upper or f'V_{p}_' in col for p in ['R1', 'R2', 'R3', 'RE', 'SE']):
                voltage_cols.append(col)

    # Fallback: positional approach
    if len(current_cols) < 3:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'time' in numeric_cols:
            numeric_cols.remove('time')
        if len(numeric_cols) >= 30:
            current_cols = numeric_cols[:15]
            voltage_cols = numeric_cols[15:30]

    # Get time values
    time_col = None
    for c in cols:
        if 'time' in c.lower():
            time_col = c
            break

    if time_col is not None:
        time_vals = df[time_col].values
        fault_mask = (time_vals >= 0.12) & (time_vals <= 0.14)
        if np.sum(fault_mask) < 10:
            fault_mask = (time_vals >= 0.08) & (time_vals <= 0.12)
    else:
        fault_mask = slice(None)

    features = {}

    # Extract phasor features for currents (R1, R2 - protection relays)
    for point_prefix in ['R1', 'R2', 'R3', 'RE', 'SE']:
        for phase in ['A', 'B', 'C']:
            col_name = None
            for col in current_cols:
                # Match patterns like I_R1_A, I_R1_A, I_R1_phA, etc.
                col_upper = col.upper()
                if (f'{point_prefix}_' in col_upper or f'{point_prefix}' in col_upper) and phase in col_upper:
                    col_name = col
                    break

            if col_name is None:
                continue

            signal = df[col_name].values
            if len(signal) == 0:
                continue

            if isinstance(fault_mask, np.ndarray) and len(signal) == len(fault_mask):
                signal_window = signal[fault_mask]
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
                col_upper = col.upper()
                if (f'{point_prefix}_' in col_upper or f'{point_prefix}' in col_upper) and phase in col_upper:
                    col_name = col
                    break

            if col_name is None:
                continue

            signal = df[col_name].values
            if len(signal) == 0:
                continue

            if isinstance(fault_mask, np.ndarray) and len(signal) == len(fault_mask):
                signal_window = signal[fault_mask]
            else:
                signal_window = signal

            if len(signal_window) < 5:
                continue

            mag, ang = compute_phasor(signal_window, FS)
            features[f'V_{point_prefix}_{phase}_fault_mag'] = mag
            features[f'V_{point_prefix}_{phase}_fault_ang'] = ang

    # Compute sequence components for R1 and R2 currents
    for point_prefix in ['R1', 'R2']:
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

    # Additional features: RMS differences between phases
    for point_prefix in ['R1', 'R2']:
        for window in ['fault']:
            rms_vals = []
            for phase in ['A', 'B', 'C']:
                key = f'{point_prefix}_{phase}_{window}_rms'
                if key in features:
                    rms_vals.append(features[key])
            if len(rms_vals) == 3:
                features[f'{point_prefix}_{window}_rms_std'] = np.std(rms_vals)
                features[f'{point_prefix}_{window}_rms_max'] = np.max(rms_vals)
                features[f'{point_prefix}_{window}_rms_min'] = np.min(rms_vals)
                features[f'{point_prefix}_{window}_rms_ratio'] = np.max(rms_vals) / np.min(rms_vals) if np.min(rms_vals) > 0 else 0

    return features


def compute_idmt_trip_times(df):
    """Compute IDMT trip times using IEC standard formula."""
    t_R1 = []
    t_R2 = []

    for _, row in df.iterrows():
        # Get fault current magnitudes at R1 and R2 (max of three phases)
        try:
            I_R1_vals = [row.get(f'R1_{p}_fault_mag', 0) for p in ['A', 'B', 'C']]
            I_R2_vals = [row.get(f'R2_{p}_fault_mag', 0) for p in ['A', 'B', 'C']]
            I_R1 = np.max(I_R1_vals)
            I_R2 = np.max(I_R2_vals)
        except Exception:
            t_R1.append(-1.0)
            t_R2.append(-1.0)
            continue

        # IDMT trip time function
        def trip_time(I, Is, TMS):
            if I <= Is:
                return -1.0
            M = I / Is
            if M <= 1.0:
                return -1.0
            t = TMS * K_SI / (M**ALPHA_SI - 1.0)
            return max(0.04, t)

        t1 = trip_time(I_R1, Is_R1, TMS_R1)
        t2 = trip_time(I_R2, Is_R2, TMS_R2)

        t_R1.append(t1)
        t_R2.append(t2)

    return pd.Series(t_R1), pd.Series(t_R2)


# ============================================================
# DATASET CREATION
# ============================================================
def create_dataset():
    """Create ML-ready dataset from all CSV files."""
    print("=" * 60)
    print("CREATING ML-READY DATASET")
    print("=" * 60)

    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} CSV files")

    rows = []
    for i, filename in enumerate(csv_files):
        print(f"  [{i+1}/{len(csv_files)}] {filename}")

        file_path = DATA_DIR / filename
        metadata = parse_filename(filename)
        if metadata is None:
            print(f"    SKIP: Could not parse filename")
            continue

        df = load_csv_file(file_path)
        if df is None:
            print(f"    SKIP: Could not load file")
            continue

        features = extract_features_from_dataframe(df)
        if features is None:
            print(f"    SKIP: Feature extraction failed")
            continue

        row = {**metadata, **features}
        rows.append(row)
        print(f"    OK: zone={metadata['zone']}, pos={metadata['position']}, fault={metadata['fault_type']}")

    print(f"\nSuccessfully processed: {len(rows)} files")

    if len(rows) == 0:
        raise ValueError("No valid data files processed!")

    df = pd.DataFrame(rows)
    df = df.fillna(0)

    print(f"\nFinal dataset shape: {df.shape}")

    print("\nFault type distribution:")
    print(df['fault_type'].value_counts().to_string())
    print("\nZone distribution:")
    print(df['zone'].value_counts().sort_index().to_string())

    return df


# ============================================================
# MODEL TRAINING
# ============================================================
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
    print("  Using Leave-One-Out CV (small dataset)")

    rf_fault = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=5,
            min_samples_split=2
        )
    )

    # Leave-One-Out CV for small dataset
    loo = LeaveOneOut()
    cv_scores = cross_val_score(rf_fault, X, y_fault_enc, cv=loo, scoring='accuracy', n_jobs=-1)

    print(f"LOO CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"Correct: {cv_scores.sum():.0f}/{len(cv_scores)}")

    rf_fault.fit(X, y_fault_enc)
    models['fault_classification'] = rf_fault
    metrics['fault_classification'] = {
        'cv_acc_mean': cv_scores.mean(),
        'cv_acc_std': cv_scores.std(),
        'cv_correct': int(cv_scores.sum()),
        'cv_total': len(cv_scores)
    }

    # ============================================================
    # TASK 2: ZONE DETECTION
    # ============================================================
    print("\n--- TASK 2: ZONE DETECTION ---")
    print("  Using Leave-One-Out CV (small dataset)")

    rf_zone = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
            max_depth=5,
            min_samples_split=2
        )
    )

    cv_scores = cross_val_score(rf_zone, X, y_zone_enc, cv=loo, scoring='accuracy', n_jobs=-1)

    print(f"LOO CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")

    rf_zone.fit(X, y_zone_enc)
    models['zone_detection'] = rf_zone
    metrics['zone_detection'] = {
        'cv_acc_mean': cv_scores.mean(),
        'cv_acc_std': cv_scores.std(),
        'cv_correct': int(cv_scores.sum()),
        'cv_total': len(cv_scores)
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

    if r1_mask.sum() > 3:
        rf_t_R1 = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_depth=5,
                min_samples_split=2
            )
        )

        # Use LOO for small dataset
        loo = LeaveOneOut()
        cv_mae = -cross_val_score(rf_t_R1, X[r1_mask], y_t_R1[r1_mask], cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_r2 = cross_val_score(rf_t_R1, X[r1_mask], y_t_R1[r1_mask], cv=loo, scoring='r2', n_jobs=-1)

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

    if r2_mask.sum() > 3:
        rf_t_R2 = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                max_depth=5,
                min_samples_split=2
            )
        )

        cv_mae = -cross_val_score(rf_t_R2, X[r2_mask], y_t_R2[r2_mask], cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_r2 = cross_val_score(rf_t_R2, X[r2_mask], y_t_R2[r2_mask], cv=loo, scoring='r2', n_jobs=-1)

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

    # Save feature list for consistent feature extraction in demo
    with open(RESULTS_DIR / 'features.json', 'w') as f:
        json.dump(feature_cols, f, indent=2)

    print(f"  Saved {len(models)} models")

    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)

    print(f"\nDataset: {df.shape[0]} scenarios x {df.shape[1]} features")
    print(f"Features used: {X.shape[1]}")

    fault_m = metrics['fault_classification']
    zone_m = metrics['zone_detection']

    print(f"\nT1 Fault Classification:")
    print(f"  LOO Accuracy: {fault_m['cv_acc_mean']:.3f}")
    print(f"  Correct: {fault_m['cv_correct']}/{fault_m['cv_total']}")

    print(f"\nT2 Zone Detection:")
    print(f"  LOO Accuracy: {zone_m['cv_acc_mean']:.3f}")
    print(f"  Correct: {zone_m['cv_correct']}/{zone_m['cv_total']}")

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
    print("CORRECT ML PIPELINE FOR MATLAB DATASET")
    print("=" * 60)
    print("Processing real Simulink CSV data for protection tasks")
    print()

    df = create_dataset()
    train_models(df)

    print("\nPipeline completed!")


if __name__ == '__main__':
    main()