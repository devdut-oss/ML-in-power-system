"""
ml_real_pipeline_final.py
=========================
Final ML pipeline that properly handles real MATLAB-derived CSV data.

Issues with original data:
- Malformed CSV files with comment lines (e.g., "# faultType=LL position=25.0 zone=1.0")
- Different structure for different fault types
- Some files are just comments without actual data

SOLUTION: Parse comment lines, handle malformed files, create unified dataset.
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    train_test_split, cross_val_score,
    StratifiedKFold, KFold
)
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    mean_absolute_error, r2_score,
    classification_report
)

warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\Copy of Fault_Dataset_All\Fault_Dataset_All\data set')
RESULTS_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\results_final')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Time windows for feature extraction
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
    'AG': 'AG',
    'BG': 'BC',  # B-C
    'CG': 'BC',  # B-C
    'AB': 'BC',  # B-C
    'BC': 'BC',
    'CA': 'BC',  # B-C
    'ABG': 'BCG',
    'BCG': 'BCG',
    'CAG': 'BCG',
    'ABC': 'ABC',
    'ABCG': 'ABC',
}

# Zone mapping
ZONE_MAP = {'Z1': 1, 'Z2': 2, 'Z3': 3}


# ============================================================
# DATA LOADING AND PREPROCESSING
# ============================================================
def load_and_parse_file(file_path):
    """Load a CSV file, handling malformed files with comment lines."""
    filename = file_path.name

    # Parse filename to extract metadata
    if '_' not in filename:
        print(f"WARNING: {filename} has unexpected format")
        return None, None, None, None

    parts = filename.replace('_Data.csv', '').replace('_merged', '').split('_')
    zone_str = parts[0]  # e.g., Z1_25
    zone = ZONE_MAP.get(zone_str[:2], 0)
    position = int(zone_str[2:]) if len(zone_str) > 2 else 0
    fault_raw = parts[1] if len(parts) > 1 else 'UNKNOWN'
    fault_type = FAULT_MAP.get(fault_raw, fault_raw)

    # Read file with comment handling
    try:
        # First, read the first few lines to check for comments
        with open(file_path, 'r') as f:
            first_lines = []
            for i, line in enumerate(f):
                if i >= 10:  # Limit to first 10 lines
                    break
                if line.strip() and not line.startswith('#'):
                    first_lines.append(line)
                # Stop at blank line after potential comment
                if line.strip() == '' and len(first_lines) > 1:
                    break

        # If file starts with comment line, try to detect format
        if len(first_lines) > 0 and first_lines[0].startswith('#'):
            # Try to parse comment line for metadata
            comment_line = first_lines[0]
            if 'faultType=' in comment_line:
                # Parse fault type from comment
                for item in comment_line.split():
                    if 'faultType=' in item:
                        fault_type_from_comment = item.split('=')[1].split()[0]
                        # Map to standard
                        for key, val in FAULT_MAP.items():
                            if key in fault_type_from_comment or fault_type_from_comment in key:
                                fault_type = val
                                break
            print(f"  Parsed {filename}: zone={zone}, pos={position}, fault={fault_type}")

            # For malformed files, try to fix by skipping comment lines
            df = pd.read_csv(file_path, comment='#', skipinitialspace=True)

            # If the file has expected columns, use them
            expected_cols = ['time'] + [f'{p}_{q}' for p in CURRENT_POINTS for q in PHASES] + \
                           [f'{p}_{q}' for p in VOLTAGE_POINTS for q in PHASES]

            missing = [c for c in expected_cols if c not in df.columns]
            if missing:
                # If columns are missing, this is likely a problematic file
                print(f"    WARNING: {filename} missing columns {missing[:3]}...")
                return None, None, None, None

        else:
            # Normal CSV file
            df = pd.read_csv(file_path)

    except Exception as e:
        print(f"    ERROR loading {filename}: {e}")
        return None, None, None, None

    return df, zone, position, fault_type


def create_features_from_clean_csv(df, zone, position, fault_type):
    """Create ML features from a properly formatted CSV."""
    if df.empty:
        return None

    # Sampling frequency
    if len(df) > 1:
        fs = 1 / (df['time'].iloc[1] - df['time'].iloc[0])
    else:
        return None

    t = df['time'].values

    # Define windows
    windows = {
        'pre': (PRE_FAULT_START, PRE_FAULT_END),
        'trans': (TRANSIENT_START, TRANSIENT_END),
        'fault': (FAULT_START, FAULT_END),
    }

    features = {}

    # Extract phasors for each measurement point and phase
    for point in CURRENT_POINTS:
        for phase in PHASES:
            col_name = f'{point}_{phase}'
            if col_name not in df.columns:
                continue

            signal_data = df[col_name].values

            for window_name, (w_start, w_end) in windows.items():
                mask = (t >= w_start) & (t <= w_end)
                if np.sum(mask) < 20:  # Minimum samples
                    continue

                features_dict = extract_signal_features(signal_data[mask], t[mask], fs, window_name)
                for feat_name, value in features_dict.items():
                    features[f'{point}_{phase}_{feat_name}'] = value

    # Add metadata
    features['zone'] = zone
    features['position'] = position
    features['fault_type_raw'] = fault_type

    return features


def extract_signal_features(signal_data, t, fs, window_name):
    """Extract features from a signal window."""
    features = {}

    # Basic statistics
    features[f'{window_name}_mean'] = np.mean(signal_data)
    features[f'{window_name}_std'] = np.std(signal_data)
    features[f'{window_name}_rms'] = np.sqrt(np.mean(signal_data**2))
    features[f'{window_name}_kurt'] = kurtosis(signal_data) if len(signal_data) > 3 else 0
    features[f'{window_name}_skew'] = skew(signal_data) if len(signal_data) > 2 else 0

    # Peak features
    features[f'{window_name}_peak'] = np.max(np.abs(signal_data))
    features[f'{window_name}_peak_pos'] = t[np.argmax(np.abs(signal_data))]
    features[f'{window_name}_crest'] = features[f'{window_name}_peak'] / features[f'{window_name}_rms'] if features[f'{window_name}_rms'] > 0 else 0

    # Zero crossing rate
    zero_crossings = np.sum(np.diff(np.sign(signal_data)) != 0)
    features[f'{window_name}_zcr'] = zero_crossings / (len(signal_data) * fs)

    # FFT features
    if len(signal_data) > 20:
        fft_vals = np.abs(np.fft.rfft(signal_data))
        fft_freq = np.fft.rfftfreq(len(signal_data), 1/fs)

        # Energy in different frequency bands
        fund_mask = (fft_freq >= 49) & (fft_freq <= 51)
        harm2_mask = (fft_freq >= 99) & (fft_freq <= 101)
        harm3_mask = (fft_freq >= 149) & (fft_freq <= 151)

        features[f'{window_name}_energy_fund'] = np.sum(fft_vals[fund_mask]**2) if np.any(fund_mask) else 0
        features[f'{window_name}_energy_harm2'] = np.sum(fft_vals[harm2_mask]**2) if np.any(harm2_mask) else 0
        features[f'{window_name}_energy_harm3'] = np.sum(fft_vals[harm3_mask]**2) if np.any(harm3_mask) else 0

        if features[f'{window_name}_energy_fund'] > 0:
            features[f'{window_name}_harmonic_ratio'] = features[f'{window_name}_energy_harm2'] / features[f'{window_name}_energy_fund']
        else:
            features[f'{window_name}_harmonic_ratio'] = 0

    return features


def compute_sequence_components(Ia, Ib, Ic):
    """Compute symmetrical components (I0, I1, I2)."""
    a = np.exp(1j * 2 * np.pi / 3)
    A = np.array([[1, 1, 1], [1, a**2, a], [1, a, a**2]], dtype=complex)
    Iabc = np.array([Ia, Ib, Ic])
    I012 = np.linalg.solve(A, Iabc)
    return I012[0], I012[1], I012[2]


def extract_sequence_features(df):
    """Extract symmetrical components from phasor data."""
    seq_features = {}

    # Use only R1 and R2 for sequence analysis (protection relays)
    for point in ['R1', 'R2']:
        for phase in ['A', 'B', 'C']:
            col_name = f'{point}_{phase}_fault_magnitude'
            if col_name not in df.columns:
                continue

            # Get phasors (magnitude and angle)
            mag = df[col_name].values
            ang = df[f'{point}_{phase}_fault_angle'].values
            phasors = [mag[i] * np.exp(1j * np.deg2rad(ang[i])) for i in range(len(mag))]

            # Average across all samples for each measurement
            seq_features[f'{point}_I0_mean'] = np.abs(np.mean([p[0] for p in phasors]))
            seq_features[f'{point}_I1_mean'] = np.abs(np.mean([p[1] for p in phasors]))
            seq_features[f'{point}_I2_mean'] = np.abs(np.mean([p[2] for p in phasors]))

    return seq_features


def load_dataset():
    """Load and process all CSV files into a unified dataset."""
    print("=" * 70)
    print("LOADING REAL DATASET FROM MATLAB CSV FILES")
    print("=" * 70)

    csv_files = sorted(DATA_DIR.glob('*.csv'))
    print(f"Found {len(csv_files)} CSV files")

    processed_count = 0
    skipped_count = 0
    rows = []

    for i, file_path in enumerate(csv_files):
        print(f"  [{i+1}/{len(csv_files)}] {file_path.name}")

        # Try to load and parse
        df, zone, position, fault_type = load_and_parse_file(file_path)

        if df is not None and len(df) > 100:  # Has data
            features = create_features_from_clean_csv(df, zone, position, fault_type)
            if features:
                rows.append(features)
                processed_count += 1
            else:
                skipped_count += 1
        else:
            skipped_count += 1

    print(f"\nDataset processed successfully!")
    print(f"  Processed files: {processed_count}")
    print(f"  Skipped files: {skipped_count}")

    if not rows:
        raise ValueError("No valid data files found!")

    df = pd.DataFrame(rows)

    # Convert zone and position to numeric
    df['zone'] = pd.to_numeric(df['zone'], errors='coerce')
    df['position'] = pd.to_numeric(df['position'], errors='coerce')

    # Create fault type for ML
    df['ml_fault_type'] = df['fault_type_raw'].apply(lambda x: x if x in ['AG', 'BC', 'ABC'] else 'OTHER')

    # Create target variables
    df['target_zone'] = df['zone']

    # Compute operating times using physics-based IDMT formula
    df['target_t_R1'], df['target_t_R2'] = compute_idmt_trip_times(df)

    print(f"\nFinal dataset:")
    print(f"  Shape: {df.shape}")
    print(f"  Fault types: {df['ml_fault_type'].value_counts().to_dict()}")
    print(f"  Zones: {df['target_zone'].value_counts().sort_index().to_dict()}")

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
        # Get fault current magnitudes
        try:
            I_R1 = max(row.get(f'R1_{p}_fault_magnitude', 0) for p in ['A', 'B', 'C'])
            I_R2 = max(row.get(f'R2_{p}_fault_magnitude', 0) for p in ['A', 'B', 'C'])
        except:
            t_R1.append(-1.0)
            t_R2.append(-1.0)
            continue

        # IDMT trip time function
        def trip_time(I, Is, TMS):
            if I <= Is:
                return -1.0
            M = I / Is
            t = TMS * k / (M**alpha - 1.0)
            return max(0.04, t)  # minimum 2 cycles

        t1 = trip_time(I_R1, Is_R1, TMS_R1)
        t2 = trip_time(I_R2, Is_R2, TMS_R2)

        t_R1.append(t1)
        t_R2.append(t2)

    return pd.Series(t_R1), pd.Series(t_R2)


def select_features_for_ml(df):
    """Select optimal features for ML models."""
    # Focus on phasor magnitudes, angles, and sequence components
    feature_cols = []

    # Current phasors (R1, R2 - protection relays)
    for point in ['R1', 'R2']:
        for phase in ['A', 'B', 'C']:
            for window in ['pre', 'trans', 'fault']:
                feature_cols.extend([
                    f'{point}_{phase}_{window}_magnitude',
                    f'{point}_{phase}_{window}_angle',
                    f'{point}_{phase}_{window}_crest',
                    f'{point}_{phase}_{window}_zcr',
                ])

    # Sequence components
    feature_cols.extend([
        'R1_I0_mean', 'R1_I1_mean', 'R1_I2_mean',
        'R2_I0_mean', 'R2_I1_mean', 'R2_I2_mean',
    ])

    # Filter to available columns
    available = [c for c in feature_cols if c in df.columns]
    print(f"Selected {len(available)} ML features out of {len(feature_cols)} requested")

    return available


def train_all_models(df):
    """Train all three protection models."""
    from sklearn.preprocessing import LabelEncoder
    import joblib

    print("\n" + "=" * 70)
    print("TRAINING ML MODELS")
    print("=" * 70)

    # Select features
    feature_cols = select_features_for_ml(df)
    X = df[feature_cols].values

    # Prepare targets
    y_fault = df['ml_fault_type'].values
    y_zone = df['target_zone'].values
    y_t_R1 = df['target_t_R1'].values
    y_t_R2 = df['target_t_R2'].values

    print(f"\nFeature matrix: {X.shape}")
    print(f"Fault classes: {len(np.unique(y_fault))}")
    print(f"Zone classes: {len(np.unique(y_zone))}")

    # Encode categorical targets
    le_fault = LabelEncoder()
    y_fault_enc = le_fault.fit_transform(y_fault)

    le_zone = LabelEncoder()
    y_zone_enc = le_zone.fit_transform(y_zone)

    models = {}
    metrics = {}

    # ============================================================
    # TASK 1: FAULT CLASSIFICATION
    # ============================================================
    print("\n--- TASK 1: FAULT CLASSIFICATION ---")

    # Use cross-validation for better estimates
    from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict

    rf_fault = make_pipeline(
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

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_fault, X, y_fault_enc, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
    print(f"Individual folds: {cv_scores}")

    # Train on full data
    rf_fault.fit(X, y_fault_enc)

    # Get predictions for analysis
    y_pred = cross_val_predict(rf_fault, X, y_fault_enc, cv=cv, n_jobs=-1)

    from sklearn.metrics import classification_report
    print(f"\nClassification Report:")
    print(classification_report(y_fault, le_fault.inverse_transform(y_pred), zero_division=0))

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
            n_estimators=1000,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced',
            max_depth=None,
            min_samples_split=3
        )
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_zone, X, y_zone_enc, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

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

    # Prepare data for regression (only operating cases)
    r1_mask = (y_t_R1 > 0) & ~np.isnan(y_t_R1)
    r2_mask = (y_t_R2 > 0) & ~np.isnan(y_t_R2)

    if r1_mask.sum() > 5:
        rf_r1 = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=1000,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=3
            )
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(rf_r1, X[r1_mask], y_t_R1[r1_mask],
                                 cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_r2 = cross_val_score(rf_r1, X[r1_mask], y_t_R1[r1_mask],
                               cv=cv, scoring='r2', n_jobs=-1)

        print(f"R1 Trip Time CV: MAE={cv_mae.mean():.4f}s (+/- {cv_mae.std()*2:.4f}s), R²={cv_r2.mean():.3f} (+/- {cv_r2.std()*2:.3f})")

        rf_r1.fit(X[r1_mask], y_t_R1[r1_mask])
        models['R1_trip_time'] = rf_r1
        metrics['R1_trip_time'] = {
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std(),
        }
    else:
        print("  Insufficient R1 operating cases")
        metrics['R1_trip_time'] = {}

    if r2_mask.sum() > 5:
        rf_r2 = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=1000,
                random_state=42,
                n_jobs=-1,
                max_depth=None,
                min_samples_split=3
            )
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(rf_r2, X[r2_mask], y_t_R2[r2_mask],
                                 cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_r2 = cross_val_score(rf_r2, X[r2_mask], y_t_R2[r2_mask],
                               cv=cv, scoring='r2', n_jobs=-1)

        print(f"R2 Trip Time CV: MAE={cv_mae.mean():.4f}s (+/- {cv_mae.std()*2:.4f}s), R²={cv_r2.mean():.3f} (+/- {cv_r2.std()*2:.3f})")

        rf_r2.fit(X[r2_mask], y_t_R2[r2_mask])
        models['R2_trip_time'] = rf_r2
        metrics['R2_trip_time'] = {
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std(),
        }
    else:
        print("  Insufficient R2 operating cases")
        metrics['R2_trip_time'] = {}

    # ============================================================
    # SAVE MODELS
    # ============================================================
    print(f"\nSaving models to {RESULTS_DIR}...")

    joblib.dump(models, RESULTS_DIR / 'models.joblib')
    joblib.dump({'fault': le_fault, 'zone': le_zone}, RESULTS_DIR / 'label_encoders.joblib')

    with open(RESULTS_DIR / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"  Saved {len(models)} models")

    # ============================================================
    # PRINT SUMMARY
    # ============================================================
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 70)

    print(f"\nDataset size: {len(df):,} samples x {X.shape[1]} features")
    print(f"Fault type classes: {list(le_fault.classes_)}")
    print(f"Zone classes: {sorted(list(np.unique(y_zone)))}")

    # Task 1 results
    fault_metrics = metrics['fault_classification']
    print(f"\nT1 Fault Classification:")
    print(f"  CV Accuracy: {fault_metrics['cv_acc_mean']:.3f} (+/- {fault_metrics['cv_acc_std']:.3f})")

    # Task 2 results
    zone_metrics = metrics['zone_detection']
    print(f"\nT2 Zone Detection:")
    print(f"  CV Accuracy: {zone_metrics['cv_acc_mean']:.3f} (+/- {zone_metrics['cv_acc_std']:.3f})")

    # Task 3 results
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

    return models, metrics


def main():
    start_time = time.time()

    print("FINAL ML PIPELINE FOR REAL MATLAB DATASET")
    print("=" * 70)
    print("\nThis pipeline processes ~4.4M time-domain samples from")
    print("Simulink simulations, handling malformed CSV files.")
    print("\nTasks:")
    print("1. Fault Classification (LG, LL, LLG, LLL)")
    print("2. Zone Detection (Z1, Z2, Z3)")
    print("3. Operating Time Regression (IDMT trip time)")

    # Step 1: Load and preprocess data
    df = load_dataset()

    # Step 2: Train models
    train_all_models(df)

    elapsed = time.time() - start_time
    print(f"\nTotal processing time: {elapsed:.1f}s")

if __name__ == '__main__':
    main()