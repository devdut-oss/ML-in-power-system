"""
simple_ml_pipeline.py
=====================
Simple ML pipeline for real MATLAB-derived CSV data.
Handles malformed files and creates feature dataset for training.

Tasks:
1. Fault Classification (AG, BC, ABC)
2. Zone Detection (1, 2, 3)
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
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\Copy of Fault_Dataset_All\Fault_Dataset_All\data set')
RESULTS_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\results_simple')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Sampling frequency (from data inspection)
FS = 500000  # Hz
F0 = 50.0    # Fundamental frequency

# ============================================================
# DATA LOADING
# ============================================================
def load_csv_file(file_path):
    """Load CSV file, handling various formats."""
    filename = file_path.name

    # Try different loading strategies
    strategies = [
        # Strategy 1: Normal CSV
        lambda: pd.read_csv(file_path),
        # Strategy 2: Comment lines
        lambda: pd.read_csv(file_path, comment='#'),
        # Strategy 3: Skip first line if it's a header comment
        lambda: pd.read_csv(file_path, skiprows=1),
        # Strategy 4: No header
        lambda: pd.read_csv(file_path, header=None),
    ]

    for i, strategy in enumerate(strategies):
        try:
            df = strategy()
            # Check if we got reasonable data
            if df is not None and len(df) > 10:
                # Check if it has numeric columns
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 5:
                    return df
        except Exception as e:
            continue

    # If all strategies failed, return None
    return None


def extract_features_from_dataframe(df):
    """Extract features from a loaded dataframe."""
    if df is None or len(df) < 10:
        return None

    # Get column names
    cols = df.columns.tolist()

    # Identify current and voltage columns (look for patterns)
    current_cols = [c for c in cols if any(p in c.upper() for p in ['I_R1', 'I_R2', 'I_R3', 'I_RE', 'I_SE'])]
    voltage_cols = [c for c in cols if any(p in c.upper() for p in ['V_R1', 'V_R2', 'V_R3', 'V_RE', 'V_SE'])]

    # If we can't find the expected columns, try positional approach
    if len(current_cols) < 3 and len(voltage_cols) < 3:
        # Assume first 15 columns are currents, next 15 are voltages (after time)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'time' in numeric_cols:
            numeric_cols.remove('time')

        if len(numeric_cols) >= 30:
            current_cols = numeric_cols[:15]
            voltage_cols = numeric_cols[15:30]
        else:
            return None

    # Extract phasor features from steady-state window (0.12-0.14s)
    if 'time' in df.columns:
        time_vals = df['time'].values
        # Steady-state fault window
        fault_mask = (time_vals >= 0.12) & (time_vals <= 0.14)
        if np.sum(fault_mask) < 10:
            # Try transient window
            fault_mask = (time_vals >= 0.08) & (time_vals <= 0.12)
        if np.sum(fault_mask) < 10:
            # Use last 20% of data
            fault_mask = (time_vals >= np.percentile(time_vals, 80))
    else:
        # If no time column, use all data
        fault_mask = slice(None)

    features = {}

    # Process current measurements
    for point_prefix in ['R1', 'R2', 'R3', 'RE', 'SE']:
        for phase in ['A', 'B', 'C']:
            # Look for column with this pattern
            col_name = None
            for col in current_cols:
                if point_prefix in col and phase in col:
                    col_name = col
                    break

            if col_name is None:
                continue

            # Get signal data
            signal = df[col_name].values
            if len(signal) == 0:
                continue

            # Apply window if we have time info
            if 'time' in df.columns and isinstance(fault_mask, np.ndarray):
                signal_window = signal[fault_mask] if len(signal) == len(fault_mask) else signal
            else:
                signal_window = signal

            if len(signal_window) < 5:
                continue

            # Compute phasor (simplified - just use RMS and mean for now)
            # In reality we'd do proper DFT, but for now use statistical features
            rms = np.sqrt(np.mean(signal_window**2))
            mean_abs = np.mean(np.abs(signal_window))
            std_dev = np.std(signal_window)
            peak = np.max(np.abs(signal_window))

            features[f'{point_prefix}_{phase}_rms'] = rms
            features[f'{point_prefix}_{phase}_mean_abs'] = mean_abs
            features[f'{point_prefix}_{phase}_std'] = std_dev
            features[f'{point_prefix}_{phase}_peak'] = peak
            features[f'{point_prefix}_{phase}_crest'] = peak / rms if rms > 0 else 0

    # Process voltage measurements similarly
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

            if 'time' in df.columns and isinstance(fault_mask, np.ndarray):
                signal_window = signal[fault_mask] if len(signal) == len(fault_mask) else signal
            else:
                signal_window = signal

            if len(signal_window) < 5:
                continue

            rms = np.sqrt(np.mean(signal_window**2))
            mean_abs = np.mean(np.abs(signal_window))
            std_dev = np.std(signal_window)
            peak = np.max(np.abs(signal_window))

            features[f'V_{point_prefix}_{phase}_rms'] = rms
            features[f'V_{point_prefix}_{phase}_mean_abs'] = mean_abs
            features[f'V_{point_prefix}_{phase}_std'] = std_dev
            features[f'V_{point_prefix}_{phase}_peak'] = peak
            features[f'V_{point_prefix}_{phase}_crest'] = peak / rms if rms > 0 else 0

    return features


def parse_filename(filename):
    """Parse metadata from filename."""
    # Remove extensions
    name = filename.replace('_Data.csv', '').replace('_merged.csv', '').replace('_merged', '')

    # Split by underscore
    parts = name.split('_')
    if len(parts) < 2:
        return None

    # Zone and position (e.g., Z1_25)
    zone_part = parts[0]
    pos_part = parts[1] if len(parts) > 1 else '0'

    zone = int(zone_part[1:]) if zone_part.startswith('Z') and len(zone_part) > 1 else 0
    position = int(pos_part) if pos_part.isdigit() else 0

    # Fault type (remaining parts)
    fault_raw = '_'.join(parts[2:]) if len(parts) > 2 else 'UNKNOWN'

    # Map to standard fault types
    fault_map = {
        'LG': 'AG',
        'LL': 'BC',
        'LLG': 'ABC',  # Simplified - treat as ABC
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


def create_dataset():
    """Create ML-ready dataset from all CSV files."""
    print("=" * 60)
    print("CREATING DATASET FROM CSV FILES")
    print("=" * 60)

    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])
    print(f"Found {len(csv_files)} CSV files")

    rows = []
    skipped = 0

    for i, filename in enumerate(csv_files):
        if i % 5 == 0:
            print(f"  Processed {i}/{len(csv_files)} files...")

        file_path = DATA_DIR / filename

        # Parse filename for metadata
        metadata = parse_filename(filename)
        if metadata is None:
            skipped += 1
            continue

        # Load CSV file
        df = load_csv_file(file_path)
        if df is None:
            skipped += 1
            continue

        # Extract features
        features = extract_features_from_dataframe(df)
        if features is None:
            skipped += 1
            continue

        # Combine metadata and features
        row = {**metadata, **features}
        rows.append(row)

    print(f"  Successfully processed: {len(rows)} files")
    print(f"  Skipped: {skipped} files")

    if len(rows) == 0:
        raise ValueError("No valid data files processed!")

    df = pd.DataFrame(rows)

    # Fill NaN values with 0
    df = df.fillna(0)

    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    # Show distributions
    print("\nFault type distribution:")
    print(df['fault_type'].value_counts().to_string())
    print("\nZone distribution:")
    print(df['zone'].value_counts().sort_index().to_string())
    print("\nPosition distribution:")
    print(df['position'].value_counts().sort_index().head().to_string())

    return df


def train_models(df):
    """Train ML models for the three tasks."""
    print("\n" + "=" * 60)
    print("TRAINING ML MODELS")
    print("=" * 60)

    # Prepare feature matrix (exclude metadata columns)
    exclude_cols = ['zone', 'position', 'fault_type', 'fault_raw']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    X = df[feature_cols].values

    print(f"Feature matrix: {X.shape[0]} samples x {X.shape[1]} features")

    # Prepare targets
    y_fault = df['fault_type'].values
    y_zone = df['zone'].values

    # Encode categorical targets
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

    # Use a smaller forest for speed (can increase later)
    rf_fault = make_pipeline(
        StandardScaler(),
        RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5
        )
    )

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf_fault, X, y_fault_enc, cv=cv, scoring='accuracy', n_jobs=-1)

    print(f"CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std()*2:.3f})")
    print(f"Folds: {['{:.3f}'.format(s) for s in cv_scores]}")

    # Train on full data
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
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5
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
    # TASK 3: OPERATING TIME REGRESSION (Simplified)
    # ============================================================
    print("\n--- TASK 3: OPERATING TIME REGRESSION ---")
    print("  Creating approximate trip time targets from RMS currents...")

    # Create approximate trip time targets based on fault current magnitude
    # Higher current = shorter trip time (inverse relationship)
    # We'll create a synthetic target for demonstration

    # Calculate approximate fault current (R1 RMS)
    r1_rms_cols = [c for c in df.columns if 'R1' in c and '_rms' in c and ('A' in c or 'B' in c or 'C' in c)]
    if len(r1_rms_cols) >= 3:
        # Use max of three phases as fault current
        fault_current = df[r1_rms_cols].max(axis=1).values

        # Inverse relationship: higher current = shorter trip time
        # Scale to reasonable range (0.05s to 1.0s)
        y_t_R1 = 1.0 / (fault_current / 1000.0 + 0.1)  # Simple inverse
        y_t_R1 = np.clip(y_t_R1, 0.05, 2.0)  # Reasonable bounds

        # For R2, typically half the time of R1 (coordination)
        y_t_R2 = y_t_R1 * 0.5

        # Train R1 model
        rf_t_R1 = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=10,
                min_samples_split=5
            )
        )

        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        cv_mae = -cross_val_score(rf_t_R1, X, y_t_R1, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_r2 = cross_val_score(rf_t_R1, X, y_t_R1, cv=cv, scoring='r2', n_jobs=-1)

        print(f"  R1 Trip Time: MAE={cv_mae.mean():.4f}s, R²={cv_r2.mean():.3f}")

        rf_t_R1.fit(X, y_t_R1)
        models['R1_trip_time'] = rf_t_R1
        metrics['R1_trip_time'] = {
            'cv_mae_mean': cv_mae.mean(),
            'cv_mae_std': cv_mae.std(),
            'cv_r2_mean': cv_r2.mean(),
            'cv_r2_std': cv_r2.std()
        }

        # Train R2 model
        rf_t_R2 = make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
                max_depth=10,
                min_samples_split=5
            )
        )

        cv_mae_r2 = -cross_val_score(rf_t_R2, X, y_t_R2, cv=cv, scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_r2_r2 = cross_val_score(rf_t_R2, X, y_t_R2, cv=cv, scoring='r2', n_jobs=-1)

        print(f"  R2 Trip Time: MAE={cv_mae_r2.mean():.4f}s, R²={cv_r2_r2.mean():.3f}")

        rf_t_R2.fit(X, y_t_R2)
        models['R2_trip_time'] = rf_t_R2
        metrics['R2_trip_time'] = {
            'cv_mae_mean': cv_mae_r2.mean(),
            'cv_mae_std': cv_mae_r2.std(),
            'cv_r2_mean': cv_r2_r2.mean(),
            'cv_r2_std': cv_r2_r2.std()
        }
    else:
        print("  Could not compute trip time targets - insufficient current columns")
        metrics['R1_trip_time'] = {}
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
    print("SIMPLE ML PIPELINE FOR MATLAB DATASET")
    print("=" * 60)
    print("Processing real Simulink CSV data for protection tasks")
    print("Tasks: Fault Classification | Zone Detection | Trip Time Regression")
    print()

    # Load and process data
    df = create_dataset()

    # Train models
    train_models(df)

    print("\nPipeline completed successfully!")


if __name__ == '__main__':
    main()