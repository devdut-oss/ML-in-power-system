"""
demo_final.py
=============
Demo script for the trained protection ML models.

Usage:
    python src/demo_final.py
    python src/demo_final.py AG 1 25
    python src/demo_final.py BC 2 50
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add src to path to import pipeline functions
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from correct_ml_pipeline import (
    load_csv_file, parse_filename,
    extract_features_from_dataframe, compute_phasor, compute_sequence_components
)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\results_correct')
DATA_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\Copy of Fault_Dataset_All\Fault_Dataset_All\data set')

# Load models and encoders
models = joblib.load(RESULTS_DIR / 'models.joblib')
encoders = joblib.load(RESULTS_DIR / 'encoders.joblib')

with open(RESULTS_DIR / 'metrics.json') as f:
    metrics = json.load(f)

print("=" * 70)
print("PROTECTION ML MODELS - DEMO")
print("=" * 70)
print(f"\nFault Classification Accuracy: {metrics['fault_classification']['cv_acc_mean']:.3f}")
print(f"Zone Detection Accuracy:       {metrics['zone_detection']['cv_acc_mean']:.3f}")

def predict_protection(file_path):
    """Run protection predictions on a fault scenario file."""
    filename = os.path.basename(file_path)
    metadata = parse_filename(filename)
    zone = metadata['zone']
    position = metadata['position']
    fault_type = metadata['fault_type']

    df = load_csv_file(file_path)
    features = extract_features_from_dataframe(df)
    if features is None:
        print(f"Could not extract features from {filename}")
        return None

    # Convert to feature vector matching training
    with open(RESULTS_DIR / 'features.json') as f:
        feature_cols = json.load(f)
    X = np.array([[features.get(c, 0.0) for c in feature_cols]], dtype=float)

    # Make predictions
    fault_pred_enc = models['fault_classification'].predict(X)[0]
    fault_label = encoders['fault'].inverse_transform([fault_pred_enc])[0]

    zone_pred_enc = models['zone_detection'].predict(X)[0]
    zone_pred = encoders['zone'].inverse_transform([zone_pred_enc])[0]

    t_R1 = models.get('R1_trip_time')
    t_R2 = models.get('R2_trip_time')

    t_R1_pred = t_R1.predict(X)[0] if t_R1 is not None else None
    t_R2_pred = t_R2.predict(X)[0] if t_R2 is not None else None

    return {
        'filename': filename,
        'zone': zone,
        'position': position,
        'actual_fault_type': fault_type,
        'predicted_fault_type': fault_label,
        'actual_zone': zone,
        'predicted_zone': int(zone_pred),
        'R1_trip_time': t_R1_pred,
        'R2_trip_time': t_R2_pred,
        'fault_match': fault_label == fault_type,
        'zone_match': int(zone_pred) == zone
    }


def main():
    print("\n" + "=" * 70)
    print("PROTECTION ML MODELS - INTERACTIVE DEMO")
    print("=" * 70)

    # Find all CSV files
    csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.csv')])

    if len(sys.argv) == 4:
        # Specific case: python demo_final.py AG 1 25
        fault_type = sys.argv[1]
        zone = int(sys.argv[2])
        position = int(sys.argv[3])

        # Find matching file
        fault_map = {'AG': 'LG', 'BC': 'LL', 'ABC': 'LLL'}
        fault_folder = fault_map.get(fault_type, fault_type)
        zone_str = f'Z{zone}'
        filename = f"{zone_str}_{position:02d}_{fault_folder}_Data.csv"

        file_path = DATA_DIR / filename
        if not file_path.exists():
            # Try without leading zero
            filename = f"{zone_str}_{position}_{fault_folder}_Data.csv"
            file_path = DATA_DIR / filename
    else:
        # Random case
        import random
        file_path = DATA_DIR / random.choice(csv_files)

    print(f"\nProcessing: {os.path.basename(file_path)}")

    result = predict_protection(file_path)
    if result is None:
        return

    # Display results
    print("\n" + "-" * 70)
    print("PROTECTION RESULTS")
    print("-" * 70)

    print(f"\nScenario: Zone {result['zone']}, Position {result['position']}%")
    print(f"Fault Type: {result['actual_fault_type']} (actual) -> {result['predicted_fault_type']} (predicted)")
    print(f"  Match: {'YES' if result['fault_match'] else 'NO'}")

    print(f"\nZone Detection: Zone {result['actual_zone']} (actual) -> Zone {result['predicted_zone']} (predicted)")
    print(f"  Match: {'YES' if result['zone_match'] else 'NO'}")

    if result['R1_trip_time'] is not None:
        print(f"\nOperating Time:")
        print(f"  R1 Trip Time: {result['R1_trip_time']:.4f} s")
        print(f"  R2 Trip Time: {result['R2_trip_time']:.4f} s")

        # Determine which relay operates
        if result['R1_trip_time'] > 0 and result['R2_trip_time'] > 0:
            if result['R1_trip_time'] < result['R2_trip_time']:
                relay = 'R1'
            else:
                relay = 'R2'
            print(f"  Relay that operates: {relay}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()