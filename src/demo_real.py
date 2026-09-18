"""
demo_real.py
============
Interactive demonstration of the three ML protection tasks on REAL dataset:
1. Fault Classification (fault type)
2. Zone Detection (zone of operation)
3. Operating Time Regression (IDMT trip time)

Usage:
    python src/demo_real.py                    # random fault from real dataset
    python src/demo_real.py AG 1 0.25          # specify: fault_type zone position
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RESULTS_DIR = os.path.join(ROOT, 'results_real')
DATA_DIR = os.path.join(ROOT, 'data')

sys.path.insert(0, HERE)
from power_system import PowerSystem
from idmt_relay import coordinate, relay_decision
from convert_matlab_dataset import extract_features_from_mat, FAULT_MAP, SIGNAL_REFS, MATLAB_ROOT


def load_models():
    """Load trained models and encoders."""
    models = joblib.load(os.path.join(RESULTS_DIR, 'models_real.joblib'))
    encoders = joblib.load(os.path.join(RESULTS_DIR, 'encoders_real.joblib'))
    return models, encoders


def load_real_dataset():
    """Load the real dataset for random sampling."""
    df = pd.read_csv(os.path.join(DATA_DIR, 'fault_dataset_real.csv'))
    return df


def predict_fault_type(models, encoders, features):
    """Predict fault type."""
    model = models['T1_fault_type']
    le = encoders['fault_type']
    X = features.reshape(1, -1)
    pred_encoded = model.predict(X)[0]
    return le.inverse_transform([pred_encoded])[0]


def predict_zone(models, features):
    """Predict zone of operation."""
    model = models['T2_zone']
    X = features.reshape(1, -1)
    return int(model.predict(X)[0])


def predict_trip_times(models, features):
    """Predict R1 and R2 trip times."""
    X = features.reshape(1, -1)
    t_R1 = models['T3a_t_R1'].predict(X)[0] if models['T3a_t_R1'] else -1
    t_R2 = models['T3b_t_R2'].predict(X)[0] if models['T3b_t_R2'] else -1
    return max(0, t_R1), max(0, t_R2)


def get_features_from_matlab(fault_type, zone, position_pct):
    """Extract features from MATLAB file for given fault parameters."""
    # Find matching file
    zone_str = f"Z{zone}_{int(position_pct*100):02d}"
    # Map fault type to folder name
    folder_map = {v: k for k, v in FAULT_MAP.items()}
    fault_folder = folder_map.get(fault_type, fault_type)

    mat_path = MATLAB_ROOT / zone_str / fault_folder / f"{zone_str}_{fault_folder}_Data.mat"
    if not mat_path.exists():
        # Try alternative naming
        for f in (MATLAB_ROOT / zone_str).rglob("*_Data.mat"):
            if fault_folder in f.name:
                mat_path = f
                break

    if not mat_path.exists():
        raise FileNotFoundError(f"No MATLAB file for {fault_type} zone {zone} pos {position_pct}")

    features, _, _, _ = extract_features_from_mat(mat_path)
    return features


def get_physics_baseline(fault_type, zone, position_pct):
    """Get physics-based ground truth using power system model."""
    ps = PowerSystem()
    R1, R2 = coordinate(ps)

    if zone == 1:
        dist_km = position_pct * ps.L1
        sim_zone = 1
    elif zone == 2:
        dist_km = ps.L1 + position_pct * ps.L2
        sim_zone = 2
    else:
        dist_km = ps.L1 + ps.L2 + position_pct * 40.0
        sim_zone = 2

    Rf = 0.01
    lf = 1.0
    meas = ps.simulate_fault(fault_type, sim_zone, min(dist_km, ps.L1) if sim_zone == 1 else dist_km, Rf, lf)
    dec = relay_decision(R1, R2, meas, sim_zone, fault_type)

    return {
        'correct_relay': dec['correct_relay'],
        't_R1_physics': dec['t_R1'],
        't_R2_physics': dec['t_R2'],
        'coordinated': dec['coordinated'],
        'distance_km': dist_km,
    }


def run_demo(fault_type=None, zone=None, position_pct=None):
    """Run demonstration."""
    print("=" * 70)
    print("REAL DATASET ML PROTECTION DEMO")
    print("=" * 70)

    # Load models
    models, encoders = load_models()
    df_real = load_real_dataset()

    # Select case
    if fault_type is None:
        # Random fault case
        fault_cases = df_real[df_real['fault'] == 1]
        row = fault_cases.sample(1, random_state=np.random.randint(1000)).iloc[0]
        fault_type = row['fault_type']
        zone = int(row['zone'])
        position_pct = 0.25 if '25' in str(row.name) else (0.5 if '50' in str(row.name) else 0.75)
        print(f"\nRandom case selected: {fault_type}, Zone {zone}, Pos ~{position_pct*100:.0f}%")
    else:
        print(f"\nSpecified case: {fault_type}, Zone {zone}, Pos {position_pct*100:.0f}%")

    # Get features from MATLAB file
    print("\nExtracting features from MATLAB/Simulink data...")
    try:
        features_dict = get_features_from_matlab(fault_type, zone, position_pct)
        # Convert to array in correct order
        FEATURES = [
            'R1_Ia', 'R1_Ib', 'R1_Ic', 'R1_I0', 'R1_I1', 'R1_I2',
            'R1_Ia_ang', 'R1_Ib_ang', 'R1_Ic_ang',
            'R2_Ia', 'R2_Ib', 'R2_Ic', 'R2_I0', 'R2_I1', 'R2_I2',
            'R2_Ia_ang', 'R2_Ib_ang', 'R2_Ic_ang',
            'VA_a', 'VA_b', 'VA_c', 'VA_V0', 'VA_V2',
            'VB_a', 'VB_b', 'VB_c', 'VB_V0', 'VB_V2',
        ]
        features = np.array([features_dict[f] for f in FEATURES])
        print("Features extracted successfully.")
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Get physics baseline
    physics = get_physics_baseline(fault_type, zone, position_pct)

    # ============================================================
    # PREDICTIONS
    # ============================================================
    print("\n" + "=" * 70)
    print("ML PREDICTIONS vs PHYSICS GROUND TRUTH")
    print("=" * 70)

    # 1. Fault Classification
    pred_type = predict_fault_type(models, encoders, features)
    print(f"\n1. FAULT CLASSIFICATION")
    print(f"   Physics (label):  {fault_type}")
    print(f"   ML Prediction:    {pred_type}")
    print(f"   Match: {'[OK]' if pred_type == fault_type else '[FAIL]'}")

    # 2. Zone Detection
    pred_zone = predict_zone(models, features)
    print(f"\n2. ZONE DETECTION (Zone of Operation)")
    print(f"   Physics (label):  Zone {zone}")
    print(f"   ML Prediction:    Zone {pred_zone}")
    print(f"   Match: {'[OK]' if pred_zone == zone else '[FAIL]'}")

    # 3. Operating Time
    t_R1_ml, t_R2_ml = predict_trip_times(models, features)
    print(f"\n3. OPERATING TIME REGRESSION (IDMT Trip Time)")
    print(f"   Physics R1: {physics['t_R1_physics']:.4f} s  |  ML R1: {t_R1_ml:.4f} s  |  Error: {abs(t_R1_ml - physics['t_R1_physics'])*1000:.1f} ms")
    print(f"   Physics R2: {physics['t_R2_physics']:.4f} s  |  ML R2: {t_R2_ml:.4f} s  |  Error: {abs(t_R2_ml - physics['t_R2_physics'])*1000:.1f} ms")

    # Relay Selection
    print(f"\n4. RELAY SELECTION (which relay operates)")
    print(f"   Physics: {physics['correct_relay']}")
    # Determine from trip times
    if t_R1_ml < t_R2_ml and t_R1_ml > 0:
        pred_relay = 'R1'
    elif t_R2_ml < t_R1_ml and t_R2_ml > 0:
        pred_relay = 'R2'
    else:
        pred_relay = 'NONE'
    print(f"   ML (from times): {pred_relay}")
    print(f"   Match: {'[OK]' if pred_relay == physics['correct_relay'] else '[FAIL]'}")

    # Coordination check
    print(f"\n5. COORDINATION CHECK")
    ml_margin = t_R1_ml - t_R2_ml if (t_R1_ml > 0 and t_R2_ml > 0) else 'N/A'
    phys_margin = physics['t_R1_physics'] - physics['t_R2_physics'] if (physics['t_R1_physics'] > 0 and physics['t_R2_physics'] > 0) else 'N/A'
    print(f"   Physics margin (R1-R2): {phys_margin:.4f} s" if isinstance(phys_margin, float) else f"   Physics margin: {phys_margin}")
    print(f"   ML margin (R1-R2):      {ml_margin:.4f} s" if isinstance(ml_margin, float) else f"   ML margin: {ml_margin}")
    print(f"   Physics coordinated: {physics['coordinated']}")

    # Feature importance (for insight)
    print(f"\n6. TOP FEATURES (Random Forest importance)")
    rf_model = models['T1_fault_type'].named_steps['randomforestclassifier']
    importances = rf_model.feature_importances_
    FEATURES = [
        'R1_Ia', 'R1_Ib', 'R1_Ic', 'R1_I0', 'R1_I1', 'R1_I2',
        'R1_Ia_ang', 'R1_Ib_ang', 'R1_Ic_ang',
        'R2_Ia', 'R2_Ib', 'R2_Ic', 'R2_I0', 'R2_I1', 'R2_I2',
        'R2_Ia_ang', 'R2_Ib_ang', 'R2_Ic_ang',
        'VA_a', 'VA_b', 'VA_c', 'VA_V0', 'VA_V2',
        'VB_a', 'VB_b', 'VB_c', 'VB_V0', 'VB_V2',
    ]
    idx = np.argsort(importances)[::-1][:10]
    for i in idx:
        print(f"   {FEATURES[i]:<15} {importances[i]:.4f}")

    print("\n" + "=" * 70)


if __name__ == '__main__':
    if len(sys.argv) == 4:
        fault_type = sys.argv[1]
        zone = int(sys.argv[2])
        position_pct = float(sys.argv[3])
        run_demo(fault_type, zone, position_pct)
    else:
        run_demo()