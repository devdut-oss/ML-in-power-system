"""
convert_matlab_dataset.py
=========================
Convert MATLAB v7.3 .mat files (from Simulink) to CSV dataset for ML training.

Data structure:
- 3 zones (Z1, Z2, Z3) × 3 positions (25%, 50%, 75%) = 9 locations
- 5 fault types (LG, LL, LLG, LLL, LLLG) per location
- Total: 45 .mat files
- Each file: 5 current measurement points (R1, R2, R3, Sending, Receiving) × 3 phases
- Time-domain: 0-0.2s at 500 kHz (100001 samples)
- Fault inception ~0.08-0.10s, steady-state fault 0.12-0.2s
"""

import os
import sys
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path for power_system, idmt_relay
sys.path.insert(0, os.path.dirname(__file__))
from power_system import PowerSystem
from idmt_relay import coordinate, relay_decision

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MATLAB_ROOT = Path(r"C:\Users\sunda\PowerSystemProtection_ML\Copy of Fault_Dataset_All\Fault_Dataset_All")
OUTPUT_CSV = Path(r"C:\Users\sunda\PowerSystemProtection_ML\data\fault_dataset_real.csv")

# Fault mapping from folder names to standard labels
FAULT_MAP = {
    'LG': 'AG',      # Single line-ground (phase A reference)
    'LL': 'BC',      # Line-line (B-C reference)
    'LLG': 'BCG',    # Double line-ground (B-C-G reference)
    'LLL': 'ABC',    # Three-phase
    'LLLG': 'ABC',   # Three-phase-ground -> treat as ABC (ground doesn't change 3ph)
}

# Zone mapping
ZONE_MAP = {
    'Z1': 1,
    'Z2': 2,
    'Z3': 3,
}

# Position mapping (percentage of line length)
POS_MAP = {
    '25': 0.25,
    '50': 0.50,
    '75': 0.75,
}

# Signal reference IDs in .mat files (identified from exploration)
SIGNAL_REFS = {
    'R1_I': '2c',      # Relay 1 currents
    'R2_I': 'Bb',      # Relay 2 currents
    'R3_I': 'Fc',      # Relay 3 currents
    'SEND_I': 'Jd',    # Sending end currents
    'RECV_I': 'nd',    # Receiving end currents
}

# Time window for steady-state fault phasor extraction (seconds)
FAULT_WINDOW_START = 0.12
FAULT_WINDOW_END = 0.14

# Sampling frequency (from time vector)
FS = 500000  # Hz
F0 = 50      # Fundamental frequency


# ---------------------------------------------------------------------------
# Phasor computation
# ---------------------------------------------------------------------------
def compute_phasor(signal, fs=FS, f0=F0):
    """Compute phasor (magnitude, angle in degrees) using DFT at fundamental frequency."""
    N = len(signal)
    t = np.arange(N) / fs
    cos = np.cos(2 * np.pi * f0 * t)
    sin = np.sin(2 * np.pi * f0 * t)
    real = 2.0 / N * np.sum(signal * cos)
    imag = -2.0 / N * np.sum(signal * sin)
    mag = np.sqrt(real**2 + imag**2)
    ang = np.angle(real + 1j * imag, deg=True)
    return mag, ang


def compute_sequence_components(Ia, Ib, Ic):
    """Compute symmetrical components from phase phasors (complex)."""
    a = np.exp(1j * 2 * np.pi / 3)
    A = np.array([[1, 1, 1],
                  [1, a**2, a],
                  [1, a, a**2]], dtype=complex)
    Iabc = np.array([Ia, Ib, Ic])
    I012 = np.linalg.solve(A, Iabc)
    return I012[0], I012[1], I012[2]  # I0, I1, I2


def extract_features_from_mat(mat_path):
    """Extract phasor features from a single .mat file."""
    with h5py.File(mat_path, 'r') as f:
        refs = f['#refs#']
        tout = refs['Kd']['tout'][0]

        # Find fault window indices
        fault_idx = np.where((tout >= FAULT_WINDOW_START) & (tout < FAULT_WINDOW_END))[0]
        if len(fault_idx) == 0:
            raise ValueError(f"No samples in fault window for {mat_path}")

        # Extract phasors for each measurement point
        features = {}
        phasors = {}  # Store complex phasors for sequence components

        for meas_name, ref_key in SIGNAL_REFS.items():
            sig = refs[ref_key][:, fault_idx]  # shape (3, N)

            # Compute phasor for each phase
            Ia_mag, Ia_ang = compute_phasor(sig[0])
            Ib_mag, Ib_ang = compute_phasor(sig[1])
            Ic_mag, Ic_ang = compute_phasor(sig[2])

            # Store magnitude and angle features - use training-compatible names
            # Map measurement names to training feature prefixes
            name_map = {
                'R1_I': 'R1',
                'R2_I': 'R2',
                'R3_I': 'R3',      # Not used in training
                'SEND_I': 'SEND',  # Not used in training
                'RECV_I': 'RECV',  # Not used in training
            }
            prefix = name_map.get(meas_name, meas_name)

            features[f'{prefix}_Ia'] = Ia_mag
            features[f'{prefix}_Ib'] = Ib_mag
            features[f'{prefix}_Ic'] = Ic_mag
            features[f'{prefix}_Ia_ang'] = Ia_ang
            features[f'{prefix}_Ib_ang'] = Ib_ang
            features[f'{prefix}_Ic_ang'] = Ic_ang

            # Store complex phasors for sequence calculation
            phasors[meas_name] = {
                'Ia': Ia_mag * np.exp(1j * np.deg2rad(Ia_ang)),
                'Ib': Ib_mag * np.exp(1j * np.deg2rad(Ib_ang)),
                'Ic': Ic_mag * np.exp(1j * np.deg2rad(Ic_ang)),
            }

        # Compute sequence components for each measurement point
        name_map = {
            'R1_I': 'R1',
            'R2_I': 'R2',
            'R3_I': 'R3',
            'SEND_I': 'SEND',
            'RECV_I': 'RECV',
        }
        for meas_name in SIGNAL_REFS.keys():
            prefix = name_map.get(meas_name, meas_name)
            I0, I1, I2 = compute_sequence_components(
                phasors[meas_name]['Ia'],
                phasors[meas_name]['Ib'],
                phasors[meas_name]['Ic']
            )
            features[f'{prefix}_I0'] = np.abs(I0)
            features[f'{prefix}_I1'] = np.abs(I1)
            features[f'{prefix}_I2'] = np.abs(I2)

        # Voltage features - not available in .mat files, set to nominal
        # In real deployment, these would come from voltage channels
        Vnom = 132e3 / np.sqrt(3)  # 76.21 kV phase voltage
        for vname in ['VA', 'VB']:
            features[f'{vname}_a'] = Vnom
            features[f'{vname}_b'] = Vnom
            features[f'{vname}_c'] = Vnom
            features[f'{vname}_V0'] = 0.0
            features[f'{vname}_V2'] = 0.0

        # Metadata
        fault_type_raw = f['faultType'][()].flatten()
        fault_type_str = ''.join([chr(int(c)) for c in fault_type_raw if c > 0])
        zone = int(f['zone'][()].flatten()[0])
        position_pct = float(f['position'][()].flatten()[0])

        return features, fault_type_str, zone, position_pct


def get_line_length(zone, ps):
    """Get line length for a zone."""
    if zone == 1:
        return ps.L1
    elif zone == 2:
        return ps.L2
    else:
        # Zone 3 - assume another 40km section beyond L2
        return 40.0


def compute_labels(features, fault_type_str, zone, position_pct, ps, R1, R2):
    """Compute ground-truth labels using physics model + relay coordination."""
    # Map fault type
    fault_type = FAULT_MAP.get(fault_type_str, fault_type_str)

    # Distance from Bus A
    if zone == 1:
        dist_km = position_pct * ps.L1
        sim_zone = 1
    elif zone == 2:
        dist_km = ps.L1 + position_pct * ps.L2
        sim_zone = 2
    else:
        # Zone 3: beyond L2, simulate as zone 2 with extended distance
        dist_km = ps.L1 + ps.L2 + position_pct * 40.0
        sim_zone = 2  # simulate as zone 2 fault for relay decision

    # Simulate fault using physics model to get relay currents
    # Use the same fault resistance as training (0.01 ohm bolted)
    Rf = 0.01
    lf = 1.0  # nominal loading

    # Get simulated measurements for this fault
    meas_sim = ps.simulate_fault(fault_type, sim_zone, dist_km if sim_zone == 2 else min(dist_km, ps.L1), Rf, lf)

    # Relay decision
    dec = relay_decision(R1, R2, meas_sim, sim_zone, fault_type)

    labels = {
        'fault': 1,
        'fault_type': fault_type,
        'zone': zone,
        'distance_km': dist_km,
        'Rf': Rf,
        'load_factor': lf,
        'correct_relay': dec['correct_relay'],
        't_R1': dec['t_R1'],
        't_R2': dec['t_R2'],
        'coordinated': dec['coordinated'],
    }
    return labels


def compute_labels(features, fault_type_str, zone, position_pct, ps, R1, R2):
    """Compute ground-truth labels using physics model + relay coordination."""
    # Map fault type
    fault_type = FAULT_MAP.get(fault_type_str, fault_type_str)

    # Distance from Bus A
    if zone == 1:
        dist_km = position_pct * ps.L1
    elif zone == 2:
        dist_km = ps.L1 + position_pct * ps.L2
    else:
        dist_km = ps.L1 + ps.L2 + position_pct * 40.0

    # Simulate fault using physics model to get relay currents
    # Use the same fault resistance as training (0.01 ohm bolted)
    Rf = 0.01
    lf = 1.0  # nominal loading

    # Get simulated measurements for this fault
    meas_sim = ps.simulate_fault(fault_type, zone, dist_km, Rf, lf)

    # Relay decision
    dec = relay_decision(R1, R2, meas_sim, zone, fault_type)

    labels = {
        'fault': 1,
        'fault_type': fault_type,
        'zone': zone,
        'distance_km': dist_km,
        'Rf': Rf,
        'load_factor': lf,
        'correct_relay': dec['correct_relay'],
        't_R1': dec['t_R1'],
        't_R2': dec['t_R2'],
        'coordinated': dec['coordinated'],
    }
    return labels


def main():
    print("=" * 60)
    print("Converting MATLAB dataset to CSV")
    print("=" * 60)

    # Initialize physics model and relays for label generation
    ps = PowerSystem()
    R1, R2 = coordinate(ps)
    print(f"Relay coordination: R1 TMS={R1.TMS:.3f}, R2 TMS={R2.TMS:.3f}")

    # Find all .mat files
    mat_files = list(MATLAB_ROOT.rglob("*_Data.mat"))
    print(f"Found {len(mat_files)} .mat files")

    rows = []
    for mat_path in mat_files:
        try:
            # Parse path: .../Z1_25/LG/Z1_25_LG_Data.mat
            rel_path = mat_path.relative_to(MATLAB_ROOT)
            parts = rel_path.parts
            zone_str = parts[0]      # e.g., Z1_25
            fault_folder = parts[1]  # e.g., LG

            zone = ZONE_MAP.get(zone_str[:2], 1)
            position_pct = POS_MAP.get(zone_str[3:], 0.5)

            print(f"Processing {rel_path}...", end=" ")

            # Extract features
            features, fault_type_str, zone_parsed, pos_parsed = extract_features_from_mat(mat_path)

            # Verify zone/position match
            if zone_parsed != zone or abs(pos_parsed - position_pct * 100) > 1:
                print(f"WARNING: metadata mismatch (zone={zone_parsed}, pos={pos_parsed})")

            # Compute labels
            labels = compute_labels(features, fault_type_str, zone, position_pct, ps, R1, R2)

            # Combine
            row = {**features, **labels}
            rows.append(row)
            print(f"OK (fault={fault_type_str}, zone={zone}, pos={position_pct})")

        except Exception as e:
            print(f"FAILED: {e}")
            continue

    # Create DataFrame
    df = pd.DataFrame(rows)

    # Add healthy cases (no fault) - simulate using physics model
    print("\nGenerating healthy cases...")
    healthy_rows = []
    for _ in range(100):
        lf = np.random.uniform(0.3, 1.2)
        meas = ps.simulate_fault('NONE', 1, 0.001, 0.0, lf)
        # Add small noise
        for k in meas:
            if k.endswith('_ang'):
                meas[k] += np.random.normal(0, 0.5)
            else:
                meas[k] *= (1 + np.random.normal(0, 0.01))
        dec = relay_decision(R1, R2, meas, 0, 'NONE')
        row = {**meas, 'fault': 0, 'fault_type': 'NONE', 'zone': 0,
               'distance_km': -1.0, 'Rf': 0.0, 'load_factor': lf, **dec}
        healthy_rows.append(row)

    df_healthy = pd.DataFrame(healthy_rows)
    df = pd.concat([df, df_healthy], ignore_index=True)

    # Ensure column order matches training script
    FEATURES = [
        'R1_Ia', 'R1_Ib', 'R1_Ic', 'R1_I0', 'R1_I1', 'R1_I2',
        'R1_Ia_ang', 'R1_Ib_ang', 'R1_Ic_ang',
        'R2_Ia', 'R2_Ib', 'R2_Ic', 'R2_I0', 'R2_I1', 'R2_I2',
        'R2_Ia_ang', 'R2_Ib_ang', 'R2_Ic_ang',
        'VA_a', 'VA_b', 'VA_c', 'VA_V0', 'VA_V2',
        'VB_a', 'VB_b', 'VB_c', 'VB_V0', 'VB_V2',
    ]

    # Map our feature names to training feature names
    # Our measurements: R1_I, R2_I, R3_I, SEND_I, RECV_I
    # Training expects: R1, R2 only
    # We'll use R1_I and R2_I as R1 and R2 (primary protection relays)
    rename_map = {}
    for meas_old, meas_new in [('R1_I', 'R1'), ('R2_I', 'R2')]:
        for suffix in ['_Ia', '_Ib', '_Ic', '_I0', '_I1', '_I2',
                       '_Ia_ang', '_Ib_ang', '_Ic_ang']:
            rename_map[f'{meas_old}{suffix}'] = f'{meas_new}{suffix}'

    df = df.rename(columns=rename_map)

    # Select only the features used in training + labels
    label_cols = ['fault', 'fault_type', 'zone', 'distance_km', 'Rf', 'load_factor',
                  'correct_relay', 't_R1', 't_R2', 'coordinated']
    final_cols = FEATURES + label_cols

    # Check for missing columns
    missing = [c for c in final_cols if c not in df.columns]
    if missing:
        print(f"WARNING: Missing columns: {missing}")
        for c in missing:
            df[c] = 0.0

    df = df[final_cols]

    # Save
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(df)} rows x {len(df.columns)} cols to {OUTPUT_CSV}")
    print(f"Fault type distribution:")
    print(df['fault_type'].value_counts().to_string())
    print(f"\nCorrect relay distribution:")
    print(df['correct_relay'].value_counts().to_string())
    print(f"\nZone distribution:")
    print(df['zone'].value_counts().sort_index().to_string())


if __name__ == '__main__':
    main()