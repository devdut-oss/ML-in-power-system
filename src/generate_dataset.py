"""
generate_dataset.py
====================
Scenario sweep -> labelled ML dataset.

Every row = one steady-state fault (or healthy) case with the phasor
measurements both relays would extract, plus ground-truth labels:

  LABELS
    fault        : 0/1                     (detection)
    fault_type   : NONE + 10 classes       (classification)
    zone         : 0 (none) / 1 / 2        (zone identification)
    correct_relay: NONE / R1 / R2          (relay-selection - the prof's
                                            "model decides which relay
                                            operates" objective)
    distance_km  : fault distance from Bus A along the feeder (regression)
    t_R1, t_R2   : IDMT operating times    (trip-time regression - the
                                            "IDMT characteristic via ML")

  SWEEP GRID
    fault types  : 10                      (AG BG CG AB BC CA ABG BCG CAG ABC)
    zones        : 2
    locations    : 12 per zone             (2% ... 98% of section length)
    fault Rf     : 6                       (0.01, 1, 5, 10, 25, 50 ohm)
    loading      : 3                       (0.6, 1.0, 1.2 pu)
    measurement noise : 1% multiplicative Gaussian on magnitudes,
                        0.5 deg on angles  (CT/CVT + DFT estimation error)
    -> 10*2*12*6*3 = 4320 fault rows + 480 healthy rows = 4800 rows

Run:  python src/generate_dataset.py
"""

import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from power_system import PowerSystem
from idmt_relay import coordinate, relay_decision

rng = np.random.default_rng(42)

MAG_NOISE = 0.01          # 1 % on magnitudes
ANG_NOISE = 0.5           # deg on angles

FAULT_TYPES = ['AG', 'BG', 'CG', 'AB', 'BC', 'CA', 'ABG', 'BCG', 'CAG', 'ABC']
RF_VALUES = [0.01, 1.0, 5.0, 10.0, 25.0, 50.0]
LOADINGS = [0.6, 1.0, 1.2]
N_LOCS = 12


def add_noise(meas):
    out = {}
    for k, v in meas.items():
        if k.endswith('_ang'):
            out[k] = v + rng.normal(0, ANG_NOISE)
        else:
            out[k] = v * (1 + rng.normal(0, MAG_NOISE))
    return out


def main():
    ps = PowerSystem()
    print("--- relay coordination ---")
    R1, R2 = coordinate(ps)
    print("--- sweeping scenarios ---")

    rows = []
    for ftype in FAULT_TYPES:
        for zone in (1, 2):
            L = ps.L1 if zone == 1 else ps.L2
            for frac in np.linspace(0.02, 0.98, N_LOCS):
                d = frac * L
                for Rf in RF_VALUES:
                    for lf in LOADINGS:
                        meas = ps.simulate_fault(ftype, zone, d, Rf, lf)
                        meas = add_noise(meas)
                        dec = relay_decision(R1, R2, meas, zone, ftype)
                        dist_from_A = d if zone == 1 else ps.L1 + d
                        rows.append({**meas,
                                     'fault': 1,
                                     'fault_type': ftype,
                                     'zone': zone,
                                     'distance_km': dist_from_A,
                                     'Rf': Rf,
                                     'load_factor': lf,
                                     **dec})

    # healthy rows: varied loading, no fault
    for _ in range(480):
        lf = rng.uniform(0.3, 1.2)
        meas = add_noise(ps.simulate_fault('NONE', 1, 0.001, 0.0, lf))
        dec = relay_decision(R1, R2, meas, 0, 'NONE')
        rows.append({**meas, 'fault': 0, 'fault_type': 'NONE', 'zone': 0,
                     'distance_km': -1.0, 'Rf': 0.0, 'load_factor': lf, **dec})

    df = pd.DataFrame(rows)
    os.makedirs(os.path.join(os.path.dirname(__file__), '..', 'data'),
                exist_ok=True)
    out = os.path.join(os.path.dirname(__file__), '..', 'data',
                       'fault_dataset.csv')
    df.to_csv(out, index=False)
    print(f"dataset: {df.shape[0]} rows x {df.shape[1]} cols -> {out}")
    print(df['fault_type'].value_counts().to_string())
    print("\ncorrect_relay distribution:")
    print(df['correct_relay'].value_counts().to_string())
    print(f"\ncoordination held in {df['coordinated'].mean()*100:.1f} % of cases")


if __name__ == '__main__':
    main()
