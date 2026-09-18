"""
demo.py — live demonstration script (run this in front of the professor).

Simulates a random (or user-chosen) fault on the 132 kV feeder, shows what the
relays measure, and lets the trained ML models act as a "smart relay":
detect -> classify -> locate -> select the correct relay -> predict trip time.

Usage:
    python src/demo.py                      # random fault
    python src/demo.py AG 2 25 10           # AG fault, zone 2, 25 km, Rf=10 ohm
"""

import os
import sys
sys.stdout.reconfigure(encoding='utf-8')      # Windows console safety
import numpy as np
import joblib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ROOT = os.path.dirname(HERE)

from power_system import PowerSystem
from idmt_relay import coordinate
from train_models import FEATURES

rng = np.random.default_rng()


def main():
    ps = PowerSystem()
    R1, R2 = coordinate(ps, verbose=False)
    models = joblib.load(os.path.join(ROOT, 'results', 'models.joblib'))

    if len(sys.argv) == 5:
        ftype, zone, d, rf = (sys.argv[1].upper(), int(sys.argv[2]),
                              float(sys.argv[3]), float(sys.argv[4]))
    else:
        ftype = rng.choice(list(ps.FAULT_MAP))
        zone = int(rng.integers(1, 3))
        d = float(rng.uniform(1, ps.L1 if zone == 1 else ps.L2))
        rf = float(rng.choice([0.01, 1, 5, 10, 25, 50]))

    print("=" * 62)
    print("  ML-BASED PROTECTION DEMO — 132 kV radial feeder")
    print("=" * 62)
    print(f"  APPLIED FAULT   : {ftype}  in Zone {zone}, "
          f"{d:.1f} km from {'Bus A' if zone == 1 else 'Bus B'}, Rf = {rf} Ω")

    meas = ps.simulate_fault(ftype, zone, d, rf)
    # 1 % measurement noise, like the training data
    noisy = {k: (v * (1 + rng.normal(0, 0.01)) if not k.endswith('_ang')
                 else v + rng.normal(0, 0.5)) for k, v in meas.items()}
    x = np.array([[noisy[f] for f in FEATURES]])

    print(f"\n  RELAY MEASUREMENTS (phasor magnitudes):")
    print(f"    R1: Ia={noisy['R1_Ia']:8.1f}  Ib={noisy['R1_Ib']:8.1f}  "
          f"Ic={noisy['R1_Ic']:8.1f}  3I0={3*noisy['R1_I0']:8.1f} A")
    print(f"    R2: Ia={noisy['R2_Ia']:8.1f}  Ib={noisy['R2_Ib']:8.1f}  "
          f"Ic={noisy['R2_Ic']:8.1f}  3I0={3*noisy['R2_I0']:8.1f} A")

    print(f"\n  ML PROTECTION PIPELINE:")
    det = models['M1'].predict(x)[0]
    print(f"    [M1] Fault detected?      -> {'YES' if det else 'no'}")
    if det:
        ft = models['M2'].predict(x)[0]
        rl = models['M3'].predict(x)[0]
        loc = models['M4'].predict(x)[0]
        tt = models['M5'].predict(x)[0]
        true_from_A = d if zone == 1 else ps.L1 + d
        print(f"    [M2] Fault type           -> {ft}"
              f"     (actual: {ftype})")
        print(f"    [M3] Relay to operate     -> {rl}"
              f"     (actual: {'R1' if zone == 1 else 'R2'})")
        print(f"    [M4] Fault location       -> {loc:6.1f} km from Bus A"
              f"  (actual: {true_from_A:.1f} km, err {abs(loc-true_from_A):.2f} km)")
        Imax = max(noisy['R1_Ia'], noisy['R1_Ib'], noisy['R1_Ic'])
        t_true = R1.trip_time(Imax)
        print(f"    [M5] R1 IDMT trip time    -> {tt:6.3f} s"
              f"    (IEC formula: {t_true:.3f} s)" if np.isfinite(t_true)
              else f"    [M5] R1 below pickup")
    print("=" * 62)


if __name__ == '__main__':
    main()
