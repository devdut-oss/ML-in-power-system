# Power System Protection using ML — Project Summary

## Overview
ML-based protection for a **132 kV radial transmission feeder** (two-section: 50 km + 40 km) with IDMT overcurrent relays at Bus A (R1) and Bus B (R2). All models trained on **physics-simulated dataset** (4800 cases).

---

## System Model (`src/power_system.py`)
- **Voltage**: 132 kV LL, 50 Hz
- **Source**: 3500 MVA, X/R = 10 (Thevenin equivalent)
- **Lines**: ACSR Panther class, z₁ = 0.12+j0.40 Ω/km, z₀ = 0.35+j1.25 Ω/km
- **Load**: 40 MVA, 0.85 pf lagging at Bus C (constant-Z model)
- **CTs**: R1 = 400/1, R2 = 300/1
- **Fault analysis**: Symmetrical components (Fortescue) — 10 shunt fault types + healthy

| Fault Category | Types |
|----------------|-------|
| L-G (80% real) | AG, BG, CG |
| L-L | AB, BC, CA |
| L-L-G | ABG, BCG, CAG |
| L-L-L | ABC |

---

## IDMT Relay & Coordination (`src/idmt_relay.py`)
- **Curve**: IEC 60255 Standard Inverse (SI) — k=0.14, α=0.02
- **Pickup**: 1.25 × max load current (R1 slightly higher for selectivity)
- **Coordination**: Time grading, CTI = 0.3 s
  - R2 (downstream): TMS = 0.05 (fastest)
  - R1 (upstream): TMS auto-calculated from grading point (max 3φ fault at Bus B)

---

## Simulated Dataset (`src/generate_dataset.py`)

| Parameter | Values | Count |
|-----------|--------|-------|
| Fault types | 10 | 10 |
| Zones | 1, 2 | 2 |
| Locations/zone | 2% … 98% (12 steps) | 12 |
| Fault resistance (Rf) | 0.01, 1, 5, 10, 25, 50 Ω | 6 |
| Loading | 0.6, 1.0, 1.2 pu | 3 |
| **Fault cases** | | **4,320** |
| Healthy cases | varied loading (0.3–1.2 pu) | 480 |
| **Total rows** | | **4,800** |

**Noise**: 1% multiplicative Gaussian on magnitudes, 0.5° on angles (CT/CVT + DFT error)

### Features (28) — Only relay-measurable quantities
```
R1_Ia, R1_Ib, R1_Ic, R1_I0, R1_I1, R1_I2,
R1_Ia_ang, R1_Ib_ang, R1_Ic_ang,
R2_Ia, R2_Ib, R2_Ic, R2_I0, R2_I1, R2_I2,
R2_Ia_ang, R2_Ib_ang, R2_Ic_ang,
VA_a, VA_b, VA_c, VA_V0, VA_V2,
VB_a, VB_b, VB_c, VB_V0, VB_V2
```

### Labels (7)
| Label | Type | Description |
|-------|------|-------------|
| `fault` | binary | 0 = healthy, 1 = fault |
| `fault_type` | 11-class | NONE + 10 fault types |
| `zone` | 3-class | 0=none, 1=Line1, 2=Line2 |
| `correct_relay` | 3-class | NONE, R1, R2 (which relay should operate) |
| `distance_km` | regression | Distance from Bus A (0–90 km) |
| `t_R1`, `t_R2` | regression | IDMT operating times (s) |
| `coordinated` | binary | Coordination held? |

---

## ML Tasks & Results (`src/train_models.py`)

| Task | Type | Best Model | Test Result |
|------|------|------------|-------------|
| **M1** Fault Detection | Binary classification | DecisionTree | 100% acc |
| **M2** Fault Classification | 11-class | RandomForest | 100% acc, CV 100% |
| **M3** Relay Selection | 3-class | MLP | 100% test / 99.6% CV |
| **M4** Fault Location | Regression (km) | MLP | MAE 0.75 km, R² 0.998 |
| **M5** IDMT Trip Time | Regression (s) | RandomForest | MAE 1.9 ms, R² 0.999 |

**Training protocol**: Stratified 80/20 split + 5-fold CV  
**Models compared**: DecisionTree, RandomForest, SVM-RBF, MLP (classification); RF + MLP (regression)  
**Saved artifacts**: `results/models.joblib`, `results/test_splits.joblib`, `results/metrics.json`

---

## Outputs
- **Figures** (7, 300 dpi): `results/figures/fig1–fig7.png`
- **Demo**: `python src/demo.py` (random or user-specified fault → full ML pipeline response)
- **Report**: `PowerSystemProtection_ML_Report.pdf` (built from `docs/` via `build_pdf.py`)

---

## Key Assumptions / Limitations
1. **Single-source radial system** — no parallel infeeds, no mesh
2. **Steady-state phasors only** — no transients, no harmonics, no CT saturation
3. **Perfect phasor estimation** — only 1%/0.5° noise added post-simulation
4. **Constant-Z load** — no motor contribution, no dynamic load models
5. **Bolted fault at busbar for coordination** — grading point = max 3φ at Bus B
6. **R2 sees zero fault current for Zone-1 faults** — simplified (load only at 30%)

---

## Ready for Real Data
The pipeline is **modular** — to switch to real data:
1. Replace `data/fault_dataset.csv` with your dataset (same 28 features + 7 labels)
2. Retrain: `python src/train_models.py`
3. Regenerate plots: `python src/make_plots.py`
4. Test demo: `python src/demo.py`

**Next steps**: Share your real dataset format and requirements for the new model.