# Project Proposal

## Machine-Learning-Based Protection of a 132 kV Transmission Line: Fault Classification, Zone Discrimination, Relay Selection and IDMT Characteristic Emulation

**Major Project — Electrical Engineering (Power Systems)**

---

## 1. Motivation

Overcurrent (OC) protection with IDMT (Inverse Definite Minimum Time) relays is
the workhorse of sub-transmission and distribution protection. But conventional
IDMT relays have well-known limitations:

1. **Fixed settings.** Pickup and TMS are computed off-line for one network
   condition. Load growth, high fault resistance, or changed source impedance
   can cause under-reach (relay never picks up) or loss of coordination.
2. **Slow backup clearing.** Time grading means an upstream relay waits
   0.3–0.4 s *per downstream stage* — remote faults on long radials are
   cleared slowly.
3. **No intelligence.** A conventional relay only compares |I| to a threshold.
   It cannot say *what type* of fault occurred, *where* it is, or *whether it —
   and not its neighbour — is the right relay to trip.

A data-driven (ML) relay can extract all of that from the same CT/VT
measurements the conventional relay already has. This project demonstrates the
complete chain: **simulate → generate data → learn → protect**.

## 2. Objectives

| # | Objective | ML task |
|---|-----------|---------|
| O1 | Model a 132 kV radial transmission system (2 line sections, 2 IDMT OC relays) and perform short-circuit analysis for all 10 shunt fault types | — (physics simulation) |
| O2 | Generate a labelled dataset by sweeping fault type × location × fault resistance × loading (4 800 cases) | — (dataset engineering) |
| O3 | Detect the presence of a fault from relay phasor measurements | Binary classification |
| O4 | Classify the fault type (AG, BG, CG, AB, BC, CA, ABG, BCG, CAG, ABC, none) | 11-class classification |
| O5 | **Decide which relay must operate (R1 / R2 / none)** — i.e. the ML model replaces the coordination logic | 3-class classification |
| O6 | Estimate the fault location (km from the source bus) | Regression |
| O7 | Emulate the IEC 60255 IDMT characteristic — the ML model reproduces the correct inverse-time trip delay | Regression |

Objective O5 is the core novelty: instead of *time-graded* coordination (where
selectivity is bought with delay), the ML model identifies the faulted zone
**instantaneously from a single set of synchronized measurements**, so the
correct relay can trip without waiting out a grading margin.

## 3. System Under Study

```
 Grid source          Line 1 (50 km)          Line 2 (40 km)
 3500 MVA  ----[Bus A]================[Bus B]================[Bus C]
 X/R = 10       |R1: IDMT OC             |R2: IDMT OC           |
                |CT 400/1                |CT 300/1              Load 40 MVA
                                                                0.85 pf lag
   132 kV, 50 Hz  |  Zone 1 = Line 1 (R1 primary)
                  |  Zone 2 = Line 2 (R2 primary, R1 backup)
```

- Line constants (typical 132 kV ACSR): z₁ = 0.12 + j0.40 Ω/km, z₀ = 0.35 + j1.25 Ω/km
- Fault analysis by **symmetrical components** (the same phasor method used in
  ETAP / MATLAB phasor mode), implemented from first principles in Python.
- Relays coordinated per IEC 60255 Standard-Inverse: t = TMS·0.14/(M^0.02 − 1),
  CTI = 0.3 s. Computed settings: R2 (Is = 204 A, TMS = 0.05),
  R1 (Is = 234 A, TMS = 0.161).

## 4. Dataset

4 800 labelled cases = 10 fault types × 2 zones × 12 locations × 6 fault
resistances (0–50 Ω) × 3 load levels, + 480 healthy cases; 1 % Gaussian noise
on magnitudes and 0.5° on angles to emulate CT/CVT and DFT estimation error.
28 features per case — exactly the quantities a numerical relay measures
(phase & sequence current magnitudes/angles at both relays, bus voltages).

## 5. Results Obtained (already implemented and running)

| Task | Best model | Result (20 % held-out test) |
|------|-----------|------------------------------|
| M1 Fault detection | Random Forest | **100 % accuracy** (5-fold CV 100 %) |
| M2 Fault-type classification (11-class) | Random Forest | **100 % accuracy** (5-fold CV 100 %) |
| M3 **Relay selection (R1/R2/none)** | MLP / Decision Tree | **100 % test, 99.6 % CV** |
| M4 Fault location | MLP neural network | **MAE = 0.75 km** (R² = 0.998) on a 90 km feeder |
| M5 IDMT trip-time emulation | Random Forest | **MAE = 1.9 ms** (R² = 0.999) |

The trained model demonstrably *reproduces the IEC inverse-time curve* it was
never shown analytically (Fig. 7) — i.e. the IDMT characteristic has been
**learned from data**, which is the enabler for adaptive/self-setting relays.

## 6. Why this can become a publishable paper

- Most published work does *one* of: detection, classification, or location.
  Our pipeline integrates **five protection functions from one dataset and one
  measurement set**, including relay selection — a step toward
  "coordination-free" protection.
- Fully reproducible open-source Python implementation (no proprietary
  simulator needed) — attractive to reviewers.
- Clear extension path (Section 7) gives enough depth for a journal/conference
  submission (e.g. IEEE SPICES / NPSC / MDPI Energies).

## 7. Future Scope (semester roadmap)

1. **EMTP-level realism** — regenerate the dataset from PSCAD/Simulink
   time-domain waveforms (DFT phasor extraction, CT saturation, decaying DC
   offset) and retrain; compare degradation.
2. **Adaptive settings** — let the ML propose TMS/pickup updates when network
   topology or load changes (reinforcement-learning or re-optimization loop).
3. **More topology** — parallel lines / ring main, where conventional OC
   grading genuinely fails and directional + ML shines.
4. **Wavelet / travelling-wave features** for sub-cycle detection.
5. **Hardware-in-loop demo** — Raspberry Pi + relay board executing the
   trained model in real time.

## 8. Deliverables (already complete)

- `src/power_system.py` — 132 kV system + symmetrical-component fault engine (validated: 3-φ fault at Bus A = 15.4 kA vs 15.3 kA hand calculation)
- `src/idmt_relay.py` — IEC 60255 relay model + automatic time-grading coordination
- `src/generate_dataset.py` — 4 800-case labelled dataset generator
- `src/train_models.py` — trains/evaluates DT, RF, SVM, MLP on all 5 tasks
- `src/make_plots.py` — 7 publication-quality figures (300 dpi)
- `src/demo.py` — live demo: applies a random fault, ML pipeline responds
- `data/fault_dataset.csv`, `results/metrics.json`, `results/figures/`
- `docs/LITERATURE_SURVEY.md` — surveyed papers with citations
- `docs/METHODOLOGY.md` — full mathematical formulation
