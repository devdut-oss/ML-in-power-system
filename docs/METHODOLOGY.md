# Methodology — Mathematical Formulation

This document records every equation implemented in the code, so the report /
paper can be written directly from it.

## 1. Fault analysis by symmetrical components

Any unbalanced three-phase set is decomposed with the Fortescue transform,
a = 1∠120°:

```
[I0]   1 [1  1   1 ] [Ia]
[I1] = - [1  a   a²] [Ib]        Iabc = A · I012,  A = [[1,1,1],[1,a²,a],[1,a,a²]]
[I2]   3 [1  a²  a ] [Ic]
```

### 1.1 Thevenin sequence impedances to the fault point

Source: SCC = 3500 MVA at 132 kV ⇒ |Zs1| = V²/SCC = 4.98 Ω, X/R = 10;
Zs0 = 1.5 · Zs1. Line: z₁ = 0.12+j0.40 Ω/km, z₀ = 0.35+j1.25 Ω/km.

Fault at distance *d* into zone *z*:

```
Z1 = Zs1 + z1·(L1·[z=2] + d)        Z2 = Z1        Z0 = Zs0 + z0·(L1·[z=2] + d)
```

### 1.2 Sequence networks per fault type (fault impedance Zf = Rf)

| Fault | Connection | Sequence currents |
|-------|-----------|-------------------|
| 3-φ (ABC) | pos. seq. only | I1 = E/(Z1+Zf); I2 = I0 = 0 |
| L-G (AG) | series | I1 = I2 = I0 = E/(Z1+Z2+Z0+3Zf) |
| L-L (BC) | pos ∥ neg | I1 = −I2 = E/(Z1+Z2+Zf); I0 = 0 |
| L-L-G (BCG) | pos + (neg ∥ zero) | I1 = E/(Z1 + Z2(Z0+3Zf)/(Z2+Z0+3Zf)), I2, I0 by current divider |

Faults on other phases: cyclic rotation of the phase labels (a BG fault is an
AG fault with phases rolled by one).

Phase currents: Iabc = A·[I0,I1,I2]ᵀ, then rolled by the rotation index.
Load current superposed (pre-fault load at the applied loading factor).

### 1.3 Relay-point voltages

Sequence voltages at measuring bus behind impedance Zs (source→bus):

```
V1 = E − I1·Z1(source→bus)    V2 = −I2·Z2(→bus)    V0 = −I0·Z0(→bus)
```

## 2. IDMT relay model (IEC 60255-151)

```
t_op = TMS · k / ((I/Is)^α − 1),   I > Is
Standard Inverse: k = 0.14, α = 0.02  (min operating time clamped at 40 ms)
```

### 2.1 Coordination procedure (implemented in `coordinate()`)

1. Pickup: Is = 1.25 × I_load,max (R2 = 204 A; R1 = 234 A with 15 % selectivity margin).
2. Downstream relay fastest allowed: TMS_R2 = 0.05.
3. Grading point: maximum 3-φ fault at Bus B (I_gp = 3 082 A).
   t_R2(I_gp) = 0.125 s ⇒ t_R1,req = t_R2 + CTI = 0.425 s.
4. Solve TMS_R1 = t_R1,req·(M^α − 1)/k = **0.161** where M = I_gp/Is_R1.
5. Verified: at remote end of Line 2, margin = 0.371 s ≥ 0.3 s ✔

## 3. Dataset construction

Sweep grid (faulted rows): 10 types × 2 zones × 12 locations (2–98 % of
section) × Rf ∈ {0.01, 1, 5, 10, 25, 50} Ω × load ∈ {0.6, 1.0, 1.2} pu
= 4 320 rows; + 480 healthy rows (random load 0.3–1.2 pu). Noise: magnitudes
×(1+N(0, 0.01)), angles +N(0, 0.5°).

**Feature vector (28):** |Ia|,|Ib|,|Ic|,|I0|,|I1|,|I2|,∠Ia,∠Ib,∠Ic at R1 and
R2; |Va|,|Vb|,|Vc|,|V0|,|V2| at Bus A and Bus B.

**Labels:** fault (0/1); fault_type (11); zone (0/1/2); correct_relay
(NONE/R1/R2); distance_km; t_R1, t_R2 (IEC formula ground truth).

## 4. ML models & evaluation

- Classifiers: Decision Tree (depth ≤ 12), Random Forest (200 trees),
  SVM-RBF (C = 10, standardized), MLP (64–32, Adam, standardized).
- Regressors: Random Forest (300 trees), MLP (128–64).
- Split: stratified 80/20 hold-out **plus** 5-fold cross-validation.
- Metrics: accuracy, macro-F1, confusion matrices; MAE and R² for regression.

## 5. Results summary

See `results/metrics.json` and `results/figures/`. Headline numbers:
detection 100 %, 11-class type 100 % (CV 100 %), relay selection 100 % test /
99.6 % CV, location MAE 0.75 km, IDMT trip-time MAE 1.9 ms (R² 0.999) —
the ML model reproduces the analytical IEC curve (Fig. 7).

## 6. Assumptions & limitations (state these honestly in the report)

1. Steady-state phasor analysis — no DC offset, no CT saturation, no
   travelling-wave transients. (Future work: EMTP-level simulation.)
2. Radial single-source system — no infeed from the load side.
3. Zone-1 fault: downstream relay's current approximated as partial load
   (voltage collapse at the fault reduces downstream flow).
4. Constant-impedance load model.
5. Perfect phasor synchronization between R1 and R2 measurements assumed
   (in practice: IEC 61850 process bus / PMU alignment).
