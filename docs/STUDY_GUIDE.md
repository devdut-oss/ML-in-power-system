# Study Guide — Understand Everything You Built

*Read this end-to-end once and you can explain the whole project to your
professor or in a viva. It goes concept → code → results, with the questions
you're most likely to be asked at the end.*

---

## Part 1 — The Power System Background

### 1.1 Why transmission lines need protection

A short circuit (fault) on a 132 kV line pushes currents of **kilo-amperes**
through equipment rated for hundreds of amps. If not cleared within a few
hundred milliseconds: conductors anneal, transformers cook, generators lose
synchronism, and the outage cascades. Protection = **detect the fault and trip
only the faulted section, as fast as selectivity allows.**

### 1.2 Our system (memorise this diagram)

```
 Grid                Line 1 (50 km)            Line 2 (40 km)
 3500 MVA ---[Bus A]====================[Bus B]==================[Bus C]
 X/R=10       |R1: IDMT OC, CT 400/1     |R2: IDMT OC, CT 300/1   |
              |Zone 1 primary            |Zone 2 primary          Load 40 MVA
                                         |(R1 = Zone 2 BACKUP)    0.85 pf lag
 132 kV line-line, 50 Hz. Radial (power flows one way only).
```

Numbers you should know cold:
- Phase voltage = 132/√3 = **76.2 kV**
- Full-load current = 40 MVA/(√3·132 kV) ≈ **175 A** nominal; with the
  constant-impedance load model and source drop, the computed value is **163 A**
- Max fault current (3-φ at Bus A) = **15.4 kA**; at Bus B = **3.1 kA**
- Line constants: z₁ = 0.12 + j0.40 Ω/km, z₀ = 0.35 + j1.25 Ω/km (typical
  132 kV ACSR "Panther")

### 1.3 The 10 fault types

| Group | Types | Share of real faults | Severity |
|---|---|---|---|
| L-G (single line-ground) | AG, BG, CG | ~80 % | lowest current (usually) |
| L-L (line-line) | AB, BC, CA | ~10 % | ≈ 87 % of 3-φ current |
| L-L-G (double line-ground) | ABG, BCG, CAG | ~8 % | between L-L and 3-φ |
| L-L-L (three-phase) | ABC | ~2 % | highest, balanced |

### 1.4 Symmetrical components — the engine of the whole simulation

Fortescue's theorem: any unbalanced 3-phase phasor set = sum of three balanced
sets — **positive** (normal rotation), **negative** (reverse rotation), **zero**
(all three in phase). With a = 1∠120°:

```
[Ia]   [1  1   1 ] [I0]
[Ib] = [1  a²  a ] [I1]      (phase ← sequence,  Iabc = A·I012)
[Ic]   [1  a   a²] [I2]
```

Each fault type corresponds to a specific **interconnection of the three
sequence networks** (this is the classic exam material):

| Fault | Network connection | Formula for I₁ |
|---|---|---|
| ABC | positive network alone | I₁ = E/(Z₁+Zf) |
| AG | all three **in series** | I₁ = I₂ = I₀ = E/(Z₁+Z₂+Z₀+3Zf) |
| BC | positive ∥ negative | I₁ = −I₂ = E/(Z₁+Z₂+Zf) |
| BCG | positive + (negative ∥ zero) | I₁ = E/(Z₁ + Z₂(Z₀+3Zf)/(Z₂+Z₀+3Zf)) |

Two facts the ML models exploit (and you should quote in the viva):
- **I₀ ≠ 0 only when ground is involved** → I₀ separates AG/ABG-type faults
  from AB/ABC-type. (I₀ flows only if the fault path touches earth.)
- **I₂ ≠ 0 only when the fault is unbalanced** → I₂ = 0 for ABC and healthy
  operation. So (I₀, I₂) is almost a fingerprint of the fault group; which
  *phases* are involved comes from the per-phase magnitudes and angles.

### 1.5 Fault resistance Rf — why it matters

An arcing fault or a tree-contact fault adds 1–50 Ω in the fault path. This
**reduces the fault current dramatically** (see Fig. 2: a 50 Ω AG fault gives
~1.5 kA instead of 12 kA). A plain overcurrent relay with a fixed pickup may
then **fail to see the fault at all** — this is the classic weakness of OC
protection and one justification for the ML approach.

---

## Part 2 — IDMT Overcurrent Protection

### 2.1 The IDMT law (IEC 60255-151)

```
t_op = TMS · k / ((I/Is)^α − 1)        for I > Is
Standard Inverse (used here): k = 0.14, α = 0.02
```

- **Is (pickup / plug setting):** the current above which the relay starts.
  Set at 1.25 × max load so normal load never trips it.
- **TMS (time multiplier 0.025–1.2):** scales the whole curve up/down in time.
- **Inverse** = bigger fault current → faster trip. A 10 kA fault trips in
  ~0.3 s, a 300 A overload takes tens of seconds.

### 2.2 Coordination (time grading) — how R1 and R2 avoid tripping together

Rule: for any fault in Zone 2, R2 must trip **first**; R1 waits by at least the
**CTI = 0.3 s** (breaker interrupting time + relay overshoot + safety margin).
Procedure implemented in `coordinate()` in `idmt_relay.py`:

1. R2 gets the fastest curve: TMS = 0.05, Is = 1.25 × 163 A = **204 A**.
2. Find the **grading point** = worst case for coordination = maximum fault
   current just past R2 = 3-φ fault at Bus B = **3082 A**.
3. There, t_R2 = 0.125 s ⇒ R1 must take ≥ 0.425 s.
4. Invert the IDMT formula for TMS: TMS_R1 = 0.425·(M^0.02−1)/0.14 = **0.161**
   (M = 3082/234).
5. Verify at the remote end of Line 2: margin 0.371 s ✔ (Fig. 3 shows this).

### 2.3 The cost of coordination — the problem ML attacks

Time grading buys selectivity **with delay**. Every upstream relay is 0.3 s
slower than the one below it; in a 5-stage radial feeder the source-end relay
may need >1.5 s. Also, if the ML can tell *instantly* which zone the fault is
in (our task M3 does this at 100 %), the correct relay could trip with **no
grading delay at all**. That is the "coordination-free protection" argument in
the proposal.

---

## Part 3 — How Each Source File Works

### 3.1 `power_system.py` (the simulator)

Class `PowerSystem`. Key methods:
- `__init__`: builds source impedance from SCC (Zs1 = V²/SCC ∠tan⁻¹10),
  line constants, constant-impedance load.
- `_thevenin_to_fault(zone, d)`: adds up source + line impedance from the
  source to the fault point, per sequence.
- `_sequence_fault_currents(...)`: the four textbook formulas of §1.4.
- `simulate_fault(ftype, zone, d, Rf, load_factor)`: the public API. Computes
  sequence currents → phase currents (`A @ I012`), **rotates phase labels**
  for B/C-referenced faults (a BG fault is an AG fault with phases rolled),
  superposes load current, computes bus voltages, and returns a dict of the
  **28 phasor measurements** a numerical relay would extract.
- Currents split by topology: zone-1 fault → fault current only through R1;
  zone-2 fault → same fault current through both R1 and R2. This asymmetry is
  exactly what lets ML task M3 distinguish the zones.

Validation built into `__main__`: bolted 3-φ fault at Bus A returns 15.41 kA
vs the hand formula SCC/(√3·V) = 15.31 kA (0.7 % difference = source R vs X
detail). **Quote this when asked "how do you know your simulator is right?"**

### 3.2 `idmt_relay.py` (the protection layer)

- `IDMTRelay.trip_time(I)`: the IEC formula, vectorised, ∞ below pickup,
  clamped at 40 ms (numerical relays have a minimum operating time).
- `coordinate(ps)`: the 5-step grading procedure of §2.2 — computes settings
  *from the system itself*, nothing hard-coded.
- `relay_decision(...)`: produces the **ground-truth labels**: which relay
  should clear this fault (`correct_relay`), both trip times, and whether
  coordination held. This function is the "teacher" for the ML models.

### 3.3 `generate_dataset.py` (dataset engineering)

Nested sweep: 10 types × 2 zones × 12 locations × 6 Rf × 3 loads = 4320 fault
rows + 480 healthy rows = **4800 rows × 38 columns**. Adds measurement noise
(1 % magnitude, 0.5° angle — the realistic error of CT/CVT + DFT phasor
estimation) so the ML never sees mathematically perfect data. Fixed random
seed (42) → fully reproducible.

### 3.4 `train_models.py` (the ML layer)

28 features — **only quantities a real relay measures** (no cheating with Rf
or distance as inputs). Five tasks:

| Task | Type | Output |
|---|---|---|
| M1 detection | binary | fault / no fault |
| M2 fault type | 11-class | NONE, AG … ABC |
| M3 relay selection | 3-class | NONE / R1 / R2 |
| M4 location | regression | km from Bus A |
| M5 IDMT trip time | regression | seconds |

Four classifier families compared (Decision Tree, Random Forest, SVM-RBF,
MLP) — showing a *comparison* rather than a single model is what makes it a
project rather than a tutorial. Evaluation: stratified 80/20 hold-out **plus
5-fold cross-validation** (the CV numbers are the honest ones). SVM and MLP
are wrapped in a `StandardScaler` pipeline (kA next to angles in degrees needs
scaling); trees don't care.

### 3.5 `make_plots.py`, `demo.py`

Seven 300-dpi figures (see Part 5). `demo.py` is the live show: applies a
fault you choose (or random), adds fresh noise, runs all five models, prints
predicted vs actual. Run `python src/demo.py AG 2 25 10` in front of the prof.

---

## Part 4 — The ML Concepts You Must Be Able to Explain

**Random Forest** — many decision trees, each trained on a bootstrap sample
with random feature subsets; prediction by majority vote / averaging. Wins
here because fault classes are separated by **threshold logic** (I₀ high? I₂
high? which phase largest?) which is exactly what trees encode. Also gives
**feature importance** (Fig. 6) — expect I₀ and per-phase currents on top.

**SVM (RBF)** — finds the maximum-margin boundary after implicitly mapping to
a higher-dimensional space. Competitive but needs scaling and tuning; its CV
score (94 %) trails RF — worth mentioning as an honest comparison.

**MLP** — two hidden layers (64, 32), ReLU, Adam. Wins the regressions and
generalises smoothly between grid points (location MAE 0.75 km) because the
current→distance map is a smooth nonlinear function — ideal neural-net
territory.

**Why can ML "learn" the IDMT curve (M5)?** The trip time is a deterministic
function t = 0.161·0.14/((I/234)^0.02−1) of the measured current. Regression
on (features → t) recovers this mapping from examples alone — Fig. 7 shows
the learned dots lying on the analytical curve. The *point* is not that ML
beats the formula (it can't); it's that **a data-driven relay can acquire any
characteristic — including adaptive, nonstandard ones — without being given
the formula.** That's the door to self-setting protection.

**Why are accuracies ~100 %? (They WILL ask.)** Steady-state phasor
simulation gives cleanly separable classes: zone decided by the R1/R2 current
ratio, ground faults flagged by I₀, etc. Small noise doesn't blur the
boundaries. This is expected at phase 1 and consistent with published phasor-
based studies; the CV scores (99.5–100 %) confirm it isn't overfitting. The
numbers will honestly degrade when we move to time-domain waveforms with CT
saturation and DC offset — that is precisely phase 2 of the roadmap and the
gap a journal paper needs us to explore.

---

## Part 5 — Reading the Figures

- **Fig. 1** — the system. Start every presentation with it.
- **Fig. 2** — fault current vs location: decays with distance (more line
  impedance); ABC highest, AG lowest; the dashed 50 Ω AG curve shows the
  high-resistance blind-spot argument; pickup line shows why the relay still
  (barely) sees everything in this system.
- **Fig. 3** — the two IDMT curves on log-log axes with the red CTI = 0.30 s
  bar at the grading point (3082 A). This one figure *is* relay coordination.
- **Fig. 4** — confusion matrices, all mass on the diagonal (M2: 11 classes,
  M3: relay selection).
- **Fig. 5** — predicted vs actual scatter hugging the y = x line for location
  (MAE 0.75 km) and trip time (MAE 1.9 ms).
- **Fig. 6** — RF feature importances: sequence and phase currents dominate —
  matches the physics of §1.4, which is your "the model learned real physics,
  not artifacts" argument.
- **Fig. 7** — ML dots on the analytical IEC curve = "the relay characteristic
  has been learned from data."

---

## Part 6 — Likely Viva Questions (with answers)

1. **Why 132 kV?** Indian sub-transmission level where IDMT OC relays are the
   standard primary protection (EHV lines ≥ 220 kV use distance/differential).
2. **Why is Z₀ ≠ Z₁ for a line?** Zero-sequence currents are in phase in all
   three conductors and return through earth/ground wire — different flux
   linkage path, so higher R and X (here z₀ ≈ 3·z₁).
3. **Why is the L-L fault current 87 % of the 3-φ?** √3/2 factor:
   I_LL = √3·E/(Z₁+Z₂) vs I_3φ = E/Z₁ with Z₁ = Z₂.
4. **What if both relays fail?** Upstream grid protection (not modelled) —
   acknowledge as a limitation.
5. **Your model is steady-state. What about the first cycle after the fault?**
   DC offset and transients are absent; numerical relays filter them with a
   full-cycle DFT anyway, so phasor analysis approximates the post-transient
   values the relay algorithm actually uses. Full EMTP realism is phase 2.
6. **Could the ML misoperate on load swings / inrush?** Present dataset has no
   inrush or power-swing cases — good honest answer: add them as a "no-trip"
   class in phase 2.
7. **Why not deep learning / CNN-LSTM?** 4800 phasor rows don't need it;
   CNN/LSTM become appropriate when inputs are raw waveforms (phase 2). Right
   tool for the data size — reviewers like that reasoning.
8. **How would this deploy in a real relay?** Trained model exported (e.g.
   ONNX) into a numerical relay's processor or a substation edge device; both
   relays' phasors arrive via IEC 61850-9-2 sampled values / GOOSE.
9. **What's novel vs the literature?** Papers do detection OR classification
   OR location OR coordination separately; we chain **five functions from one
   measurement set on one dataset**, including relay selection (see
   LITERATURE_SURVEY.md, gap section).
10. **Is 100 % believable?** Yes for phasor-domain data (cite the CV scores
    and the separability physics of §1.4) — and we say so honestly and put
    waveform-level realism in the roadmap.

---

## Part 7 — One-Minute Summary (say this to the prof)

> "We modelled a 132 kV two-section radial feeder in Python using symmetrical
> components, validated it against hand calculations, and coordinated two IEC
> IDMT overcurrent relays classically (CTI 0.3 s). Sweeping 10 fault types,
> 24 locations, six fault resistances and three load levels gave a 4800-case
> labelled dataset of exactly the phasors a numerical relay measures. On it we
> trained five ML tasks: fault detection, 11-class type classification, fault
> location (MAE 0.75 km), IDMT trip-time emulation (MAE 1.9 ms — the model
> reproduces the IEC curve it was never shown), and most importantly **relay
> selection** — the model instantly names the relay that should operate,
> which conventional protection can only do by waiting out grading delays.
> Phase 2 adds waveform-level realism and adaptive settings, which is the
> journal-paper direction."
