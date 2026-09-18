# Power System Protection using Machine Learning

ML-based protection of a 132 kV radial transmission feeder: fault detection,
classification, location, **relay selection**, and IDMT characteristic
emulation — all trained on a physics-based simulated dataset.

## Quick start

```bash
pip install numpy pandas scikit-learn matplotlib joblib

python src/generate_dataset.py   # build the 4800-case dataset (runs coordination too)
python src/train_models.py       # train + evaluate all 5 ML tasks
python src/make_plots.py         # produce the 7 figures
python src/demo.py               # live demo: random fault -> ML pipeline responds
python src/demo.py AG 2 25 10    # or choose: type zone dist_km Rf
```

## Project layout

```
src/power_system.py       132 kV system + symmetrical-component fault engine
src/idmt_relay.py         IEC 60255 IDMT relay + auto coordination (CTI 0.3 s)
src/generate_dataset.py   scenario sweep -> data/fault_dataset.csv (4800 rows)
src/train_models.py       DT / RF / SVM / MLP on 5 protection tasks
src/make_plots.py         publication figures (300 dpi) -> results/figures/
src/demo.py               interactive demonstration script
docs/PROJECT_PROPOSAL.md  <- show this to the professor
docs/METHODOLOGY.md       full mathematical formulation
docs/LITERATURE_SURVEY.md surveyed papers
data/                     generated dataset
results/                  metrics.json, saved models, figures
```

## Headline results (20 % held-out test set)

| Task | Best model | Result |
|------|-----------|--------|
| Fault detection | RF | 100 % |
| Fault type (11-class) | RF | 100 % (CV 100 %) |
| **Relay selection (R1/R2/none)** | MLP | 100 % test / 99.6 % CV |
| Fault location | MLP | MAE 0.75 km (90 km feeder) |
| IDMT trip time | RF | MAE 1.9 ms, R² 0.999 |
