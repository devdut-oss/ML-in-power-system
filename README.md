# Power System Protection using Machine Learning

ML-based protection of a 132 kV radial transmission feeder: fault detection,
classification, location, **relay selection**, and IDMT characteristic
emulation — all trained on a physics-based simulated dataset.


```bash
pip install numpy pandas scikit-learn matplotlib joblib

python src/generate_dataset.py   # build the 4800-case dataset (runs coordination too)
python src/train_models.py       # train + evaluate all 5 ML tasks
python src/make_plots.py         # produce the 7 figures
python src/demo.py               # live demo: random fault -> ML pipeline responds
python src/demo.py AG 2 25 10    # or choose: type zone dist_km Rf
```

