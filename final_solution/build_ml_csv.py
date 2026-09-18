"""
build_ml_csv.py - Combine all 44 MATLAB CSV files into one ML-ready sheet.
Uses correct_ml_pipeline feature extraction which already handles all formats.
Output: data/real_fault_dataset.csv (44 rows x 123 cols)
"""
import sys, os, json
import numpy as np
import pandas as pd
from pathlib import Path

# Add src to path so we can use the working functions
SRC = Path(r'C:\Users\sunda\PowerSystemProtection_ML\src')
sys.path.insert(0, str(SRC))
from correct_ml_pipeline import create_dataset, parse_filename, load_csv_file, extract_features_from_dataframe
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\Copy of Fault_Dataset_All\Fault_Dataset_All\data set')
FINAL_DIR = Path(r'C:\Users\sunda\PowerSystemProtection_ML\final one')
RESULTS_DIR = FINAL_DIR / 'results'
DATA_OUT = Path(r'C:\Users\sunda\PowerSystemProtection_ML\data')

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# STEP 1: Build consolidated ML-ready CSV from all 44 files
# ============================================================
print("=" * 70)
print("STEP 1: Building consolidated ML CSV from all 44 files")
print("=" * 70)

csv_files = sorted([f for f in DATA_DIR.iterdir() if f.is_file()])
print(f"Found {len(csv_files)} files in data directory")

rows = []
skipped = []
for i, fp in enumerate(csv_files, 1):
    filename = fp.name
    metadata = parse_filename(filename)
    if metadata is None:
        skipped.append((filename, "parse fail"))
        continue

    df = load_csv_file(fp)
    if df is None or len(df) < 10:
        skipped.append((filename, "load fail"))
        continue

    features = extract_features_from_dataframe(df)
    if features is None:
        skipped.append((filename, "feature fail"))
        continue

    row = {**metadata, **features}
    rows.append(row)
    print(f"  [{i}/{len(csv_files)}] {filename}: zone={metadata['zone']}, pos={metadata['position']}, fault={metadata['fault_type']}")

print(f"\nProcessed: {len(rows)} files successfully")
if skipped:
    print(f"Skipped: {skipped}")

df_all = pd.DataFrame(rows)
df_all = df_all.fillna(0)

# Save consolidated CSV
csv_path = DATA_OUT / 'real_fault_dataset.csv'
df_all.to_csv(csv_path, index=False)
print(f"\nSaved consolidated CSV: {csv_path}")
print(f"Shape: {df_all.shape} (rows x cols)")
print(f"Fault types: {df_all['fault_type'].value_counts().to_dict()}")
print(f"Zones: {df_all['zone'].value_counts().sort_index().to_dict()}")

# ============================================================
# STEP 2: Train ML models on the 44 scenarios
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Training ML models")
print("=" * 70)

exclude = ['zone', 'position', 'fault_type', 'fault_raw']
feature_cols = [c for c in df_all.columns if c not in exclude]
X = df_all[feature_cols].values
y_fault = df_all['fault_type'].values
y_zone = df_all['zone'].values

le_fault = LabelEncoder()
y_fault_enc = le_fault.fit_transform(y_fault)
le_zone = LabelEncoder()
y_zone_enc = le_zone.fit_transform(y_zone)

loo = LeaveOneOut()

# Task 1: Fault Classification
print("\nTask 1: Fault Classification")
rf_fault = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=5))
scores = cross_val_score(rf_fault, X, y_fault_enc, cv=loo, scoring='accuracy', n_jobs=-1)
rf_fault.fit(X, y_fault_enc)
print(f"  LOO Accuracy: {scores.mean():.3f} ({int(scores.sum())}/{len(scores)})")

# Task 2: Zone Detection
print("\nTask 2: Zone Detection")
rf_zone = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1, max_depth=5))
scores = cross_val_score(rf_zone, X, y_zone_enc, cv=loo, scoring='accuracy', n_jobs=-1)
rf_zone.fit(X, y_zone_enc)
print(f"  LOO Accuracy: {scores.mean():.3f} ({int(scores.sum())}/{len(scores)})")

# Task 3: IDMT Trip Time Regression
print("\nTask 3: Operating Time (IDMT)")
def compute_idmt(row):
    def trip(I, Is, TMS):
        if I <= Is: return -1.0
        M = I / Is
        if M <= 1.0: return -1.0
        t = 0.161 * 0.14 / (M**0.02 - 1.0) if False else 0.05 * 0.14 / (M**0.02 - 1.0)
        # Use correct TMS per relay
        return t
    I_R1 = max([row.get(f'R1_{p}_fault_mag', 0) for p in 'ABC'])
    I_R2 = max([row.get(f'R2_{p}_fault_mag', 0) for p in 'ABC'])
    t1 = compute_idmt_single(I_R1, 234.3, 0.161)
    t2 = compute_idmt_single(I_R2, 203.7, 0.05)
    return t1, t2

def compute_idmt_single(I, Is, TMS):
    if I <= Is: return -1.0
    M = I / Is
    if M <= 1.0: return -1.0
    t = TMS * 0.14 / (M**0.02 - 1.0)
    return max(0.04, t)

t_R1 = [compute_idmt_single(max([r.get(f'R1_{p}_fault_mag', 0) for p in 'ABC']), 234.3, 0.161) for _, r in df_all.iterrows()]
t_R2 = [compute_idmt_single(max([r.get(f'R2_{p}_fault_mag', 0) for p in 'ABC']), 203.7, 0.05) for _, r in df_all.iterrows()]
t_R1 = pd.Series(t_R1)
t_R2 = pd.Series(t_R2)

r1_mask = t_R1 > 0
r2_mask = t_R2 > 0
print(f"  R1 operating cases: {r1_mask.sum()}")
print(f"  R2 operating cases: {r2_mask.sum()}")

rf_r1 = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=5))
if r1_mask.sum() > 3:
    cv_mae = -cross_val_score(rf_r1, X[r1_mask], t_R1[r1_mask], cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf_r1.fit(X[r1_mask], t_R1[r1_mask])
    print(f"  R1 Trip Time MAE: {cv_mae.mean():.6f}s")

rf_r2 = make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, max_depth=5))
if r2_mask.sum() > 3:
    cv_mae = -cross_val_score(rf_r2, X[r2_mask], t_R2[r2_mask], cv=loo, scoring='neg_mean_absolute_error', n_jobs=-1)
    rf_r2.fit(X[r2_mask], t_R2[r2_mask])
    print(f"  R2 Trip Time MAE: {cv_mae.mean():.6f}s")

# Save models and encoders
joblib.dump({
    'fault_classification': rf_fault,
    'zone_detection': rf_zone,
    'R1_trip_time': rf_r1 if r1_mask.sum() > 3 else None,
    'R2_trip_time': rf_r2 if r2_mask.sum() > 3 else None,
}, RESULTS_DIR / 'models.joblib')

joblib.dump({'fault': le_fault, 'zone': le_zone}, RESULTS_DIR / 'encoders.joblib')

metrics = {
    'fault_classification': {'cv_acc_mean': float(scores.mean()), 'cv_correct': int(scores.sum()), 'cv_total': len(scores)},
    'zone_detection': {'cv_acc_mean': float(scores.mean()), 'cv_correct': int(scores.sum()), 'cv_total': len(scores)},
    'R1_trip_time': {'cv_mae_mean': float(cv_mae.mean()) if r1_mask.sum() > 3 else 0},
    'R2_trip_time': {'cv_mae_mean': float(cv_mae.mean()) if r2_mask.sum() > 3 else 0},
    'features_used': feature_cols,
}
json.dump(metrics, open(RESULTS_DIR / 'metrics.json', 'w'), indent=2)
json.dump(feature_cols, open(RESULTS_DIR / 'features.json', 'w'))

print(f"\nModels saved to {RESULTS_DIR}")

# ============================================================
# STEP 3: Generate distance/differential relay characteristics
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Distance & Differential Relay Characteristics")
print("=" * 70)

# Distance relay: R = V / I at fault location, compare to zone settings
# For each scenario, compute fault impedance from R1 voltage and current
dist_data = []
for _, row in df_all.iterrows():
    # R1 voltage magnitude (phase A) and R1 current magnitude (phase A)
    V_R1_A = row.get('R1_A_fault_mag', 0)
    I_R1_A = row.get('R1_A_fault_rms', 0)
    if I_R1_A > 0 and V_R1_A > 0:
        Z_R1 = V_R1_A / I_R1_A  # Fault impedance in ohms (per unit scaling)
    else:
        Z_R1 = 0

    # Distance relay reaches if Z_fault < Z_setting
    # Zone 1: 0-25% of line length ~ Z_set = 0.25 * Z_total
    # Zone 2: 25-50% ~ Z_set = 0.50 * Z_total
    # Zone 3: 50-75% ~ Z_set = 0.75 * Z_total
    zone = int(row['zone'])
    Z_setting = {1: 0.25, 2: 0.50, 3: 0.75}[zone]
    operates = 1 if Z_R1 <= Z_setting and Z_R1 > 0 else 0

    dist_data.append({
        'zone': zone,
        'position': int(row['position']),
        'fault_type': row['fault_type'],
        'Z_fault': round(Z_R1, 6),
        'Z_setting_zone': Z_setting,
        'distance_relay_operates': operates,
        'V_R1_A_mag': round(V_R1_A, 4),
        'I_R1_A_rms': round(I_R1_A, 4),
    })

df_dist = pd.DataFrame(dist_data)
df_dist.to_csv(RESULTS_DIR / 'distance_relay_results.csv', index=False)
print(f"Saved distance relay results: {RESULTS_DIR / 'distance_relay_results.csv'}")
print(f"Zone 1 operated: {df_dist[df_dist['zone']==1]['distance_relay_operates'].sum()}")
print(f"Zone 2 operated: {df_dist[df_dist['zone']==2]['distance_relay_operates'].sum()}")
print(f"Zone 3 operated: {df_dist[df_dist['zone']==3]['distance_relay_operates'].sum()}")

# Differential relay: compare current entering vs leaving (R1 vs SE)
diff_data = []
for _, row in df_all.iterrows():
    I_in = row.get('R1_A_fault_rms', 0)
    I_out = row.get('SE_A_fault_rms', 0)
    if I_in > 0 and I_out > 0:
        diff_current = abs(I_in - I_out) / max(I_in, I_out)  # Normalized imbalance
    else:
        diff_current = 0
    operates = 1 if diff_current < 0.05 else 0  # 5% threshold
    diff_data.append({
        'zone': int(row['zone']),
        'position': int(row['position']),
        'fault_type': row['fault_type'],
        'I_in_R1_A': round(I_in, 4),
        'I_out_SE_A': round(I_out, 4),
        'differential_ratio': round(diff_current, 6),
        'diff_relay_operates': operates,
    })

df_diff = pd.DataFrame(diff_data)
df_diff.to_csv(RESULTS_DIR / 'differential_relay_results.csv', index=False)
print(f"Saved differential relay results: {RESULTS_DIR / 'differential_relay_results.csv'}")

# ============================================================
# STEP 4: Generate IDMT curves plot
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Generating IDMT curves")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# IEC 60255 Standard Inverse curve parameters
K_SI, ALPHA_SI = 0.14, 0.02
TMS_R1, TMS_R2 = 0.161, 0.05
IS_R1, IS_R2 = 234.3, 203.7

def idmt_trip_time(I_primary, Is, TMS):
    if I_primary <= Is: return np.inf
    M = I_primary / Is
    t = TMS * K_SI / (M**ALPHA_SI - 1.0)
    return max(0.04, t)

M_range = np.linspace(1.05, 20, 500)
t_R1_curve = [idmt_trip_time(M*IS_R1, IS_R1, TMS_R1) for M in M_range]
t_R2_curve = [idmt_trip_time(M*IS_R2, IS_R2, TMS_R2) for M in M_range]

fig, ax = plt.subplots(figsize=(12, 8))
ax.plot(M_range, t_R1_curve, 'b-', linewidth=2.5, label=f'R1 (TMS={TMS_R1}, Is={IS_R1:.0f}A)')
ax.plot(M_range, t_R2_curve, 'r-', linewidth=2.5, label=f'R2 (TMS={TMS_R2}, Is={IS_R2:.0f}A)')

# Standard curves
t_SI_005 = [0.05 * K_SI / (M**ALPHA_SI - 1.0) for M in M_range]
t_SI_025 = [0.25 * K_SI / (M**ALPHA_SI - 1.0) for M in M_range]
ax.plot(M_range, t_SI_005, 'b--', alpha=0.5, label='SI Standard (TMS=0.05)')
ax.plot(M_range, t_SI_025, 'k--', alpha=0.5, label='SI Standard (TMS=0.25)')

# Coordination lines
for M in [2, 5, 10]:
    t_R2_gp = idmt_trip_time(M*IS_R2, IS_R2, TMS_R2)
    t_R1_req = t_R2_gp + 0.3
    ax.plot([M, M], [0, t_R2_gp], 'r--', alpha=0.3)
    ax.plot([M, M], [t_R2_gp, t_R1_req], 'k-', alpha=0.5, linewidth=3)
    ax.text(M, t_R2_gp/2, f'CTI\nM={M}', ha='center', fontsize=8)

ax.set_xlim(1.0, 20)
ax.set_ylim(0, 2.5)
ax.set_xlabel('Current Multiplier (M = I/Is)', fontsize=12)
ax.set_ylabel('Operating Time (seconds)', fontsize=12)
ax.set_title('IEC 60255 Standard Inverse IDMT Characteristics\nR1, R2 Coordination Comparison', fontsize=14)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(y=0.04, color='k', linestyle='--', alpha=0.5, label='Min (2 cycles)')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'idmt_curves.png', dpi=300, bbox_inches='tight')

# Operating times from dataset
scenarios = []
for _, row in df_all.iterrows():
    zone = int(row['zone'])
    I_R1 = max([row.get(f'R1_{p}_fault_mag', 0) for p in 'ABC'])
    I_R2 = max([row.get(f'R2_{p}_fault_mag', 0) for p in 'ABC'])
    t1 = idmt_trip_time(I_R1, IS_R1, TMS_R1) if I_R1 > IS_R1 else None
    t2 = idmt_trip_time(I_R2, IS_R2, TMS_R2) if I_R2 > IS_R2 else None
    if t1 and t2:
        relay = 'R1' if t1 < t2 else 'R2'
        scenarios.append((f'Z{zone}_{int(row["position"])}', zone, round(t1, 4), round(t2, 4), relay))

fig2, ax2 = plt.subplots(figsize=(14, 6))
labels = [s[0] for s in scenarios]
t1s = [s[2] for s in scenarios]
t2s = [s[3] for s in scenarios]
x = np.arange(len(scenarios))
w = 0.35
ax2.bar(x - w/2, t1s, w, label='R1', color='blue', alpha=0.7)
ax2.bar(x + w/2, t2s, w, label='R2', color='red', alpha=0.7)
ax2.set_xticks(x)
ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('Operating Time (seconds)')
ax2.set_title('Computed Operating Times from Real Dataset (IDMT)', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(RESULTS_DIR / 'idmt_operating_times.png', dpi=300, bbox_inches='tight')
print("IDMT curves saved")

# ============================================================
# STEP 5: Summary
# ============================================================
print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"Consolidated CSV: {csv_path}")
print(f"Shape: {df_all.shape[0]} scenarios x {df_all.shape[1]} features")
print(f"Fault types: {df_all['fault_type'].value_counts().to_dict()}")
print(f"Zones: {df_all['zone'].value_counts().sort_index().to_dict()}")
print(f"Models: {RESULTS_DIR}/models.joblib")
print(f"Metrics: {RESULTS_DIR}/metrics.json")
print(f"Distance relay: {RESULTS_DIR}/distance_relay_results.csv")
print(f"Differential relay: {RESULTS_DIR}/differential_relay_results.csv")
print(f"IDMT curves: {RESULTS_DIR}/idmt_curves.png")
print(f"IDMT operating times: {RESULTS_DIR}/idmt_operating_times.png")

# Save dataset as well for reference
df_all.to_csv(FINAL_DIR / 'ml_dataset.csv', index=False)
print(f"\nFull dataset saved to: {FINAL_DIR / 'ml_dataset.csv'}")
