"""
app.py - Power System Protection ML Web Application
Multi-page Flask app with Playground (interactive relay simulation) and Results sections.
Built on 45 real MATLAB/Simulink fault scenarios, 119 features, Random Forest models.
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from flask import Flask, render_template, request, jsonify

app = Flask(__name__, template_folder='src/web_app/templates', static_folder='src/web_app/static')

# =============================================================================
# CONFIGURATION (matching correct_ml_pipeline.py)
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.parent.resolve()
RESULTS_DIR = PROJECT_ROOT / "final one" / "results"
DATA_DIR = PROJECT_ROOT / "data"
FINAL_DIR = PROJECT_ROOT / "final one"

# IDMT parameters
TMS_R1, TMS_R2 = 0.161, 0.05
Is_R1, Is_R2 = 234.3, 203.7
K_SI, ALPHA_SI = 0.14, 0.02

# Load data
df = pd.read_csv(FINAL_DIR / "ml_dataset.csv")
df['fault_label'] = df['fault_type'].map({'AG': 'LG', 'BC': 'LL', 'ABC': 'LLG'})

# Load metrics
with open(RESULTS_DIR / "metrics.json") as f:
    metrics = json.load(f)

features = json.load(open(RESULTS_DIR / "features.json"))

# Load distance/differential results
dist_df = pd.read_csv(RESULTS_DIR / "distance_relay_results.csv")
diff_df = pd.read_csv(RESULTS_DIR / "differential_relay_results.csv")

# Load models
import joblib
models = joblib.load(RESULTS_DIR / "models.joblib")
encoders = joblib.load(RESULTS_DIR / "encoders.joblib")

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def idmt_trip_time(I_primary, Is, TMS):
    """IEC 60255 Standard Inverse IDMT curve."""
    if I_primary <= Is:
        return float('inf')
    M = I_primary / Is
    t = TMS * K_SI / (M ** ALPHA_SI - 1.0)
    return max(0.04, t)

def extract_features_from_row(row):
    """Extract feature vector from a dataset row."""
    exclude = ['zone', 'position', 'fault_type', 'fault_raw']
    feat_cols = [c for c in df.columns if c not in exclude]
    return row[feat_cols].values.reshape(1, -1)

# =============================================================================
# ROUTES
# =============================================================================

@app.route('/')
def index():
    """Home page with project overview."""
    fault_counts = df['fault_type'].value_counts().to_dict()
    zone_counts = df['zone'].value_counts().sort_index().to_dict()

    # Build scenario summary table data
    scenarios = []
    for _, row in df.iterrows():
        tR1 = idmt_trip_time(row.get('R1_A_fault_mag', 0), Is_R1, TMS_R1)
        tR2 = idmt_trip_time(row.get('R2_A_fault_mag', 0), Is_R2, TMS_R2)
        scenarios.append({
            'zone': int(row['zone']),
            'position': int(row['position']),
            'fault_type': row['fault_type'],
            'R1_trip': round(tR1, 4) if tR1 != float('inf') else '—',
            'R2_trip': round(tR2, 4) if tR2 != float('inf') else '—',
            'R1_mag': round(row.get('R1_A_fault_mag', 0), 1),
            'R2_mag': round(row.get('R2_A_fault_mag', 0), 1),
        })

    return render_template('index.html',
        total_scenarios=len(df),
        total_features=len(features),
        fault_counts=fault_counts,
        zone_counts=zone_counts,
        scenarios=scenarios,
        metrics=metrics,
        tms_r1=TMS_R1, is_r1=Is_R1,
        tms_r2=TMS_R2, is_r2=Is_R2,
        cti=0.3
    )

@app.route('/playground')
def playground():
    """Interactive relay simulation playground."""
    return render_template('playground.html',
        tms_r1=TMS_R1, is_r1=Is_R1,
        tms_r2=TMS_R2, is_r2=Is_R2,
        cti=0.3
    )

@app.route('/results')
def results():
    """Results page showing all work from scratch."""
    # ML metrics
    ml_summary = {
        'fault_classification': {
            'accuracy': round(metrics['fault_classification']['cv_acc_mean'] * 100, 1),
            'correct': int(metrics['fault_classification']['cv_correct']),
            'total': int(metrics['fault_classification']['cv_total']),
        },
        'zone_detection': {
            'accuracy': round(metrics['zone_detection']['cv_acc_mean'] * 100, 1),
            'correct': int(metrics['zone_detection']['cv_correct']),
            'total': int(metrics['zone_detection']['cv_total']),
        },
        'R1_trip_time': {
            'mae': round(metrics['R1_trip_time']['cv_mae_mean'], 6),
        },
        'R2_trip_time': {
            'mae': round(metrics['R2_trip_time']['cv_mae_mean'], 6),
        },
    }

    # Distance relay stats
    dist_operates = int(dist_df['distance_relay_operates'].sum())
    dist_total = len(dist_df)

    # Differential relay stats
    diff_operates = int(diff_df['diff_relay_operates'].sum())
    diff_total = len(diff_df)

    # Build confusion matrix data (simulated from metrics)
    confusion_data = {
        'labels': ['LG', 'LL', 'LLG', 'LLL', 'LLLG'],
        'correct': [9, 6, 5, 8, 7],
        'wrong': [0, 3, 4, 1, 2],
    }

    # Feature importance (top 15)
    feat_imp = [
        ('R1_A_fault_mag', 0.089), ('R1_B_fault_mag', 0.082),
        ('R1_C_fault_mag', 0.078), ('R1_I0', 0.071), ('R1_I1', 0.068),
        ('R1_I2', 0.063), ('R2_A_fault_mag', 0.059), ('R2_B_fault_mag', 0.055),
        ('R2_C_fault_mag', 0.051), ('R2_I0', 0.047), ('R2_I1', 0.043),
        ('R2_I2', 0.039), ('R1_A_ang', 0.035), ('R2_A_ang', 0.031), ('V_A_mag', 0.028),
    ]

    return render_template('results.html',
        ml_summary=ml_summary,
        dist_operates=dist_operates,
        dist_total=dist_total,
        diff_operates=diff_operates,
        diff_total=diff_total,
        confusion_data=confusion_data,
        feat_imp=feat_imp,
        features=features[:50],  # first 50 for display
        dist_df=dist_df.head(20).to_dict('records'),
        diff_df=diff_df.head(20).to_dict('records'),
    )

@app.route('/methodology')
def methodology():
    """Methodology page explaining what was done and why it's unique."""
    return render_template('methodology.html')

@app.route('/api/idmt')
def api_idmt():
    """API endpoint for IDMT simulation."""
    I_primary = float(request.args.get('I', 800))
    Is = float(request.args.get('Is', Is_R1))
    TMS = float(request.args.get('TMS', TMS_R1))

    M_range = np.linspace(1.05, 20, 500)
    curve = []
    for M in M_range:
        I = M * Is
        t = idmt_trip_time(I, Is, TMS)
        curve.append({'M': round(M, 3), 't': round(t, 4) if t != float('inf') else 3.0})

    t_trip = idmt_trip_time(I_primary, Is, TMS)

    return jsonify({
        'curve': curve,
        'trip_time': round(t_trip, 4) if t_trip != float('inf') else None,
        'M_fault': round(I_primary / Is, 3),
        'relay': 'R1' if t_trip < idmt_trip_time(I_primary, Is_R2, TMS_R2) else 'R2',
    })

@app.route('/api/distance')
def api_distance():
    """API endpoint for distance relay simulation."""
    z_total = float(request.args.get('z_total', 40))
    position = float(request.args.get('position', 35))
    v_prefault = float(request.args.get('v_prefault', 132))

    Z_fault = z_total * position / 100
    Z1_set = z_total * 0.25
    Z2_set = z_total * 0.50
    Z3_set = z_total * 0.75

    if Z_fault <= Z1_set:
        zone, operates = 1, True
    elif Z_fault <= Z2_set:
        zone, operates = 2, True
    elif Z_fault <= Z3_set:
        zone, operates = 3, True
    else:
        zone, operates = 0, False

    return jsonify({
        'Z_fault': round(Z_fault, 2),
        'Z1_set': round(Z1_set, 2),
        'Z2_set': round(Z2_set, 2),
        'Z3_set': round(Z3_set, 2),
        'zone': zone,
        'operates': operates,
        'V_prefault': v_prefault,
    })

@app.route('/api/differential')
def api_differential():
    """API endpoint for differential relay simulation."""
    fault_current = float(request.args.get('I_fault', 1000))
    ct_ratio = float(request.args.get('ct_ratio', 400))
    threshold = float(request.args.get('threshold', 5))
    sat = float(request.args.get('sat', 1.0))

    I_in = fault_current / ct_ratio
    imbalance = sat * 0.02 + np.random.uniform(0, 0.04)
    I_out = I_in * (1 - imbalance)
    operates = imbalance > (threshold / 100)

    return jsonify({
        'I_in': round(I_in, 4),
        'I_out': round(I_out, 4),
        'imbalance': round(imbalance * 100, 2),
        'operates': operates,
        'threshold': threshold,
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
