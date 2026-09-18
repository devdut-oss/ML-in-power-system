"""
make_plots.py
==============
Publication-quality figures (300 dpi PNG -> results/figures/):

  fig1_system_diagram.png      single-line diagram of the study system
  fig2_fault_currents.png      fault current vs location per fault type
  fig3_idmt_curves.png         coordinated IDMT curves R1/R2 + grading point
  fig4_confusion_matrices.png  M2 fault-type + M3 relay-selection confusion
  fig5_regression.png          M4 location + M5 trip-time predicted vs actual
  fig6_feature_importance.png  Random-Forest feature importances (M2)
  fig7_idmt_ml_vs_analytical.png  ML-learned IDMT curve overlaid on IEC formula

Run AFTER train_models.py:  python src/make_plots.py
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from power_system import PowerSystem
from idmt_relay import coordinate, IDMTRelay

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, 'results', 'figures')
os.makedirs(FIG, exist_ok=True)

# ------------- palette (validated categorical set, light mode) -------------
C = {'blue': '#2a78d6', 'aqua': '#1baf7a', 'yellow': '#eda100',
     'green': '#008300', 'violet': '#4a3aa7', 'red': '#e34948',
     'magenta': '#e87ba4', 'orange': '#eb6834'}
INK, INK2, MUTED = '#0b0b0b', '#52514e', '#898781'
GRID, SURFACE = '#e1e0d9', '#fcfcfb'
# one-hue sequential ramp for heatmaps (white -> deep blue)
BLUES = LinearSegmentedColormap.from_list(
    'seq_blue', ['#fcfcfb', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])

plt.rcParams.update({
    'figure.facecolor': SURFACE, 'axes.facecolor': SURFACE,
    'savefig.facecolor': SURFACE, 'font.family': 'DejaVu Sans',
    'axes.edgecolor': '#c3c2b7', 'axes.labelcolor': INK,
    'xtick.color': INK2, 'ytick.color': INK2,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.axisbelow': True, 'font.size': 9.5,
    'axes.titlesize': 10.5, 'axes.titleweight': 'bold',
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})


def save(fig, name):
    fig.savefig(os.path.join(FIG, name))
    plt.close(fig)
    print(f"  wrote {name}")


# ------------------------------------------------------------- fig 1: SLD --
def fig1():
    fig, ax = plt.subplots(figsize=(9, 2.8))
    ax.set_xlim(0, 10); ax.set_ylim(0, 3); ax.axis('off')
    y = 1.5
    # source
    ax.add_patch(plt.Circle((0.7, y), 0.32, fill=False, lw=2, color=INK))
    ax.text(0.7, y, '~', ha='center', va='center', fontsize=16, color=INK)
    ax.text(0.7, 2.35, 'Grid source\n3500 MVA, X/R=10', ha='center',
            fontsize=8, color=INK2)
    # buses
    for x, name in [(2.2, 'Bus A'), (5.8, 'Bus B'), (9.3, 'Bus C')]:
        ax.plot([x, x], [y - 0.55, y + 0.55], lw=3.5, color=INK)
        ax.text(x, 2.35, name, ha='center', fontsize=9, weight='bold', color=INK)
    ax.plot([1.02, 2.2], [y, y], lw=1.8, color=INK)
    # lines
    ax.plot([2.2, 5.8], [y, y], lw=1.8, color=C['blue'])
    ax.plot([5.8, 9.3], [y, y], lw=1.8, color=C['aqua'])
    ax.text(4.0, 1.72, 'Line 1 — 50 km  (Zone 1)', ha='center', fontsize=8.5,
            color=C['blue'], weight='bold')
    ax.text(7.55, 1.72, 'Line 2 — 40 km  (Zone 2)', ha='center', fontsize=8.5,
            color='#128a60', weight='bold')
    # relays (squares just after each bus)
    for x, name, ct in [(2.75, 'R1', 'CT 400/1'), (6.35, 'R2', 'CT 300/1')]:
        ax.add_patch(plt.Rectangle((x - 0.17, y - 0.17), 0.34, 0.34,
                                   fill=True, fc=SURFACE, ec=C['red'], lw=2))
        ax.text(x, y - 0.55, f'{name}\nIDMT OC\n{ct}', ha='center', va='top',
                fontsize=7.5, color=INK2)
    # load arrow
    ax.annotate('', xy=(9.3, 0.45), xytext=(9.3, 0.95),
                arrowprops=dict(arrowstyle='-|>', lw=1.8, color=INK))
    ax.text(9.3, 0.30, 'Load 40 MVA\n0.85 pf', ha='center', va='top',
            fontsize=8, color=INK2)
    ax.text(0.25, 0.25, '132 kV, 50 Hz radial system', fontsize=9,
            style='italic', color=MUTED)
    ax.set_title('Fig. 1 — Single-line diagram of the study system')
    save(fig, 'fig1_system_diagram.png')


# ------------------------------------- fig 2: fault current vs location ----
def fig2():
    ps = PowerSystem()
    d = np.linspace(0.5, 89.5, 120)
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    series = [('ABC', C['red'], '3-phase (ABC)'),
              ('AB', C['orange'], 'L-L (AB)'),
              ('ABG', C['violet'], 'L-L-G (ABG)'),
              ('AG', C['blue'], 'L-G (AG), Rf=0'),
              ]
    for ftype, col, lab in series:
        I = [ps.simulate_fault(ftype, 1 if x <= 50 else 2,
                               x if x <= 50 else x - 50, 0.01)['R1_Ia']
             for x in d]
        ax.plot(d, np.array(I) / 1e3, lw=2, color=col, label=lab)
    I = [ps.simulate_fault('AG', 1 if x <= 50 else 2,
                           x if x <= 50 else x - 50, 50.0)['R1_Ia'] for x in d]
    ax.plot(d, np.array(I) / 1e3, lw=2, ls='--', color=C['blue'],
            label='L-G (AG), Rf=50 Ω')
    ax.axvline(50, color=MUTED, lw=1, ls=':')
    ax.text(50, ax.get_ylim()[1] * 0.93, ' Bus B', color=MUTED, fontsize=8.5)
    ax.axhline(0.2343, color=INK2, lw=1, ls='-.')
    ax.text(66, 0.4, 'R1 pickup (234 A)', color=INK2, fontsize=8)
    ax.set_xlabel('Fault distance from Bus A (km)')
    ax.set_ylabel('Fault current at R1 (kA)')
    ax.set_title('Fig. 2 — Fault current seen by relay R1 vs fault location')
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, 'fig2_fault_currents.png')


# ----------------------------------------- fig 3: coordinated IDMT curves --
def fig3():
    ps = PowerSystem()
    R1, R2 = coordinate(ps, verbose=False)
    I = np.logspace(np.log10(250), np.log10(16000), 400)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    ax.loglog(I, R1.trip_time(I), lw=2.2, color=C['blue'],
              label=f'R1 (Is={R1.Is:.0f} A, TMS={R1.TMS})')
    ax.loglog(I, R2.trip_time(I), lw=2.2, color=C['aqua'],
              label=f'R2 (Is={R2.Is:.0f} A, TMS={R2.TMS})')
    # grading point
    fb = ps.simulate_fault('ABC', 2, 0.001, 0.0)
    Igp = fb['R2_Ia']
    t2, t1 = R2.trip_time(Igp), R1.trip_time(Igp)
    ax.plot([Igp, Igp], [t2, t1], color=C['red'], lw=1.6)
    for t in (t1, t2):
        ax.plot(Igp, t, 'o', ms=7, color=C['red'], mec=SURFACE, mew=1.5)
    ax.annotate(f'CTI = {t1 - t2:.2f} s\n@ Bus B max fault ({Igp:.0f} A)',
                xy=(Igp, (t1 + t2) / 2), xytext=(Igp * 1.6, 0.9),
                fontsize=8.5, color=INK,
                arrowprops=dict(arrowstyle='->', color=INK2, lw=1))
    ax.set_xlabel('Current (A, primary)')
    ax.set_ylabel('Operating time (s)')
    ax.set_title('Fig. 3 — Coordinated IEC-SI IDMT characteristics (CTI = 0.3 s)')
    ax.legend(frameon=False, fontsize=9)
    ax.set_ylim(0.03, 30)
    save(fig, 'fig3_idmt_curves.png')


# ------------------------------------------ fig 4: confusion matrices -----
def _plot_cm(ax, yte, pred, labels, title):
    cm = confusion_matrix(yte, pred, labels=labels)
    im = ax.imshow(cm, cmap=BLUES)
    ax.set_xticks(range(len(labels)), labels, rotation=45, ha='right',
                  fontsize=7.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=7.5)
    thresh = cm.max() * 0.55
    for i in range(len(labels)):
        for j in range(len(labels)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha='center', va='center', fontsize=7,
                        color=SURFACE if cm[i, j] > thresh else INK)
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title); ax.grid(False)
    return im


def fig4():
    splits = joblib.load(os.path.join(ROOT, 'results', 'test_splits.joblib'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6),
                             gridspec_kw={'width_ratios': [1.55, 1]})
    yte, pred = splits['M2']
    labs = ['NONE', 'AG', 'BG', 'CG', 'AB', 'BC', 'CA', 'ABG', 'BCG', 'CAG', 'ABC']
    _plot_cm(axes[0], yte, pred, labs, 'M2: fault type (11-class, RF)')
    yte, pred = splits['M3']
    _plot_cm(axes[1], yte, pred, ['NONE', 'R1', 'R2'],
             'M3: relay selection (which relay operates)')
    fig.suptitle('Fig. 4 — Confusion matrices on the 20 % held-out test set',
                 fontsize=11, weight='bold')
    fig.tight_layout()
    save(fig, 'fig4_confusion_matrices.png')


# ------------------------------------------------ fig 5: regressions ------
def fig5():
    splits = joblib.load(os.path.join(ROOT, 'results', 'test_splits.joblib'))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, key, unit, title, col in [
            (axes[0], 'M4', 'km', 'M4: fault location (MLP)', C['blue']),
            (axes[1], 'M5', 's', 'M5: R1 IDMT trip time (RF)', C['violet'])]:
        yte, pred = splits[key]
        ax.scatter(yte, pred, s=14, alpha=0.45, color=col, edgecolors='none')
        lo, hi = min(yte.min(), pred.min()), max(yte.max(), pred.max())
        ax.plot([lo, hi], [lo, hi], lw=1.4, color=INK2, ls='--',
                label='ideal (y = x)')
        mae = np.mean(np.abs(yte - pred))
        r2 = 1 - np.sum((yte - pred) ** 2) / np.sum((yte - yte.mean()) ** 2)
        ax.set_xlabel(f'Actual ({unit})'); ax.set_ylabel(f'Predicted ({unit})')
        ax.set_title(title)
        ax.text(0.04, 0.93, f'MAE = {mae:.3f} {unit}\nR² = {r2:.4f}',
                transform=ax.transAxes, fontsize=9, va='top', color=INK)
        ax.legend(frameon=False, fontsize=8.5, loc='lower right')
    fig.suptitle('Fig. 5 — Regression performance (held-out test set)',
                 fontsize=11, weight='bold')
    fig.tight_layout()
    save(fig, 'fig5_regression.png')


# -------------------------------------- fig 6: feature importance ---------
def fig6():
    from train_models import FEATURES
    models = joblib.load(os.path.join(ROOT, 'results', 'models.joblib'))
    rf = models['M2']
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()[-14:]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    ax.barh(imp.index, imp.values, height=0.62, color=C['blue'])
    ax.set_xlabel('Importance (Gini)')
    ax.set_title('Fig. 6 — Top features for fault-type classification (RF)')
    ax.grid(axis='y', visible=False)
    save(fig, 'fig6_feature_importance.png')


# ------------------------- fig 7: ML-learned IDMT vs analytical curve -----
def fig7():
    """Sweep AG faults along the feeder, compare analytical IEC trip time of
    R1 with the ML model's prediction -> shows the ML relay has *learned* the
    IDMT characteristic."""
    from train_models import FEATURES
    ps = PowerSystem()
    R1, R2 = coordinate(ps, verbose=False)
    models = joblib.load(os.path.join(ROOT, 'results', 'models.joblib'))
    m5 = models['M5']

    rows, Iplot, t_true = [], [], []
    for x in np.linspace(1, 88, 60):
        zone = 1 if x <= 50 else 2
        d = x if x <= 50 else x - 50
        meas = ps.simulate_fault('AG', zone, d, 1.0)
        Imax = max(meas['R1_Ia'], meas['R1_Ib'], meas['R1_Ic'])
        if Imax <= R1.Is * 1.05:
            continue
        rows.append([meas[f] for f in FEATURES])
        Iplot.append(Imax)
        t_true.append(R1.trip_time(Imax))
    t_ml = m5.predict(np.array(rows))
    order = np.argsort(Iplot)
    Iplot = np.array(Iplot)[order]
    t_true = np.array(t_true)[order]; t_ml = t_ml[order]

    fig, ax = plt.subplots(figsize=(7, 4.4))
    I = np.logspace(np.log10(R1.Is * 1.05), np.log10(Iplot.max() * 1.1), 300)
    ax.loglog(I, R1.trip_time(I), lw=2, color=INK2,
              label='Analytical IEC-SI curve (R1)')
    ax.loglog(Iplot, t_ml, 'o', ms=6, color=C['red'], mec=SURFACE, mew=0.8,
              label='ML prediction (Random Forest)')
    ax.set_xlabel('Fault current at R1 (A)')
    ax.set_ylabel('Operating time (s)')
    ax.set_title('Fig. 7 — ML model reproduces the IDMT characteristic\n'
                 '(AG faults swept along the feeder, Rf = 1 Ω)')
    ax.legend(frameon=False, fontsize=9)
    save(fig, 'fig7_idmt_ml_vs_analytical.png')


if __name__ == '__main__':
    print("generating figures ...")
    fig1(); fig2(); fig3(); fig4(); fig5(); fig6(); fig7()
    print(f"all figures in {FIG}")
