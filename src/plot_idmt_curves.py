"""
plot_idmt_curves.py
===================
Generate and plot IDMT operating time curves for R1, R2, R3 relays
and compare with standard IEC 60255 characteristics.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# IEC 60255 Standard Inverse curve parameters
K_SI = 0.14       # Standard Inverse
ALPHA_SI = 0.02   # Standard Inverse

# Relay settings
TMS_R1 = 0.161
TMS_R2 = 0.050
TMS_R3 = 0.050  # Assuming R3 same as R2

IS_R1 = 234.3  # Pickup currents (A)
IS_R2 = 203.7
IS_R3 = 203.7

def idmt_trip_time(I_primary, Is, TMS, curve='SI'):
    """
    Compute IDMT operating time.

    IEC 60255 Standard Inverse: t = TMS * k / ((I/Is)^alpha - 1)
    k = 0.14, alpha = 0.02 for SI
    """
    if I_primary <= Is:
        return np.inf
    M = I_primary / Is
    t = TMS * K_SI / (M**ALPHA_SI - 1.0)
    return max(0.04, t)  # Minimum 2 cycles (40ms)

# Current multiplier range
M_range = np.linspace(1.05, 20, 500)

# Compute time curves
def compute_curve(Is, TMS, M_range):
    return [idmt_trip_time(M*Is, Is, TMS) for M in M_range]

t_R1 = compute_curve(IS_R1, TMS_R1, M_range)
t_R2 = compute_curve(IS_R2, TMS_R2, M_range)
t_R3 = compute_curve(IS_R3, TMS_R3, M_range)

# Standard IDMT curves with same TMS
t_SI_005 = compute_curve(1.0, 0.05, M_range)  # Standard SI with TMS=0.05
t_SI_010 = compute_curve(1.0, 0.10, M_range)  # Standard SI with TMS=0.10
t_SI_025 = compute_curve(1.0, 0.25, M_range)  # Standard SI with TMS=0.25

# Create figure
fig, ax = plt.subplots(figsize=(10, 7))

# Plot relay curves
ax.plot(M_range, t_R1, 'b-', linewidth=2.5, label=f'R1 (TMS=0.161, Is={IS_R1:.0f}A)')
ax.plot(M_range, t_R2, 'r-', linewidth=2.5, label=f'R2 (TMS=0.05, Is={IS_R2:.0f}A)')
ax.plot(M_range, t_R3, 'g-', linewidth=2.5, label=f'R3 (TMS=0.05, Is={IS_R3:.0f}A)')

# Plot standard curves for comparison
ax.plot(M_range, t_SI_005, 'b--', linewidth=1.5, label='SI (TMS=0.05) - Standard')
ax.plot(M_range, t_SI_025, 'k--', linewidth=1.5, alpha=0.7, label='SI (TMS=0.25)')

# Highlight coordination
# Time grading between R2 and R1
coord_margin = 0.3  # seconds
for M in [2, 5, 10]:
    I_gp = M * IS_R2  # Grading point: max fault at Bus B
    t_R2_gp = idmt_trip_time(I_gp, IS_R2, TMS_R2)
    t_R1_required = t_R2_gp + coord_margin
    ax.axvline(x=M, color='gray', linestyle=':', alpha=0.3)
    ax.plot([M, M], [0, t_R2_gp], 'r--', alpha=0.3)
    ax.plot([M, M], [t_R2_gp, t_R1_required], 'k-', alpha=0.5, linewidth=3)
    ax.text(M, t_R2_gp/2, f'CTI\nM={M}', ha='center', va='center', fontsize=8)

# Formatting
ax.set_xlim(1.0, 20)
ax.set_ylim(0, 2.5)
ax.set_xlabel('Current Multiplier (M = I/Is)', fontsize=12)
ax.set_ylabel('Operating Time (seconds)', fontsize=12)
ax.set_title('IEC 60255 Standard Inverse IDMT Characteristics\nR1, R2, R3 Coordination Comparison', fontsize=14)
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Add annotation for min operating time
ax.axhline(y=0.04, color='k', linestyle='--', alpha=0.5, label='Min (2 cycles)')

plt.tight_layout()
plt.savefig('results_final_protection/idmt_curves.png', dpi=300, bbox_inches='tight')

# Create a second plot showing operating times from the real dataset
fig2, ax2 = plt.subplots(figsize=(12, 6))

# From real dataset analysis
scenarios = [
    ('Z1 AG 25%', 1, 0.47),  # R1 operates
    ('Z1 BC 25%', 1, 0.47),  # R1 operates
    ('Z2 AG 50%', 2, 0.20),  # R2 operates
    ('Z2 BC 50%', 2, 0.20),  # R2 operates
    ('Z3 AG 75%', 3, 0.15),  # R2 operates
    ('Z3 BC 75%', 3, 0.15),  # R2 operates
]

zones = [s[1] for s in scenarios]
times = [s[2] for s in scenarios]

colors = ['blue' if z == 1 else 'red' if z == 2 else 'green' for z in zones]
bars = ax2.bar(range(len(scenarios)), times, color=colors, alpha=0.7)

# Add min coordination margin
ax2.axhline(y=0.20, color='r', linestyle='--', alpha=0.5, label='R2 Coordination Min')
ax2.axhline(y=0.35, color='b', linestyle='--', alpha=0.5, label='R1 Coordination Min (R2 + CTI=0.3)')

ax2.set_xticks(range(len(scenarios)))
ax2.set_xticklabels([s[0] for s in scenarios], rotation=45, ha='right')
ax2.set_ylabel('Operating Time (seconds)')
ax2.set_title('Computed Operating Times from Real Dataset', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, time in zip(bars, times):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{time:.2f}s', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('results_final_protection/idmt_operating_times.png', dpi=300, bbox_inches='tight')

print("IDMT curves generated:")
print("  - results_final_protection/idmt_curves.png (standard IDMT curves)")
print("  - results_final_protection/idmt_operating_times.png (real data operating times)")