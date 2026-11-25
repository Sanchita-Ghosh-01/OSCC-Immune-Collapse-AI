import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

# =====================================================================
# STATISTICAL DISCLAIMER:
# Disease stages (No Dysplasia, Mild, Moderate, Severe, OSCC) are ordinal
# categories. Mapping them onto an equidistant grid (0, 1, 2, 3, 4) via
# natural cubic splines provides a heuristic visual summary of directional
# phase shifts (spike and crash), not true continuous-time kinetics.
# =====================================================================

# Set up portable paths (consistent with main pipeline & correlation script)
USER_PATH = Path(os.environ.get("OSCC_DATA_DIR", Path(__file__).resolve().parent))
INPUT_FILE = USER_PATH / "processed_synergy_data_final.xlsx"


def analyze_spline_trajectory(df, score_col, stages, stage_names, n_boot=1000, seed=42):
    """Fit a natural cubic spline to stage means, find exact analytical extrema,
    and run a stage-stratified bootstrap to quantify peak stability."""
    df_sub = df.dropna(subset=['Stage_Ordinal', score_col]).copy()
    df_sub['Stage_Ordinal'] = df_sub['Stage_Ordinal'].astype(int)

    stats = df_sub.groupby('Stage_Ordinal')[score_col].agg(['count', 'mean', 'sem']).reindex(stages)
    if stats['mean'].isna().any():
        missing = stats[stats['mean'].isna()].index.tolist()
        raise ValueError(f"Missing stage(s) {missing} in {score_col}; cannot fit spline.")

    means = stats['mean'].values
    sems = stats['sem'].fillna(0.0).values

    # Fit natural cubic spline and exact analytical 1st & 2nd derivatives
    spline = CubicSpline(stages, means, bc_type='natural')
    d1_spline = spline.derivative(1)
    d2_spline = spline.derivative(2)

    x_min, x_max = float(stages[0]), float(stages[-1])
    roots = d1_spline.roots(extrapolate=False)

    # Exact interior local maxima (S'(x) = 0 and S''(x) < 0) and minima (S''(x) > 0)
    local_maxima = sorted([float(r) for r in roots if (x_min < r < x_max) and (d2_spline(r) < 0)])
    local_minima = sorted([float(r) for r in roots if (x_min < r < x_max) and (d2_spline(r) > 0)])

    grid = np.linspace(x_min, x_max, 400)
    curve = spline(grid)
    deriv = d1_spline(grid)

    strictly_increasing = bool(np.all(deriv > 0))
    strictly_decreasing = bool(np.all(deriv < 0))

    # Stage-stratified bootstrap (1,000 iterations) to test peak stability
    rng = np.random.RandomState(seed)
    stage_arrays = [df_sub.loc[df_sub['Stage_Ordinal'] == s, score_col].values for s in stages]
    boot_peaks = []
    boot_strictly_inc = 0

    for _ in range(n_boot):
        b_means = np.array([rng.choice(arr, size=len(arr), replace=True).mean() for arr in stage_arrays])
        b_spline = CubicSpline(stages, b_means, bc_type='natural')
        b_d1 = b_spline.derivative(1)
        b_d2 = b_spline.derivative(2)
        if np.all(b_d1(grid) > 0):
            boot_strictly_inc += 1
        b_roots = b_d1.roots(extrapolate=False)
        b_maxima = [float(r) for r in b_roots if (x_min < r < x_max) and (b_d2(r) < 0)]
        if b_maxima:
            boot_peaks.append(max(b_maxima))  # last pre-OSCC peak

    summary = stats.copy()
    summary.insert(0, 'Stage_Name', stage_names)

    return {
        "summary": summary,
        "means": means,
        "sems": sems,
        "grid": grid,
        "curve": curve,
        "deriv": deriv,
        "local_maxima": local_maxima,
        "local_minima": local_minima,
        "peak": local_maxima[-1] if local_maxima else None,
        "strictly_increasing": strictly_increasing,
        "strictly_decreasing": strictly_decreasing,
        "min_deriv": float(deriv.min()),
        "max_deriv": float(deriv.max()),
        "boot_peak_rate": len(boot_peaks) / n_boot,
        "boot_peak_ci": np.percentile(boot_peaks, [2.5, 97.5]) if boot_peaks else None,
        "boot_strictly_inc_rate": boot_strictly_inc / n_boot,
    }

print("Loading pre-processed data...")
try:
    xls = pd.ExcelFile(INPUT_FILE)
    df_clinical = pd.read_excel(xls, sheet_name='Local_Clinical_Proteins')
    df_geo = pd.read_excel(xls, sheet_name='Global_GEO_mRNA')
except Exception as e:
    print(f"Error loading data: {e}")
    raise SystemExit(1)

# =====================================================================
# PART 1: CLINICAL SERUM PROTEINS (i-IPS)
# =====================================================================
print("\n--- INITIATING ANALYSIS: CLINICAL SERUM PROTEINS (i-IPS) ---")

stages_clin = np.array([0, 1, 2, 3, 4])
names_clin = ['No Dysplasia', 'Mild\nDysplasia', 'Moderate\nDysplasia', 'Severe\nDysplasia', 'OSCC']
res_clin = analyze_spline_trajectory(df_clinical, 'i_IPS', stages_clin, [n.replace('\n', ' ') for n in names_clin])

print("Empirical Stage Summary (i-IPS):")
print(res_clin["summary"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"Derivative Range across [0, 4]: [{res_clin['min_deriv']:.4f}, {res_clin['max_deriv']:.4f}]")
if res_clin["local_minima"]:
    print(f"-> Analytical Local Minimum (Trough) at Pseudo-Time: {[round(x, 4) for x in res_clin['local_minima']]}")
if res_clin["peak"] is not None:
    print(f"-> Exact Analytical Systemic Peak (Crash Onset) at Pseudo-Time: {res_clin['peak']:.4f}")
    print(f"-> Bootstrap Stability (1,000 stratified resamples): Peak detected in {res_clin['boot_peak_rate']*100:.1f}% of resamples")
    if res_clin["boot_peak_ci"] is not None:
        print(f"-> Bootstrap 95% CI for Systemic Peak: [{res_clin['boot_peak_ci'][0]:.2f}, {res_clin['boot_peak_ci'][1]:.2f}]")


# =====================================================================
# PART 2: GSE30784 TISSUE mRNA (g-IPS)
# =====================================================================
print("\n--- INITIATING ANALYSIS: TISSUE mRNA (g-IPS) ---")

stages_geo = np.array([0, 1, 2])
names_geo = ['Normal', 'Dysplasia', 'OSCC']
res_geo = analyze_spline_trajectory(df_geo, 'g_IPS', stages_geo, names_geo)

print("Empirical Stage Summary (g-IPS):")
print(res_geo["summary"].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print(f"Derivative Range across [0, 2]: [{res_geo['min_deriv']:.4f}, {res_geo['max_deriv']:.4f}]")
if res_geo["peak"] is not None:
    print(f"-> Exact Analytical Tissue Peak at Pseudo-Time: {res_geo['peak']:.4f}")
elif res_geo["strictly_increasing"]:
    print("-> RESULT: Strictly monotonic escalation confirmed (S'(x) > 0 everywhere on [0, 2]; no collapse peak).")
    print(f"-> Bootstrap Stability (1,000 stratified resamples): Strictly escalating in {res_geo['boot_strictly_inc_rate']*100:.1f}% of resamples")
elif res_geo["strictly_decreasing"]:
    print("-> RESULT: Strictly decreasing trajectory (S'(x) < 0 everywhere).")
else:
    print("-> RESULT: Non-monotonic trajectory without an interior maximum.")


# =====================================================================
# 3. VISUALIZATIONS (Outputting two separate figures)
# =====================================================================
print("\nGenerating visualizations...")

# --- Figure 1: Clinical Proteins ---
fig1, ax1 = plt.subplots(1, 2, figsize=(14, 5))
ax1[0].plot(res_clin["grid"], res_clin["curve"], color='blue', linewidth=2.5, label="Natural Cubic Spline")
ax1[0].errorbar(
    stages_clin, res_clin["means"], yerr=res_clin["sems"],
    fmt='o', color='red', ecolor='darkred', elinewidth=1.5, capsize=4,
    markersize=7, label='Empirical Means (± SEM)', zorder=5
)
if res_clin["peak"] is not None:
    ax1[0].axvline(res_clin["peak"], color='red', linestyle='--', linewidth=1.8, alpha=0.7,
                   label=f'Heuristic Peak ({res_clin["peak"]:.2f})')
ax1[0].set_title("Systemic Proteins: Heuristic Pseudo-Time Trajectory", fontweight='bold')
ax1[0].set_xticks(stages_clin)
ax1[0].set_xticklabels(names_clin)
ax1[0].set_xlabel("Histopathological Stage")
ax1[0].set_ylabel("i-IPS Synergy Score")
ax1[0].legend()
ax1[0].grid(True, linestyle='--', alpha=0.6)

ax1[1].plot(res_clin["grid"], res_clin["deriv"], color='purple', linewidth=2.5, label="Analytical 1st Derivative")
ax1[1].axhline(0, color='black', linestyle='-', alpha=0.8)
if res_clin["peak"] is not None:
    ax1[1].axvline(res_clin["peak"], color='red', linestyle='--', linewidth=2.5,
                   label=f'Crash Onset ({res_clin["peak"]:.2f})')
ax1[1].set_title("Heuristic Velocity (Pseudo-Time Derivative)", fontweight='bold')
ax1[1].set_xticks(stages_clin)
ax1[1].set_xticklabels(names_clin)
ax1[1].set_xlabel("Histopathological Stage")
ax1[1].set_ylabel("Rate of Change (Δ i-IPS)")
ax1[1].legend()
ax1[1].grid(True, linestyle='--', alpha=0.6)

fig1.tight_layout()
fig1.savefig(USER_PATH / "Functional_Trajectory_Clinical.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig1)

# --- Figure 2: GEO Tissue mRNA ---
fig2, ax2 = plt.subplots(1, 2, figsize=(14, 5))
ax2[0].plot(res_geo["grid"], res_geo["curve"], color='green', linewidth=2.5, label="Natural Cubic Spline")
ax2[0].errorbar(
    stages_geo, res_geo["means"], yerr=res_geo["sems"],
    fmt='o', color='darkred', ecolor='darkred', elinewidth=1.5, capsize=4,
    markersize=7, label='Empirical Means (± SEM)', zorder=5
)
if res_geo["peak"] is not None:
    ax2[0].axvline(res_geo["peak"], color='red', linestyle='--', linewidth=1.8, alpha=0.7,
                   label=f'Heuristic Peak ({res_geo["peak"]:.2f})')
ax2[0].set_title("GSE30784: Heuristic Pseudo-Time Trajectory", fontweight='bold')
ax2[0].set_xticks(stages_geo)
ax2[0].set_xticklabels(names_geo)
ax2[0].set_xlabel("Histopathological Stage")
ax2[0].set_ylabel("g-IPS Synergy Score")
ax2[0].legend()
ax2[0].grid(True, linestyle='--', alpha=0.6)

ax2[1].plot(res_geo["grid"], res_geo["deriv"], color='orange', linewidth=2.5, label="Analytical 1st Derivative")
ax2[1].axhline(0, color='black', linestyle='-', alpha=0.8)
if res_geo["peak"] is not None:
    ax2[1].axvline(res_geo["peak"], color='red', linestyle='--', linewidth=2.5,
                   label=f'Crash Onset ({res_geo["peak"]:.2f})')
ax2[1].set_title("GSE30784 Heuristic Velocity (Pseudo-Time Derivative)", fontweight='bold')
ax2[1].set_xticks(stages_geo)
ax2[1].set_xticklabels(names_geo)
ax2[1].set_xlabel("Histopathological Stage")
ax2[1].set_ylabel("Rate of Change (Δ g-IPS)")
ax2[1].legend()
ax2[1].grid(True, linestyle='--', alpha=0.6)

fig2.tight_layout()
fig2.savefig(USER_PATH / "Functional_Trajectory_GEO.png", dpi=300, bbox_inches='tight', facecolor='white')
plt.close(fig2)

print(f"SUCCESS! Both analytics complete. Visualizations saved to:\n{USER_PATH}")