import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from scipy.stats import f_oneway, shapiro, levene, kruskal
import scikit_posthocs as sp
import statsmodels.api as sm
from statsmodels.stats.power import TTestIndPower, FTestAnovaPower
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.utils import resample
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    average_precision_score,
    confusion_matrix,
)

import pwlf
import shap

# Set global seed for reproducible ML and bootstrapping
np.random.seed(42)


# --- 1. CONFIGURATION ---
DEFAULT_PATH = Path(r"E:\Documents\RESEARCH_Anusandhan\Medical Technology\DATA\Required data set")
USER_PATH = Path(os.environ.get("OSCC_DATA_DIR", DEFAULT_PATH if DEFAULT_PATH.exists() else Path(__file__).resolve().parent))
warnings.simplefilter(action='ignore', category=FutureWarning)

if not USER_PATH.exists():
    print(f" Warning: The path {USER_PATH} was not found. Please check the address.")


# --- 2. HELPER FUNCTIONS ---
def clean_numeric(val):
    if pd.isna(val) or val == '': return np.nan
    try:
        s = str(val).lower().replace('years', '').replace('yr', '').replace('kg', '').replace(',', '.').strip()
        # Safely keep numbers, decimals, negative signs, scientific 'e', and '<' signs
        s = re.sub(r'[^\d.\-eE<]', '', s)
        if '<' in s:
            return float(s.replace('<', '')) / 2.0  # Impute below limit-of-detection
        return float(s)
    except Exception:
        return np.nan

def categorize_diagnosis(diag):
    if pd.isna(diag): return 'No Dysplasia'
    d = str(diag).lower()
    if any(kw in d for kw in ['oscc', 'malignant', 'cancer']): return 'OSCC'
    if any(kw in d for kw in ['severe', 'high grade']): return 'Sev Dysplasia'
    if 'moderate' in d: return 'Mod Dysplasia'
    if any(kw in d for kw in ['mild', 'olkp', 'low grade']): return 'Mild Dysplasia'
    return 'No Dysplasia'

# --- 3. DATA LOADING ---
print(f" Loading Excel files from: {USER_PATH}...")

try:
    df1 = pd.read_excel(USER_PATH / "clean-dataset.xlsx")

    master = pd.DataFrame({
        'Patient_Name': df1['Sample No.'],
        'Sex': df1['Sex'],
        'Age': df1['Age'].apply(clean_numeric),
        'BMI': df1['BMI'].apply(clean_numeric),
        'IL-8': df1['IL-8 Mean'].apply(clean_numeric),
        'sPD-L1': df1['sPD-L1 Mean'].apply(clean_numeric),
        'Diagnosis': df1['Diagnosis']
    })

    master['Group'] = master['Diagnosis'].apply(categorize_diagnosis)
    print(f" Master data created: {len(master)} rows.")

except Exception as e:
    print(f" Error loading files: {e}")
    exit()

# --- 4. DATA CLEANING & RESIDUAL ANALYSIS ---
print(f"Original Cohort Size: {len(master)}")

master_both = master.dropna(subset=['Age', 'BMI', 'IL-8', 'sPD-L1', 'Sex']).copy()
print(f"Cohort Size after dropping missing demographics/biomarkers: {len(master_both)}")

master_both = master_both[(master_both['IL-8'] > 0) & (master_both['sPD-L1'] > 0)].copy()

def encode_sex(val):
    s = str(val).strip().upper()
    if s in ['M', 'MALE']: return 1
    if s in ['F', 'FEMALE']: return 0
    return np.nan  # Safely catches ambiguous/missing data

master_both['Sex_Binary'] = master_both['Sex'].apply(encode_sex)
master_both = master_both.dropna(subset=['Sex_Binary']).copy()
master_both['Sex_Binary'] = master_both['Sex_Binary'].astype(int)
print(f"Final Analytical Cohort Size (Positive biomarkers & valid Sex): {len(master_both)}")

def calculate_residuals_and_betas(df, target):
    X = df[['Age', 'BMI', 'Sex_Binary']]
    y = np.log1p(df[target])
    model = LinearRegression().fit(X, y)
    resid = y - model.predict(X)
    return resid, (model.intercept_, model.coef_[0], model.coef_[1], model.coef_[2])


master_both['IL-8_adj'], beta_il8 = calculate_residuals_and_betas(master_both, 'IL-8')
master_both['sPD-L1_adj'], beta_pdl1 = calculate_residuals_and_betas(master_both, 'sPD-L1')

master_both['i_IPS'] = master_both['IL-8_adj'] + master_both['sPD-L1_adj']

print("\n--- MULTICOLLINEARITY (VIF) CHECK ---")
X_vif = master_both[['Age', 'BMI', 'Sex_Binary']].copy()
X_vif_const = sm.add_constant(X_vif)
vif_df = pd.DataFrame({
    'Feature': X_vif_const.columns,
    'VIF': [variance_inflation_factor(X_vif_const.values, i) for i in range(X_vif_const.shape[1])]
})
vif_df = vif_df[vif_df['Feature'] != 'const']
print(vif_df)

print("Note: VIF < 5 indicates no severe multicollinearity among clinical covariates.")

# --- 4b. COHORT DEMOGRAPHICS & POWER ANALYSIS ---
print("\n--- COHORT DEMOGRAPHICS & CLASS DISTRIBUTION ---")
print(f"Total Valid N = {len(master_both)}")
print("\nHistological Groups (n per group):")
print(master_both['Group'].value_counts())

master_both['is_HighRisk'] = master_both['Group'].isin(['Sev Dysplasia', 'OSCC']).astype(int)
n_high = int(master_both['is_HighRisk'].sum())
n_low = int((master_both['is_HighRisk'] == 0).sum())

# 1. Target-effect sensitivity power for primary 5-group One-Way ANOVA (Large Cohen's f = 0.40)
anova_power_calc = FTestAnovaPower()
power_anova_target = anova_power_calc.solve_power(
    effect_size=0.40, nobs=len(master_both), alpha=0.05, k_groups=5
)
print(f"-> 5-Group ANOVA Power to detect a Large Effect (f=0.40, N={len(master_both)}): {power_anova_target:.4f}")

# 2. Target-effect sensitivity power for 2-group High-Risk vs Low-Risk (Large Cohen's d = 0.80)
ttest_power_calc = TTestIndPower()
power_ttest_target = ttest_power_calc.solve_power(
    effect_size=0.80, nobs1=n_high, ratio=n_low / n_high, alpha=0.05
)
print(f"-> 2-Group Power (High-Risk n={n_high} vs Low-Risk n={n_low}, d=0.80): {power_ttest_target:.4f}")

# Safe initializations
tukey_df, dunn_df, geo_dunn = None, None, None

# --- 5. STATISTICAL CALCULATIONS & TABLES ---
print("\n" + "=" * 50)
print("--- CLINICAL MODEL COEFFICIENTS ---")
print(
    f"IL-8   Baseline: B0={beta_il8[0]:.4f}, B1(Age)={beta_il8[1]:.4f}, B2(BMI)={beta_il8[2]:.4f}, B3(Sex)={beta_il8[3]:.4f}")
print(
    f"sPD-L1 Baseline: B0={beta_pdl1[0]:.4f}, B1(Age)={beta_pdl1[1]:.4f}, B2(BMI)={beta_pdl1[2]:.4f}, B3(Sex)={beta_pdl1[3]:.4f}")
print("\n--- OLS RESIDUAL NORMALITY CHECKS ---")
print(f"IL-8_adj OLS Residuals Shapiro-Wilk:   p={shapiro(master_both['IL-8_adj'])[1]:.4f}")
print(f"sPD-L1_adj OLS Residuals Shapiro-Wilk: p={shapiro(master_both['sPD-L1_adj'])[1]:.4f}")

group_order = ['No Dysplasia', 'Mild Dysplasia', 'Mod Dysplasia', 'Sev Dysplasia', 'OSCC']

anova_groups = [master_both[master_both['Group'] == g]['i_IPS'] for g in group_order if len(master_both[master_both['Group'] == g]) >= 2]
print("\n--- STATISTICAL ASSUMPTION CHECKS ---")
# Group-wise Shapiro (only on groups with >=3 samples)
shapiro_ps = [shapiro(g)[1] for g in anova_groups if len(g) >= 3]
p_shapiro_min = min(shapiro_ps) if shapiro_ps else 0
stat_levene, p_levene = levene(*anova_groups)

print(f"Min Group-wise Shapiro (Normality): p={p_shapiro_min:.4f}")
print(f"Levene's Test (Equal Variance): p={p_levene:.4f}")

if p_shapiro_min > 0.05 and p_levene > 0.05:
    f_stat, p_val_anova = f_oneway(*anova_groups)
    df_between, df_within = len(anova_groups) - 1, len(master_both) - len(anova_groups)
    eta_sq = (f_stat * df_between) / ((f_stat * df_between) + df_within)
    cohens_f_obs = np.sqrt(eta_sq / (1.0 - eta_sq))
    obs_anova_power = anova_power_calc.solve_power(
        effect_size=cohens_f_obs, nobs=len(master_both), alpha=0.05, k_groups=len(anova_groups)
    )
    print(f"\n--- ONE-WAY ANOVA (Parametric Omnibus) ---")
    print(f"F: {f_stat:.4f}, p: {p_val_anova:.4f}, Effect Size (η²): {eta_sq:.4f}, Observed Power: {obs_anova_power:.4f}")

    print("\n--- TUKEY'S HSD POST-HOC COMPARISONS (Parametric) ---")
    tukey_res = pairwise_tukeyhsd(endog=master_both['i_IPS'], groups=master_both['Group'], alpha=0.05)
    print(tukey_res)
    tukey_df = pd.DataFrame(
        data=tukey_res._results_table.data[1:],
        columns=tukey_res._results_table.data[0]
    )

    # Non-parametric sensitivity check (properly paired with Kruskal-Wallis)
    h_stat_sens, p_kw_sens = kruskal(*anova_groups)
    print(f"\n--- NON-PARAMETRIC SENSITIVITY CHECK (Kruskal-Wallis + Dunn's FDR-BH) ---")
    print(f"Kruskal-Wallis H: {h_stat_sens:.4f}, p: {p_kw_sens:.4f}")
    dunn_df = sp.posthoc_dunn(master_both, val_col='i_IPS', group_col='Group', p_adjust='fdr_bh')
    print(dunn_df)
else:
    h_stat, p_val_anova = kruskal(*anova_groups)
    k, n = len(anova_groups), len(master_both)
    epsilon_sq = (h_stat - k + 1) / (n - k)
    print(f"\n--- KRUSKAL-WALLIS (Non-Parametric Omnibus) ---")
    print(f"H: {h_stat:.4f}, p: {p_val_anova:.4f}, Effect Size (ε²): {epsilon_sq:.4f}")

    print("\n--- DUNN'S POST-HOC COMPARISONS (FDR-BH) ---")
    dunn_df = sp.posthoc_dunn(master_both, val_col='i_IPS', group_col='Group', p_adjust='fdr_bh')
    print(dunn_df)
print("\nClass Distribution (High-Risk vs Low-Risk):")
print(master_both['is_HighRisk'].value_counts(normalize=True).apply(lambda x: f"{x*100:.1f}%"))

# --- 5-FOLD STRATIFIED CV FOR i-IPS DIAGNOSTIC POWER ---
# OLS covariate adjustment and Youden cutoffs are fit inside each training fold.
X_cov_all = master_both[['Age', 'BMI', 'Sex_Binary']].values
y_il8_log_all = np.log1p(master_both['IL-8'].values)
y_pdl1_log_all = np.log1p(master_both['sPD-L1'].values)
y_hr_all = master_both['is_HighRisk'].values

skf_sec5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
oof_i_ips = np.zeros(len(master_both))
oof_hr_preds = np.zeros(len(master_both), dtype=int)
fold_hr_aucs = []
fold_hr_thresholds = []

for tr_idx, te_idx in skf_sec5.split(X_cov_all, y_hr_all):
    # 1. Fit OLS on training fold
    ols_il8 = LinearRegression().fit(X_cov_all[tr_idx], y_il8_log_all[tr_idx])
    ols_pdl1 = LinearRegression().fit(X_cov_all[tr_idx], y_pdl1_log_all[tr_idx])

    # 2. Compute training-fold residuals & learn Youden threshold  on training fold
    tr_i_ips = (
        (y_il8_log_all[tr_idx] - ols_il8.predict(X_cov_all[tr_idx]))
        + (y_pdl1_log_all[tr_idx] - ols_pdl1.predict(X_cov_all[tr_idx]))
    )
    fpr_tr, tpr_tr, thr_tr = roc_curve(y_hr_all[tr_idx], tr_i_ips)
    valid_tr = np.isfinite(thr_tr)
    fold_thr = float(thr_tr[valid_tr][np.argmax((tpr_tr - fpr_tr)[valid_tr])])
    fold_hr_thresholds.append(fold_thr)

    # 3. Apply frozen training OLS models & threshold to unseen test fold
    te_i_ips = (
        (y_il8_log_all[te_idx] - ols_il8.predict(X_cov_all[te_idx]))
        + (y_pdl1_log_all[te_idx] - ols_pdl1.predict(X_cov_all[te_idx]))
    )
    oof_i_ips[te_idx] = te_i_ips
    oof_hr_preds[te_idx] = (te_i_ips >= fold_thr).astype(int)
    fold_hr_aucs.append(roc_auc_score(y_hr_all[te_idx], te_i_ips))

# Pooled OOF ROC & Stratified Bootstrap 95% CI (feeds Section 8 ROC plots automatically)
fpr, tpr, _ = roc_curve(y_hr_all, oof_i_ips)
auc_val = roc_auc_score(y_hr_all, oof_i_ips)

n_bootstraps = 1000
bootstrapped_scores = []
rng = np.random.RandomState(42)
for _ in range(n_bootstraps):
    indices = resample(np.arange(len(y_hr_all)), replace=True, stratify=y_hr_all, random_state=rng)
    bootstrapped_scores.append(roc_auc_score(y_hr_all[indices], oof_i_ips[indices]))

ci_lower, ci_upper = np.percentile(bootstrapped_scores, [2.5, 97.5])

tn5, fp5, fn5, tp5 = confusion_matrix(y_hr_all, oof_hr_preds, labels=[0, 1]).ravel()
sensitivity = tp5 / (tp5 + fn5) if (tp5 + fn5) > 0 else 0.0
specificity = tn5 / (tn5 + fp5) if (tn5 + fp5) > 0 else 0.0
ppv_sec5 = tp5 / (tp5 + fp5) if (tp5 + fp5) > 0 else 0.0
npv_sec5 = tn5 / (tn5 + fn5) if (tn5 + fn5) > 0 else 0.0
acc_sec5 = (tp5 + tn5) / len(y_hr_all)

print(f"\n--- DIAGNOSTIC POWER (LEAKAGE-FREE 5-FOLD OUT-OF-FOLD CV) ---")
print(f"NOTE: OLS covariate adjustment and Youden cutoffs are fit strictly inside each training fold.")
print(f"Clinical Target: High-Risk Transformation (Sev Dysplasia + OSCC)")
print(f"Out-of-Fold i-IPS AUC: {auc_val:.3f} (95% CI: {ci_lower:.3f} - {ci_upper:.3f}) | Mean 5-Fold AUC: {np.mean(fold_hr_aucs):.3f}")
print(f"Learned Clinical Thresholds (5-Fold Mean ± SD): {np.mean(fold_hr_thresholds):.4f} ± {np.std(fold_hr_thresholds, ddof=1):.4f}")
print(f" -> Out-of-Fold Sensitivity: {sensitivity*100:.1f}% ({tp5}/{tp5+fn5})")
print(f" -> Out-of-Fold Specificity: {specificity*100:.1f}% ({tn5}/{tn5+fp5})")
print(f" -> Out-of-Fold PPV / NPV:   {ppv_sec5*100:.1f}% / {npv_sec5*100:.1f}% | Accuracy: {acc_sec5*100:.1f}%")
print("=" * 50)


# ==============================================================================
# --- 7. ADVANCED MATHEMATICAL SCOPES ---
# ==============================================================================
print("\n--- ADVANCED SCOPE 1: IMMUNE EVASION RATIO (IEI) ---")
master_both['IEI_Log_Ratio'] = master_both['sPD-L1_adj'] - master_both['IL-8_adj']
iei_groups = [master_both[master_both['Group'] == g]['IEI_Log_Ratio'] for g in group_order if len(master_both[master_both['Group'] == g]) >= 2]
h_stat_iei, p_val_iei = kruskal(*iei_groups)
print(f"Kruskal-Wallis test for IEI (Log-Ratio): p={p_val_iei:.4f}")

print("\n--- ADVANCED SCOPE 2: PIECEWISE REGRESSION (Biological Breakpoint) ---")
print("NOTE: Disease stages are mapped as a continuous linear pseudo-time (0 to 4).")
print("This assumes equal biological progression distance between clinical stages.")
stage_map = {g: i for i, g in enumerate(group_order)}
master_both['Stage_Ordinal'] = master_both['Group'].map(stage_map)

x = master_both['Stage_Ordinal'].values
y = master_both['i_IPS'].values
my_pwlf = pwlf.PiecewiseLinFit(x, y)

# 1-Segment (Linear) BIC
my_pwlf.fit(1)
bic_1_segment = len(y) * np.log(my_pwlf.ssr / len(y)) + 2 * np.log(len(y))

# 2-Segment (Biphasic) BIC ( seeded for reproducibility)
breakpoints = my_pwlf.fit(2, seed=42)
bic_2_segment = len(y) * np.log(my_pwlf.ssr / len(y)) + 4 * np.log(len(y))

print("\n--- PIECEWISE REGRESSION VALIDATION (BIC) ---")
print(f"BIC (Linear / 1-Segment): {bic_1_segment:.2f}")
print(f"BIC (Biphasic / 2-Segment): {bic_2_segment:.2f}")
if bic_2_segment < bic_1_segment:
    print("-> BIC favors the 2-Segment Biphasic model.")
else:
    print("-> NOTE: BIC favors the 1-Segment Linear model (2-segment knot is exploratory only).")

optimal_knot = breakpoints[1]
print(f"-> Algorithm heuristically estimated the optimal knot at Pseudo-Time Coordinate: {optimal_knot:.4f}")

# --- BOOTSTRAPPED KNOT CONFIDENCE (Sensitivity Analysis) ---
print("\n--- KNOT SENSITIVITY ANALYSIS (Bootstrapping) ---")
knot_samples = []
rng_knot = np.random.RandomState(42)
for b in range(200):
    idx = resample(np.arange(len(x)), replace=True, random_state=rng_knot)
    try:
        boot_pwlf = pwlf.PiecewiseLinFit(x[idx], y[idx])
        boot_pwlf.fit(2, seed=42 + b)
        knot_samples.append(boot_pwlf.fit_breaks[1])
    except Exception:
        continue
if knot_samples:
    knot_ci = np.percentile(knot_samples, [2.5, 97.5])
    print(f"-> Heuristic Knot 95% CI: {knot_ci[0]:.2f} - {knot_ci[1]:.2f}")

# --- ADVANCED SCOPE 2b: PURE RAW DATA TIPPING POINT (Dedent to top level) ---
print("\n--- ADVANCED SCOPE 2b: PURE RAW DATA TIPPING POINT (No Human Bias) ---")
print("NOTE: Evaluating if an 'immune collapse' exists in the pure raw biomarkers before age/demographic adjustment.")

master_both['raw_i_IPS'] = np.log1p(master_both['IL-8']) + np.log1p(master_both['sPD-L1'])
x_raw = master_both['Stage_Ordinal'].values
y_raw = master_both['raw_i_IPS'].values

pwlf_raw = pwlf.PiecewiseLinFit(x_raw, y_raw)
pwlf_raw.fit(1)
bic_1_raw = len(y_raw) * np.log(pwlf_raw.ssr / len(y_raw)) + 2 * np.log(len(y_raw))

breaks_raw = pwlf_raw.fit(2, seed=42)
bic_2_raw = len(y_raw) * np.log(pwlf_raw.ssr / len(y_raw)) + 4 * np.log(len(y_raw))

print(f"RAW BIC (Linear / 1-Segment): {bic_1_raw:.2f}")
print(f"RAW BIC (Biphasic / 2-Segment): {bic_2_raw:.2f}")

if bic_2_raw < bic_1_raw:
    print("\n -> DISCOVERY: A non-linear phase shift (collapse) is estimated even in RAW data!")
    print(f" -> Algorithm heuristically located the cliff at Pseudo-Time Coordinate: {breaks_raw[1]:.4f}")
else:
    print("\n -> CONCLUSION: No immune collapse detected in RAW data. BIC favors linear progression.")

# ==============================================================================
# --- 7b. INTEGRATED 5-FOLD CV: THE "EARLY WARNING" AI MODEL ---
# ==============================================================================
print("\n" + "=" * 50)
print("--- 7b. VALIDATING THE EARLY WARNING AI (Predicting the Immune Peak) ---")
print("NOTE: OSCC patients are excluded here. The AI is trained exclusively on pre-malignant")
print("patients to detect the Severe Dysplasia 'Spike' before invasion occurs.")

# 1. Isolate the Pre-Malignant Cohort
ml_cohort = master_both[master_both['Group'] != 'OSCC'].copy()

# 2. Define the new Biological Target: The Immune Peak (Severe Dysplasia)
ml_cohort['is_ImmunePeak'] = ml_cohort['Group'].isin(['Sev Dysplasia']).astype(int)

X_raw_cv = ml_cohort[['Age', 'BMI', 'Sex_Binary', 'IL-8', 'sPD-L1']]
y_cv = ml_cohort['is_ImmunePeak'].values
print(f"Pre-Malignant ML Cohort: N={len(ml_cohort)} (Severe Dysplasia={int(y_cv.sum())}, Non-Severe={int((y_cv == 0).sum())})")

def compute_clinical_metrics(y_true_arr, y_pred_arr, label, threshold_desc):
    tn, fp, fn, tp = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    bal_acc = 0.5 * (sens + spec)
    acc = (tp + tn) / len(y_true_arr)
    return {
        'Operating_Point': label,
        'Decision_Threshold': threshold_desc,
        'TP_Captured': int(tp),
        'FN_Missed': int(fn),
        'FP_FalseAlarm': int(fp),
        'TN_Cleared': int(tn),
        'Sensitivity_Capture_Pct': round(sens * 100, 2),
        'Specificity_Pct': round(spec * 100, 2),
        'PPV_Precision_Pct': round(ppv * 100, 2),
        'NPV_Pct': round(npv * 100, 2),
        'Balanced_Accuracy_Pct': round(bal_acc * 100, 2),
        'Overall_Accuracy_Pct': round(acc * 100, 2),
    }

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

fold_aucs, fold_accs, fold_briers, fold_pr_aucs = [], [], [], []
fold_opt_thresholds, fold_sens_def, fold_sens_tuned, fold_spec_def, fold_spec_tuned = [], [], [], [], []
fold_shap_means = []
all_y_true_cv, all_y_probs_cv = [], []
oof_probs = np.zeros(len(X_raw_cv))
oof_preds_default = np.zeros(len(X_raw_cv), dtype=int)
oof_preds_tuned = np.zeros(len(X_raw_cv), dtype=int)
oof_thresholds = np.zeros(len(X_raw_cv))
cv_shap_values_global = np.zeros((len(X_raw_cv), 5))  # 5 features in the Random Forest

for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw_cv, y_cv), 1):
    X_train_raw, X_test_raw = X_raw_cv.iloc[train_idx].copy(), X_raw_cv.iloc[test_idx].copy()
    y_train, y_test = y_cv[train_idx], y_cv[test_idx]

    # Log-transform biomarkers alongside covariates (row-wise, zero leakage)
    X_train_adj = X_train_raw.copy()
    X_test_adj = X_test_raw.copy()

    X_train_adj['IL-8_log'] = np.log1p(X_train_adj['IL-8'])
    X_train_adj['sPD-L1_log'] = np.log1p(X_train_adj['sPD-L1'])
    X_test_adj['IL-8_log'] = np.log1p(X_test_adj['IL-8'])
    X_test_adj['sPD-L1_log'] = np.log1p(X_test_adj['sPD-L1'])

    cols = ['Age', 'BMI', 'Sex_Binary', 'IL-8_log', 'sPD-L1_log']
    X_train_adj = X_train_adj[cols]
    X_test_adj = X_test_adj[cols]

    # 1. Train the Random Forest Classifier on training fold
    rf_fold = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42)
    rf_fold.fit(X_train_adj, y_train)

    probs = rf_fold.predict_proba(X_test_adj)[:, 1]
    preds = rf_fold.predict(X_test_adj)

    # 2. Nested Inner 4-Fold CV Threshold Tuning (inside training fold)
    inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    inner_oof_probs = np.zeros(len(y_train))
    for in_tr, in_val in inner_skf.split(X_train_adj, y_train):
        rf_inner = RandomForestClassifier(n_estimators=100, max_depth=4, class_weight='balanced', random_state=42)
        rf_inner.fit(X_train_adj.iloc[in_tr], y_train[in_tr])
        inner_oof_probs[in_val] = rf_inner.predict_proba(X_train_adj.iloc[in_val])[:, 1]

    fpr_in, tpr_in, thr_in = roc_curve(y_train, inner_oof_probs)
    valid_in = np.isfinite(thr_in)
    fold_opt_thresh = float(thr_in[valid_in][np.argmax((tpr_in - fpr_in)[valid_in])])
    preds_tuned = (probs >= fold_opt_thresh).astype(int)

    # 3. Record Outer-Fold Evaluation Metrics
    fold_aucs.append(roc_auc_score(y_test, probs))
    fold_accs.append(accuracy_score(y_test, preds))
    fold_briers.append(brier_score_loss(y_test, probs))
    fold_pr_aucs.append(average_precision_score(y_test, probs))
    fold_opt_thresholds.append(fold_opt_thresh)

    tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_test, preds, labels=[0, 1]).ravel()
    tn_t, fp_t, fn_t, tp_t = confusion_matrix(y_test, preds_tuned, labels=[0, 1]).ravel()
    fold_sens_def.append(tp_d / (tp_d + fn_d) if (tp_d + fn_d) > 0 else 0.0)
    fold_spec_def.append(tn_d / (tn_d + fp_d) if (tn_d + fp_d) > 0 else 0.0)
    fold_sens_tuned.append(tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0.0)
    fold_spec_tuned.append(tn_t / (tn_t + fp_t) if (tn_t + fp_t) > 0 else 0.0)

    all_y_true_cv.extend(y_test)
    all_y_probs_cv.extend(probs)
    oof_probs[test_idx] = probs
    oof_preds_default[test_idx] = preds
    oof_preds_tuned[test_idx] = preds_tuned
    oof_thresholds[test_idx] = fold_opt_thresh

    # 4. Extract Out-of-Fold SHAP values safely
    explainer_fold = shap.TreeExplainer(rf_fold)
    shap_f = explainer_fold.shap_values(X_test_adj)
    if isinstance(shap_f, list):
        shap_f = shap_f[1]
    elif len(shap_f.shape) == 3:
        shap_f = shap_f[:, :, 1]

    cv_shap_values_global[test_idx, :] = shap_f
    fold_shap_means.append(np.abs(shap_f).mean(axis=0))

    print(f"  Fold {fold} Complete: Peak-Detection AUC = {fold_aucs[-1]:.3f} | Nested Tuned Cutoff = {fold_opt_thresh*100:.1f}%")

cv_stability_results = pd.DataFrame({
    'Metric': ['Mean ROC-AUC', 'Mean PR-AUC (Imbalance-Robust)', 'Mean Brier Score', 'Mean Accuracy'],
    'Value': [np.mean(fold_aucs), np.mean(fold_pr_aucs), np.mean(fold_briers), np.mean(fold_accs)]
})

prevalence = y_cv.mean()
brier_null = prevalence * (1.0 - prevalence)
brier_skill = 1.0 - (np.mean(fold_briers) / brier_null)

clinical_capture_df = pd.DataFrame([
    compute_clinical_metrics(y_cv, oof_preds_default, "Default Classifier Cutoff", "50.0% (Fixed)"),
    compute_clinical_metrics(
        y_cv,
        oof_preds_tuned,
        "Leakage-Free Nested-CV Screening Cutoff",
        f"{np.mean(fold_opt_thresholds)*100:.1f}% ± {np.std(fold_opt_thresholds, ddof=1)*100:.1f}% (Inner 4-Fold Youden)",
    ),
])

print("\n--- EARLY WARNING AI PERFORMANCE (Pre-Malignant Cohort Only) ---")
print(cv_stability_results)
print(f"Baseline (Null) Brier Score: {brier_null:.4f} | Brier Skill Score: {brier_skill:.4f}")
print("\n--- LEAKAGE-FREE CLINICAL CAPTURE COMPARISON (Severe Dysplasia Detection) ---")
print(clinical_capture_df.to_string(index=False))

# --- REPEATED STRATIFIED CV (10x5 = 50 Folds: Split & Seed Stability Check) ---
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
X_rep = X_raw_cv.copy()
X_rep['IL-8_log'] = np.log1p(X_rep['IL-8'])
X_rep['sPD-L1_log'] = np.log1p(X_rep['sPD-L1'])
X_rep = X_rep[['Age', 'BMI', 'Sex_Binary', 'IL-8_log', 'sPD-L1_log']]

rep_aucs, rep_pr_aucs, rep_briers, rep_accs = [], [], [], []
for split_i, (tr_idx, te_idx) in enumerate(rskf.split(X_rep, y_cv)):
    rf_rep = RandomForestClassifier(
        n_estimators=100, max_depth=4, class_weight='balanced', random_state=42 + split_i
    )
    rf_rep.fit(X_rep.iloc[tr_idx], y_cv[tr_idx])
    p_rep = rf_rep.predict_proba(X_rep.iloc[te_idx])[:, 1]
    y_hat_rep = rf_rep.predict(X_rep.iloc[te_idx])

    rep_aucs.append(roc_auc_score(y_cv[te_idx], p_rep))
    rep_pr_aucs.append(average_precision_score(y_cv[te_idx], p_rep))
    rep_briers.append(brier_score_loss(y_cv[te_idx], p_rep))
    rep_accs.append(accuracy_score(y_cv[te_idx], y_hat_rep))

# Reshape to (10 repeats, 5 folds) to get true repeat-level stability
def summarize_repeats(vals):
    repeat_means = np.array(vals).reshape(10, 5).mean(axis=1)
    return repeat_means.mean(), repeat_means.std(ddof=1), np.percentile(vals, 2.5), np.percentile(vals, 97.5)

print("\n--- 10x5 REPEATED STRATIFIED CV STABILITY (50 Folds Total) ---")
for name, vals in [
    ("ROC-AUC", rep_aucs),
    ("PR-AUC", rep_pr_aucs),
    ("Brier Score", rep_briers),
    ("Accuracy", rep_accs),
]:
    mean_v, sd_rep, p025, p975 = summarize_repeats(vals)
    print(f"  {name:12s}: Mean = {mean_v:.4f} ± {sd_rep:.4f} (SD across 10 repeats) | 50-fold range [2.5-97.5%]: [{p025:.3f}, {p975:.3f}]")

# --- Q1 LINEAR BASELINE BENCHMARK (L2 Logistic Regression) ---
lr_pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),
])
scoring = {'auc': 'roc_auc', 'pr': 'average_precision', 'brier': 'neg_brier_score'}

lr_5 = cross_validate(lr_pipe, X_rep, y_cv, cv=skf, scoring=scoring)
lr_50 = cross_validate(lr_pipe, X_rep, y_cv, cv=rskf, scoring=scoring)

rf_auc_m, rf_auc_sd, _, _ = summarize_repeats(rep_aucs)
rf_pr_m, rf_pr_sd, _, _ = summarize_repeats(rep_pr_aucs)
lr_auc_m, lr_auc_sd, _, _ = summarize_repeats(lr_50['test_auc'])
lr_pr_m, lr_pr_sd, _, _ = summarize_repeats(lr_50['test_pr'])

model_benchmark_df = pd.DataFrame([
    {
        'Model': 'Random Forest (Early Warning AI)',
        'CV5_AUC': round(float(np.mean(fold_aucs)), 4),
        'CV5_PR': round(float(np.mean(fold_pr_aucs)), 4),
        'CV5_Brier': round(float(np.mean(fold_briers)), 4),
        'Rep10x5_AUC': f"{rf_auc_m:.4f} ± {rf_auc_sd:.4f}",
        'Rep10x5_PR': f"{rf_pr_m:.4f} ± {rf_pr_sd:.4f}",
    },
    {
        'Model': 'L2 Logistic Regression (Baseline)',
        'CV5_AUC': round(float(lr_5['test_auc'].mean()), 4),
        'CV5_PR': round(float(lr_5['test_pr'].mean()), 4),
        'CV5_Brier': round(float(-lr_5['test_brier'].mean()), 4),
        'Rep10x5_AUC': f"{lr_auc_m:.4f} ± {lr_auc_sd:.4f}",
        'Rep10x5_PR': f"{lr_pr_m:.4f} ± {lr_pr_sd:.4f}",
    },
])
print("\n--- Q1 LINEAR BASELINE BENCHMARK (RF vs. L2 Logistic Regression) ---")
print(model_benchmark_df.to_string(index=False))

# Generate Calibration Curve (explicit axes avoids stray empty figure & title clipping)
fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
CalibrationDisplay.from_predictions(
    all_y_true_cv, all_y_probs_cv, n_bins=5, strategy='uniform', ax=ax_cal
)
ax_cal.set_title("Early Warning AI: Calibration Curve", pad=12)
fig_cal.tight_layout()
fig_cal.savefig(USER_PATH / 'calibration_curve.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig_cal)

# Define X_cv on the ML COHORT so the SHAP visualizer has the correct shape
ml_cohort['IL-8_log'] = np.log1p(ml_cohort['IL-8'])
ml_cohort['sPD-L1_log'] = np.log1p(ml_cohort['sPD-L1'])
X_cv = ml_cohort[['Age', 'BMI', 'Sex_Binary', 'IL-8_log', 'sPD-L1_log']]
clean_shap_names = ['Age', 'BMI', 'Sex', 'ln(IL-8 + 1)', 'ln(sPD-L1 + 1)']

# 1. Save Static SHAP Summary (publication-ready labels)
plt.figure(figsize=(10, 6))
shap.summary_plot(cv_shap_values_global, X_cv, feature_names=clean_shap_names, show=False)
plt.title("Early Warning AI: SHAP Feature Importance")
plt.savefig(USER_PATH / 'cv_stable_shap_summary.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

# --- SHAP STABILITY CHECK (True Standard Error Across the 5 CV Folds) ---
fold_shap_means = np.array(fold_shap_means)  # shape: (5 folds, 5 features)
cv_shap_mean_impact = np.abs(cv_shap_values_global).mean(axis=0)
cv_shap_std_error = fold_shap_means.std(axis=0, ddof=1) / np.sqrt(len(fold_shap_means))

df_cv_shap_sum = pd.DataFrame({
    'Feature': X_cv.columns,
    'Mean_Impact': cv_shap_mean_impact,
    'Std_Error_5Fold': cv_shap_std_error
}).sort_values(by='Mean_Impact', ascending=True)

print("\n--- EARLY WARNING AI: SHAP FEATURE IMPORTANCE ---")
print(df_cv_shap_sum.sort_values(by='Mean_Impact', ascending=False).to_string(index=False))


# 2. Interactive Bar Chart
fig_cv_shap_bar = px.bar(df_cv_shap_sum, x='Mean_Impact', y='Feature', orientation='h',
                         title="Early Warning AI: Cross-Validated Feature Importance",
                         color='Mean_Impact', color_continuous_scale='Viridis')
fig_cv_shap_bar.write_html(USER_PATH / 'cv_stable_shap_summary_interactive.html')

# 3. Interactive Beeswarm Plot
X_cv_scaled = pd.DataFrame(StandardScaler().fit_transform(X_cv), columns=X_cv.columns)

df_cv_shap_melt = pd.DataFrame(cv_shap_values_global, columns=X_cv.columns).melt(var_name='Feature', value_name='SHAP_Value')
df_cv_val_melt = X_cv_scaled.melt(var_name='Feature', value_name='Feature_Value')

df_cv_shap_melt['Feature_Value'] = df_cv_val_melt['Feature_Value']
df_cv_shap_melt['Patient_Name'] = ml_cohort['Patient_Name'].tolist() * len(X_cv.columns)

fig_cv_beeswarm = px.scatter(df_cv_shap_melt, x='SHAP_Value', y='Feature', color='Feature_Value',
                             hover_name='Patient_Name', title="Early Warning AI: Interactive SHAP Beeswarm",
                             color_continuous_scale='Bluered')

fig_cv_beeswarm.update_traces(opacity=0.8, marker=dict(size=7, line=dict(width=0.5, color='DarkSlateGrey')))
fig_cv_beeswarm.write_html(USER_PATH / 'cv_stable_shap_beeswarm_interactive.html')

# ==============================================================================
# --- 8. ALL VISUALIZATIONS (CLEANED & MATHEMATICALLY HONEST) ---
# ==============================================================================
print("\n--- 8. GENERATING VISUALIZATIONS ---")
color_map = {'No Dysplasia': 'green', 'Mild Dysplasia': 'gold', 'Mod Dysplasia': 'orange', 'Sev Dysplasia': 'red', 'OSCC': 'darkred'}

# --- Safely Calculate PCA if missing ---
if 'PC1' not in master_both.columns:
    scaler = StandardScaler()
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaler.fit_transform(master_both[['IL-8_adj', 'sPD-L1_adj']]))
    master_both['PC1'] = pca_result[:, 0]
    master_both['PC2'] = pca_result[:, 1]

# 1. PCA Clustering (Proves chaos/independence in early stages)
pc1_label = f"PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% Variance)"
pc2_label = f"PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% Variance)"

plt.figure(figsize=(10, 6))
sns.scatterplot(
    x='PC1', y='PC2', hue='Group', hue_order=group_order,
    data=master_both, palette=color_map, s=100, alpha=0.8, edgecolor='w', zorder=2
)
plt.title('PCA Clustering of Immune Microenvironment')
plt.xlabel(pc1_label)
plt.ylabel(pc2_label)
plt.legend(title='Histopathological Stage', bbox_to_anchor=(1.02, 1), loc='upper left')
plt.savefig(USER_PATH / 'adv1_pca_clustering.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

fig_pca = px.scatter(
    master_both, x='PC1', y='PC2', color='Group', hover_name='Patient_Name',
    category_orders={'Group': group_order}, color_discrete_map=color_map,
    labels={'PC1': pc1_label, 'PC2': pc2_label},
    title='Interactive PCA Clustering'
)
fig_pca.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
fig_pca.write_html(USER_PATH / 'adv1_pca_clustering_interactive.html')

# 2. IEI Log-Ratio (Proves hot/cold ratio shift is false)
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='IEI_Log_Ratio', data=master_both, order=group_order, palette=color_map)
plt.title('Immune Evasion-to-Inflammation Log-Ratio (IEI)')
plt.xlabel('Histopathological Stage')
plt.ylabel(r'IEI Log-Ratio (sPD-L1$_{\mathrm{adj}}$ - IL-8$_{\mathrm{adj}}$)')
plt.savefig(USER_PATH / 'adv2_iei_ratio_boxplot.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

px.box(master_both, x='Group', y='IEI_Log_Ratio', color='Group', hover_name='Patient_Name',
       category_orders={'Group': group_order}, color_discrete_map=color_map).write_html(
    USER_PATH / 'adv2_iei_ratio_interactive.html')

# 3. The "Spike and Crash" Trajectory (Connecting the Stage Means)
plt.figure(figsize=(10, 6))

# Plot the individual patient scatter points (Stripplot)
sns.stripplot(x='Stage_Ordinal', y='i_IPS', data=master_both, hue='Group',
              palette=color_map, alpha=0.5, jitter=True)

# Plot the trajectory of the MEANS to reveal the Spike and Crash
sns.pointplot(x='Stage_Ordinal', y='i_IPS', data=master_both, color='black',
              markers='D', linestyles='-', errorbar=None, label='Mean Trajectory')

plt.xticks(ticks=[0, 1, 2, 3, 4], labels=group_order)
plt.title('Tumor Microenvironment Trajectory: The Spike and Crash Anomaly')
plt.ylabel('i-IPS Score')
plt.xlabel('Histopathological Stage')


handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[:5] + [handles[-1]], labels[:5] + ['Mean Trajectory'],
           title="Disease Stage", bbox_to_anchor=(1.05, 1), loc='upper left')

plt.savefig(USER_PATH / 'adv3_spike_and_crash_trajectory.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

# --- INTERACTIVE VERSION ---
# Calculate means for the plotly line
stage_means = master_both.groupby('Stage_Ordinal')['i_IPS'].mean().reset_index()

fig_traj = px.scatter(master_both, x='Stage_Ordinal', y='i_IPS', color='Group', hover_name='Patient_Name',
                      color_discrete_map=color_map)


fig_traj.add_trace(go.Scatter(x=stage_means['Stage_Ordinal'], y=stage_means['i_IPS'],
                              mode='lines+markers', name='Mean Trajectory',
                              line=dict(color='black', width=4),
                              marker=dict(size=10, symbol='diamond', color='black')))

fig_traj.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
fig_traj.update_layout(title="Interactive Trajectory: The Spike and Crash Anomaly",
                       xaxis=dict(tickvals=[0, 1, 2, 3, 4], ticktext=group_order),
                       xaxis_title="Histopathological Stage", yaxis_title="i-IPS Score")
fig_traj.write_html(USER_PATH / 'adv3_spike_and_crash_trajectory_interactive.html')

# 4. i-IPS Progression Boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Group', y='i_IPS', data=master_both, order=group_order, palette=color_map)
plt.title('i-IPS Progression Mapping')
plt.xlabel('Histopathological Stage')
plt.ylabel('i-IPS Synergy Score')
plt.savefig(USER_PATH / 'orig1_progression_boxplot.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

px.box(master_both, x='Group', y='i_IPS', color='Group', hover_name='Patient_Name',
       category_orders={'Group': group_order}, color_discrete_map=color_map).write_html(
    USER_PATH / 'orig1_progression_boxplot_interactive.html')

# 5. ROC Curve (Proves linear thresholds fail)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'i-IPS (AUC = {auc_val:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Chance (AUC = 0.500)')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.title('ROC Curve Analysis (Leakage-Free 5-Fold OOF i-IPS)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.4)
plt.savefig(USER_PATH / 'orig2_roc_curve.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

fig_roc = go.Figure()
fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'i-IPS (AUC={auc_val:.2f})', line=dict(color='darkorange', width=4)))
fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Chance', line=dict(dash='dash', color='navy')))
fig_roc.write_html(USER_PATH / 'orig2_roc_curve_interactive.html')

# 6. 3D Synergy Landscape Plot (orig3)
# Static Matplotlib 3D Plot
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

for group, color in color_map.items():
    subset = master_both[master_both['Group'] == group]
    ax.scatter(
        subset['IL-8_adj'],
        subset['sPD-L1_adj'],
        subset['i_IPS'],
        c=color,
        label=group,
        s=60,
        edgecolors='black',
        linewidths=0.5,
        alpha=0.85
    )

ax.set_xlabel('Adjusted IL-8')
ax.set_ylabel('Adjusted sPD-L1')
ax.set_zlabel('Synergy Score (i-IPS)')
ax.set_title('3D Synergy Landscape')
ax.view_init(elev=20, azim=45)
ax.legend(loc='upper right', title="Diagnosis")

plt.savefig(USER_PATH / 'orig3_synergy_3d.png', dpi=300, bbox_inches='tight', transparent=False, facecolor='white')
plt.close()

# Interactive Plotly 3D Plot
fig_3d = px.scatter_3d(master_both,
                       x='IL-8_adj',
                       y='sPD-L1_adj',
                       z='i_IPS',
                       color='Group',
                       color_discrete_map=color_map,
                       hover_name='Patient_Name',
                       title="Interactive 3D Synergy Landscape")

fig_3d.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
fig_3d.write_html(USER_PATH / 'orig3_synergy_3d_interactive.html')


# ==============================================================================
# --- 9. CROSS-PLATFORM TRANSCRIPTOMIC VALIDATION (GSE30784) ---
# ==============================================================================
print("\n" + "=" * 60)
print("--- 9. CROSS-PLATFORM TRANSCRIPTOMIC VALIDATION (GSE30784) ---")
print("NOTE: Validating directional trends of serum proteins against tissue mRNA (CD274/CXCL8).")

group_order_geo = ['Normal', 'Dysplasia', 'OSCC']
color_map_geo = {'Normal': 'green', 'Dysplasia': 'orange', 'OSCC': 'darkred'}

geo1_path = USER_PATH / "GSE30784_series_matrix.txt"
geo1_df = pd.DataFrame()

if geo1_path.exists():
    try:
        print(f" Processing GSE30784 (Affymetrix)...")
        statuses1, il8_data1, pdl1_data1, sample_ids1 = [], [], [], []
        with open(geo1_path, 'r') as f:
            for line in f:
                if line.startswith('!Sample_title'):
                    sample_ids1 = [s.strip().replace('"', '') for s in line.strip().split('\t')[1:]]
                elif line.startswith('!Sample_characteristics_ch1') and 'status:' in line:
                    statuses1 = [s.strip().replace('"', '').replace('status: ', '') for s in
                                 line.strip().split('\t')[1:]]
                elif line.startswith('"202859_x_at"'):
                    il8_data1 = [float(x) for x in line.strip().split('\t')[1:]]
                elif line.startswith('"227458_at"'):
                    pdl1_data1 = [float(x) for x in line.strip().split('\t')[1:]]

        if il8_data1 and pdl1_data1:
            geo1_df = pd.DataFrame({'Sample_ID': sample_ids1, 'Diagnosis_Raw': statuses1, 'IL-8_mRNA': il8_data1,
                                    'sPD-L1_mRNA': pdl1_data1})
            geo1_df['Group'] = geo1_df['Diagnosis_Raw'].map(
                {'control': 'Normal', 'dysplasia': 'Dysplasia', 'cancer': 'OSCC'})
            geo1_df = geo1_df[geo1_df['Group'].isin(group_order_geo)]

            geo1_df['g_IPS'] = geo1_df['IL-8_mRNA'] + geo1_df['sPD-L1_mRNA']
            geo1_df['Stage_Ordinal'] = geo1_df['Group'].map({g: i for i, g in enumerate(group_order_geo)})

            gips_groups = [geo1_df[geo1_df['Group'] == g]['g_IPS'] for g in group_order_geo if
                           len(geo1_df[geo1_df['Group'] == g]) >= 2]
            print(f" GSE30784 Cohort Size: N={len(geo1_df)}")
            print(geo1_df['Group'].value_counts())
            h_stat_gips, p_val_gips = kruskal(*gips_groups)
            print(f" GSE30784 Genomic Synergy (g-IPS) Kruskal-Wallis: H={h_stat_gips:.4f}, p={p_val_gips:.4e}")

            print("\n--- GSE30784 DUNN'S POST-HOC (Proving the OSCC Cliff) ---")
            geo_dunn = sp.posthoc_dunn(geo1_df, val_col='g_IPS', group_col='Group', p_adjust='fdr_bh')
            print(geo_dunn)

            # --- TRUE ITERATIVE PIECEWISE FOR GLOBAL DATA ---
            x_geo, y_geo = geo1_df['Stage_Ordinal'].values, geo1_df['g_IPS'].values
            pwlf_geo = pwlf.PiecewiseLinFit(x_geo, y_geo)
            n_geo = len(y_geo)

            # 1. Global Linear BIC
            pwlf_geo.fit(1)
            bic_1_geo = n_geo * np.log(pwlf_geo.ssr / n_geo) + 2 * np.log(n_geo)

            # 2. Global Biphasic BIC
            breaks_geo = pwlf_geo.fit(2, seed=42)
            bic_2_geo = n_geo * np.log(pwlf_geo.ssr / n_geo) + 4 * np.log(n_geo)

            print("\n--- GLOBAL PIECEWISE REGRESSION VALIDATION (BIC) ---")
            print(f"BIC (Linear / 1-Segment): {bic_1_geo:.2f}")
            print(f"BIC (Biphasic / 2-Segment): {bic_2_geo:.2f}")

            optimal_knot_geo = breaks_geo[1]
            print(f" -> Global GSE30784 optimal knot heuristically discovered at Pseudo-Time: {optimal_knot_geo:.4f}")

            # --- GEO VISUALIZATIONS (Mathematically Sound) ---
            # 1. Global Boxplot
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Group', y='g_IPS', data=geo1_df, order=group_order_geo, palette=color_map_geo)
            plt.title('In-Silico Validation (GSE30784): Genomic Synergy (g-IPS)')
            plt.xlabel('Histopathological Stage')
            plt.ylabel('g-IPS Score (Tissue mRNA)')
            plt.savefig(USER_PATH / 'geo1_GSE30784_synergy_boxplot.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.close()

            px.box(geo1_df, x='Group', y='g_IPS', color='Group', hover_name='Sample_ID',
                   category_orders={'Group': group_order_geo}, color_discrete_map=color_map_geo,
                   title="Interactive GSE30784 Synergy Validation").write_html(
                USER_PATH / 'geo1_GSE30784_synergy_interactive.html')

            # 2. Global Tissue Trajectory (Connecting the Means)
            plt.figure(figsize=(10, 6))

            # Plot the individual patient scatter points (Stripplot)
            sns.stripplot(x='Stage_Ordinal', y='g_IPS', data=geo1_df, hue='Group',
                          palette=color_map_geo, alpha=0.5, jitter=True)

            # Plot the trajectory of the MEANS to reveal the pure upward trend
            sns.pointplot(x='Stage_Ordinal', y='g_IPS', data=geo1_df, color='black',
                          markers='D', linestyles='-', errorbar=None, label='Mean Trajectory')

            plt.xticks(ticks=[0, 1, 2], labels=group_order_geo)
            plt.title('Global Tissue Trajectory (GSE30784): Linear Escalation (No Crash)')
            plt.ylabel('g-IPS Score (Tissue mRNA)')
            plt.xlabel('Histopathological Stage')


            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[:3] + [handles[-1]], labels[:3] + ['Mean Trajectory'],
                       title="Disease Stage", bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.savefig(USER_PATH / 'geo2_GSE30784_trajectory.png', dpi=300, bbox_inches='tight', transparent=True)
            plt.close()

            # --- INTERACTIVE GLOBAL TRAJECTORY ---
            stage_means_geo = geo1_df.groupby('Stage_Ordinal')['g_IPS'].mean().reset_index()

            fig_geo_traj = px.scatter(geo1_df, x='Stage_Ordinal', y='g_IPS', color='Group',
                                      hover_name='Sample_ID', color_discrete_map=color_map_geo)

            fig_geo_traj.add_trace(go.Scatter(x=stage_means_geo['Stage_Ordinal'], y=stage_means_geo['g_IPS'],
                                              mode='lines+markers', name='Mean Trajectory',
                                              line=dict(color='black', width=4),
                                              marker=dict(size=10, symbol='diamond', color='black')))

            fig_geo_traj.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')),
                                       selector=dict(mode='markers'))
            fig_geo_traj.update_layout(title="Interactive GSE30784 Trajectory: Tissue Escalation",
                                       xaxis=dict(tickvals=[0, 1, 2], ticktext=group_order_geo),
                                       xaxis_title="Histopathological Stage", yaxis_title="g-IPS Score (Tissue mRNA)")
            fig_geo_traj.write_html(USER_PATH / 'geo2_GSE30784_trajectory_interactive.html')

    except Exception as e:
        print(f" Error processing GSE30784: {e}")
else:
    print(f" GSE30784 not found. Skipping.")

# ==============================================================================
# --- 10. FINAL MULTI-COHORT EXCEL EXPORT ---
# ==============================================================================
print("\n--- 10. SAVING MULTI-COHORT MASTER EXCEL FILE ---")

meta_info = pd.DataFrame({
    'Parameter': ['Date of Analysis', 'Total Valid Samples', 'Features Analysed', 'Software Seed'],
    'Value': [pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'), len(master_both), master_both.shape[1], 42]
})

export_clinical = master_both.drop(columns=['Sex_Binary', 'is_HighRisk', 'is_ImmunePeak', 'raw_i_IPS'], errors='ignore')
final_excel_path = USER_PATH / 'processed_synergy_data_final.xlsx'

try:
    with pd.ExcelWriter(final_excel_path, engine='openpyxl') as writer:
        meta_info.to_excel(writer, sheet_name='Metadata', index=False)
        export_clinical.to_excel(writer, sheet_name='Local_Clinical_Proteins', index=False)

        if tukey_df is not None:
            tukey_df.to_excel(writer, sheet_name='Local_Tukey_HSD', index=False)

        if dunn_df is not None:
            dunn_df.to_excel(writer, sheet_name='Local_Dunns_P_Values')

        cv_stability_results.to_excel(writer, sheet_name='CV_Stability_Metrics', index=False)
        clinical_capture_df.to_excel(writer, sheet_name='AI_Clinical_Capture_Metrics', index=False)

        model_benchmark_df.to_excel(writer, sheet_name='Model_Benchmark_RF_vs_LR', index=False)

        pd.DataFrame({
            'Fold': range(1, 6),
            'AUC': fold_aucs,
            'PR_AUC': fold_pr_aucs,
            'Brier_Score': fold_briers,
            'Accuracy_Default50': fold_accs,
            'Nested_Tuned_Cutoff_Pct': np.round(np.array(fold_opt_thresholds) * 100, 2),
            'Sens_Default50_Pct': np.round(np.array(fold_sens_def) * 100, 2),
            'Spec_Default50_Pct': np.round(np.array(fold_spec_def) * 100, 2),
            'Sens_Tuned_Pct': np.round(np.array(fold_sens_tuned) * 100, 2),
            'Spec_Tuned_Pct': np.round(np.array(fold_spec_tuned) * 100, 2),
        }).to_excel(writer, sheet_name='Individual_Fold_Data', index=False)

        if not geo1_df.empty:
            geo1_df.drop(columns=['Knot_Term'], errors='ignore').to_excel(
                writer, sheet_name='Global_GEO_mRNA', index=False
            )
            if geo_dunn is not None:
                geo_dunn.to_excel(writer, sheet_name='Global_GEO_Dunns_P_Values')

        print(f" SUCCESS! All Data, Post-Hoc matrices, and CV Metrics embedded into: {final_excel_path.name}")
        print(f" Saved to folder: {USER_PATH}")

except PermissionError:
    print(f"\n ERROR: Permission Denied!")
    print(f" Please CLOSE the Excel file '{final_excel_path.name}' and run the script again.")