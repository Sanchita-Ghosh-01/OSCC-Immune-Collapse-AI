import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    accuracy_score,
    brier_score_loss,
    average_precision_score,
    confusion_matrix,
)

import shap

# Set global seed for reproducibility
np.random.seed(42)
warnings.simplefilter(action='ignore', category=FutureWarning)


# --- 1. CONFIGURATION ---
USER_PATH = Path(os.environ.get("OSCC_DATA_DIR", Path(__file__).resolve().parent))

DATA_FILE = USER_PATH / "clean-dataset.xlsx"
if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Could not find '{DATA_FILE}'. "
        "Place 'clean-dataset.xlsx' in the same folder as this script or set OSCC_DATA_DIR."
    )


# --- 2. HELPER FUNCTIONS (Identical to Main Pipeline) ---
def clean_numeric(val):
    if pd.isna(val) or val == '':
        return np.nan
    try:
        s = (
            str(val)
            .lower()
            .replace('years', '')
            .replace('yr', '')
            .replace('kg', '')
            .replace(',', '.')
            .strip()
        )
        # Safely keep numbers, decimals, negative signs, scientific 'e', and '<' signs
        s = re.sub(r'[^\d.\-eE<]', '', s)
        if '<' in s:
            return float(s.replace('<', '')) / 2.0  # Impute below limit-of-detection
        return float(s)
    except Exception:
        return np.nan


def categorize_diagnosis(diag):
    if pd.isna(diag):
        return 'No Dysplasia'
    d = str(diag).lower()
    if any(kw in d for kw in ['oscc', 'malignant', 'cancer']):
        return 'OSCC'
    if any(kw in d for kw in ['severe', 'high grade']):
        return 'Sev Dysplasia'
    if 'moderate' in d:
        return 'Mod Dysplasia'
    if any(kw in d for kw in ['mild', 'olkp', 'low grade']):
        return 'Mild Dysplasia'
    return 'No Dysplasia'


def encode_sex(val):
    s = str(val).strip().upper()
    if s in ['M', 'MALE']:
        return 1
    if s in ['F', 'FEMALE']:
        return 0
    return np.nan


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


# --- 3. DATA LOADING & COHORT FILTERING ---
print(f"Loading Excel file from: {DATA_FILE}...")
df1 = pd.read_excel(DATA_FILE)

master = pd.DataFrame({
    'Patient_Name': df1['Sample No.'],
    'Sex': df1['Sex'],
    'Age': df1['Age'].apply(clean_numeric),
    'BMI': df1['BMI'].apply(clean_numeric),
    'IL-8': df1['IL-8 Mean'].apply(clean_numeric),
    'sPD-L1': df1['sPD-L1 Mean'].apply(clean_numeric),
    'Diagnosis': df1['Diagnosis'],
})
master['Group'] = master['Diagnosis'].apply(categorize_diagnosis)

print(f"Original Cohort Size: {len(master)}")
master_both = master.dropna(subset=['Age', 'BMI', 'IL-8', 'sPD-L1', 'Sex']).copy()
print(f"Cohort Size after dropping missing demographics/biomarkers: {len(master_both)}")

master_both = master_both[(master_both['IL-8'] > 0) & (master_both['sPD-L1'] > 0)].copy()
master_both['Sex_Binary'] = master_both['Sex'].apply(encode_sex)
master_both = master_both.dropna(subset=['Sex_Binary']).copy()
master_both['Sex_Binary'] = master_both['Sex_Binary'].astype(int)
print(f"Final Analytical Cohort Size (Positive biomarkers & valid Sex): {len(master_both)}")


# --- 4. EARLY WARNING AI MODEL (LEAKAGE-FREE 5-FOLD STRATIFIED CV + SHAP) ---
print("\n" + "=" * 60)
print("--- VALIDATING THE EARLY WARNING AI (Predicting the Immune Peak) ---")
print("NOTE: OSCC patients are excluded here. The AI is trained exclusively on pre-malignant")
print("patients to detect the Severe Dysplasia 'Spike' before invasion occurs.")

# Isolate the Pre-Malignant Cohort (N=65: 11 Severe Dysplasia vs 54 Non-Severe)
ml_cohort = master_both[master_both['Group'] != 'OSCC'].copy()
ml_cohort['is_ImmunePeak'] = ml_cohort['Group'].isin(['Sev Dysplasia']).astype(int)

X_raw_cv = ml_cohort[['Age', 'BMI', 'Sex_Binary', 'IL-8', 'sPD-L1']]
y_cv = ml_cohort['is_ImmunePeak'].values
print(
    f"Pre-Malignant ML Cohort: N={len(ml_cohort)} "
    f"(Severe Dysplasia={int(y_cv.sum())}, Non-Severe={int((y_cv == 0).sum())})"
)

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

feature_cols = ['Age', 'BMI', 'Sex_Binary', 'IL-8_log', 'sPD-L1_log']

for fold, (train_idx, test_idx) in enumerate(skf.split(X_raw_cv, y_cv), 1):
    X_train_raw, X_test_raw = X_raw_cv.iloc[train_idx].copy(), X_raw_cv.iloc[test_idx].copy()
    y_train, y_test = y_cv[train_idx], y_cv[test_idx]

    # Log-transform biomarkers alongside clinical covariates (row-wise)
    X_train_adj = X_train_raw.copy()
    X_test_adj = X_test_raw.copy()

    X_train_adj['IL-8_log'] = np.log1p(X_train_adj['IL-8'])
    X_train_adj['sPD-L1_log'] = np.log1p(X_train_adj['sPD-L1'])
    X_test_adj['IL-8_log'] = np.log1p(X_test_adj['IL-8'])
    X_test_adj['sPD-L1_log'] = np.log1p(X_test_adj['sPD-L1'])

    X_train_adj = X_train_adj[feature_cols]
    X_test_adj = X_test_adj[feature_cols]

    # 1. Train Random Forest Classifier (on training fold)
    rf_fold = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        class_weight='balanced',
        random_state=42,
    )
    rf_fold.fit(X_train_adj, y_train)

    probs = rf_fold.predict_proba(X_test_adj)[:, 1]
    preds = rf_fold.predict(X_test_adj)

    # 2. Leakage-Free Nested Inner 4-Fold CV Threshold Tuning (inside training fold)
    inner_skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    inner_oof_probs = np.zeros(len(y_train))
    for in_tr, in_val in inner_skf.split(X_train_adj, y_train):
        rf_inner = RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            class_weight='balanced',
            random_state=42,
        )
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

    # 4. Extract Exact Out-of-Fold TreeSHAP Values
    explainer_fold = shap.TreeExplainer(rf_fold)
    shap_f = explainer_fold.shap_values(X_test_adj)
    if isinstance(shap_f, list):
        shap_f = shap_f[1]
    elif len(shap_f.shape) == 3:
        shap_f = shap_f[:, :, 1]

    cv_shap_values_global[test_idx, :] = shap_f
    fold_shap_means.append(np.abs(shap_f).mean(axis=0))

    print(
        f"  Fold {fold} Complete: Peak-Detection AUC = {fold_aucs[-1]:.3f} "
        f"| Nested Tuned Cutoff = {fold_opt_thresh*100:.1f}%"
    )

cv_stability_results = pd.DataFrame({
    'Metric': ['Mean ROC-AUC', 'Mean PR-AUC (Imbalance-Robust)', 'Mean Brier Score', 'Mean Accuracy'],
    'Value': [np.mean(fold_aucs), np.mean(fold_pr_aucs), np.mean(fold_briers), np.mean(fold_accs)],
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


# --- 5. REPEATED STRATIFIED CV (10x5 = 50 FOLDS STABILITY CHECK) ---
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)
X_rep = X_raw_cv.copy()
X_rep['IL-8_log'] = np.log1p(X_rep['IL-8'])
X_rep['sPD-L1_log'] = np.log1p(X_rep['sPD-L1'])
X_rep = X_rep[feature_cols]

rep_aucs, rep_pr_aucs, rep_briers, rep_accs = [], [], [], []
for split_i, (tr_idx, te_idx) in enumerate(rskf.split(X_rep, y_cv)):
    rf_rep = RandomForestClassifier(
        n_estimators=100,
        max_depth=4,
        class_weight='balanced',
        random_state=42 + split_i,
    )
    rf_rep.fit(X_rep.iloc[tr_idx], y_cv[tr_idx])
    p_rep = rf_rep.predict_proba(X_rep.iloc[te_idx])[:, 1]
    y_hat_rep = rf_rep.predict(X_rep.iloc[te_idx])

    rep_aucs.append(roc_auc_score(y_cv[te_idx], p_rep))
    rep_pr_aucs.append(average_precision_score(y_cv[te_idx], p_rep))
    rep_briers.append(brier_score_loss(y_cv[te_idx], p_rep))
    rep_accs.append(accuracy_score(y_cv[te_idx], y_hat_rep))


def summarize_repeats(vals):
    repeat_means = np.array(vals).reshape(10, 5).mean(axis=1)
    return repeat_means.mean(), repeat_means.std(ddof=1), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


print("\n--- 10x5 REPEATED STRATIFIED CV STABILITY (50 Folds Total) ---")
repeated_rows = []
for name, vals in [
    ("ROC-AUC", rep_aucs),
    ("PR-AUC", rep_pr_aucs),
    ("Brier Score", rep_briers),
    ("Accuracy", rep_accs),
]:
    mean_v, sd_rep, p025, p975 = summarize_repeats(vals)
    repeated_rows.append({
        'Metric': name,
        'Mean_10Repeats': mean_v,
        'SD_10Repeats': sd_rep,
        'Percentile_2.5': p025,
        'Percentile_97.5': p975,
    })
    print(
        f"  {name:12s}: Mean = {mean_v:.4f} ± {sd_rep:.4f} (SD across 10 repeats) "
        f"| 50-fold range [2.5-97.5%]: [{p025:.3f}, {p975:.3f}]"
    )
repeated_cv_df = pd.DataFrame(repeated_rows)

# --- 5b. LINEAR BASELINE COMPARISON (Leakage-Free L2 Logistic Regression) ---
# Evaluated on the exact same 5-fold and 10x5 splits; StandardScaler is fit inside each training fold via Pipeline.
print("\n--- 5b. LINEAR BASELINE BENCHMARK (L2 Logistic Regression vs. Random Forest) ---")

lr_5fold_aucs, lr_5fold_pr_aucs, lr_5fold_briers, lr_5fold_accs = [], [], [], []
lr_oof_preds_tuned = np.zeros(len(y_cv), dtype=int)
lr_fold_cutoffs = []

for tr_idx, te_idx in skf.split(X_rep, y_cv):
    X_tr, X_te = X_rep.iloc[tr_idx], X_rep.iloc[te_idx]
    y_tr, y_te = y_cv[tr_idx], y_cv[te_idx]

    lr_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),
    ])
    lr_pipe.fit(X_tr, y_tr)
    p_lr = lr_pipe.predict_proba(X_te)[:, 1]
    y_hat_lr = lr_pipe.predict(X_te)

    # 4-fold Youden cutoff strictly inside training fold
    inner_skf_lr = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
    in_oof_lr = np.zeros(len(y_tr))
    for in_tr, in_val in inner_skf_lr.split(X_tr, y_tr):
        pipe_in = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42)),
        ])
        pipe_in.fit(X_tr.iloc[in_tr], y_tr[in_tr])
        in_oof_lr[in_val] = pipe_in.predict_proba(X_tr.iloc[in_val])[:, 1]

    fpr_lr_in, tpr_lr_in, thr_lr_in = roc_curve(y_tr, in_oof_lr)
    valid_lr_in = np.isfinite(thr_lr_in)
    lr_cut = float(thr_lr_in[valid_lr_in][np.argmax((tpr_lr_in - fpr_lr_in)[valid_lr_in])])
    lr_fold_cutoffs.append(lr_cut)

    lr_5fold_aucs.append(roc_auc_score(y_te, p_lr))
    lr_5fold_pr_aucs.append(average_precision_score(y_te, p_lr))
    lr_5fold_briers.append(brier_score_loss(y_te, p_lr))
    lr_5fold_accs.append(accuracy_score(y_te, y_hat_lr))
    lr_oof_preds_tuned[te_idx] = (p_lr >= lr_cut).astype(int)

# 10x5 Repeated Stratified CV on the exact same 50 splits
lr_rep_aucs, lr_rep_pr_aucs = [], []
for split_i, (tr_idx, te_idx) in enumerate(rskf.split(X_rep, y_cv)):
    lr_rep = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42 + split_i)),
    ])
    lr_rep.fit(X_rep.iloc[tr_idx], y_cv[tr_idx])
    p_lr_rep = lr_rep.predict_proba(X_rep.iloc[te_idx])[:, 1]
    lr_rep_aucs.append(roc_auc_score(y_cv[te_idx], p_lr_rep))
    lr_rep_pr_aucs.append(average_precision_score(y_cv[te_idx], p_lr_rep))

rf_rep_auc_m, rf_rep_auc_sd, _, _ = summarize_repeats(rep_aucs)
rf_rep_pr_m, rf_rep_pr_sd, _, _ = summarize_repeats(rep_pr_aucs)
lr_rep_auc_m, lr_rep_auc_sd, _, _ = summarize_repeats(lr_rep_aucs)
lr_rep_pr_m, lr_rep_pr_sd, _, _ = summarize_repeats(lr_rep_pr_aucs)

rf_tuned_row = compute_clinical_metrics(y_cv, oof_preds_tuned, "Random Forest", "Nested")
lr_tuned_row = compute_clinical_metrics(y_cv, lr_oof_preds_tuned, "Logistic Regression", "Nested")

model_benchmark_df = pd.DataFrame([
    {
        'Model': 'Random Forest (Early Warning AI)',
        'CV5_ROC_AUC': round(float(np.mean(fold_aucs)), 4),
        'CV5_PR_AUC': round(float(np.mean(fold_pr_aucs)), 4),
        'CV5_Brier': round(float(np.mean(fold_briers)), 4),
        'Rep10x5_ROC_AUC': f"{rf_rep_auc_m:.4f} ± {rf_rep_auc_sd:.4f}",
        'Rep10x5_PR_AUC': f"{rf_rep_pr_m:.4f} ± {rf_rep_pr_sd:.4f}",
        'Nested_Sens_Pct': rf_tuned_row['Sensitivity_Capture_Pct'],
        'Nested_Spec_Pct': rf_tuned_row['Specificity_Pct'],
        'Nested_NPV_Pct': rf_tuned_row['NPV_Pct'],
    },
    {
        'Model': 'L2 Logistic Regression (Linear Baseline)',
        'CV5_ROC_AUC': round(float(np.mean(lr_5fold_aucs)), 4),
        'CV5_PR_AUC': round(float(np.mean(lr_5fold_pr_aucs)), 4),
        'CV5_Brier': round(float(np.mean(lr_5fold_briers)), 4),
        'Rep10x5_ROC_AUC': f"{lr_rep_auc_m:.4f} ± {lr_rep_auc_sd:.4f}",
        'Rep10x5_PR_AUC': f"{lr_rep_pr_m:.4f} ± {lr_rep_pr_sd:.4f}",
        'Nested_Sens_Pct': lr_tuned_row['Sensitivity_Capture_Pct'],
        'Nested_Spec_Pct': lr_tuned_row['Specificity_Pct'],
        'Nested_NPV_Pct': lr_tuned_row['NPV_Pct'],
    },
])
print(model_benchmark_df.to_string(index=False))

# --- 6. CALIBRATION & EXPLAINABLE AI (SHAP) OUTPUTS ---
# 1. Calibration Curve
fig_cal, ax_cal = plt.subplots(figsize=(7, 6))
CalibrationDisplay.from_predictions(
    all_y_true_cv, all_y_probs_cv, n_bins=5, strategy='uniform', ax=ax_cal
)
ax_cal.set_title("Early Warning AI: Calibration Curve", pad=12)
fig_cal.tight_layout()
fig_cal.savefig(USER_PATH / 'calibration_curve.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig_cal)

# Prepare SHAP feature matrix
ml_cohort['IL-8_log'] = np.log1p(ml_cohort['IL-8'])
ml_cohort['sPD-L1_log'] = np.log1p(ml_cohort['sPD-L1'])
X_cv = ml_cohort[feature_cols]
clean_shap_names = ['Age', 'BMI', 'Sex', 'ln(IL-8 + 1)', 'ln(sPD-L1 + 1)']

# 2. Static SHAP Summary Beeswarm Plot (Publication-Ready)
plt.figure(figsize=(10, 6))
shap.summary_plot(cv_shap_values_global, X_cv, feature_names=clean_shap_names, show=False)
plt.title("Early Warning AI: SHAP Feature Importance")
plt.savefig(USER_PATH / 'cv_stable_shap_summary.png', dpi=300, bbox_inches='tight', transparent=True)
plt.close()

# 3. SHAP Stability Table (True Standard Error Across the 5 CV Folds)
fold_shap_means = np.array(fold_shap_means)  # shape: (5 folds, 5 features)
cv_shap_mean_impact = np.abs(cv_shap_values_global).mean(axis=0)
cv_shap_std_error = fold_shap_means.std(axis=0, ddof=1) / np.sqrt(len(fold_shap_means))

df_cv_shap_sum = pd.DataFrame({
    'Feature': X_cv.columns,
    'Mean_Impact': cv_shap_mean_impact,
    'Std_Error_5Fold': cv_shap_std_error,
}).sort_values(by='Mean_Impact', ascending=True)

print("\n--- EARLY WARNING AI: SHAP FEATURE IMPORTANCE ---")
print(df_cv_shap_sum.sort_values(by='Mean_Impact', ascending=False).to_string(index=False))

# 4. Interactive SHAP Bar Chart
fig_cv_shap_bar = px.bar(
    df_cv_shap_sum,
    x='Mean_Impact',
    y='Feature',
    orientation='h',
    title="Early Warning AI: Cross-Validated Feature Importance",
    color='Mean_Impact',
    color_continuous_scale='Viridis',
)
fig_cv_shap_bar.write_html(USER_PATH / 'cv_stable_shap_summary_interactive.html')

# 5. Interactive SHAP Beeswarm Plot
X_cv_scaled = pd.DataFrame(StandardScaler().fit_transform(X_cv), columns=X_cv.columns)
df_cv_shap_melt = pd.DataFrame(cv_shap_values_global, columns=X_cv.columns).melt(
    var_name='Feature', value_name='SHAP_Value'
)
df_cv_val_melt = X_cv_scaled.melt(var_name='Feature', value_name='Feature_Value')

df_cv_shap_melt['Feature_Value'] = df_cv_val_melt['Feature_Value']
df_cv_shap_melt['Patient_Name'] = ml_cohort['Patient_Name'].tolist() * len(X_cv.columns)

fig_cv_beeswarm = px.scatter(
    df_cv_shap_melt,
    x='SHAP_Value',
    y='Feature',
    color='Feature_Value',
    hover_name='Patient_Name',
    title="Early Warning AI: Interactive SHAP Beeswarm",
    color_continuous_scale='Bluered',
)
fig_cv_beeswarm.update_traces(
    opacity=0.8, marker=dict(size=7, line=dict(width=0.5, color='DarkSlateGrey'))
)
fig_cv_beeswarm.write_html(USER_PATH / 'cv_stable_shap_beeswarm_interactive.html')


# --- 7. EXPORT STANDALONE AI REPORT TO EXCEL ---
oof_predictions_df = pd.DataFrame({
    'Patient_Name': ml_cohort['Patient_Name'].values,
    'Actual_Group': ml_cohort['Group'].values,
    'Actual_Severe_Dysplasia': y_cv,
    'Predicted_Risk_Probability_Pct': np.round(oof_probs * 100, 2),
    'Predicted_Default_50Pct': oof_preds_default,
    'Fold_Nested_Tuned_Cutoff_Pct': np.round(oof_thresholds * 100, 2),
    'Predicted_Nested_Tuned': oof_preds_tuned,
})

ai_excel_path = USER_PATH / 'early_warning_ai_results.xlsx'
try:
    with pd.ExcelWriter(ai_excel_path, engine='openpyxl') as writer:
        cv_stability_results.to_excel(writer, sheet_name='CV_5Fold_Summary', index=False)
        clinical_capture_df.to_excel(writer, sheet_name='Clinical_Capture_Metrics', index=False)
        pd.DataFrame({
            'Fold': range(1, 6),
            'ROC_AUC': fold_aucs,
            'PR_AUC': fold_pr_aucs,
            'Brier_Score': fold_briers,
            'Accuracy_Default50': fold_accs,
            'Nested_Tuned_Cutoff_Pct': np.round(np.array(fold_opt_thresholds) * 100, 2),
            'Sens_Default50_Pct': np.round(np.array(fold_sens_def) * 100, 2),
            'Spec_Default50_Pct': np.round(np.array(fold_spec_def) * 100, 2),
            'Sens_Tuned_Pct': np.round(np.array(fold_sens_tuned) * 100, 2),
            'Spec_Tuned_Pct': np.round(np.array(fold_spec_tuned) * 100, 2),
        }).to_excel(writer, sheet_name='Individual_Fold_Data', index=False)
        repeated_cv_df.to_excel(writer, sheet_name='Repeated_CV_10x5', index=False)
        df_cv_shap_sum.sort_values(by='Mean_Impact', ascending=False).to_excel(
            writer, sheet_name='SHAP_Feature_Importance', index=False
        )
        oof_predictions_df.to_excel(writer, sheet_name='Patient_OOF_Predictions', index=False)
        model_benchmark_df.to_excel(writer, sheet_name='Model_Benchmark_RF_vs_LR', index=False)

    print(f"\nSUCCESS! Standalone Early Warning AI outputs saved to: {USER_PATH}")
    print(f" - Calibration plot:       calibration_curve.png")
    print(f" - Static SHAP summary:    cv_stable_shap_summary.png")
    print(f" - Interactive SHAP bar:   cv_stable_shap_summary_interactive.html")
    print(f" - Interactive beeswarm:   cv_stable_shap_beeswarm_interactive.html")
    print(f" - Excel metrics & probs:  {ai_excel_path.name}")
except PermissionError:
    print(f"\nERROR: Please close '{ai_excel_path.name}' and re-run the script.")