import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import (
    kendalltau, kruskal, chi2_contingency,
    pearsonr, spearmanr, pointbiserialr
)
import pingouin as pg
import dcor
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Suppress warnings for clean console output
warnings.filterwarnings("ignore")


def print_header(title):
    print("\n" + "=" * 85 + f"\n{title}\n" + "=" * 85)


def compute_comprehensive_correlations(df, var1, var2, dcor_resamples=1000):
    """Compute linear, monotonic, ordinal, and non-linear (distance) correlations with p-values."""
    temp = df[[var1, var2]].dropna()
    if temp.empty or len(temp) < 3:
        return None

    x_arr = np.asarray(temp[var1], dtype=float)
    y_arr = np.asarray(temp[var2], dtype=float)

    # 1. Pearson (Linear)
    pear_r, pear_p = pearsonr(x_arr, y_arr)
    # 2. Spearman (Monotonic Rank)
    spea_r, spea_p = spearmanr(x_arr, y_arr)
    # 3. Kendall's Tau (Ordinal Concordance)
    ken_tau, ken_p = kendalltau(x_arr, y_arr)
    # 4. Distance Correlation + Permutation Independence Test
    dc_val = float(dcor.distance_correlation(x_arr, y_arr))
    dc_test = dcor.independence.distance_covariance_test(
        x_arr, y_arr, num_resamples=dcor_resamples, random_state=42
    )
    dc_p = float(getattr(dc_test, "pvalue", getattr(dc_test, "p_value", np.nan)))

    return {
        "Pair": f"{var1} vs {var2}",
        "Pearson_r": pear_r, "Pearson_p": pear_p,
        "Spearman_rho": spea_r, "Spearman_p": spea_p,
        "Kendall_tau": ken_tau, "Kendall_p": ken_p,
        "Distance_Cor": dc_val, "Distance_Cor_p": dc_p
    }


# ============================================================
# 1. FILE PATHS & DATA LOADING
# ============================================================
DEFAULT_PATH = Path(r"E:\Documents\RESEARCH_Anusandhan\Medical Technology\DATA\Required data set\Nov-2026_results_immune_collapse")
USER_PATH = Path(os.environ.get("OSCC_DATA_DIR", DEFAULT_PATH if DEFAULT_PATH.exists() else Path(__file__).resolve().parent))
INPUT_FILE = USER_PATH / "processed_synergy_data_final.xlsx"
OUTPUT_FILE = USER_PATH / "advanced_statistical_results_Raw_&_Processed.xlsx"

print_header("1. LOADING DATASETS")
try:
    xls = pd.ExcelFile(INPUT_FILE)
    df_clinical = pd.read_excel(xls, sheet_name='Local_Clinical_Proteins')
    has_geo = 'Global_GEO_mRNA' in xls.sheet_names
    if has_geo:
        df_geo = pd.read_excel(xls, sheet_name='Global_GEO_mRNA')
        print(f"Loaded 'Local_Clinical_Proteins' ({len(df_clinical)} rows) and 'Global_GEO_mRNA' ({len(df_geo)} rows).")
    else:
        print(f"Loaded 'Local_Clinical_Proteins' ({len(df_clinical)} rows). 'Global_GEO_mRNA' not found.")
        df_geo = pd.DataFrame()
except FileNotFoundError:
    print(f"Error: Could not find {INPUT_FILE}. Please run your main pipeline first.")
    exit()


# ============================================================
# 2. DATA CLEANING & STAGE MAPPING
# ============================================================
def encode_sex(value):
    """Match Immune_Collapse_15-11-2026.py strictly: Male = 1, Female = 0."""
    if pd.isna(value):
        return np.nan
    val = str(value).strip().lower()
    return 1 if val in ["male", "m"] else (0 if val in ["female", "f"] else np.nan)


stage_mapping = {'No Dysplasia': 0, 'Mild Dysplasia': 1, 'Mod Dysplasia': 2, 'Sev Dysplasia': 3, 'OSCC': 4}

df_clean = pd.DataFrame()
df_clean["Patient_Name"] = df_clinical["Patient_Name"]
df_clean["Age"] = df_clinical["Age"].astype(float)
df_clean["BMI"] = df_clinical["BMI"].astype(float)
df_clean["Sex"] = df_clinical["Sex"].apply(encode_sex)
df_clean["Group"] = df_clinical["Group"]
df_clean["Stage_Rank"] = df_clinical["Group"].map(stage_mapping)

# RAW & LOG-TRANSFORMED Proteins (matches OLS & ML scale in main pipeline)
df_clean["IL-8"] = df_clinical["IL-8"].astype(float)
df_clean["sPD-L1"] = df_clinical["sPD-L1"].astype(float)
df_clean["IL-8_log"] = np.log1p(df_clean["IL-8"])
df_clean["sPD-L1_log"] = np.log1p(df_clean["sPD-L1"])
df_clean["raw_i_IPS"] = df_clean["IL-8_log"] + df_clean["sPD-L1_log"]

# ADJUSTED Proteins
df_clean["IL-8_adj"] = df_clinical["IL-8_adj"].astype(float)
df_clean["sPD-L1_adj"] = df_clinical["sPD-L1_adj"].astype(float)
df_clean["i_IPS"] = df_clinical["i_IPS"].astype(float)

# ============================================================
# 3. COMPREHENSIVE CORRELATION ON RAW DATA
# ============================================================
print_header("3. COMPREHENSIVE CORRELATIONS: RAW DATA")

raw_pairs = [
    ("IL-8", "sPD-L1"), ("Age", "IL-8"), ("Age", "sPD-L1"),
    ("BMI", "IL-8"), ("BMI", "sPD-L1"), ("Age", "BMI")
]

raw_corr_results = []
for v1, v2 in raw_pairs:
    res = compute_comprehensive_correlations(df_clean, v1, v2)
    if res: raw_corr_results.append(res)

df_raw_corr = pd.DataFrame(raw_corr_results)
print("--- RAW DATA: MULTI-METRIC PAIRWISE CORRELATIONS ---")
print(df_raw_corr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n--- RAW & LOG-TRANSFORMED PARTIAL CORRELATIONS (Controlling for Covariates) ---")
partial_results = []
temp_raw = df_clean[['Age', 'BMI', 'Sex', 'IL-8', 'sPD-L1', 'IL-8_log', 'sPD-L1_log']].dropna()

for scale_label, il8_col, pdl1_col in [
    ("Raw", "IL-8", "sPD-L1"),
    ("Log1p", "IL-8_log", "sPD-L1_log")
]:
    pcor_bio = pg.partial_corr(data=temp_raw, x=il8_col, y=pdl1_col, covar=['Age', 'BMI', 'Sex'])
    pcol = 'p_val' if 'p_val' in pcor_bio.columns else 'p-val'
    partial_results.append({
        "Scale": scale_label, "Target": f"{il8_col} vs {pdl1_col}",
        "Covariates": "Age, BMI, Sex",
        "r": pcor_bio['r'].values[0], "p-value": pcor_bio[pcol].values[0]
    })
    for bio in [il8_col, pdl1_col]:
        pcor_age = pg.partial_corr(data=temp_raw, x='Age', y=bio, covar=['BMI', 'Sex'])
        partial_results.append({
            "Scale": scale_label, "Target": f"Age vs {bio}",
            "Covariates": "BMI, Sex",
            "r": pcor_age['r'].values[0], "p-value": pcor_age[pcol].values[0]
        })
        pcor_bmi = pg.partial_corr(data=temp_raw, x='BMI', y=bio, covar=['Age', 'Sex'])
        partial_results.append({
            "Scale": scale_label, "Target": f"BMI vs {bio}",
            "Covariates": "Age, Sex",
            "r": pcor_bmi['r'].values[0], "p-value": pcor_bmi[pcol].values[0]
        })

df_raw_partial = pd.DataFrame(partial_results)
print(df_raw_partial.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# 4. COMPREHENSIVE CORRELATION ON PROCESSED DATA
# ============================================================
print_header("4. COMPREHENSIVE CORRELATIONS: PROCESSED DATA (RESIDUALS)")
print("Goal: Demographic noise vs Adjusted Biomarkers should collapse to ~0.00")

proc_pairs = [
    ("IL-8_adj", "sPD-L1_adj"), ("Age", "IL-8_adj"), ("Age", "sPD-L1_adj"),
    ("BMI", "IL-8_adj"), ("BMI", "sPD-L1_adj"), ("Age", "i_IPS"), ("BMI", "i_IPS")
]

proc_corr_results = []
for v1, v2 in proc_pairs:
    res = compute_comprehensive_correlations(df_clean, v1, v2)
    if res: proc_corr_results.append(res)

df_proc_corr = pd.DataFrame(proc_corr_results)
print("--- PROCESSED DATA: MULTI-METRIC PAIRWISE CORRELATIONS ---")
print(df_proc_corr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# 5. DISEASE STAGE DYNAMICS (Kendall's Tau & Kruskal-Wallis)
# ============================================================
print_header("5. CLINICAL PROGRESSION: VARIABLES VS DISEASE STAGE")
stage_results = []
temp_stage = df_clean.dropna(
    subset=['Age', 'BMI', 'Sex', 'Stage_Rank', 'IL-8', 'sPD-L1', 'i_IPS', 'IL-8_adj', 'sPD-L1_adj'])

# A. Continuous Variables vs Ordinal Stage
for var in ["Age", "BMI", "IL-8", "sPD-L1", "IL-8_adj", "sPD-L1_adj", "i_IPS"]:
    tau, p_tau = kendalltau(temp_stage[var], temp_stage['Stage_Rank'])
    stage_results.append({"Variable": var, "Test": "Kendall's Tau vs Stage", "Statistic": tau, "p-value": p_tau})

# B. Global Variance across stages (Kruskal-Wallis)
for var in ["Age", "BMI"]:
    groups = [temp_stage[temp_stage['Stage_Rank'] == i][var] for i in range(5)]
    h_stat, p_kw = kruskal(*[g for g in groups if len(g) > 0])
    stage_results.append(
        {"Variable": var, "Test": "Kruskal-Wallis across Stages", "Statistic": h_stat, "p-value": p_kw})

# C. Categorical vs Ordinal (Chi-Square + Permutation p-value if expected cell count < 5)
contingency = pd.crosstab(temp_stage['Sex'], temp_stage['Stage_Rank'])
chi2, p_chi2, dof, expected = chi2_contingency(contingency)
min_expected = float(expected.min())
stage_results.append({"Variable": "Sex", "Test": "Chi-Square vs Stage (Asymptotic)", "Statistic": chi2, "p-value": p_chi2})

if min_expected < 5.0:
    rng_chi = np.random.RandomState(42)
    sex_vals = temp_stage['Sex'].values.copy()
    stage_vals = temp_stage['Stage_Rank'].values
    n_perm = 5000
    exceed = 0
    for _ in range(n_perm):
        perm_table = pd.crosstab(rng_chi.permutation(sex_vals), stage_vals)
        perm_chi2, _, _, _ = chi2_contingency(perm_table)
        if perm_chi2 >= chi2 - 1e-12:
            exceed += 1
    p_chi2_perm = (exceed + 1) / (n_perm + 1)
    print(f"Note: Chi-Square min expected cell count = {min_expected:.2f} (< 5). Adding 5,000-permutation p-value.")
    stage_results.append({
        "Variable": "Sex", "Test": "Chi-Square vs Stage (Permutation, n=5000)",
        "Statistic": chi2, "p-value": p_chi2_perm
    })

df_stage = pd.DataFrame(stage_results)
print(df_stage.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# 6. HIERARCHICAL VARIANCE PARTITIONING (Delta R-Squared)
# ============================================================
print_header("6. VARIANCE PARTITIONING (Demographics vs Linear & Non-Linear Stage)")
temp_vp = df_clean[['Age', 'BMI', 'Sex', 'Stage_Rank', 'IL-8_log', 'sPD-L1_log', 'raw_i_IPS']].dropna()

vp_rows = []
for target_col, target_label in [
    ("IL-8_log", "ln(IL-8 + 1)"),
    ("sPD-L1_log", "ln(sPD-L1 + 1)"),
    ("raw_i_IPS", "Raw Synergy [ln(IL-8+1) + ln(sPD-L1+1)]")
]:
    m_demo = smf.ols(f"Q('{target_col}') ~ Age + BMI + Sex", data=temp_vp).fit()
    m_lin = smf.ols(f"Q('{target_col}') ~ Age + BMI + Sex + Stage_Rank", data=temp_vp).fit()
    m_cat = smf.ols(f"Q('{target_col}') ~ Age + BMI + Sex + C(Stage_Rank)", data=temp_vp).fit()

    f_lin, p_lin, _ = m_lin.compare_f_test(m_demo)
    f_cat, p_cat, _ = m_cat.compare_f_test(m_demo)

    vp_rows.append({
        "Target": target_label,
        "R2_Demographics": m_demo.rsquared,
        "R2_Demo_plus_LinearStage": m_lin.rsquared,
        "Delta_R2_LinearStage": m_lin.rsquared - m_demo.rsquared,
        "p_LinearStage": p_lin,
        "R2_Demo_plus_CategoricalStage": m_cat.rsquared,
        "Delta_R2_CategoricalStage": m_cat.rsquared - m_demo.rsquared,
        "p_CategoricalStage": p_cat
    })

df_vp = pd.DataFrame(vp_rows)
print(df_vp.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# 7. AIC EFFICACY TEST (Scale-Matched & Leak-Free Comparison)
# ============================================================
print_header("7. AIC EFFICACY (Predicting the Pre-Malignant Peak, N=65)")
ml_cohort = df_clean[df_clean['Stage_Rank'] < 4].dropna(
    subset=['Age', 'BMI', 'Sex', 'IL-8', 'sPD-L1', 'IL-8_log', 'sPD-L1_log', 'Stage_Rank']
).copy()
y_target = (ml_cohort['Stage_Rank'] == 3).astype(int)

# Refit demographic residuals strictly inside the pre-malignant cohort (zero OSCC leakage)
ml_cohort['IL-8_adj_pm'] = smf.ols("Q('IL-8_log') ~ Age + BMI + Sex", data=ml_cohort).fit().resid
ml_cohort['sPD-L1_adj_pm'] = smf.ols("Q('sPD-L1_log') ~ Age + BMI + Sex", data=ml_cohort).fit().resid

m_raw = sm.Logit(y_target, sm.add_constant(ml_cohort[['IL-8', 'sPD-L1']])).fit(disp=False)
m_log = sm.Logit(y_target, sm.add_constant(ml_cohort[['IL-8_log', 'sPD-L1_log']])).fit(disp=False)
m_adj_pm = sm.Logit(y_target, sm.add_constant(ml_cohort[['IL-8_adj_pm', 'sPD-L1_adj_pm']])).fit(disp=False)
m_full = sm.Logit(y_target, sm.add_constant(ml_cohort[['IL-8_log', 'sPD-L1_log', 'Age', 'BMI', 'Sex']])).fit(disp=False)

df_aic = pd.DataFrame([
    {"Model": "1. Raw Linear (IL-8, sPD-L1)", "Params_k": len(m_raw.params), "LogLik": m_raw.llf, "AIC": m_raw.aic},
    {"Model": "2. Log1p Only (IL-8_log, sPD-L1_log)", "Params_k": len(m_log.params), "LogLik": m_log.llf, "AIC": m_log.aic},
    {"Model": "3. Pre-Malignant Residuals (unpenalized OLS)", "Params_k": len(m_adj_pm.params), "LogLik": m_adj_pm.llf, "AIC": m_adj_pm.aic},
    {"Model": "4. Joint Multivariable (Log1p + Age, BMI, Sex)", "Params_k": len(m_full.params), "LogLik": m_full.llf, "AIC": m_full.aic},
])
print(df_aic.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# 8. GEO TRANSCRIPTOMIC VALIDATION (GSE30784)
# ============================================================
print_header("8. GSE30784 TISSUE mRNA CORRELATION (Compartmental Divergence)")
geo_results = []
if has_geo and not df_geo.empty:
    temp_geo = df_geo.dropna(subset=['Stage_Ordinal', 'IL-8_mRNA', 'sPD-L1_mRNA', 'g_IPS'])
    for bio in ['IL-8_mRNA', 'sPD-L1_mRNA', 'g_IPS']:
        tau, p_tau = kendalltau(temp_geo['Stage_Ordinal'], temp_geo[bio])
        geo_results.append({"Pair": f"Stage vs {bio}", "Kendall_tau": tau, "Kendall_p": p_tau})

    geo_pair = compute_comprehensive_correlations(temp_geo, 'IL-8_mRNA', 'sPD-L1_mRNA')
    if geo_pair:
        geo_results.append(geo_pair)

df_geo_res = pd.DataFrame(geo_results)
print(df_geo_res.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# 8b. GENDER (POINT-BISERIAL) & COMPOSITE COLLINEARITY
# ============================================================
print_header("8b. GENDER POINT-BISERIAL & COMPONENT COLLINEARITY")

gender_pairs = ['Age', 'BMI', 'IL-8', 'sPD-L1', 'IL-8_log', 'sPD-L1_log', 'IL-8_adj', 'sPD-L1_adj', 'i_IPS']
gender_results = []
temp_gender = df_clean.dropna(subset=['Sex'] + gender_pairs)

for var in gender_pairs:
    pbs_r, pbs_p = pointbiserialr(temp_gender['Sex'], temp_gender[var])
    gender_results.append({
        "Pair": f"Sex vs {var}",
        "Point-Biserial_r": pbs_r,
        "p-value": pbs_p
    })

df_gender_corr = pd.DataFrame(gender_results)
print("--- POINT-BISERIAL CORRELATIONS (1=Male, 0=Female) ---")
print(df_gender_corr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

comp_pairs = [("i_IPS", "IL-8_adj"), ("i_IPS", "sPD-L1_adj")]
comp_results = []
for v1, v2 in comp_pairs:
    res = compute_comprehensive_correlations(df_clean, v1, v2)
    if res:
        comp_results.append(res)

df_comp_corr = pd.DataFrame(comp_results)
print("\n--- COMPOSITE VS COMPONENT CORRELATIONS ---")
print(df_comp_corr.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ============================================================
# 9. RIGOROUS EXPORT TO EXCEL
# ============================================================
print_header("9. EXPORTING RESULTS TO EXCEL")
try:
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_raw_corr.to_excel(writer, sheet_name="Raw_Correlations_All", index=False)
        df_raw_partial.to_excel(writer, sheet_name="Raw_Partial_Correlations", index=False)
        df_proc_corr.to_excel(writer, sheet_name="Processed_Correlations_All", index=False)
        df_stage.to_excel(writer, sheet_name="Stage_Progression_Metrics", index=False)
        df_vp.to_excel(writer, sheet_name="Variance_Partitioning", index=False)
        df_aic.to_excel(writer, sheet_name="AIC_Efficacy_Test", index=False)
        df_gender_corr.to_excel(writer, sheet_name="Gender_PointBiserial", index=False)
        df_comp_corr.to_excel(writer, sheet_name="Composite_Collinearity", index=False)
        if not df_geo_res.empty:
            df_geo_res.to_excel(writer, sheet_name="GSE30784_Correlations", index=False)
        df_clean.to_excel(writer, sheet_name="Cleaned_Clinical_Data", index=False)
    print(f"SUCCESS! All comprehensive matrices exported rigorously to:\n{OUTPUT_FILE}")
except PermissionError:
    print(f"\nERROR: Permission Denied! Please CLOSE '{OUTPUT_FILE.name}' and run the script again.")