from numpy import ndarray
import pandas as pd
import os
from pandas import DataFrame, read_csv, Series
from matplotlib.pyplot import savefig, show, figure
import dslabs as dslabs
from dslabs.dslabs_functions import (
    evaluate_approach,
    plot_multibar_chart,
    CLASS_EVAL_METRICS,
    run_NB,
    run_KNN,
    get_variable_types,
    plot_multiline_chart
)
from sklearn.model_selection import train_test_split
from math import ceil

# --- Load data ---
data_path = os.getcwd()
df_fraud = pd.read_csv(os.path.join(data_path, 'client_pos_prep.csv'))
df_clients = df_fraud.drop(columns=["client_id"])

# --- Settings ---
target = "target"
file_tag = "client"

# --- Split dataset ---
train, test = train_test_split(df_clients, test_size=0.3, random_state=42, stratify=df_clients[target])

# --- 1. Study low variance thresholds ---
def study_variance_for_feature_selection(train, test, target, max_threshold=1.0, lag=0.05, metric="accuracy", file_tag=""):
    options = [round(i * lag, 3) for i in range(1, ceil(max_threshold / lag + lag))]
    results = {"NB": [], "KNN": []}
    summary5 = train.describe()

    for thresh in options:
        vars2drop = summary5.columns[summary5.loc["std"] ** 2 < thresh]
        if target in vars2drop:
            vars2drop = vars2drop.drop(target)

        trn = train.drop(columns=vars2drop)
        tst = test.drop(columns=vars2drop)

        if trn.drop(columns=[target]).shape[1] == 0:
            print(f"Skipping threshold {thresh} — would result in 0 features.")
            continue

        eval_res = evaluate_approach(trn.copy(), tst.copy(), target=target, metric=metric)
        results["NB"].append(eval_res[metric][0])
        results["KNN"].append(eval_res[metric][1])

    plot_multiline_chart(
        options[:len(results["NB"])],
        results,
        title=f"{file_tag} variance threshold study ({metric})",
        xlabel="Variance Threshold",
        ylabel=metric,
        percentage=True
    )
    savefig(f"images/{file_tag}_low_var_threshold_study.png")
    show()
    return results

# --- 2. Select variables to drop ---
def select_low_variance_variables(data, max_threshold, target):
    summary5 = data.describe()
    vars2drop = summary5.columns[summary5.loc["std"] ** 2 < max_threshold]
    if target in vars2drop:
        vars2drop = vars2drop.drop(target)
    return list(vars2drop)

# --- 3. Apply FS and save new datasets ---
def apply_feature_selection(train, test, vars2drop, filename="", tag=""):
    train_fs = train.drop(columns=vars2drop)
    test_fs = test.drop(columns=vars2drop)
    train_fs.to_csv(f"{filename}_train_{tag}.csv", index=False)
    test_fs.to_csv(f"{filename}_test_{tag}.csv", index=False)
    return train_fs, test_fs

# --- Run low variance feature selection ---
figure(figsize=(10, 4))
study_variance_for_feature_selection(train, test, target=target, max_threshold=0.05, lag=0.005, metric="recall", file_tag=file_tag)
best_threshold = 0.02
vars_low_var = select_low_variance_variables(train, best_threshold, target=target)
train_fs, test_fs = apply_feature_selection(train, test, vars_low_var, filename="data/client", tag="lowvar")

# --- Clean dataset for redundancy step ---
df_cleaned = df_clients.drop(columns=vars_low_var)
train, test = train_test_split(df_cleaned, test_size=0.3, random_state=42, stratify=df_cleaned[target])

# --- 4. Redundant variables analysis ---
def select_redundant_variables(data, min_threshold=0.90, target="target"):
    df = data.drop(columns=[target])
    corr_matrix = abs(df.corr())
    variables = corr_matrix.columns
    vars2drop = []
    for v1 in variables:
        vars_corr = corr_matrix[v1][corr_matrix[v1] >= min_threshold].drop(v1, errors="ignore")
        for v2 in vars_corr.index:
            if v2 not in vars2drop:
                vars2drop.append(v2)
    return vars2drop


def study_redundancy_for_feature_selection(train, test, target="target", min_threshold=0.50, lag=0.05, metric="accuracy", file_tag=""):
    options = [round(min_threshold + i * lag, 3) for i in range(ceil((1 - min_threshold) / lag) + 1)]
    df = train.drop(columns=[target])
    corr_matrix = abs(df.corr())
    variables = corr_matrix.columns
    results = {"NB": [], "KNN": []}

    for thresh in options:
        vars2drop = []
        for v1 in variables:
            vars_corr = corr_matrix[v1][corr_matrix[v1] >= thresh].drop(v1, errors="ignore")
            for v2 in vars_corr.index:
                if v2 not in vars2drop:
                    vars2drop.append(v2)
        trn = train.drop(columns=vars2drop)
        tst = test.drop(columns=vars2drop)
        eval = evaluate_approach(trn.copy(), tst.copy(), target=target, metric=metric)
        results["NB"].append(eval[metric][0])
        results["KNN"].append(eval[metric][1])

    plot_multiline_chart(
        options,
        results,
        title=f"{file_tag} redundancy study ({metric})",
        xlabel="correlation threshold",
        ylabel=metric,
        percentage=True
    )
    savefig(f"images/{file_tag}_fs_redundancy_{metric}_study.png")
    show()
    return results

# --- Run redundancy study ---
figure(figsize=(10, 5))
study_redundancy_for_feature_selection(train, test, target=target, min_threshold=0.5, lag=0.05, metric="recall", file_tag=file_tag)

# --- Final correlation threshold ---
final_corr_threshold = 0.75
vars_corr_drop = select_redundant_variables(train, min_threshold=final_corr_threshold, target=target)
train_final, test_final = apply_feature_selection(train, test, vars_corr_drop, filename="data/client", tag="redundant")

print(f"Vars to drop (corr > {final_corr_threshold}):", vars_corr_drop)
print(f"Train before: {train.shape}, after: {train_final.shape}")
print(f"Test before: {test.shape}, after: {test_final.shape}")


# Final union of both drop lists
all_vars_to_drop = list(set(vars_low_var + vars_corr_drop))

# Drop from the original df_clients
df_final = df_fraud.drop(columns=all_vars_to_drop)

# Save to CSV
df_final.to_csv("client_pos_selected.csv", index=False)

print(f"\nFinal dataset saved with shape: {df_final.shape}")