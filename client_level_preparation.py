from numpy import ndarray
import pandas as pd
import os
from pandas import DataFrame, read_csv, Series
from matplotlib.pyplot import savefig, show, figure
import dslabs as dslabs
from dslabs.dslabs_functions import plot_multibar_chart, CLASS_EVAL_METRICS, run_NB, run_KNN, get_variable_types
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, KBinsDiscretizer


def evaluate_approach(
    train: DataFrame, test: DataFrame, target: str = "class", metric: str = "accuracy"
) -> dict[str, list]:
    trnY = train.pop(target).values
    trnX: ndarray = train.values
    tstY = test.pop(target).values
    tstX: ndarray = test.values
    eval: dict[str, list] = {}

    eval_NB: dict[str, float] = run_NB(trnX, trnY, tstX, tstY, metric=metric)
    eval_KNN: dict[str, float] = run_KNN(trnX, trnY, tstX, tstY, metric=metric)
    if eval_NB != {} and eval_KNN != {}:
        for met in CLASS_EVAL_METRICS:
            eval[met] = [eval_NB[met], eval_KNN[met]]
    return eval


# --- Load data ---
data_path = os.getcwd()
df_fraud = pd.read_csv(os.path.join(data_path, 'client_level_dataset.csv'))
df_clients = df_fraud.drop(columns=["client_id"])
client_ids = df_fraud[["client_id"]].reset_index(drop=True)

# make sure invoice_date_std is in timedelta format
df_clients['invoice_date_std'] = pd.to_timedelta(df_clients['invoice_date_std'], errors='coerce').dt.total_seconds() / (24 * 3600)

# Encode symbolic
variable_types = get_variable_types(df_clients)
to_encode = variable_types["binary"] + variable_types["symbolic"]
print(f"\nFound {len(to_encode)} to encode variables:")
for var in to_encode:
    unique_vals = df_clients[var].nunique()
    print(f"\n{var} — {unique_vals} unique values")
    print(df_clients[var].value_counts(dropna=False).head(10))

df_clients['counter_type_get_mode'] = df_clients['counter_type_get_mode'].map({'ELEC': 0, 'GAZ': 1})
df_clients['target'] = df_clients['target'].astype(int)


# Missing Summary
missing_counts = df_clients.isnull().sum()
missing_percent = (missing_counts / len(df_clients)) * 100
missing_summary = missing_counts[missing_counts > 0].sort_values(ascending=False)
print("Missing Values Summary:")
print(pd.DataFrame({"Missing Count": missing_summary, "Missing %": missing_percent[missing_summary.index].round(2)}))

# Drop datetime
datetime_cols = df_clients.select_dtypes(include=['datetime64']).columns.tolist()
df_clients = df_clients.drop(columns=datetime_cols)


# Strategy 1: median fill
#df_median = df_clients.copy()
#num_with_nan = df_median.columns[df_median.isnull().any()].tolist()
#median_imputer = SimpleImputer(strategy='median')
#df_median[num_with_nan] = median_imputer.fit_transform(df_median[num_with_nan])

## Strategy 2: KNN fill
#df_knn = df_clients.copy()  
#knn_imputer = KNNImputer(n_neighbors=5)
#df_knn[df_knn.columns] = knn_imputer.fit_transform(df_knn)

target = "target"

## Strategy 3: Drop rows with missing values
#df_dropna = df_clients.dropna()

## Train-test split
#trn_median, tst_median = train_test_split(df_median, test_size=0.3, random_state=42, stratify=df_median[target])
#trn_knn, tst_knn = train_test_split(df_knn, test_size=0.3, random_state=42, stratify=df_knn[target])
#trn_dropna, tst_dropna = train_test_split(df_dropna, test_size=0.3, random_state=42, stratify=df_dropna[target])

#figure(figsize=(6, 3))
#eval1 = evaluate_approach(trn_median.copy(), tst_median.copy(), target=target)
#plot_multibar_chart(["NB", "KNN"], eval1, title="Median Imputation Evaluation", percentage=True)
#show()

#figure(figsize=(6, 3))
#eval2 = evaluate_approach(trn_knn.copy(), tst_knn.copy(), target=target)
#plot_multibar_chart(["NB", "KNN"], eval2, title="KNN Imputation Evaluation", percentage=True)
#show()

#figure(figsize=(6, 3))
#eval3 = evaluate_approach(trn_dropna.copy(), tst_dropna.copy(), target=target)
#plot_multibar_chart(["NB", "KNN"], eval3, title="Dropna Evaluation", percentage=True)
#show()

## Before dropping
#print("Before dropna:")
#print(df_clients['target'].value_counts())


## After dropping missing values
#print("\nAfter dropna:")
#print(df_dropna['target'].value_counts())

# Drop rows with NaNs
df_clients = df_clients.dropna()

# ------------------------------------------------------------------
# Outlier Detection 
NR_STDEV = 2
IQR_FACTOR = 1.5

def determine_outlier_thresholds_for_var(summary: Series, std_based=True, threshold=NR_STDEV) -> tuple[float, float]:
    if std_based:
        std = threshold * summary["std"]
        return summary["mean"] + std, summary["mean"] - std
    else:
        iqr = summary["75%"] - summary["25%"]
        return summary["75%"] + IQR_FACTOR * iqr, summary["25%"] - IQR_FACTOR * iqr

numeric_vars = get_variable_types(df_clients)["numeric"]
summary5 = df_clients[numeric_vars].describe()
df_truncate_outliers = df_clients.copy()
for var in numeric_vars:
    top, bottom = determine_outlier_thresholds_for_var(summary5[var])
    df_truncate_outliers[var] = df_truncate_outliers[var].apply(lambda x: top if x > top else bottom if x < bottom else x)
df_clients = df_truncate_outliers.copy()


# # --- Strategy A: Do Nothing (Already Done) ---
# df_no_outlier = df_clients.copy()

# # --- Strategy B: Drop Outliers ---
# df_drop_outliers = df_clients.copy()
# for var in numeric_vars:
#     top, bottom = determine_outlier_thresholds_for_var(summary5[var])
#     df_drop_outliers = df_drop_outliers[(df_drop_outliers[var] <= top) & (df_drop_outliers[var] >= bottom)]

# # --- Strategy C: Replace Outliers with Median ---
# df_replace_outliers = df_clients.copy()
# for var in numeric_vars:
#     top, bottom = determine_outlier_thresholds_for_var(summary5[var])
#     median = df_replace_outliers[var].median()
#     df_replace_outliers[var] = df_replace_outliers[var].apply(lambda x: median if x > top or x < bottom else x)

# --- Strategy D: Truncate (Clip) Outliers ---
# df_truncate_outliers = df_clients.copy()
# for var in numeric_vars:
#     top, bottom = determine_outlier_thresholds_for_var(summary5[var])
#     df_truncate_outliers[var] = df_truncate_outliers[var].apply(lambda x: top if x > top else bottom if x < bottom else x)

# datasets = {
#     "No Outlier Treatment": df_no_outlier,
#     "Drop Outliers": df_drop_outliers,
#     "Replace with Median": df_replace_outliers,
#     "Truncate": df_truncate_outliers
# }

# for label, df in datasets.items():
#     print(f"\n--- {label} ---")
#     trn, tst = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target])
#     eval_result = evaluate_approach(trn.copy(), tst.copy(), target=target)
#     figure(figsize=(6, 3))
#     plot_multibar_chart(["NB", "KNN"], eval_result, title=f"{label} Evaluation", percentage=True)
#     show()


# -----------------------------------------------------------
# Feature Scaling

# Identify numeric features (excluding target)
numeric_vars = get_variable_types(df_clients)["numeric"]
X = df_clients.drop(columns=["target"])
y = df_clients["target"]

# # --- Scaling: QuantileTransformer ---
# scaler_quantile = QuantileTransformer(output_distribution='uniform', random_state=42)
# X_qt_scaled = pd.DataFrame(scaler_quantile.fit_transform(X[numeric_vars]), columns=numeric_vars)
# X_qt_scaled["target"] = y.values

# --- Scaling: MinMaxScaler ---
# scaler_mm = MinMaxScaler()
# X_mm_scaled = pd.DataFrame(scaler_mm.fit_transform(X[numeric_vars]), columns=numeric_vars)
# X_mm_scaled["target"] = y.values

# # No scaling baseline
# X_noscale = df_clients[numeric_vars].copy()
# X_noscale["target"] = df_clients["target"].values


# #--- Evaluation Function ---
# def scale_and_evaluate(df_scaled: DataFrame, title: str):
#     trn, tst = train_test_split(df_scaled, test_size=0.3, random_state=42, stratify=df_scaled["target"])
#     eval_res = evaluate_approach(trn.copy(), tst.copy(), target="target")
#     figure(figsize=(6, 3))
#     plot_multibar_chart(["NB", "KNN"], eval_res, title=title, percentage=True)
#     show()

# --- Evaluate ---
# scale_and_evaluate(X_noscale, "No Scaling (Raw Values)")
#scale_and_evaluate(X_mm_scaled, "Min-Max Scaling")
# scale_and_evaluate(X_qt_scaled, "Quantile (Uniform) Scaling")

# import matplotlib.pyplot as plt
# import seaborn as sns

# # Select numeric variables with most unique values
# most_variable = df_clients[numeric_vars].nunique().sort_values(ascending=False).head(5).index.tolist()

# # Create boxplots before and after scaling
# fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# # Raw values
# sns.boxplot(data=df_clients[most_variable], ax=axs[0])
# axs[0].set_title("Raw Values")

# # Min-Max Scaled
# sns.boxplot(data=X_mm_scaled[most_variable], ax=axs[1])
# axs[1].set_title("Min-Max Scaled")

# # Quantile Scaled
# sns.boxplot(data=X_qt_scaled[most_variable], ax=axs[2])
# axs[2].set_title("Quantile Scaled")

# for ax in axs:
#     ax.tick_params(axis='x', rotation=45)

# plt.tight_layout()
# plt.show()

# Aplicar Min-Max Scaling apenas nas variáveis numéricas
scaler_mm = MinMaxScaler()
X_scaled = pd.DataFrame(scaler_mm.fit_transform(df_clients[numeric_vars]), columns=numeric_vars)

# Reanexar a coluna target
df_clients = pd.concat([X_scaled, df_clients["target"].reset_index(drop=True)], axis=1)

# balancing
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

# --- CLASS BALANCING ---

target = "target"
# # Use the preprocessed and scaled df_clients for balancing experiments
# df_bal = df_clients.copy() 

# # Separate features (X) and target (y) for balancing
# X_bal = df_bal.drop(columns=[target])
# y_bal = df_bal[target]

# print(f"\nClass distribution BEFORE balancing: {Counter(y_bal)}") # Shows the initial imbalance

# Split data into training and test sets BEFORE applying any balancing technique.
# # It's crucial to balance ONLY the training set to prevent data leakage.
# trn_X, tst_X, trn_y, tst_y = train_test_split(X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal)

# # Recombine the test data (which will not be balanced) for evaluation
# # The test set remains unchanged across all balancing experiments for fair comparison.
# tst_final = pd.concat([pd.DataFrame(tst_X, columns=tst_X.columns), pd.Series(tst_y, name=target)], axis=1)

# --- 1. No Balancing (Baseline after Min-Max Scaling) ---
# print(f"\n--- Evaluation: No Balancing ---")
# print(f"Class distribution in training set: {Counter(trn_y)}")
# # Recombine the original (unbalanced) training data for evaluation
# trn_no_bal = pd.concat([pd.DataFrame(trn_X, columns=trn_X.columns), pd.Series(trn_y, name=target)], axis=1)
# eval_no_bal = evaluate_approach(trn_no_bal.copy(), tst_final.copy(), target=target)
# figure(figsize=(8, 4))
# plot_multibar_chart(["NB", "KNN"], eval_no_bal, title=f"No Balancing", percentage=True)
# show()

# --- 2. Random Under-sampling ---
# print(f"\n--- Evaluation: Random Under-sampling ---")
# # Initialize RandomUnderSampler
# rus = RandomUnderSampler(random_state=42)
# # Apply RUS to the training set
# trn_X_rus, trn_y_rus = rus.fit_resample(trn_X, trn_y)
# print(f"Class distribution in training set AFTER RUS: {Counter(trn_y_rus)}")
# Recombine the undersampled training data for evaluation
# trn_rus = pd.concat([pd.DataFrame(trn_X_rus, columns=trn_X.columns), pd.Series(trn_y_rus, name=target)], axis=1)
# eval_rus = evaluate_approach(trn_rus.copy(), tst_final.copy(), target=target)
# figure(figsize=(8, 4))
# plot_multibar_chart(["NB", "KNN"], eval_rus, title=f"Random Under-sampling", percentage=True)
# show()

# # --- 3. Random Over-sampling ---
# print(f"\n--- Evaluation: Random Over-sampling ---")
# # Initialize RandomOverSampler
# ros = RandomOverSampler(random_state=42)
# # Apply ROS to the training set
# trn_X_ros, trn_y_ros = ros.fit_resample(trn_X, trn_y)
# print(f"Class distribution in training set AFTER ROS: {Counter(trn_y_ros)}")
# # Recombine the oversampled training data for evaluation
# trn_ros = pd.concat([pd.DataFrame(trn_X_ros, columns=trn_X.columns), pd.Series(trn_y_ros, name=target)], axis=1)
# eval_ros = evaluate_approach(trn_ros.copy(), tst_final.copy(), target=target)
# figure(figsize=(8, 4))
# plot_multibar_chart(["NB", "KNN"], eval_ros, title=f"Random Over-sampling", percentage=True)
# show()

# # --- 4. SMOTE (Synthetic Minority Over-sampling Technique) ---
# print(f"\n--- Evaluation: SMOTE ---")
# # Initialize SMOTE
# smote = SMOTE(random_state=42)
# # Apply SMOTE to the training set
# trn_X_smote, trn_y_smote = smote.fit_resample(trn_X, trn_y)
# print(f"Class distribution in training set AFTER SMOTE: {Counter(trn_y_smote)}")
# # Recombine the SMOTE-treated training data for evaluation
# trn_smote = pd.concat([pd.DataFrame(trn_X_smote, columns=trn_X.columns), pd.Series(trn_y_smote, name=target)], axis=1)
# eval_smote = evaluate_approach(trn_smote.copy(), tst_final.copy(), target=target)
# figure(figsize=(8, 4))
# plot_multibar_chart(["NB", "KNN"], eval_smote, title=f"SMOTE", percentage=True)
# show()

# --- Balancing ---
X_bal = df_clients.drop(columns=["target"])
y_bal = df_clients["target"]
rus = RandomUnderSampler(random_state=42)
X_rus, y_rus = rus.fit_resample(X_bal, y_bal)
df_clients = pd.concat([pd.DataFrame(X_rus, columns=X_bal.columns), pd.Series(y_rus, name="target")], axis=1)
print("New class distribution after undersampling:")
print(Counter(df_clients["target"]))

# # --- Discretization ---
# # Define numerical variables to discretize (excluding target and already binary ones)
# variable_types_after_sampling = get_variable_types(df_clients) 
# discretize_vars = [
#     var for var in variable_types_after_sampling["numeric"] 
#     if var not in ['target', 'counter_type_nunique', 'counter_type_get_mode']
# ]

# print(f"\nVariables selected for discretization: {len(discretize_vars)} variables.")

# # Define a list of n_bins values to try
# n_bins_to_try = [3, 5, 7, 10] # Example: Try 3, 5, 7, and 10 bins

# # Separate X and y for train-test split for discretization experiments
# X_discretize = df_clients.drop(columns=[target])
# y_discretize = df_clients[target]

# # --- Train-Test Split (Crucial to do before discretizing train set) ---
# trn_X_orig, tst_X_orig, trn_y_orig, tst_y_orig = train_test_split(X_discretize, y_discretize, test_size=0.3, random_state=42, stratify=y_discretize)

# # Recombine test data (this will be the same for all experiments, as test set is NOT discretized)
# tst_final_discretization = pd.concat([pd.DataFrame(tst_X_orig, columns=tst_X_orig.columns), pd.Series(tst_y_orig, name=target)], axis=1)


# # --- 1. No Discretization (Baseline) ---
# print(f"\n--- Evaluation: No Discretization (Baseline) ---")
# trn_no_disc = pd.concat([pd.DataFrame(trn_X_orig, columns=trn_X_orig.columns), pd.Series(trn_y_orig, name=target)], axis=1)
# eval_no_disc = evaluate_approach(trn_no_disc.copy(), tst_final_discretization.copy(), target=target)
# figure(figsize=(8, 4))
# plot_multibar_chart(["NB", "KNN"], eval_no_disc, title=f"No Discretization", percentage=True)
# show()


# # --- Iterate through different n_bins for Equal-Width Discretization ---
# for n_bins in n_bins_to_try:
#     print(f"\n--- Evaluation: Equal-Width Discretization with {n_bins} Bins ---")
    
#     # Initialize Equal-Width Discretizer
#     discretizer_ew = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform', subsample=None, random_state=42)
    
#     # Apply to training features only
#     trn_X_ew = trn_X_orig.copy()
#     # Only transform the selected variables for discretization
#     trn_X_ew[discretize_vars] = discretizer_ew.fit_transform(trn_X_orig[discretize_vars])

#     # Apply the SAME fitted discretizer to test features (transform only)
#     tst_X_ew = tst_X_orig.copy()
#     tst_X_ew[discretize_vars] = discretizer_ew.transform(tst_X_orig[discretize_vars]) # Use transform, not fit_transform

#     # Recombine training data for evaluation
#     trn_ew = pd.concat([pd.DataFrame(trn_X_ew, columns=trn_X_ew.columns), pd.Series(trn_y_orig, name=target)], axis=1)
#     # Recombine test data for evaluation
#     tst_ew = pd.concat([pd.DataFrame(tst_X_ew, columns=tst_X_ew.columns), pd.Series(tst_y_orig, name=target)], axis=1)

#     eval_ew = evaluate_approach(trn_ew.copy(), tst_ew.copy(), target=target)
#     figure(figsize=(8, 4))
#     plot_multibar_chart(["NB", "KNN"], eval_ew, title=f"EW Discretization (Bins={n_bins})", percentage=True)
#     show()

# # --- Iterate through different n_bins for Equal-Frequency Discretization ---
# for n_bins in n_bins_to_try:
#     print(f"\n--- Evaluation: Equal-Frequency Discretization with {n_bins} Bins ---")
    
#     # Initialize Equal-Frequency Discretizer
#     discretizer_ef = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile', subsample=None, random_state=42)
    
#     # Apply to training features only
#     trn_X_ef = trn_X_orig.copy()
#     trn_X_ef[discretize_vars] = discretizer_ef.fit_transform(trn_X_orig[discretize_vars])

#     # Apply the SAME fitted discretizer to test features (transform only)
#     tst_X_ef = tst_X_orig.copy()
#     tst_X_ef[discretize_vars] = discretizer_ef.transform(tst_X_orig[discretize_vars]) # Use transform, not fit_transform

#     # Recombine training data for evaluation
#     trn_ef = pd.concat([pd.DataFrame(trn_X_ef, columns=trn_X_ef.columns), pd.Series(trn_y_orig, name=target)], axis=1)
#     # Recombine test data for evaluation
#     tst_ef = pd.concat([pd.DataFrame(tst_X_ef, columns=tst_X_ef.columns), pd.Series(tst_y_orig, name=target)], axis=1)

#     eval_ef = evaluate_approach(trn_ef.copy(), tst_ef.copy(), target=target)
#     figure(figsize=(8, 4))
#     plot_multibar_chart(["NB", "KNN"], eval_ef, title=f"EF Discretization (Bins={n_bins})", percentage=True)
#     show()

# print("\nAll discretization strategies with various bin counts tested and evaluated.")

# Reattach client_id and save
client_ids_final = client_ids.loc[df_clients.index].reset_index(drop=True)
df_clients = pd.concat([df_clients.reset_index(drop=True), client_ids_final], axis=1)
df_clients.to_csv("client_pos_prep.csv", index=False)