import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from dslabs import config
from dslabs import dslabs_functions as dslabs
import seaborn as sns
from scipy.stats import entropy

# Load data
data_path = os.getcwd()
df_fraud = pd.read_csv(os.path.join(data_path, 'client_level_dataset.csv'))  # No index_col here

# Dataset head
print("First 5 rows:")
print(df_fraud.head(), "\n")

# ---------------------------------
# Dimensionality
print(f"Fraud data dimensionality: {df_fraud.shape}\n")
print("First 5 rows:\n", df_fraud.head(), "\n")

# Convert date columns
date_cols = [col for col in df_fraud.columns if "date" in col.lower()]
for col in date_cols:
    df_fraud[col] = pd.to_datetime(df_fraud[col], errors='coerce', dayfirst=True)

# Clean up counter_statue (if present and in wrong dtype)
if 'counter_statue_get_mode' in df_fraud.columns:
    df_fraud['counter_statue_get_mode'] = df_fraud['counter_statue_get_mode'].astype(str).str.strip()
    df_fraud['counter_statue_get_mode'] = pd.to_numeric(df_fraud['counter_statue_get_mode'], errors='coerce')

# Identify variable types
variable_types = dslabs.get_variable_types(df_fraud)
print("Variable types detected:\n", variable_types)

# Identify variable types
variable_types: dict[str, list] = dslabs.get_variable_types(df_fraud)
print(variable_types)
counts: dict[str, int] = {}
for tp in variable_types.keys():
    counts[tp] = len(variable_types[tp])

plt.figure(figsize=(4, 2))
dslabs.plot_bar_chart(
    list(counts.keys()), list(counts.values()), title="Nr of variables per type"
)

# Column types
print("Column data types:")
print(df_fraud.dtypes, "\n")

# Convert all object-type columns to category 
symbolic: list[str] = variable_types["symbolic"]
df_fraud[symbolic] = df_fraud[symbolic].apply(lambda x: x.astype("category"))

# Convert target column to int
df_fraud["target"] = df_fraud["target"].astype(int)  

# Convert binary columns (except 'target') to category
binary: list[str] = [col for col in variable_types["binary"] if col != "target"]
df_fraud[binary] = df_fraud[binary].apply(lambda x: x.astype("category"))

# Print variable types summary
variable_types: dict[str, list] = dslabs.get_variable_types(df_fraud)
print(df_fraud.dtypes, "\n")

# ---------------------------------
# Sparsity

numeric = variable_types['numeric']

# Check for missing values
missing_counts = df_fraud.isnull().sum()
missing_percent = (missing_counts / len(df_fraud)) * 100

# Filter only columns with missing values
missing_summary = missing_counts[missing_counts > 0].sort_values(ascending=False)
missing_percent = missing_percent[missing_summary.index]

print("Missing Values Summary:")
print(pd.DataFrame({
    "Missing Count": missing_summary,
    "Missing %": missing_percent.round(2)
}), "\n")


# Count of zeros in numeric columns
zero_counts = (df_fraud[numeric] == 0).sum()
zero_percent = (zero_counts / len(df_fraud)) * 100

print("Zero Values (%):")
print(pd.DataFrame({
    "Zero Count": zero_counts,
    "Zero %": zero_percent.round(2)
}).sort_values(by="Zero %", ascending=False), "\n")

# Unique values
unique_counts = df_fraud.nunique().sort_values()
print("Unique values per column:")
print(unique_counts)

# ---------------------------------
# Distributions

variable_types = dslabs.get_variable_types(df_fraud)

# --- Paginated Numeric Variable Distribution Plots ---
numeric_vars = variable_types['numeric']
chunk_size = 15
n_chunks = (len(numeric_vars) + chunk_size - 1) // chunk_size

for i in range(n_chunks):
    chunk = numeric_vars[i * chunk_size:(i + 1) * chunk_size]
    n_rows, n_cols = dslabs.define_grid(len(chunk))
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))
    axs = axs.flatten()

    for j, var in enumerate(chunk):
        ax = axs[j]
        data = df_fraud[var].dropna()
        low, high = data.quantile([0.01, 0.99])
        trimmed = data[(data >= low) & (data <= high)]
        ax.hist(trimmed, bins=30, edgecolor='black')
        ax.set_title(var, fontsize=9)
        ax.set_ylabel("Frequency")
        ax.tick_params(axis='x', labelrotation=45, labelsize=6)
        ax.tick_params(axis='y', labelsize=6)

    for j in range(len(chunk), len(axs)):
        axs[j].set_visible(False)

    fig.tight_layout()
    fig.subplots_adjust(hspace=0.8, top=0.92)
    fig.suptitle(f"Numeric Variable Distributions (Page {i+1})", fontsize=14, y=1.02)
    plt.show()


print(variable_types)

# Categorical Variables

# Skip symbolic vars with too many unique values (like IDs)
symbolic_filtered = [col for col in variable_types['symbolic'] if col != 'client_id' and col != 'invoice_id' and col != 'creation_date']

# Combine binary and filtered symbolic
categorical = variable_types['binary'] + symbolic_filtered
n_rows, n_cols = dslabs.define_grid(len(categorical))
fig_cat, axs_cat = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 3))

for i, var in enumerate(categorical):
    ax = axs_cat[i // n_cols, i % n_cols] if n_rows > 1 else axs_cat[i]
    counts = df_fraud[var].astype(str).value_counts(dropna=False).sort_index()

    dslabs.plot_bar_chart(
        xvalues=counts.index.tolist(),
        yvalues=counts.values.tolist(),
        ax=ax,
        title=var,
        xlabel=var,
        ylabel="Count",
        percentage=False
    )

    ax.tick_params(axis='x', labelrotation=45, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

for j in range(len(categorical), len(axs_cat.flatten())):
    axs_cat.flatten()[j].set_visible(False)


fig_cat.tight_layout()
fig_cat.suptitle("Categorical Variable Distributions (Independent Axes)", fontsize=14)
fig_cat.subplots_adjust(top=0.8)
plt.show()

# ---------------------------------
# Granularity

from dslabs import dslabs_functions as dslabs

# Temporal Granularity: derive and analyze invoice_date and creation_date ---
date_vars = [col for col in variable_types["date"] if col in df_fraud.columns]

# Derive new granularity columns: year, quarter, month, day
df_fraud = dslabs.derive_date_variables(df_fraud, date_vars)

# Plot granularity distributions
for date_col in date_vars:
    axs = dslabs.analyse_date_granularity(df_fraud, date_col, ["year", "quarter", "month", "day"])
    plt.tight_layout()
    plt.show()

# --- Spatial Granularity: analyze hierarchy if any exists ---
# If region and disrict and client_catg imply a spatial hierarchy, use:
spatial_hierarchy = [var for var in ["region", "disrict", "client_catg"] if var in df_fraud.columns]

if spatial_hierarchy:
    axs = dslabs.analyse_property_granularity(df_fraud, "client_location", spatial_hierarchy)
    plt.tight_layout()
    plt.show()

# target distribution
target_counts = df_fraud['target'].value_counts().sort_index()
plt.figure(figsize=(4, 3))
dslabs.plot_bar_chart(
    xvalues=target_counts.index.tolist(),
    yvalues=target_counts.values.tolist(),
    title="Target Distribution (Fraud vs Non-Fraud)",
    xlabel="Target",
    ylabel="Number of Clients",
    percentage=False
)
#plt.savefig("images/client_target_distribution.png")
plt.show()


# plot top features
top_features = ['n_invoices', 'activity_days', 'account_age_days',
                'consommation_level_1_mean', 'months_number_mean']

# --- Select top 24 numeric variables with highest cardinality ---
top_features = df_fraud[numeric_vars].nunique().sort_values(ascending=False).head(12).index.tolist()

n_rows, n_cols = dslabs.define_grid(len(top_features), vars_per_row=4)

fig, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
axs = axs.flatten()

for i, var in enumerate(top_features):
    ax = axs[i]
    df_fraud[[var]].boxplot(ax=ax)
    ax.set_title(var)
    ax.tick_params(axis='x', labelrotation=45, labelsize=7)
    ax.tick_params(axis='y', labelsize=7)

# Hide unused subplots if any
for j in range(len(top_features), len(axs)):
    axs[j].set_visible(False)

# Final layout adjustments
fig.tight_layout()
fig.suptitle("Boxplot of Top 24 Numeric Features", fontsize=14, y=1.02)
plt.show()

num_rows, num_cols = df_fraud.shape

plt.figure(figsize=(6, 4))
plt.bar(["Invoices", "Variables"], [num_rows, num_cols])
plt.title("Invoices vs Features")
plt.ylabel("Number of Items")
plt.xticks(rotation=45) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
for i, val in enumerate([num_rows, num_cols]):
    plt.text(i, val + max(num_rows, num_cols) * 0.02, str(val), ha='center')
plt.tight_layout()
plt.show()

missing_counts = df_fraud.isnull().sum()
missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
missing_counts.plot(kind='bar')
plt.title("Missing values per feature")
plt.ylabel("Number of missing values")
plt.xlabel("Features")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Select numeric variables
numeric_vars = df_fraud.select_dtypes(include=[np.number])

# Compute correlation matrix
corr_matrix = numeric_vars.corr()

# Plot heatmap without annotations
plt.figure(figsize=(16, 12))
sns.heatmap(
    corr_matrix,
    cmap="coolwarm",
    linewidths=0.5,
    cbar_kws={"shrink": 0.8}
)

plt.title("Correlation Heatmap of Numeric Features", fontsize=14)
plt.xticks(rotation=90, fontsize=7)
plt.yticks(rotation=0, fontsize=7)
plt.tight_layout()
plt.show()
