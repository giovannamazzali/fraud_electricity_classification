import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from dslabs import config
from dslabs import dslabs_functions as dslabs
import matplotlib.pyplot as plt
import seaborn as sns
import calendar

# Load data
data_path = os.getcwd()
df_fraud = pd.read_csv(os.path.join(data_path, 'merged_data.csv'))  # Removed index_col for now

# Dataset head
print("First 5 rows:")
print(df_fraud.head(), "\n")

# ---------------------------------
# Dimensionality

print(f"Fraud data dimensionality: {df_fraud.shape}\n")

# Convert invoice_date to datetime format and handle parsing issues
df_fraud['invoice_date'] = pd.to_datetime(df_fraud['invoice_date'], errors='coerce', dayfirst=True)

# Clean counter_statue: unify formats but preserve values
df_fraud['counter_statue'] = df_fraud['counter_statue'].astype(str).str.strip()
df_fraud['counter_statue'] = df_fraud['counter_statue'].replace({
    '0.0': '0', '1.0': '1', '2.0': '2', '3.0': '3', '4.0': '4', '5.0': '5'
})

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

# Numeric Variables
numeric = variable_types['numeric']
n_rows, n_cols = dslabs.define_grid(len(numeric))
fig_num, axs_num = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))

for i, var in enumerate(numeric):
    ax = axs_num[i // n_cols, i % n_cols] if n_rows > 1 else axs_num[i]
    ax.hist(df_fraud[var].dropna(), bins=30, edgecolor='black')
    ax.set_title(var, fontsize=9)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', labelrotation=45, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

# Adjust layout
fig_num.tight_layout()
fig_num.subplots_adjust(hspace=0.99, top=0.88, bottom=0.05)  # hspace ↑ spacing between plots, top ↓ space for title

# Title
fig_num.suptitle("Numeric Variable Distributions", fontsize=14, y=1)

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


df_fraud.to_csv(os.path.join(data_path, 'df_fraud.csv'), index=False)