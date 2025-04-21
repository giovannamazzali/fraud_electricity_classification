import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from dslabs import config
from dslabs import dslabs_functions as dslabs
import matplotlib.pyplot as plt

# Load data
data_path = os.getcwd()
df_fraud = pd.read_csv(os.path.join(data_path, 'merged_data.csv'), index_col='client_id')
print(f"Fraud data dimensionality: {df_fraud.shape}\n")

# Column names and types
print("Column data types:")
print(df_fraud.dtypes, "\n")

# Dataset head (first 5 rows)
print("First 5 rows:")
print(df_fraud.head(), "\n")

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

# Visualize missing percentages
if not missing_summary.empty:
    fig, ax = plt.subplots(figsize=(10, 5))
    dslabs.plot_bar_chart(
        xvalues=missing_percent.index.tolist(),
        yvalues=missing_percent.tolist(),
        ax=ax,
        title="Missing Values by Column (%)",
        xlabel="Columns",
        ylabel="Percentage",
        percentage=True
    )
    plt.tight_layout()
    plt.show()
else:
    print("No missing values found.")

# Identify variables types -- numeric, categorical
var_types = dslabs.get_variable_types(df_fraud)

print("Variable Types:\n")
for var_type, columns in var_types.items():
    print(f"{var_type.title()} Variables ({len(columns)}):")
    print(columns, "\n")

print(df_fraud['counter_statue'].unique())

# Numeric Variables
numeric = var_types['numeric']
n_rows, n_cols = dslabs.define_grid(len(numeric))
fig_num, axs_num = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))

for i, var in enumerate(numeric):
    ax = axs_num[i // n_cols, i % n_cols] if n_rows > 1 else axs_num[i]
    ax.hist(df_fraud[var].dropna(), bins=30, edgecolor='black')
    ax.set_title(var, fontsize=9)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis='x', labelrotation=45, labelsize=6)
    ax.tick_params(axis='y', labelsize=6)

fig_num.tight_layout()
fig_num.suptitle("Numeric Variable Distributions (Independent Axes)", fontsize=14, y=1.02)
plt.show()


# Categorical Variables
categorical = var_types['symbolic'] + var_types['binary']
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

# Hide unused subplots (if any)
for j in range(len(categorical), len(axs_cat.flatten())):
    axs_cat.flatten()[j].set_visible(False)

fig_cat.tight_layout()
fig_cat.suptitle("Categorical Variable Distributions (Independent Axes)", fontsize=14, y=1.02)
plt.show()

