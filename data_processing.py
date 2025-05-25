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
df_raw = pd.read_csv(os.path.join(data_path, 'merged_data.csv'))

print(f"Data dimensions: {df_raw.shape}\n")

# Clean 'counter_statue'
df_raw['counter_statue'] = df_raw['counter_statue'].astype(str).str.strip()
df_raw = df_raw[df_raw['counter_statue'] != 'A']
df_raw['counter_statue'] = pd.to_numeric(df_raw['counter_statue'], errors='coerce')

# Get variable types
variable_types = dslabs.get_variable_types(df_raw)
print("Variable types detected:")
print(variable_types)

client_id_col = 'client_id'

# Check symbolic/binary consistency
symbolic_binary_vars = variable_types['symbolic'] + variable_types['binary']
inconsistent_vars = []

for var in symbolic_binary_vars:
    nunique_per_client = df_raw.groupby(client_id_col)[var].nunique()
    inconsistent_count = (nunique_per_client > 1).sum()
    
    if inconsistent_count > 0:
        inconsistent_vars.append(var)
        print(f"[Warning] Variable '{var}' has {inconsistent_count} inconsistent clients.")

# Optional: drop inconsistent clients
# (currently skipped — you can re-enable if desired)

# Update variable types after cleanup
variable_types = dslabs.get_variable_types(df_raw)

# Identify fixed variables
grouped_nunique = df_raw.groupby(client_id_col).nunique()
fixed_counts = (grouped_nunique == 1).sum()
n_clients = df_raw[client_id_col].nunique()
fixed_percent = (fixed_counts / n_clients * 100).round(2)

dtypes = df_raw.dtypes.astype(str)
fixed_summary = pd.DataFrame({
    "Type": dtypes,
    "Fixed Count": fixed_counts,
    "Fixed %": fixed_percent
}).sort_values(by="Fixed %", ascending=False)

fixed_vars = fixed_summary[fixed_summary["Fixed %"] == 100.0].index.tolist()

# Numeric aggregation
agg_numeric = [v for v in variable_types["numeric"] if v not in fixed_vars]
agg_funcs = ['mean', 'std', 'min', 'max', 'median',
             lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
agg_dict = {v: agg_funcs for v in agg_numeric}

df_numeric_agg = df_raw.groupby(client_id_col).agg(agg_dict)
df_numeric_agg.columns = ['_'.join([c[0], c[1] if isinstance(c[1], str) else c[1].__name__]) for c in df_numeric_agg.columns]

# Symbolic aggregation
def get_mode(x): return x.mode().iloc[0] if not x.mode().empty else np.nan
def get_freq_ratio(x): return x.value_counts(normalize=True).iloc[0] if not x.empty else np.nan
def get_entropy(x): return entropy(x.value_counts(normalize=True), base=2) if not x.empty else np.nan

symbolic_vars = [v for v in variable_types['symbolic'] + variable_types['binary']
                 if v not in fixed_vars and v != client_id_col]

agg_cat = {
    v: ['nunique', get_mode, get_freq_ratio, get_entropy]
    for v in symbolic_vars
}

df_cat_agg = df_raw.groupby(client_id_col).agg(agg_cat)
df_cat_agg.columns = ['_'.join([c[0], c[1] if isinstance(c[1], str) else c[1].__name__]) for c in df_cat_agg.columns]

# Fixed variables
df_fixed = df_raw.groupby(client_id_col)[fixed_vars].first()

# DATE-based aggregations

# Invoice date: activity range + number of records
df_dates = df_raw.groupby(client_id_col).agg({
    'invoice_date': ['min', 'max', 'count']
})
df_dates.columns = ['invoice_date_min', 'invoice_date_max', 'n_invoices']

# Optional: convert invoice_date range to duration (e.g., in days)
df_dates['activity_days'] = (df_dates['invoice_date_max'] - df_dates['invoice_date_min']).dt.days

# Optional: calculate account age using creation_date (only if it's fixed)
if 'creation_date' in fixed_vars:
    # Use latest invoice date as reference point
    latest_invoice = df_raw['invoice_date'].max()
    creation_dates = df_fixed['creation_date']
    df_dates['account_age_days'] = (latest_invoice - creation_dates).dt.days

# Merge everything
df_client = pd.concat([df_fixed, df_numeric_agg, df_cat_agg, df_dates], axis=1).reset_index()

print(f"\nFinal client-level dataset shape: {df_client.shape}")
print(df_client.head())

# Save to CSV
df_client.to_csv("client_level_dataset.csv", index=False)
