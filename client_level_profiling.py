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
df_fraud = pd.read_csv(os.path.join(data_path, 'df_fraud.csv'))  # No index_col here
df_fraud = df_fraud.sample(2000, random_state=42)

# ------------------------------------------------------------------
# Client-level aggregation

variable_types = dslabs.get_variable_types(df_fraud)
client_id_col = 'client_id'

# Count unique values per client
grouped_nunique = df_fraud.groupby(client_id_col).nunique()
fixed_counts = (grouped_nunique == 1).sum()
n_clients = df_fraud[client_id_col].nunique()
fixed_percent = (fixed_counts / n_clients * 100).round(2)

# Build type info
dtypes = df_fraud.dtypes.astype(str)
fixed_summary = pd.DataFrame({
    "Type": dtypes,
    "Fixed Count": fixed_counts,
    "Fixed %": fixed_percent
}).sort_values(by="Fixed %", ascending=False)
print("Variable Types and Fixedness per Client:")
print(fixed_summary)

# Get fixed variable names
fixed_vars = fixed_summary[fixed_summary["Fixed %"] == 100.0].index.tolist()

# --------------------
# NUMERIC aggregation
agg_numeric = [var for var in variable_types["numeric"] if var not in fixed_vars]
agg_funcs = ['mean', 'std', 'min', 'max', 'median',
             lambda x: x.quantile(0.25),  # q1
             lambda x: x.quantile(0.75)]  # q3
agg_dict = {var: agg_funcs for var in agg_numeric}

df_numeric_agg = df_fraud.groupby(client_id_col).agg(agg_dict)
df_numeric_agg.columns = ['_'.join([col[0], col[1] if isinstance(col[1], str) else col[1].__name__])
                          for col in df_numeric_agg.columns]

# --------------------
# SYMBOLIC aggregation with mode, freq_ratio, entropy, nunique

def get_mode(x):
    return x.mode().iloc[0] if not x.mode().empty else np.nan

def get_freq_ratio(x):
    counts = x.value_counts(normalize=True)
    return counts.iloc[0] if not counts.empty else np.nan

def get_entropy(x):
    counts = x.value_counts(normalize=True)
    return entropy(counts, base=2) if not counts.empty else np.nan

symbolic_vars = [var for var in variable_types["symbolic"] + variable_types["binary"]
                 if var not in fixed_vars and var != client_id_col]
agg_cat = {
    var: ['nunique', get_mode, get_freq_ratio, get_entropy]
    for var in symbolic_vars
}
df_cat_agg = df_fraud.groupby(client_id_col).agg(agg_cat)
df_cat_agg.columns = ['_'.join([col[0], col[1] if isinstance(col[1], str) else col[1].__name__])
                      for col in df_cat_agg.columns]

# --------------------
# FIXED vars (pick first per client)
df_fixed = df_fraud.groupby(client_id_col, as_index=True)[fixed_vars].first()

# --------------------
# FINAL MERGE (index must match: all indexed by client_id)
df_client = pd.concat([df_fixed, df_numeric_agg, df_cat_agg], axis=1).reset_index()

print(f"Final shape: {df_client.shape}")
print(df_client.head())
