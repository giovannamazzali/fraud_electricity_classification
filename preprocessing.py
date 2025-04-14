import pandas as pd
import os

# Define paths
cwd = os.getcwd()
data_path = os.path.join(cwd, 'data')

# Load and label client data
client_train = pd.read_csv(os.path.join(data_path, 'client_train.csv'), index_col='client_id')
print(f"Client train data dimensionality: {client_train.shape}")
print(client_train.head())

# Load and label invoice data
invoice_train = pd.read_csv(os.path.join(data_path, 'invoice_train.csv'), index_col='client_id', low_memory=False)
print(f"Invoice train data dimensionality: {invoice_train.shape}")
print(invoice_train.head())

# Join information keeping invoice track intact
data = invoice_train.join(client_train, how='left', rsuffix='_client')
print(f"Final train data dimensionality: {data.shape}")
print(data.head())

# Save final df to csv
data.to_csv('merged_data.csv')