# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:55:26 2024

@author: lucap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Load the data
file_path = "C:/Users/lucap/OneDrive - Politecnico di Milano/Documenti/NK_esame/GECAD renewable energy lab _Wind power data.xls"
data = pd.read_excel(file_path, sheet_name='Wind speed and Power')

# Create a datetime column
data['Datetime'] = pd.to_datetime(data[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
# Set the datetime as the index
data.set_index('Datetime', inplace=True)
# Select the wind speed data
wind_speed = data['Wind speed (m/s)']

# Resample the data to get weekly data
weekly_data = wind_speed.resample('W').apply(lambda x: x.tolist()).dropna()

## DROP NAN ROWS
# Convert to a DataFrame and pad with NaN for weeks with incomplete data
max_week_length = max(weekly_data.apply(len))
weekly_data_padded = weekly_data.apply(lambda x: x + [np.nan] * (max_week_length - len(x)))

# Convert to a 2D NumPy array and drop any rows with NaN values
X = np.array(weekly_data_padded.tolist())
X = X[~np.isnan(X).any(axis=1)]  # Remove rows with NaN

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA(n_components=49)
X_pca = pca.fit_transform(X_scaled)

# Calcola la varianza spiegata
explained_variance = pca.explained_variance_ratio_

# Crea lo scree plot
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Numero di Componenti Principali')
plt.ylabel('Varianza Spiegata')
plt.xticks(range(1, len(explained_variance) + 1))
plt.grid()
plt.show()


# Reconstruct the data
# variance_threshold = 0.90  # Define the threshold for explained variance
# # Determine the number of components to keep
# num_components = np.sum(explained_variance.cumsum() <= variance_threshold)
num_components = 8

# Perform PCA
pca = PCA(n_components=num_components)
X_pca = pca.fit_transform(X_scaled)

X_reconstructed = pca.inverse_transform(X_pca)
X_reconstructed = scaler.inverse_transform(X_reconstructed)
X_reconstructed_col = X_reconstructed.reshape(-1, 1)

start_date = 144 # '2023-01-02 00:00:00'
end_date = 49*1008+144 # '2023-10-9 23:50:00'
# Create a DataFrame for plotting
reconstructed_df = pd.DataFrame(X_reconstructed_col, columns=['Reconstructed Wind Speed (m/s)'], index=data.index[start_date:end_date])

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(data.index, wind_speed, label='Original Wind Speed', color='blue')
plt.plot(reconstructed_df.index, reconstructed_df['Reconstructed Wind Speed (m/s)'], label='Reconstructed Wind Speed', color='red', linestyle='--')
plt.title('PCA: Original vs Reconstructed Wind Speed')
plt.xlabel('Week')
plt.ylabel('Wind Speed (m/s)')
plt.legend()
plt.grid()
plt.show()
