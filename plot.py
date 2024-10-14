# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 22:21:43 2024

@author: lucap
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

def load_wind_speed_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name="Wind speed and Power")
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
file_path = "C:/Users/lucap/OneDrive - Politecnico di Milano/Documenti/NK_esame/GECAD renewable energy lab _Wind power data.xls"  # Replace with your actual file path

# Load the Excel file into a pandas DataFrame
df = pd.read_excel('C:/Users/lucap/OneDrive - Politecnico di Milano/Documenti/NK_esame/GECAD renewable energy lab _Wind power data.xls', sheet_name='Wind speed and Power')

# Create a new column for the combined datetime
df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')

# Load the data from Excel
file_path_results = r"C:\Users\lucap\OneDrive - Politecnico di Milano\Documenti\NK_esame\Risultati.xlsx" # Replace with your actual file path
data = pd.read_excel(file_path_results)

# Extract relevant columns
true_values_par = df['Wind speed (m/s)'].values  # Replace with the actual name of the true wind speed column
true_values = true_values_par[:49536-144] 
lstm_reconstruction = data['DMD_LSTM Reconstruction'].values
pca_reconstruction = data['PCA Reconstruction'].values
autoencoder_reconstruction = data['Autoencoder Reconstruction'].values
dmd_reconstruction = data['DMD Reconstruction'].values

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def smape(y_true, y_pred):
    return 100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + 1e-5))


# Calculate metrics for each reconstruction method
metrics = {
    'PCA': {
        'RMSE': np.sqrt(mean_squared_error(data["Reconstructed Wind Speed (m/s)"].values, pca_reconstruction)),
        'MAPE': smape(data["Reconstructed Wind Speed (m/s)"].values, pca_reconstruction)
    },
    'Autoencoder': {
        'RMSE': np.sqrt(mean_squared_error(true_values, autoencoder_reconstruction)),
        'MAPE': smape(true_values, autoencoder_reconstruction)
    },
    'DMD': {
        'RMSE': np.sqrt(mean_squared_error(true_values, dmd_reconstruction)),
        'MAPE': smape(true_values, dmd_reconstruction)
    },
    'DMD_LSTM': {
        'RMSE': np.sqrt(mean_squared_error(true_values, lstm_reconstruction)),
        'MAPE': smape(true_values, lstm_reconstruction)
    }
}

# Display the metrics
for method, values in metrics.items():
    print(f"{method} Reconstruction:")
    print(f"  RMSE: {values['RMSE']:.4f}")
    print(f"  MAPE: {values['MAPE']:.4f}%")
    print()
    
methods = ['PCA Reconstruction', 'Autoencoder Reconstruction', 'DMD Reconstruction', 'DMD_LSTM Reconstruction']

# Set up the plot
plt.figure(figsize=(14, 10))

# Loop through each method and plot
for i, method in enumerate(methods, start=1):
    plt.subplot(2, 2, i)  # Create a 2x2 grid of subplots
    plt.plot(data['Datetime'], true_values, label='True Wind Speed', alpha=0.5)
    plt.plot(data['Datetime'], data[method].values, label=method, color='red', alpha=0.75)
    plt.title(f'Real vs. {method}')
    plt.xlabel('Datetime')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()


# Set up the plot
plt.figure(figsize=(14, 10))

# Loop through each method and plot
for i, method in enumerate(methods, start=1):
    plt.subplot(2, 2, i)  # Create a 2x2 grid of subplots
    plt.plot(data['Datetime'][1008:2016], true_values[1008:2016], label='True Wind Speed', alpha=0.5)
    plt.plot(data['Datetime'][1008:2016], data[method].values[1008:2016], label=method, color='red', alpha=0.75)
    plt.title(f'Real vs. {method}')
    plt.xlabel('Datetime')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()



