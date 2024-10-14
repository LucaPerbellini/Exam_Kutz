# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 19:10:27 2024

@author: lucap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD

def load_wind_speed_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name="Wind speed and Power")
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_time_series_matrix(df):
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
    df = df.dropna(subset=['Datetime'])
    df.set_index('Datetime', inplace=True)
    
    wind_speed = df['Wind speed (m/s)']
    
    # Resample the data to get weekly data
    weekly_data = wind_speed.resample('W').apply(lambda x: x.tolist()).dropna()

    ## DROP NAN ROWS
    # Convert to a DataFrame and pad with NaN for weeks with incomplete data
    max_week_length = max(weekly_data.apply(len))
    weekly_data_padded = weekly_data.apply(lambda x: x + [np.nan] * (max_week_length - len(x)))

    # Convert to a 2D NumPy array and drop any rows with NaN values
    X = np.array(weekly_data_padded.tolist())
    X = X[~np.isnan(X).any(axis=1)]  # Remove rows with NaN
        
    return X

def plot_reconstruction(original_series, reconstructed_series):
    original_series = df['Wind speed (m/s)']
    original_series.index = df['Datetime'] 
    
    # Create a time index for the reconstructed series
    start_date = 144 # '2023-01-02 00:00:00'
    end_date = 49392+144 # '2023-12-15 17:50:00'
    
    # Flatten the reconstructed data and adjust for the time index
    reconstructed_series_flat = pd.DataFrame(reconstructed_series.flatten(), columns=['Reconstructed Wind Speed (m/s)'], index=original_series.index[start_date:end_date])
    
    plt.figure(figsize=(12, 6))
    plt.plot(original_series.index, original_series, label='Original Time Series', alpha=0.5)
    plt.plot(reconstructed_series_flat.index, reconstructed_series_flat, label='Reconstructed Time Series', color='red')
    plt.title('Original vs. Reconstructed Wind Speed Time Series')
    plt.xlabel('Time Index')
    plt.ylabel('Wind Speed (m/s)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Example usage:
file_path = "C:/Users/lucap/OneDrive - Politecnico di Milano/Documenti/NK_esame/GECAD renewable energy lab _Wind power data.xls"

df = load_wind_speed_data(file_path)

if df is not None:
    X = create_time_series_matrix(df)
    
    # Apply DMD
    dmd = DMD(svd_rank=0)  # Adjust svd_rank as needed
    dmd.fit(X)

    # Reconstruct the dynamics
    reconstructed_data = dmd.reconstructed_data.real

    # Plot the original and reconstructed wind speed time series
    plot_reconstruction(df, reconstructed_data)
    
    

