# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:56:03 2024

@author: lucap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

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

def build_autoencoder(latent_dim, input_shape):
    # Encoder
    encoder_input = layers.Input(shape=input_shape)
    x = layers.Dense(512, activation='relu')(encoder_input)
    x = layers.Dense(256, activation='relu')(x)
    latent_space = layers.Dense(latent_dim, activation='relu')(x)

    # Decoder
    x = layers.Dense(256, activation='relu')(latent_space)
    x = layers.Dense(512, activation='relu')(x)
    decoder_output = layers.Dense(input_shape[0], activation='sigmoid')(x)

    # Autoencoder model
    autoencoder = keras.Model(encoder_input, decoder_output)
    encoder = keras.Model(encoder_input, latent_space)

    return autoencoder, encoder

def train_autoencoder(autoencoder, X, epochs=50, batch_size=32):
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, shuffle=True, validation_split=0.2)

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
latent_dim = 8  # Define the latent space dimension

df = load_wind_speed_data(file_path)

if df is not None:
    X = create_time_series_matrix(df)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Build and train the autoencoder
    autoencoder, encoder = build_autoencoder(latent_dim, X_scaled.shape[1:])
    train_autoencoder(autoencoder, X_scaled, epochs=1000, batch_size=32)

# Example usage after training the autoencoder
if df is not None:
    # Use the autoencoder to reconstruct the data
    reconstructed_data = autoencoder.predict(X_scaled)
    reconstructed_data_original_scale = scaler.inverse_transform(reconstructed_data)
    
    # Plot the original and reconstructed wind speed time series
    plot_reconstruction(df, reconstructed_data_original_scale)

