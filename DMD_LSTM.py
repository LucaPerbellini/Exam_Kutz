# -*- coding: utf-8 -*-
"""
Created on Sun Oct 13 23:07:16 2024

@author: lucap
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from keras.models import Sequential
from keras.layers import LSTM, Dense
# from pydmd import DMD

class DMD:
    def __init__(self, svd_rank=None):
        self.svd_rank = svd_rank  # Rank for truncation (optional)
        self.A_tilde = None  # Approximation of the system matrix
        self.eigenvalues = None
        self.modes = None

    def fit(self, X):
        """
        Apply DMD to the input data X.
        X should be a matrix where columns represent snapshots of the time series.
        """
        X1 = X[:, :-1]  # First time snapshot matrix
        X2 = X[:, 1:]   # Second time snapshot matrix

        # Step 1: SVD of X1
        U, Sigma, Vh = svd(X1, full_matrices=False)

        if self.svd_rank:
            # Truncate to rank r
            U = U[:, :self.svd_rank]
            Sigma = np.diag(Sigma[:self.svd_rank])
            Vh = Vh[:self.svd_rank, :]

        # Step 2: Compute the A_tilde matrix
        self.A_tilde = U.T @ X2 @ Vh.T @ np.linalg.inv(Sigma)

        # Step 3: Eigen decomposition of A_tilde
        self.eigenvalues, W = np.linalg.eig(self.A_tilde)
        self.modes = X2 @ Vh.T @ np.linalg.inv(Sigma) @ W

    def reduce(self, X):
        """
        Reduce the dimensionality of the time series using the DMD modes.
        """
        # Ensure X has the correct shape and return the projection onto the reduced DMD modes
        return self.modes.T @ X[:, :-1]

def load_wind_speed_data(file_path):
    try:
        df = pd.read_excel(file_path, sheet_name="Wind speed and Power")
        return df
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def create_dataset(data, time_step, n_future):
    X, y = [], []
    for i in range(len(data) - time_step - n_future + 1):
        X.append(data[i:(i + time_step)])
        y.append(data[i + time_step: (i + time_step + n_future)])
    return np.array(X), np.array(y)

# Example usage:
file_path = "C:/Users/lucap/OneDrive - Politecnico di Milano/Documenti/NK_esame/GECAD renewable energy lab _Wind power data.xls"  # Replace with your actual file path
time_step = 10 # Number of time steps to use for prediction
n_future = 1   # Number of time steps to predict

df = load_wind_speed_data(file_path)
X_train = df['Wind speed (m/s)'].values
X_train = X_train.reshape(1, -1)  # Reshape into (1, n_timesteps)

# Apply DMD for dimensionality reduction
dmd = DMD(svd_rank=5)  # Reduce to 5 modes
dmd.fit(X_train)

# Reduce the original time series data
X_reduced = dmd.reduce(X_train)
print("Shape of reduced data:", X_reduced.shape)

def create_lstm_dataset(data, time_steps):
    X, y = [], []
    for i in range(len(data[0]) - time_steps):
        X.append(data[:, i:i+time_steps].T)  # Take 'time_steps' of data as input
        y.append(data[:, i+time_steps])  # Predict the next step
    return np.array(X), np.array(y)

time_steps = 10  # Choose the number of time steps for the LSTM
X_lstm_train, y_lstm_train = create_lstm_dataset(X_reduced, time_steps)

print("LSTM training data shape:", X_lstm_train.shape, y_lstm_train.shape)


# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_lstm_train.shape[1], X_lstm_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(y_lstm_train.shape[1]))  # Output layer (equal to the number of reduced dimensions)

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM
model.fit(X_lstm_train, y_lstm_train, epochs=50, batch_size=32)

# Predict future time steps
X_lstm_test, y_lstm_test = create_lstm_dataset(X_reduced, time_steps)
predicted_reduced = model.predict(X_lstm_test)

# Reconstruct the full data using DMD modes
X_reconstructed = dmd.modes @ predicted_reduced.T
X_reconstructed = X_reconstructed.T  # Transpose to get back original shape

print("Shape of reconstructed data:", X_reconstructed.shape)

# Plot the Reconstructed vs Real Time Series
plt.figure(figsize=(10, 6))
plt.plot(X_train[0, time_steps:], label='True Time Series', color='blue', alpha=0.5)
plt.plot(X_reconstructed[:, 0], label='Reconstructed Time Series', color='red', linestyle='--')
plt.xlabel('Time Step')
plt.ylabel('Value')
plt.title('True vs Reconstructed Time Series')
plt.legend()
plt.show()