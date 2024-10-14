# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 18:51:56 2024

@author: lucap
"""

import pandas as pd 


# Specify the path to the Excel file
file_path = "C:/Users/lucap/OneDrive - Politecnico di Milano/Documenti/NK_esame/Risultati.xlsx"  # Replace with your actual file path

# Name for the new column
new_column_name = 'PCA Reconstruction'  # Customize this name as needed

reconstructed_df = reconstructed_data_original_scale.flatten()
start=144
end=49248+start+144
reconstructed_series = pd.DataFrame(reconstructed_df['Reconstructed Wind Speed (m/s)'].values, columns=['Reconstructed Wind Speed (m/s)'], index=data[start:end].index)
# 
# Load existing data if the file already exists, else create a new one
try:
    existing_data = pd.read_excel(file_path, index_col=0)
    # Append new reconstructed values as a new column
    existing_data[new_column_name] = reconstructed_series['Reconstructed Wind Speed (m/s)']
    # Save the updated DataFrame back to Excel
    existing_data.to_excel(file_path)
except FileNotFoundError:
    # If the file does not exist, create a new Excel file
    reconstructed_series.to_excel(file_path)

print(f"Reconstructed values saved to {file_path} in column '{new_column_name}'.")
