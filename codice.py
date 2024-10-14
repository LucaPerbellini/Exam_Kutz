# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 23:35:40 2024

@author: lucap
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load the Excel file into a pandas DataFrame
df = pd.read_excel('C:/Users/lucap/OneDrive - Politecnico di Milano/Documenti/NK_esame/GECAD renewable energy lab _Wind power data.xls', sheet_name='Wind speed and Power')

# Create a new column for the combined datetime
df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
    
# Check for any rows with NaT (not a time) in the Datetime column
invalid_dates = df[df['Datetime'].isna()]
if not invalid_dates.empty:
    print("Warning: The following rows have invalid dates:")
    print(invalid_dates[['Year', 'Month', 'Day', 'Hour', 'Minute']])

# Drop rows with invalid dates
df = df.dropna(subset=['Datetime'])

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(df['Datetime'], df['Wind speed (m/s)'], marker='o', linestyle='-', markersize=2)
plt.title('Wind Speed Over Time')
plt.xlabel('Time')
plt.ylabel('Wind Speed (m/s)')
plt.xticks(rotation=45)
plt.grid()
plt.tight_layout()
plt.show()

# Drop all columns except for 'Datetime' and 'Wind speed (m/s)'
df_filtered = df[['Datetime', 'Wind speed (m/s)']]

def plot_average_daily_wind_speed(df):
    # Create a new column for the combined datetime
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['Datetime'])

    # Extract day and month for grouping
    df['Day'] = df['Datetime'].dt.day
    df['Month'] = df['Datetime'].dt.month

    # Group by day and calculate the average wind speed for each day of each month
    average_daily_wind_speed = df.groupby(['Month', 'Day'])['Wind speed (m/s)'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(12, 6))
    for month in average_daily_wind_speed['Month'].unique():
        monthly_data = average_daily_wind_speed[average_daily_wind_speed['Month'] == month]
        plt.plot(monthly_data['Day'], monthly_data['Wind speed (m/s)'], marker='o', label=f'Month {month}')

    plt.title('Average Daily Wind Speed by Month')
    plt.xlabel('Day of the Month')
    plt.ylabel('Average Wind Speed (m/s)')
    plt.xticks(range(1, 32))  # Show all days in a month
    plt.xlim(1, 31)  # Set x-axis limits
    plt.legend(title='Month')
    plt.grid()
    plt.tight_layout()
    plt.show()

plot_average_daily_wind_speed(df)

def calculate_average_monthly_wind_speed(df):
    # Create a new column for the combined datetime
    df['Datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour', 'Minute']], errors='coerce')
    
    # Drop rows with invalid dates
    df = df.dropna(subset=['Datetime'])

    # Extract month for grouping
    df['Month'] = df['Datetime'].dt.month

    # Group by month and calculate the average wind speed
    average_monthly_wind_speed = df.groupby('Month')['Wind speed (m/s)'].mean().reset_index()

    return average_monthly_wind_speed


average_monthly_wind_speed = calculate_average_monthly_wind_speed(df)
print(average_monthly_wind_speed)


# Group by month and hour, then calculate the average wind speed
average_monthly_hourly = df.groupby([df.Month, df.Hour])['Wind speed (m/s)'].mean()

# Convert the result to a DataFrame for easier manipulation
average_monthly_hourly_df = average_monthly_hourly.unstack(level=0)

# Rename the columns to represent the months
average_monthly_hourly_df.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

# Display the average wind speed for each hour of the average monthly day
print(average_monthly_hourly_df)


# Set the figure size
plt.figure(figsize=(12, 6))

# Plot each month's average wind speed
for month in average_monthly_hourly_df.columns:
    plt.plot(average_monthly_hourly_df.index, average_monthly_hourly_df[month], marker='o', label=month)

# Add titles and labels
plt.title('Average Hourly Wind Speed for Each Month', fontsize=16)
plt.xlabel('Hour of Day', fontsize=14)
plt.ylabel('Average Wind Speed (m/s)', fontsize=14)
plt.xticks(range(24), [f"{hour}:00" for hour in range(24)])  # Label x-axis with hours
plt.grid()
plt.legend(title='Month')
plt.tight_layout()

# Show the plot
plt.show()


