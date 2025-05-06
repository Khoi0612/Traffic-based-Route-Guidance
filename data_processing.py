import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
import os
import joblib

def process_data(filename, sheet_name='Data'):
    # Read the Excel file
    raw_data = pd.read_excel(filename, sheet_name=sheet_name, header=None)

    # Extract the list of time
    time_list = raw_data.iloc[0, 10:].tolist()

    # Initialize lists for each parameter
    timestamps = []
    scats_numbers = []
    locations = []
    traffic_flows = []

    # Start reading data from row 2 (after time and column headers)
    for row in range(2, len(raw_data)):
        scat_no = raw_data.iloc[row, 0]
        location = raw_data.iloc[row, 1]
        date = raw_data.iloc[row, 9].date()  # Date only

        # Loop through all time intervals and collect data
        for i, time in enumerate(time_list):
            timestamps.append(datetime.combine(date, time))
            scats_numbers.append(scat_no)
            locations.append(location)
            traffic_flows.append(raw_data.iloc[row, 10 + i])

    # Create the final processed DataFrame
    processed_data = pd.DataFrame({
        'timestamp': timestamps,
        'scats_number': scats_numbers,
        'location': locations,
        'traffic_flow': traffic_flows
    })

    return processed_data

def split_data_into_xlsx_file(processed_data, window_size, file_dir="data"):
    # Group data by the specified column and get a specific group
    grouped = processed_data.groupby('location')

    for group_name, group_df in grouped:
        df = group_df.copy()

        # Set timestamp as the index
        df['timestamp'] = pd.to_datetime(group_df['timestamp'])
        df.set_index('timestamp', inplace=True)

        # Extract traffic flow 
        traffic_flow = df['traffic_flow'].astype(float).values.reshape(-1, 1)
        if len(traffic_flow) <= window_size:
            print(f"Skipping group {group_name} due to insufficient data.")
            continue

        # Extract target dates
        target_dates = df.index[window_size:]

        # Normalize the traffic flow values to range [0, 1]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_traffic_flow = scaler.fit_transform(traffic_flow)

        # Prepare sliding window input (x) and corresponding labels (y)
        x, y = [], []
        for i in range(window_size, len(scaled_traffic_flow)):
            x.append(scaled_traffic_flow[i - window_size:i, 0])
            y.append(scaled_traffic_flow[i, 0])

        # Convert to DataFrame for export
        x_df = pd.DataFrame(x, columns=[f't-{j}' for j in range(window_size, 0, -1)])
        x_df['target'] = y
        x_df['timestamp'] = target_dates.values
        
        # Export to Excel
        filename = os.path.join(file_dir, f"{group_name}.xlsx")
        x_df.to_excel(filename, index=False)
        print(f"Exported group {group_name} to {filename}")
        
        # Save the scaler
        joblib.dump(scaler, 'scaler.save')

def run_process_data():
    filename = os.path.join('data', 'Scats Data October 2006.xls')
    data = process_data(filename) 
    split_data_into_xlsx_file(processed_data=data, window_size=5)

run_process_data()