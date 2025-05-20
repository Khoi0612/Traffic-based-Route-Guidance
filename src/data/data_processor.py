import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from collections import defaultdict
import re

import os
import joblib


class DataProcessor:
    
    def __init__(self, raw_data_path=None, sheet_name='Data', output_dir=None):
        self.raw_data_path = raw_data_path or os.path.join('..', 'data', 'raw', 'Scats Data October 2006.xls')
        self.sheet_name = sheet_name
        self.processed_dir = output_dir or os.path.join('..', 'data', 'processed')
        self.models_dir = os.path.join('..', 'data', 'models')
        self.data = self._load_data()
        
        # Create directories if they don't exist
        for directory in [self.processed_dir, self.models_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)

    def _load_data(self):
        return pd.read_excel(self.raw_data_path, sheet_name=self.sheet_name, header=None)
    
    def process_data(self):
        # Read the Excel file
        raw_data = self.data

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
    
    def split_data_into_xlsx_file(self, processed_data, window_size):
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
            location_safe_name = group_name.replace('/', '_').replace('\\', '_')
            filename = os.path.join(self.processed_dir, f"{location_safe_name}.xlsx")
            x_df.to_excel(filename, index=False)
            print(f"Exported group {group_name} to {filename}")
            
            # Save the scaler
            scaler_path = os.path.join(self.models_dir, f"scaler.save")
            joblib.dump(scaler, scaler_path)

    def run_process_data(self, window_size=5):
        print(f"Processing data from {self.raw_data_path}...")
        data = self.process_data() 
        print(f"Splitting data into location-specific files with window size {window_size}...")
        self.split_data_into_xlsx_file(processed_data=data, window_size=window_size)
        print("Data processing complete!")

    def extract_node_coordinates(self):
        node_coordinates = {}
        
        for row in range(2, len(self.data)):  # Start from row 2 to skip headers
            try:
                scats_id = self.data.iloc[row, 0]
                latitude = self.data.iloc[row, 3]
                longitude = self.data.iloc[row, 4]

                # Skip if any value is missing or 0
                if (
                    pd.isna(scats_id) or 
                    pd.isna(latitude) or 
                    pd.isna(longitude) or 
                    latitude == 0 or 
                    longitude == 0
                ):
                    continue

                if scats_id not in node_coordinates:
                    node_coordinates[scats_id] = {'sum_lat': 0, 'sum_long': 0, 'count': 0}
                
                node_coordinates[scats_id]['sum_lat'] += latitude
                node_coordinates[scats_id]['sum_long'] += longitude
                node_coordinates[scats_id]['count'] += 1

            except (IndexError, TypeError):
                continue

        # Compute average coordinates for each node
        nodes = {
            site: [values['sum_lat'] / values['count'], values['sum_long'] / values['count']]
            for site, values in node_coordinates.items()
        }

        return nodes
    
    def extract_street_connections(self):
        # Pattern to match location descriptions like "MAIN ST N of CROSS ST"
        pattern = r'(.+?)\s+(N|NE|E|SE|S|SW|W|NW)\s+of\s+(.+)'
        
        # Store node information: node_id -> [(main_street, location, direction)]
        node_info = defaultdict(list)
        
        # Collect all nodes with their street information
        for idx in range(2, len(self.data)):
            try:
                scats_id = self.data.iloc[idx, 0]
                location_text = str(self.data.iloc[idx, 1])
                
                # Skip if data is missing
                if pd.isna(scats_id) or pd.isna(location_text):
                    continue
                    
                match = re.match(pattern, location_text, re.IGNORECASE)
                if not match:
                    continue

                main_street = match.group(1).strip().upper()
                direction = match.group(2).strip().upper()
                reference_street = match.group(3).strip().upper()
                
                # Store main street, full location text, and direction for this node
                node_info[scats_id].append({
                    'main_street': main_street,
                    'reference_street': reference_street,
                    'direction': direction,
                    'full_location': location_text
                })
                
            except Exception as e:
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Create a mapping from main_street -> list of nodes with that main street
        main_street_to_nodes = defaultdict(list)
        for node_id, info_list in node_info.items():
            for info in info_list:
                main_street_to_nodes[info['main_street']].append({
                    'node_id': node_id,
                    'info': info
                })
        
        # Find connections where reference streets match other nodes' main streets
        # Format: node_id -> [(connected_node_id, location_info)]
        node_connections = defaultdict(list)
        
        for node_id, info_list in node_info.items():
            for info in info_list:
                reference_street = info['reference_street']
                
                # Find nodes that have this reference street as their main street
                for connecting_node_data in main_street_to_nodes.get(reference_street, []):
                    connecting_node_id = connecting_node_data['node_id']
                    connecting_info = connecting_node_data['info']
                    
                    # Skip self-connections
                    if connecting_node_id == node_id:
                        continue
                    
                    # Check if this connection already exists
                    connection_exists = False
                    for existing_conn in node_connections[node_id]:
                        if existing_conn['node_id'] == connecting_node_id:
                            connection_exists = True
                            break
                    
                    if not connection_exists:
                        # Add connection with the location information of the connecting node
                        node_connections[node_id].append({
                            'node_id': connecting_node_id,
                            'location': connecting_info['full_location'],  # Location of the connecting node
                            'direction': connecting_info['direction'],
                            'connecting_street': connecting_info['main_street']
                        })
        
        return dict(node_connections)
