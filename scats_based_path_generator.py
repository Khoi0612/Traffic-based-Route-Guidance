import pandas as pd
import os
import re
import math
import sys
from collections import defaultdict
import pickle
from datetime import datetime
from ml_models import *

def load_scats_data(filename, sheet_name='Data'):
    return pd.read_excel(filename, sheet_name=sheet_name, header=None)

def calculate_haversine_distance(coord1, coord2):
    R = 6371.0  # Radius of Earth in km
    lat1, lon1 = coord1[1], coord1[0]
    lat2, lon2 = coord2[1], coord2[0]

    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2) ** 2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def extract_node_coordinates(data):
    node_coordinates = {}
    
    for row in range(2, len(data)):  # Start from row 2 to skip headers
        try:
            scats_id = data.iloc[row, 0]
            latitude = data.iloc[row, 3]
            longitude = data.iloc[row, 4]

            # Skip if any value is missing
            if pd.isna(scats_id) or pd.isna(latitude) or pd.isna(longitude):
                continue

            if scats_id not in node_coordinates:
                node_coordinates[scats_id] = {'sum_long': 0, 'sum_lat': 0, 'count': 0}

            node_coordinates[scats_id]['sum_long'] += longitude
            node_coordinates[scats_id]['sum_lat'] += latitude
            node_coordinates[scats_id]['count'] += 1

        except (IndexError, TypeError):
            continue

    # Compute average coordinates for each node
    nodes = {
        site: (values['sum_long'] / values['count'], values['sum_lat'] / values['count'])
        for site, values in node_coordinates.items()
    }

    return nodes

def extract_street_connections(data):
    # Pattern to match location descriptions like "MAIN ST N of CROSS ST"
    pattern = r'(.+?)\s+(N|NE|E|SE|S|SW|W|NW)\s+of\s+(.+)'
    
    # Store node information: node_id -> [(main_street, location, direction)]
    node_info = defaultdict(list)
    
    # Collect all nodes with their street information
    for idx in range(2, len(data)):
        try:
            scats_id = data.iloc[idx, 0]
            location_text = str(data.iloc[idx, 1])
            
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

def generate_edges(nodes, connections, models_dir, ml_method, target_datetime=None):
    edges = {}
    
    # Set default prediction time to current time if not specified
    if target_datetime is None:
        target_datetime = datetime.now()

    
    for node, neighbors in connections.items():
        for neighbor in neighbors:
            # Extract information from the connection
            destination_node = neighbor['node_id']
            location = neighbor['location']

            # Create file path for model
            model_path = os.path.join(models_dir, f"{location}_{ml_method}.pkl")

            with open(model_path, 'rb') as f:
                trained_model = pickle.load(f)

            filename = os.path.join('data', f"{location}.xlsx")
            # Load data
            data, x_train, x_test, y_train, y_test, dates_train, dates_test = trained_model.load_data(filename)

            # Create sorted edge tuple to avoid duplicates
            edge = tuple(sorted((node, destination_node)))
            
            if edge not in edges:
                coord1 = nodes.get(node)
                coord2 = nodes.get(destination_node)
                
                if coord1 and coord2:
                    # Calculate physical distance in km
                    distance = calculate_haversine_distance(coord1, coord2)

                    # Make a prediction for a specific date
                    pred_value, travel_time = trained_model.predict(data, target_datetime, distance)
                    
                    # Store prediction as the edge weight
                    edges[edge] = travel_time
    
    return edges

def format_graph_data(nodes, edges, origin, destination):
    # Format nodes
    nodes_str = "Nodes:\n"
    for node, coords in nodes.items():
        nodes_str += f"{node}: ({coords[0]},{coords[1]})\n"

    # Format edges
    edges_str = "Edges:\n"
    for edge, distance in edges.items():
        edges_str += f"({edge[0]},{edge[1]}): {distance}\n"

    # Format origin and destination
    origin_str = f"Origin:\n{origin}\n"
    destination_str = f"Destinations:\n{destination}"

    # Combine all sections
    return nodes_str + edges_str + origin_str + destination_str

def export_to_file(content, base_name="test_"):
    i = 1  # Starting index
    while True:
        filename = f"{base_name}{i}.txt"

        # If file doesn't exist, create it and exit loop
        if not os.path.exists(filename):
            with open(filename, "w") as file:
                file.write(content)
            print(f"Exported to {filename}")
            return filename
        
        # If file exists, try next index
        i += 1

def run_generator():
    if len(sys.argv) < 4:
        print("Usage: python script.py <origin> <destination> <ml_model_type>")
        return
        
    origin = sys.argv[1]
    destination = sys.argv[2]
    ml_model_type = sys.argv[3]
    
    # Path to data file
    data_file = os.path.join('data', 'raw_data' , 'Scats Data October 2006.xls')
    
    # Load data
    scats_data = load_scats_data(data_file)
    
    # Process data
    nodes = extract_node_coordinates(scats_data)
    connections = extract_street_connections(scats_data)
    target_dt = datetime(2006, 10, 31, 8, 0, 0)
    edges = generate_edges(nodes, connections, 'trained_models', ml_model_type.lower(), target_dt)
    
    # Generate graph output
    graph_content = format_graph_data(nodes, edges, origin, destination)
    
    # Export and print results
    print(graph_content)
    export_to_file(graph_content, base_name=f"test_{ml_model_type}_")
    

run_generator()