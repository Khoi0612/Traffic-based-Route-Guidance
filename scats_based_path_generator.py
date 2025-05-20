import pandas as pd
import os
import re
import math
import sys
from collections import defaultdict
import pickle
from datetime import datetime
from ml_models import *
import plotly.graph_objects as go

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
                    print(f"Generating edge between {node} and {destination_node}")

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

def export_to_file(content, base_name, dir='tests'):
    filename_str = f"{base_name}.txt"
    filename = os.path.join(dir, filename_str)
    with open(filename, "w") as file:
        file.write(content)
    print(f"Exported to {filename}")
    return filename

def draw_graph_on_map(nodes, edges, origin, destination):
    # Create a figure
    fig = go.Figure()

    # Extract the scats number, lattitude, and longtitude from the nodes dictionary
    scats_no = list(nodes.keys())
    scats_lat = [nodes[name][0] for name in scats_no]
    scats_lon = [nodes[name][1] for name in scats_no]

    # Add markers for all scat sites
    fig.add_trace(go.Scattermap(
        lat=scats_lat,
        lon=scats_lon,
        mode='markers+text',
        text=scats_no,
        textposition='top right',
        marker=dict(size=10, color='skyblue'),
        name='Locations'
    ))

    # Draw edges connecting 2 nodes
    for (n1, n2), cost in edges.items():
        # Extract the lattitude, and longtitude from the nodes dictionary with each edge's key
        lat_pair = [nodes[n1][0], nodes[n2][0]]
        lon_pair = [nodes[n1][1], nodes[n2][1]]

        # Add lines for each edge
        fig.add_trace(go.Scattermap(
            lat=lat_pair,
            lon=lon_pair,
            mode='lines+markers',
            line=dict(width=2, color='black'),
            marker=dict(size=6),
            name=f"{cost:.0f} secs"
        ))

    # Add origin
    fig.add_trace(go.Scattermap(
        lat=[nodes[origin][0]],
        lon=[nodes[origin][1]],
        mode='markers+text',
        text=origin,
        textposition='top right',
        marker=dict(size=10, color='limegreen'),
        name='Origin'
    ))

    # Add destination
    fig.add_trace(go.Scattermap(
        lat=[nodes[destination][0]],
        lon=[nodes[destination][1]],
        mode='markers+text',
        text=destination[0],
        textposition='top right',
        marker=dict(size=10, color='orange'),
        name='Destination'
    ))

    center_lat = sum(scats_lat)/len(scats_lat)
    center_lon = sum(scats_lon)/len(scats_lon)
    zoom_level = 11.5

    legend_item_count = len([trace for trace in fig.data if trace.showlegend])
    legend_height_adjustment = 0.5 * legend_item_count / zoom_level  # Adjust based on zoom level
    
    adjusted_center_lat = center_lat - legend_height_adjustment/2

    fig.update_layout(
        autosize=True,
        hovermode='closest',
        showlegend=True,
        map=dict(
            bearing=0,
            center=dict(lat=adjusted_center_lat, lon=center_lon),
            zoom=zoom_level,
            style='open-street-map'
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5
        )
    )   

    return fig, adjusted_center_lat, center_lon, zoom_level

def draw_solution_on_map(graph_map, origin, destination, solution_path):
    nodes = graph_map.locations
    edges = {}
    for node1 in graph_map.graph_dict:
        for node2, cost in graph_map.graph_dict[node1].items():
            edges[(node1, node2)] = cost

    fig, adjusted_center_lat, center_lon, zoom_level = draw_graph_on_map(nodes, edges, origin, destination)

    # Draw solution path edges in red
    for i in range(len(solution_path) - 1):
        n1 = solution_path[i]
        n2 = solution_path[i + 1]
        
        # Get coordinates
        lat1, lon1 = nodes[n1]
        lat2, lon2 = nodes[n2]
        
        # Draw the solution edge line in red with increased width
        fig.add_trace(go.Scattermap(
            lat=[lat1, lat2],
            lon=[lon1, lon2],
            mode='lines',
            line=dict(width=5, color='red'),
            showlegend=False if i > 0 else True,
            name='Solution Path' if i == 0 else None
        ))

    fig.update_layout(
        autosize=True,
        hovermode='closest',
        showlegend=True,
        map=dict(
            bearing=0,
            center=dict(lat=adjusted_center_lat, lon=center_lon),
            zoom=zoom_level,
            style='open-street-map'
        ),
        margin={"r":0,"t":0,"l":0,"b":0},
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=0.02,
            xanchor="center",
            x=0.5
        )
    ) 

    # Show the map
    fig.show()

def run_generator(origin, destination, ml_model_type, target_dt):  
    # Path to data file
    data_file = os.path.join('data', 'raw_data' , 'Scats Data October 2006.xls')
    
    # Load data
    scats_data = load_scats_data(data_file)
    
    # Process data
    nodes = extract_node_coordinates(scats_data)
    connections = extract_street_connections(scats_data)
    
    edges = generate_edges(nodes, connections, 'trained_models', ml_model_type.lower(), target_dt)
    
    # Generate graph output
    graph_content = format_graph_data(nodes, edges, origin, destination)
    
    # Export and print results
    print(graph_content)
    target_dt_str = target_dt.strftime("%Y-%m-%d_%H-%M-%S")
    filename = export_to_file(graph_content, base_name=f"test_from_{origin}_to_{destination}_using_{ml_model_type}_at_{target_dt_str}")

    # Draw graph output
    draw_graph_on_map(nodes, edges, origin, destination)

    return filename
    