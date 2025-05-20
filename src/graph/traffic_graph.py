import os
import math
import pickle
from datetime import datetime
from src.models import *

class TrafficGraph:
    
    def __init__(self, nodes, connections):
        self.nodes = nodes
        self.connections = connections
        self.edges = {}
    
    @staticmethod
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
    
    def generate_edges(self, models_dir, ml_method, target_datetime=None):
        # Set default prediction time to current time if not specified
        if target_datetime is None:
            target_datetime = datetime.now()

        for node, neighbors in self.connections.items():
            for neighbor in neighbors:
                # Extract information from the connection
                destination_node = neighbor['node_id']
                location = neighbor['location']

                # Create file path for model
                model_path = os.path.join('..', 'data', models_dir, f"{location}_{ml_method}.pkl")

                with open(model_path, 'rb') as f:
                    trained_model = pickle.load(f)

                filename = os.path.join('..', 'data', 'processed', f"{location}.xlsx")
                # Load data
                data, x_train, x_test, y_train, y_test, dates_train, dates_test = trained_model.load_data(filename)

                # Create sorted edge tuple to avoid duplicates
                edge = tuple(sorted((node, destination_node)))
                
                if edge not in self.edges:
                    coord1 = self.nodes.get(node)
                    coord2 = self.nodes.get(destination_node)
                    
                    if coord1 and coord2:
                        print(f"Generating edge between {node} and {destination_node}")

                        # Calculate physical distance in km
                        distance = self.calculate_haversine_distance(coord1, coord2)

                        # Make a prediction for a specific date
                        pred_value, travel_time = trained_model.predict(data, target_datetime, distance)
                        
                        # Store prediction as the edge weight
                        self.edges[edge] = travel_time
        
        return self.edges
    
    def format_graph_data(self, origin, destination):
        # Format nodes
        nodes_str = "Nodes:\n"
        for node, coords in self.nodes.items():
            nodes_str += f"{node}: ({coords[0]},{coords[1]})\n"

        # Format edges
        edges_str = "Edges:\n"
        for edge, distance in self.edges.items():
            edges_str += f"({edge[0]},{edge[1]}): {distance}\n"

        # Format origin and destination
        origin_str = f"Origin:\n{origin}\n"
        destination_str = f"Destinations:\n{destination}"

        # Combine all sections
        return nodes_str + edges_str + origin_str + destination_str
    
    def export_to_file(self, content, base_name):
        filename_str = f"{base_name}.txt"
        filename = os.path.join('..', 'output', filename_str)
        with open(filename, "w") as file:
            file.write(content)
        print(f"Exported to {filename}")
        return filename