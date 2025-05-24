import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import os
import re

from src.data.data_processor import DataProcessor
from src.graph.visualizer import GraphVisualizer
from src.graph.generator import TrafficGraphGenerator
from src.path_finding.path_finding_algorithms import run_algorithm, load_graph_from_file, GraphProblem

class TrafficPredictionGUI:
    @staticmethod
    def extract_intersection_labels(excel_path, sheet_name='Data'):
        """Returns a dict like {2000: 'WARRIGAL_RD / TOORAK_RD'}"""
        labels = {}
        try:
            # Use ScatsData class from refactored code
            scats_data = DataProcessor(excel_path, sheet_name)
            connections = scats_data.extract_street_connections()

            for scats_id, info_list in connections.items():
                if info_list:
                    # Use the first connection to create a readable label
                    main_street = info_list[0]['connecting_street']
                    reference_street = info_list[0]['location'].split("of")[-1].strip()
                    labels[scats_id] = f"{main_street} / {reference_street}"
        except Exception as e:
            print(f"Failed to extract labels: {e}")
        return labels
    
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Predictor")
        self.root.geometry("600x400")
        
        # Create main container
        main_frame = ttk.Frame(root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title label
        ttk.Label(main_frame, text="Traffic Prediction System", 
                 font=('Helvetica', 16)).grid(row=0, column=0, columnspan=2, pady=10)
        
        # ML Model selection
        ttk.Label(main_frame, text="Select ML Model:").grid(row=1, column=0, sticky="w", pady=5)
        self.ml_model_var = tk.StringVar()
        ml_model_combo = ttk.Combobox(main_frame, textvariable=self.ml_model_var, 
                                      values=["LSTM", "GRU", "XGB"])
        ml_model_combo.grid(row=1, column=1, sticky="ew", pady=5)
        ml_model_combo.current(0)
        
        # Route Model selection
        ttk.Label(main_frame, text="Select Route Model:").grid(row=2, column=0, sticky="w", pady=5)
        self.route_model_var = tk.StringVar()
        route_model_combo = ttk.Combobox(main_frame, textvariable=self.route_model_var, 
                                         values=["BFS", "DFS", "GBFS", "A*", "Dijkstra", "IDA*"])
        route_model_combo.grid(row=2, column=1, sticky="ew", pady=5)
        route_model_combo.current(0)
        
        # Date selection
        ttk.Label(main_frame, text="Select Date:").grid(row=3, column=0, sticky="w", pady=5)
        self.date_var = tk.StringVar(value=datetime.now().strftime("%Y-%m-%d"))
        date_entry = ttk.Entry(main_frame, textvariable=self.date_var)
        date_entry.grid(row=3, column=1, sticky="ew", pady=5)
        
        # Time selection
        ttk.Label(main_frame, text="Select Time:").grid(row=4, column=0, sticky="w", pady=5)
        self.time_var = tk.StringVar(value=datetime.now().strftime("%H:%M"))
        time_entry = ttk.Entry(main_frame, textvariable=self.time_var)
        time_entry.grid(row=4, column=1, sticky="ew", pady=5)

        # Load SCATS labels from Excel
        excel_path = os.path.join('data', 'raw', 'Scats Data October 2006.xls')
        labels = TrafficPredictionGUI.extract_intersection_labels(excel_path)
        combo_items = [f"{sid} [intersection {name}]" for sid, name in sorted(labels.items())]
        
        # Origin dropdown
        ttk.Label(main_frame, text="Origin:").grid(row=5, column=0, sticky="w", pady=5)
        self.origin_var = tk.StringVar()
        origin_combo = ttk.Combobox(main_frame, textvariable=self.origin_var, values=combo_items)
        origin_combo.grid(row=5, column=1, sticky="ew", pady=5)
        origin_combo.current(0)

        # Destination dropdown
        ttk.Label(main_frame, text="Destination:").grid(row=6, column=0, sticky="w", pady=5)
        self.destination_var = tk.StringVar()
        destination_combo = ttk.Combobox(main_frame, textvariable=self.destination_var, values=combo_items)
        destination_combo.grid(row=6, column=1, sticky="ew", pady=5)
        destination_combo.current(1)
        
        # Find Routes button 
        find_button = ttk.Button(main_frame, text="Find Routes", command=self.find_routes)
        find_button.grid(row=7, column=0, columnspan=2, pady=20)
        
        # Results display
        self.results_text = tk.Text(main_frame, height=10, width=50)
        self.results_text.grid(row=8, column=0, columnspan=2, pady=10)
        
        # Make the input column expandable
        main_frame.columnconfigure(1, weight=1)
        
        # Initialize the graph generator
        self.data_file = excel_path
        self.graph_generator = TrafficGraphGenerator(self.data_file)
        
    def find_routes(self):
        route_model = self.route_model_var.get().strip()

        # Translate GUI-friendly names to internal algorithm codes
        translation = {
            "A*": "AS",
            "Dijkstra": "CUS1",
            "IDA*": "CUS2"
        }

        # Step 1: Get inputs from GUI
        ml_model = self.ml_model_var.get().lower()
        route_model = translation.get(route_model, route_model).upper()
        date = self.date_var.get()
        time_val = self.time_var.get()
        date_time_str = f"{date} {time_val}"
        origin_text = self.origin_var.get()
        destination_text = self.destination_var.get()

        try:
            origin_id = origin_text.split(" ")[0]
            destination_id = destination_text.split(" ")[0]
        except ValueError:
            messagebox.showerror("Invalid SCATS selection", "Failed to parse SCATS site numbers.")
            return

        # Step 2: Build graph file name based on ML model
        date_time = datetime.strptime(date_time_str, "%Y-%m-%d %H:%M")
        print(f"\n{date_time}\n")
        target_dt_str = date_time.strftime("%Y-%m-%d_%H-%M-%S")
        graph_file = None
        output_directory = os.path.join('output')

        # Look for matching files in the directory
        if os.path.exists(output_directory):
            for filename in os.listdir(output_directory):
                if filename.endswith('.txt'):
                    # Match the new filename format with origin, destination, and ml model
                    pattern = fr"test_from_{origin_id}_to_{destination_id}_using_{ml_model}_at_{target_dt_str}.txt"
                    if re.match(pattern, filename):
                        graph_file = os.path.join(output_directory, filename)
                        print(f"Using existing graph file: {graph_file}")
                        break

        # If no matching file found, generate a new one using the refactored generator
        if graph_file is None:
            graph_file = self.graph_generator.run(origin_id, destination_id, ml_model, date_time)
            print(f"Generated new graph file: {graph_file}")

        try:
            graph_map, origin_from_file, destinations_from_file = load_graph_from_file(graph_file)
        except Exception as e:
            messagebox.showerror("File Error", f"Could not load graph file: {e}")
            return

        # Step 3: Create graph search problem
        problem = GraphProblem(origin_id, [destination_id], graph_map)

        try:
            # Step 4: Run selected algorithm
            result_node, explored, runtime = run_algorithm(route_model, problem)
        except Exception as e:
            messagebox.showerror("Algorithm Error", f"Failed to run algorithm: {e}")
            return

        # Step 5: Extract path
        if result_node:
            path = [str(n.state) for n in result_node.path()]
            path_str = " → ".join(path)
            goal = result_node.state
            cost = result_node.path_cost
            
            # Use the visualizer from the refactored code
            visualizer = GraphVisualizer()
            visualizer.draw_solution_on_map(graph_map, origin_id, destination_id, path)
        else:
            path_str = "No path found."
            goal = "-"
            cost = "-"

        # Step 6: Display in GUI
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END,
            f"ML Model: {ml_model.upper()}\n"
            f"Route Model: {route_model}\n"
            f"Date: {date}\n"
            f"Time: {time_val}\n"
            f"From: {origin_text}\n"
            f"To: {destination_text}\n\n"
            f"Result:\n"
            f"  Goal reached: {goal}\n"
            f"  Nodes explored: {explored}\n"
            f"  Path cost: {cost}\n"
            f"  Runtime: {runtime:.2f} ms\n\n"
            f"Path:\n{path_str}"
        )