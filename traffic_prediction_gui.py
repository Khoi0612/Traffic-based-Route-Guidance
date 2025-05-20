import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
# from scats_based_path_generator import extract_scats_labels
import pandas as pd
from path_finding_algorithms import run_algorithm, load_graph_from_file, GraphProblem

class TrafficPredictionGUI:
    @staticmethod
    def extract_scats_labels(excel_path, sheet_name='Data'):
        labels = {}
        try:
            data = pd.read_excel(excel_path, sheet_name=sheet_name, header=None)
            for row in range(2, len(data)):
                try:
                    scats_id = data.iloc[row, 0]
                    location_text = str(data.iloc[row, 1])
                    if pd.notna(scats_id) and pd.notna(location_text):
                        if int(scats_id) not in labels:
                            labels[int(scats_id)] = location_text.strip()
                except:
                    continue
        except Exception as e:
            print(f"Failed to load SCATS label data: {e}")
        
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
                                      values=["LSTM", "GRU", "XGBoost"])
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
        
        # # Origin selection
        # ttk.Label(main_frame, text="Origin:").grid(row=5, column=0, sticky="w", pady=5)
        # self.origin_var = tk.StringVar()
        # origin_combo = ttk.Combobox(main_frame, textvariable=self.origin_var, 
        #                            values=["Downtown", "Airport", "University", "Shopping Mall", "Business District"])
        # origin_combo.grid(row=5, column=1, sticky="ew", pady=5)
        # origin_combo.current(0)
        
        # # Destination selection
        # ttk.Label(main_frame, text="Destination:").grid(row=6, column=0, sticky="w", pady=5)
        # self.destination_var = tk.StringVar()
        # destination_combo = ttk.Combobox(main_frame, textvariable=self.destination_var, 
        #                                 values=["Downtown", "Airport", "University", "Shopping Mall", "Business District"])
        # destination_combo.grid(row=6, column=1, sticky="ew", pady=5)
        # destination_combo.current(1)

        # Load SCATS labels from Excel
        labels = TrafficPredictionGUI.extract_scats_labels("data/raw_data/Scats Data October 2006.xls")
        combo_items = [f"{sid} - {name}" for sid, name in sorted(labels.items())]

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
        origin_text = self.origin_var.get()
        destination_text = self.destination_var.get()
        
        # # Normalize route model
        # if route_model == "A*":
        #     route_model = "AS"

        try:
            origin_id = int(origin_text.split(" - ")[0])
            destination_id = int(destination_text.split(" - ")[0])
        except ValueError:
            messagebox.showerror("Invalid SCATS selection", "Failed to parse SCATS site numbers.")
            return

        # Step 2: Build graph file name based on ML model
        graph_file = f"test_{ml_model}_1.txt"
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

        
    # def find_routes(self):
    #     ml_model = self.ml_model_var.get()
    #     route_model = self.route_model_var.get()
    #     date = self.date_var.get()
    #     time = self.time_var.get()
    #     origin_text = self.origin_var.get()
    #     destination_text = self.destination_var.get()
        
    #     # Extract SCATS site numbers (as integers)
    #     origin_id = int(origin_text.split(" - ")[0])
    #     destination_id = int(destination_text.split(" - ")[0])
        
    #     # Sample route finding logic
    #     routes = [
    #         f"1. Via Main Street (15 mins, 5.2 miles)",
    #         f"2. Via Highway 101 (12 mins, 7.8 miles)",
    #         f"3. Via Side Streets (18 mins, 4.3 miles)"
    #     ]
        
    #     # Display the results
    #     self.results_text.delete(1.0, tk.END)
    #     self.results_text.insert(tk.END, 
    #         f"ML Model: {ml_model}\n"
    #         f"Route Model: {route_model}\n"
    #         f"Date: {date}\n"
    #         f"Time: {time}\n"
    #         f"From: {origin_text}\n"
    #         f"To: {destination_text}\n\n"
    #         "Recommended Routes:\n" + "\n".join(routes)
    #     )
        

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficPredictionGUI(root)
    root.mainloop()