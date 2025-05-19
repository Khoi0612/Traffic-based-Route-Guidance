import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

class TrafficPredictionGUI:
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
                                         values=["BFS", "DFS", "A*", "Dijkstra"])
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
        
        # Origin selection
        ttk.Label(main_frame, text="Origin:").grid(row=5, column=0, sticky="w", pady=5)
        self.origin_var = tk.StringVar()
        origin_combo = ttk.Combobox(main_frame, textvariable=self.origin_var, 
                                   values=["Downtown", "Airport", "University", "Shopping Mall", "Business District"])
        origin_combo.grid(row=5, column=1, sticky="ew", pady=5)
        origin_combo.current(0)
        
        # Destination selection
        ttk.Label(main_frame, text="Destination:").grid(row=6, column=0, sticky="w", pady=5)
        self.destination_var = tk.StringVar()
        destination_combo = ttk.Combobox(main_frame, textvariable=self.destination_var, 
                                        values=["Downtown", "Airport", "University", "Shopping Mall", "Business District"])
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
        ml_model = self.ml_model_var.get()
        route_model = self.route_model_var.get()
        date = self.date_var.get()
        time = self.time_var.get()
        origin = self.origin_var.get()
        destination = self.destination_var.get()
        
        # Sample route finding logic
        routes = [
            f"1. Via Main Street (15 mins, 5.2 miles)",
            f"2. Via Highway 101 (12 mins, 7.8 miles)",
            f"3. Via Side Streets (18 mins, 4.3 miles)"
        ]
        
        # Display the results
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, 
            f"ML Model: {ml_model}\n"
            f"Route Model: {route_model}\n"
            f"Date: {date}\n"
            f"Time: {time}\n"
            f"From: {origin}\n"
            f"To: {destination}\n\n"
            "Recommended Routes:\n" + "\n".join(routes)
        )
        

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficPredictionGUI(root)
    root.mainloop()