import tkinter as tk
from tkinter import ttk, messagebox

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
        
        # Model selection
        ttk.Label(main_frame, text="Select Model:").grid(row=1, column=0, sticky="w")
        self.model_var = tk.StringVar()
        model_combo = ttk.Combobox(main_frame, textvariable=self.model_var,
                                  values=["LSTM", "GRU", "XGBoost"])
        model_combo.grid(row=1, column=1, sticky="ew", pady=5)
        model_combo.current(0)
        
        # Location selection
        ttk.Label(main_frame, text="Select Location:").grid(row=2, column=0, sticky="w")
        self.location_var = tk.StringVar()
        location_combo = ttk.Combobox(main_frame, textvariable=self.location_var,
                                     values=["Main Street", "Highway 101", "Downtown"])
        location_combo.grid(row=2, column=1, sticky="ew", pady=5)
        location_combo.current(0)
        
        # Time selection
        ttk.Label(main_frame, text="Select Time:").grid(row=3, column=0, sticky="w")
        self.time_var = tk.StringVar(value="08:00")
        time_entry = ttk.Entry(main_frame, textvariable=self.time_var)
        time_entry.grid(row=3, column=1, sticky="ew", pady=5)
        
        # Buttons
        ttk.Button(main_frame, text="Predict Traffic", 
                  command=self.predict_traffic).grid(row=4, column=0, pady=20)
        ttk.Button(main_frame, text="Find Routes", 
                  command=self.find_routes).grid(row=4, column=1, pady=20)
        
        # Results display
        self.results_text = tk.Text(main_frame, height=10, width=50)
        self.results_text.grid(row=5, column=0, columnspan=2)
        
        # Configure grid weights
        main_frame.columnconfigure(1, weight=1)
        
    def predict_traffic(self):
        model = self.model_var.get()
        location = self.location_var.get()
        time = self.time_var.get()
        
        # Prediction logic
        prediction = "Heavy" if "08" in time else "Moderate"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, 
                               f"Model: {model}\n"
                               f"Location: {location}\n"
                               f"Time: {time}\n\n"
                               f"Predicted Traffic: {prediction}\n"
                               f"Confidence: 85%")
        
    def find_routes(self):
        location = self.location_var.get()
        time = self.time_var.get()
        
        # Route finding
        routes = [
            "1. Main Street → 1st Ave (15 mins)",
            "2. High Street → Park Road (18 mins)",
            "3. Backroads (20 mins)"
        ]
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, 
                               f"From: {location}\n"
                               f"Departure: {time}\n\n"
                               "Recommended Routes:\n" + 
                               "\n".join(routes))

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficPredictionGUI(root)
    root.mainloop()