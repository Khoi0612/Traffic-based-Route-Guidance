import tkinter as tk
from src.ui.application import TrafficPredictionGUI

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficPredictionGUI(root)
    root.mainloop()