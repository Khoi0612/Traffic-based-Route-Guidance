import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
import joblib
import sys
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from xgboost import XGBRegressor

from ml_models import *

def run_traffic_flow_prediction():   
    # Parse CLI arguments
    if len(sys.argv) < 3:
        print("Usage: python traffic_model.py <excel_file> <model_type>")
        print("Model types: lstm, gru, xgb")
        sys.exit(1)
        
    filename = sys.argv[1]
    model_type = sys.argv[2].lower()
    
    # Create model based on the selected type
    if model_type == "lstm":
        model = LSTMTrafficPredictionModel()
    elif model_type == "gru":
        model = GRUTrafficPredictionModel()
    elif model_type == "xgb":
        model = XGBoostTrafficPredictionModel()
    else:
        print(f"Unsupported model type: {model_type}")
        print("Supported types: lstm, xgb")
        sys.exit(1)
    
    # Load data
    data, x_train, x_test, y_train, y_test, dates_train, dates_test = model.load_data(filename)
    
    # Load scaler
    model.load_scaler()
    
    # Train model
    model.train(x_train, y_train)
    
    # Evaluate model
    model.evaluate(x_test, y_test, dates_test)
    
    # Make a prediction for a specific date
    target_dt = datetime(2006, 11, 17, 8, 0, 0)
    model.predict(data, target_dt)


run_traffic_flow_prediction()
    