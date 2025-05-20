import os
import numpy as np
import pickle
from src.models import LSTMTrafficPredictionModel, GRUTrafficPredictionModel, XGBoostTrafficPredictionModel


class ModelManager:
    def __init__(self, train_data_dir=None, models_dir=None):
        self.train_data_dir = train_data_dir or os.path.join('..', 'data', 'processed')
        self.models_dir = models_dir or os.path.join('..', 'data', 'models')
        self.models_dict = {
            "lstm": LSTMTrafficPredictionModel,
            "gru": GRUTrafficPredictionModel,
            "xgb": XGBoostTrafficPredictionModel
        }
        
        # Create the models directory if it doesn't exist
        if not os.path.exists(self.models_dir):
            os.makedirs(self.models_dir)
    
    def train_all_models(self):
        for filename in os.listdir(self.train_data_dir):
            if filename.endswith(".xlsx") or filename.endswith(".xls"):
                full_path = os.path.join(self.train_data_dir, filename)
                base_name = os.path.splitext(filename)[0]

                for model_type, model_class in self.models_dict.items():
                    print(f"\nTraining {model_type.upper()} model on {filename}")

                    # Create model instance
                    model = model_class()

                    # Load data and scaler
                    data, x_train, x_test, y_train, y_test, dates_train, dates_test = model.load_data(full_path)
                    model.load_scaler()

                    # Train model
                    model.train(x_train, y_train)  # train in-place

                    # Save the entire model instance using pickle for all types
                    model_filename = f"{base_name}{model_type}.pkl"
                    model_path = os.path.join(self.models_dir, model_filename)

                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)

                    print(f"Saved {model_type.upper()} model instance to {model_path}")