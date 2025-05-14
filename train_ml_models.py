import os
import sys
import pickle

from ml_models import *

def train_all_models(input_dir, output_dir):
    if not os.path.isdir(input_dir):
        print(f"Input directory '{input_dir}' does not exist.")
        sys.exit(1)

    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    model_classes = {
        "lstm": LSTMTrafficPredictionModel,
        "gru": GRUTrafficPredictionModel,
        "xgb": XGBoostTrafficPredictionModel
    }

    for filename in os.listdir(input_dir):
        if filename.endswith(".xlsx") or filename.endswith(".xls"):
            full_path = os.path.join(input_dir, filename)
            base_name = os.path.splitext(filename)[0]

            for model_type, model_class in model_classes.items():
                print(f"\nTraining {model_type.upper()} model on {filename}")

                # Create model instance
                model = model_class()

                # Load data and scaler
                data, x_train, x_test, y_train, y_test, dates_train, dates_test = model.load_data(full_path)
                model.load_scaler()

                # Train model
                model.train(x_train, y_train)  # train in-place

                # Save the entire model instance using pickle for all types
                model_filename = f"{base_name}_{model_type}.pkl"
                model_path = os.path.join(output_dir, model_filename)

                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)

                print(f"Saved {model_type.upper()} model instance to {model_path}")


if len(sys.argv) != 3:
    print("Usage: python traffic_model.py <input_directory> <output_directory>")
    sys.exit(1)

input_dir = sys.argv[1]
output_dir = sys.argv[2]

train_all_models(input_dir, output_dir)