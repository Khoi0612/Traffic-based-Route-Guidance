import os
import numpy as np
import pickle
from src.models import LSTMTrafficPredictionModel, GRUTrafficPredictionModel, XGBoostTrafficPredictionModel
import matplotlib.pyplot as plt


class ModelManager:
    def __init__(self, train_data_dir=None, models_dir=None):
        self.train_data_dir = train_data_dir or os.path.join('data', 'processed')
        self.models_dir = models_dir or os.path.join('data', 'models')
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

                    # Train model
                    model.train(x_train, y_train)  # train in-place

                    # Save the entire model instance using pickle for all types
                    model_filename = f"{base_name}_{model_type}.pkl"
                    model_path = os.path.join(self.models_dir, model_filename)

                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)

                    print(f"Saved {model_type.upper()} model instance to {model_path}")


    def get_available_models(self):
        available_models = []
        if not os.path.exists(self.models_dir):
            return available_models
            
        for filename in os.listdir(self.models_dir):
            if filename.endswith(".pkl"):
                available_models.append(filename)
        
        return sorted(available_models)
    
    def load_model(self, model_filename):
        model_path = os.path.join(self.models_dir, model_filename)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        return model
    
    def evaluate_model(self, model_filename, data_filename=None, show_plot=False, create_plot=None):
        print(f"\nEvaluating model: {model_filename}")
        
        # Load the trained model
        model = self.load_model(model_filename)
        
        # Determine the data file to use
        if data_filename is None:
            # Extract base name from model filename
            base_name = model_filename.split('_')[0]  # Assumes format: basename_modeltype.pkl
            
            # Find corresponding data file
            data_filename = None
            for filename in os.listdir(self.train_data_dir):
                if filename.startswith(base_name) and (filename.endswith(".xlsx") or filename.endswith(".xls")):
                    data_filename = filename
                    break
            
            if data_filename is None:
                raise FileNotFoundError(f"Could not find data file for model {model_filename}")
        
        data_path = os.path.join(self.train_data_dir, data_filename)
        
        # Load test data using the model's load_data method
        data, x_train, x_test, y_train, y_test, dates_train, dates_test = model.load_data(data_path)
        
        # Determine if we should create a plot
        if create_plot is None:
            create_plot = show_plot
        
        # Evaluate the model
        rmse, predictions, fig = model.evaluate(x_test, y_test, dates_test, create_plot=create_plot)
        
        # Show plot if requested and figure exists
        if show_plot and fig is not None:
            plt.show()
        
        return rmse, predictions, y_test, dates_test, fig
    
    def evaluate_all_models(self, show_plots=False):
        available_models = self.get_available_models()
        
        if not available_models:
            print("No trained models found. Please train models first.")
            return
        
        results = {}
        figures = {}
        
        for i, model_filename in enumerate(available_models):
            try:
                print(f"\nEvaluating {i+1}/{len(available_models)}: {model_filename}")
                
                # Only create plots if we're going to show them
                rmse, predictions, y_test, dates_test, fig = self.evaluate_model(
                    model_filename, 
                    show_plot=False,  # Don't show during batch evaluation
                    create_plot=show_plots  # Only create if we need them
                )
                
                results[model_filename] = {
                    'rmse': rmse,
                    'predictions': predictions,
                    'actual': y_test,
                    'dates': dates_test
                }
                
                if show_plots and fig is not None:
                    figures[model_filename] = fig
                    # Show plot
                    plt.show()
                elif fig is not None:
                    # Close figure immediately if created but not needed
                    plt.close(fig)
                    
            except Exception as e:
                print(f"Error evaluating {model_filename}: {str(e)}")
                continue
        
        # Print summary of results
        print("\n" + "="*50)
        print("EVALUATION RESULTS SUMMARY")
        print("="*50)
        
        sorted_results = sorted(results.items(), key=lambda x: x[1]['rmse'])
        
        for model_name, result in sorted_results:
            print(f"{model_name}: RMSE = {result['rmse']:.2f}")
        
        if sorted_results:
            print(f"\nBest performing model: {sorted_results[0][0]} (RMSE: {sorted_results[0][1]['rmse']:.2f})")
        
        return results, figures