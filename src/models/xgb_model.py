import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from .base_model import BaseTrafficPredictionModel

class XGBoostTrafficPredictionModel(BaseTrafficPredictionModel):

    def __init__(self, window_size=5):
        super().__init__(window_size)
        self.model_name = "XGBoost"

    def train(self, x_train, y_train):
        self.model = XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        
        self.model.fit(x_train, y_train)
        return self.model

    def evaluate(self, x_test, y_test, dates_test):
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        if self.scaler is None:
            raise ValueError("Scaler must be loaded before evaluation")
            
        # Make predictions (XGBoost doesn't need reshaping)
        predictions = self.model.predict(x_test)
        
        # Inverse transform predictions and actual values
        predictions = self.scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
        y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        print(f'{self.model_name} RMSE: {rmse:.2f}')
        
        # Plot results
        plt.figure(figsize=(12, 6))
        plt.plot(dates_test, y_test, label='Actual Flow')
        plt.plot(dates_test, predictions, label=f'{self.model_name} Predicted Flow')
        plt.title(f'{self.model_name}: Actual vs Predicted Traffic Flow')
        plt.xlabel('Date')
        plt.ylabel('Traffic Flow (cars)')
        plt.legend()
        plt.tight_layout()
        plt.show()
        
        return rmse, predictions
    
    def predict(self, data, target_datetime, distance_km=1.4):
        if self.model is None:
            raise ValueError("Model must be trained before prediction")
        
        if self.scaler is None:
            raise ValueError("Scaler must be loaded before prediction")

        target_datetime = pd.to_datetime(target_datetime)
        last_known_time = data['timestamp'].max()

        if target_datetime <= last_known_time:
            row = data[data['timestamp'] == target_datetime]
            
            if row.empty:
                print(f"No exact match for {target_datetime}")
                return None
            x_input = row.iloc[0, 0:self.window_size].values.reshape(1, -1).astype('float32')
            prediction_scaled = self.model.predict(x_input)
            prediction = self.scaler.inverse_transform([[prediction_scaled[0]]])[0][0]
            print(f"Predicted traffic flow for {target_datetime}: {prediction:.2f} cars")
        else:
            print(f"{target_datetime} is in the future. Starting recursive prediction from {last_known_time}...")

            last_row = data[data['timestamp'] == last_known_time]
            if last_row.empty:
                print(f"No data for last known time {last_known_time}")
                return None

            x_input = last_row.iloc[0, 0:self.window_size].values.reshape(1, -1).astype('float32')
            step = pd.Timedelta(minutes=15)
            steps_needed = int((target_datetime - last_known_time) / step)

            for _ in range(steps_needed):
                prediction_scaled = self.model.predict(x_input)
                new_val = prediction_scaled[0]
                x_input = np.append(x_input[:, 1:], [[new_val]], axis=1)

            prediction = self.scaler.inverse_transform([[new_val]])[0][0]
            print(f"Predicted traffic flow for {target_datetime} (forecasted): {prediction:.2f} cars")

        flow_hourly = prediction * 4
        travel_time = self.estimate_travel_time(flow_hourly, distance_km)
        print(f"Estimated travel time: {travel_time:.2f} seconds")

        return prediction, travel_time
