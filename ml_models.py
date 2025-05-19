import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
import joblib
import math

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout
from xgboost import XGBRegressor

class BaseTrafficPredictionModel:

    def __init__(self, window_size=5):
        self.window_size = window_size
        self.model = None
        self.scaler = None

    def load_data(self, filename):
        data = pd.read_excel(filename)
        target_dates = pd.to_datetime(data['timestamp'])
        
        # Extract features and target
        x = data.iloc[:, 0:self.window_size].values
        y = data['target'].values
        
        # Split into train/test sets
        x_train, x_test, y_train, y_test, dates_train, dates_test = train_test_split(
            x, y, target_dates, test_size=0.2, shuffle=False
        )
        
        return data, x_train, x_test, y_train, y_test, dates_train, dates_test

    def load_scaler(self, scaler_path='scaler.save'):
        self.scaler = joblib.load(scaler_path)
        return self.scaler
    
    def flow_to_speed(self, flow):
        if flow <= 351:
            return 60.0
        a, b, c = -1.4648375, 93.75, -flow
        d = b**2 - 4*a*c
        if d < 0:
            return 0.0
        s1 = (-b + math.sqrt(d)) / (2 * a)
        s2 = (-b - math.sqrt(d)) / (2 * a)
        return min(s for s in (s1, s2) if s > 0)

    def estimate_travel_time(self, flow, distance_km, delay_sec=30):
        speed = self.flow_to_speed(flow)
        if speed <= 0:
            return float('inf')
        return (distance_km / speed) * 3600 + delay_sec

    def train(self, x_train, y_train):
        raise NotImplementedError("Subclasses must implement the train method")

    def evaluate(self, x_test, y_test, dates_test):
        raise NotImplementedError("Subclasses must implement the evaluate method")

    def predict(self, data, target_datetime):
        raise NotImplementedError("Subclasses must implement the predict method")


class LSTMTrafficPredictionModel(BaseTrafficPredictionModel):

    def __init__(self, window_size=5):
        super().__init__(window_size)
        self.model_name = "LSTM"

    def train(self, x_train, y_train):
        # Reshape input to be [samples, time steps, features]
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        # Define the LSTM model architecture
        self.model = Sequential()
        self.model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(units=128))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        # Compile the model
        self.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train the model
        self.model.fit(
            x_train, y_train,
            epochs=25,
            batch_size=64,
            validation_split=0.1,
            verbose=1
        )
        
        return self.model

    def evaluate(self, x_test, y_test, dates_test):
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
            
        if self.scaler is None:
            raise ValueError("Scaler must be loaded before evaluation")
            
        # Reshape test data for LSTM
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        
        # Make predictions
        predictions = self.model.predict(x_test)
        
        # Inverse transform predictions and actual values
        predictions = self.scaler.inverse_transform(predictions).flatten()
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
            # Use known row from dataset
            row = data[data['timestamp'] == target_datetime]
            
            if row.empty:
                print(f"No exact match for {target_datetime}")
                return None
        
            x_input = row.iloc[0, 0:self.window_size].values.reshape((1, self.window_size, 1)).astype('float32')
            prediction_scaled = self.model.predict(x_input, verbose=0)
            prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
            print(f"Predicted traffic flow for {target_datetime}: {prediction:.2f} cars")
        else:
            print(f"{target_datetime} is in the future. Starting recursive prediction from {last_known_time}...")

            last_row = data[data['timestamp'] == last_known_time]
            if last_row.empty:
                print(f"No data for last known time {last_known_time}")
                return None

            x_input = last_row.iloc[0, 0:self.window_size].values.reshape((1, self.window_size, 1)).astype('float32')
            step = pd.Timedelta(minutes=15)
            steps_needed = int((target_datetime - last_known_time) / step)

            for _ in range(steps_needed):
                prediction_scaled = self.model.predict(x_input, verbose=0)
                new_val = prediction_scaled[0][0]
                x_input = np.append(x_input[:, 1:, :], [[[new_val]]], axis=1)

            prediction = self.scaler.inverse_transform([[new_val]])[0][0]
            print(f"Predicted traffic flow for {target_datetime} (forecasted): {prediction:.2f} cars")

        flow_hourly = prediction * 4
        travel_time = self.estimate_travel_time(flow_hourly, distance_km)
        
        print(f"Estimated travel time: {travel_time:.2f} seconds")

        return prediction, travel_time


class GRUTrafficPredictionModel(BaseTrafficPredictionModel):

    def __init__(self, window_size=5):
        super().__init__(window_size)
        self.model_name = "GRU"

    def train(self, x_train, y_train):
        x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

        self.model = Sequential()
        self.model.add(GRU(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1)))
        self.model.add(Dropout(0.2))
        self.model.add(GRU(units=128))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(1))

        self.model.compile(optimizer='adam', loss='mean_squared_error')

        self.model.fit(
            x_train, y_train,
            epochs=25,
            batch_size=64,
            validation_split=0.1,
            verbose=1
        )
        return self.model

    def evaluate(self, x_test, y_test, dates_test):
        if self.model is None:
            raise ValueError("Model must be trained before evaluation")
        if self.scaler is None:
            raise ValueError("Scaler must be loaded before evaluation")

        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
        predictions = self.model.predict(x_test)

        predictions = self.scaler.inverse_transform(predictions).flatten()
        y_test = self.scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
        print(f'{self.model_name} RMSE: {rmse:.2f}')

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
            # Use known row from dataset
            row = data[data['timestamp'] == target_datetime]
            
            if row.empty:
                print(f"No exact match for {target_datetime}")
                return None
            
            x_input = row.iloc[0, 0:self.window_size].values.reshape((1, self.window_size, 1)).astype('float32')
            prediction_scaled = self.model.predict(x_input, verbose=0)
            prediction = self.scaler.inverse_transform(prediction_scaled)[0][0]
            print(f"Predicted traffic flow for {target_datetime}: {prediction:.2f} cars")
        else:
            print(f"{target_datetime} is in the future. Starting recursive prediction from {last_known_time}...")

            last_row = data[data['timestamp'] == last_known_time]
            if last_row.empty:
                print(f"No data for last known time {last_known_time}")
                return None

            x_input = last_row.iloc[0, 0:self.window_size].values.reshape((1, self.window_size, 1)).astype('float32')
            step = pd.Timedelta(minutes=15)
            steps_needed = int((target_datetime - last_known_time) / step)

            for _ in range(steps_needed):
                prediction_scaled = self.model.predict(x_input, verbose=0)
                new_val = prediction_scaled[0][0]
                x_input = np.append(x_input[:, 1:, :], [[[new_val]]], axis=1)

            prediction = self.scaler.inverse_transform([[new_val]])[0][0]
            print(f"Predicted traffic flow for {target_datetime} (forecasted): {prediction:.2f} cars")

        flow_hourly = prediction * 4
        travel_time = self.estimate_travel_time(flow_hourly, distance_km)
        
        print(f"Estimated travel time: {travel_time:.2f} seconds")

        return prediction, travel_time


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
