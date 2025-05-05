import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from xgboost import XGBRegressor

def process_data(filename, sheet_name='Data'):
    # Read the Excel file
    raw_data = pd.read_excel(filename, sheet_name=sheet_name, header=None)

    # Extract the list of time
    time_list = raw_data.iloc[0, 10:].tolist()

    # Initialize lists for each parameter
    timestamps = []
    scats_numbers = []
    locations = []
    traffic_flows = []

    # Start reading data from row 2 (after time and column headers)
    for row in range(2, len(raw_data)):
        scat_no = raw_data.iloc[row, 0]
        location = raw_data.iloc[row, 1]
        date = raw_data.iloc[row, 9].date()  # Date only

        # Loop through all time intervals and collect data
        for i, time in enumerate(time_list):
            timestamps.append(datetime.combine(date, time))
            scats_numbers.append(scat_no)
            locations.append(location)
            traffic_flows.append(raw_data.iloc[row, 10 + i])

    # Create the final processed DataFrame
    processed_data = pd.DataFrame({
        'timestamp': timestamps,
        'scats_number': scats_numbers,
        'location': locations,
        'traffic_flow': traffic_flows
    })

    return processed_data

def split_data(processed_data, window_size, groupby='scats_number', group_name='0970'):
    # Group data by the specified column and get a specific group
    grouped = processed_data.groupby(groupby)
    group_df = grouped.get_group(group_name).copy()

    # Set timestamp as the index
    group_df['timestamp'] = pd.to_datetime(group_df['timestamp'])
    group_df.set_index('timestamp', inplace=True)

    # Extract traffic flow 
    traffic_flow = group_df['traffic_flow'].astype(float).values.reshape(-1, 1)

    # Extract target dates
    target_dates = group_df.index[window_size:]

    # Normalize the traffic flow values to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_traffic_flow = scaler.fit_transform(traffic_flow)

    # Prepare sliding window input (x) and corresponding labels (y)
    x, y = [], []
    for i in range(window_size, len(scaled_traffic_flow)):
        x.append(scaled_traffic_flow[i - window_size:i, 0])
        y.append(scaled_traffic_flow[i, 0])

    # Convert to NumPy arrays
    x = np.array(x)
    y = np.array(y)

    # Split data into training and testing sets, and return scaler along with them
    return scaler, train_test_split(x, y, target_dates, test_size=0.2, shuffle=False)

def train_lstm_model(x_train, y_train):
    # Reshape input to be [samples, time steps, features]
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

    # Define the LSTM model architecture
    model = Sequential()
    model.add(LSTM(units=128, return_sequences=True, input_shape=(x_train.shape[1], 1))) # First layer
    model.add(Dropout(0.2))  # Prevent overfitting
    model.add(LSTM(units=128))  # Final layer
    model.add(Dropout(0.2))
    model.add(Dense(1))  # Output layer

    # Compile the model with Adam optimizer and MSE loss
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model with 10% validation split
    history = model.fit(
        x_train, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.1,
        verbose=1
    )

    return model, history

def train_xgboost_model(x_train, y_train):
    model = XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(x_train, y_train)
    return model

def evaluate_model(model, history, scaler, x_test, y_test, dates_test):
    # Reshape test data to match LSTM input
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Make predictions
    predictions = model.predict(x_test)

    #  Inverse transform to get actual traffic flow for both predictions and actuals
    predictions = scaler.inverse_transform(predictions).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    print(f'RMSE: {rmse:.2f}')

    # Plot the actual vs predicted traffic flow
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label='Actual Flow')
    plt.plot(dates_test, predictions, label='Predicted Flow')
    plt.title('Actual vs Predicted Traffic Flow')
    plt.xlabel('Date')
    plt.ylabel('Traffic Flow (cars)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
def evaluate_xgboost_model(model, scaler, x_test, y_test, dates_test):
    # XGBoost returns flat predictions
    predictions = model.predict(x_test)

    # Inverse scale
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    # Calculate RMSE
    rmse = np.sqrt(np.mean((y_test - predictions) ** 2))
    print(f'XGBoost RMSE: {rmse:.2f}')

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(dates_test, y_test, label='Actual Flow')
    plt.plot(dates_test, predictions, label='XGBoost Predicted Flow')
    plt.title('XGBoost: Actual vs Predicted Traffic Flow')
    plt.xlabel('Date')
    plt.ylabel('Traffic Flow (cars)')
    plt.legend()
    plt.tight_layout()
    plt.show()

data = process_data('Scats Data October 2006.xls')
scaler, (x_train, x_test, y_train, y_test, dates_train, dates_test) = split_data(data, 5, 'location', 'WARRIGAL_RD N of HIGH STREET_RD')
model, history = train_lstm_model(x_train, y_train)
evaluate_model(model, history, scaler, x_test, y_test, dates_test)

xgb_model = train_xgboost_model(x_train, y_train)
evaluate_xgboost_model(xgb_model, scaler, x_test, y_test, dates_test)
