from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from .base_model import BaseTrafficPredictionModel

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