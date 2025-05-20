from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout
from .base_model import BaseTrafficPredictionModel

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