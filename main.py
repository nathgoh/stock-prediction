from stock import Stock
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers

def main():
    stock = Stock('SPY')
    df = stock.get_data()
    X_train, X_test, y_train, y_test = stock.get_train_test(df)

    model = get_model(120, df.shape[1], X_train, y_test)

def get_model(train_forecast_size, num_features, X_train, y_train):
    model = Sequential()

    model.add(layers.LSTM(
        units = train_forecast_size, 
        activation= 'tanh',
        return_sequences = True,
        dropout = 0.1, 
        input_shape = (train_forecast_size, num_features)
        ))

    model.add(layers.LSTM(
        units = train_forecast_size // 2, 
        activation = 'tanh'
        ))

    model.add(layers.Dense(30))

    model.summary()

    optimizer = optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer = optimizer, loss = 'mse', metrics=['accuracy'])

if __name__ == '__main__':
    main()