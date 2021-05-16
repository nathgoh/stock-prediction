import yfinance as yf
import pandas as pd
import numpy as np
import ta

# from sklearn import preprocessing
# from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers

class Stock():
    def __init__(self, symbol):
        """
        Basic interface for stock predictor

        Parameters
        ----------
        symbol : str
            Get the stock symbol you want to predict
        """
        self.symbol = symbol

    def get_data(self, period = '2y', interval = '1d'):
        """
        Get the historical stock data using the yfinance API
        Then pre-process data to be used for training and testing

        Parameters
        ----------
        period : str
            How far back we want to get historical data using
        interval : str
            Interval of historical data
        """
        
        # Get data
        stock = yf.Ticker(self.symbol)
        df = stock.history(period = period, interval = interval)
        df.dropna(inplace = True)

        # Get the features we want to use
        df = ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna = True)
        df.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis = 1, inplace = True)
        # df_closed = df[['Close']]

        return df
    
    def get_train_test(self, df):
        df_array = df.to_numpy()
        train_forecast = 120 # How many days of past data you want to train for forecasting
        forecast = 30 # How many days we want to predict into the future

        X, y = [], []
        for j in range(len(df_array)):
            train_end = j + train_forecast
            end = train_end + forecast

            if end > len(df_array):
                break

            X.append(df_array[j:train_end, :])
            y.append(df_array[train_end:end, 0])

        X = np.array(X)
        y = np.array(y)
        return X, y

    def make_model(self, num_features, train_forecast_size = 120):
        model = Sequential()    

        model.add(layers.LSTM(
            units = train_forecast_size, 
            activation= 'tanh',
            return_sequences = True,
            input_shape = (train_forecast_size, num_features)
            ))

        model.add(layers.LSTM(
            units = int(train_forecast_size * 0.75), 
            activation = 'tanh',
            return_sequences = True,
            ))

        model.add(layers.LSTM(
            units = train_forecast_size // 2, 
            activation = 'tanh',
            return_sequences = True,
            ))

        model.add(layers.Dropout(0.2))

        model.add(layers.LSTM(
            units = train_forecast_size // 4, 
            activation = 'tanh'
            ))

        model.add(layers.Dense(30))

        optimizer = optimizers.Adam(learning_rate = 0.001)
        model.compile(optimizer = optimizer, loss = 'mse', metrics=['accuracy'])

        return model
      
