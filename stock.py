import yfinance as yf
import pandas as pd
import numpy as np
import ta

from sklearn.preprocessing import RobustScaler


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
        # df = pd.DataFrame(RobustScaler().fit_transform(df), columns = df.columns, index = df.index)

        return df
    
    def get_train_test(self, df):
        df_array = df.to_numpy()
        train_forecast = 240 # How many days of past data you want to train for forecasting
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

    def make_model(self, num_features, train_forecast = 240):
        model = Sequential()    

        model.add(layers.LSTM(
            units = train_forecast, 
            activation= 'tanh',
            return_sequences = True,
            input_shape = (train_forecast, num_features)
            ))

        model.add(layers.LSTM(
            units = int(train_forecast * 0.75), 
            activation = 'tanh',
            return_sequences = True,
            ))

        model.add(layers.LSTM(
            units = int(train_forecast * 0.6), 
            activation = 'tanh',
            return_sequences = True,
            ))

        model.add(layers.Dropout(0.1))

        model.add(layers.LSTM(
            units = train_forecast // 2, 
            activation = 'tanh',
            return_sequences = True,
            ))

        model.add(layers.Dropout(0.2))

        model.add(layers.LSTM(
            units = train_forecast // 4, 
            activation = 'tanh'
            ))

        model.add(layers.Dense(30))

        optimizer = optimizers.Adam(learning_rate = 0.001)
        model.compile(optimizer = optimizer, loss = 'mse', metrics=['accuracy'])

        return model
      
    def evaluate_model(self, df, trained_model, train_forecast = 240, forecast = 30):
        predictions = pd.DataFrame(index = df.index, columns = [df.columns[0]])
        for j in range(train_forecast, len(df) - train_forecast, forecast):
            X = df[-j - train_forecast:-j]

            # Prediction per train_forecast size
            y_hat = model.predict(np.array(X).reshape(1, train_forecast, df.shape[0]))

            # Update predictions
            pred_df = pd.DataFrame(y_hat, index=pd.date_range(start=X.index[-1], periods=len(y_hat), freq="B"), columns=[X.columns[0]])
            predictions.update(pred_df) 

            