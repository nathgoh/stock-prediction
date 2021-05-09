import yfinance as yf
import pandas as pd
import numpy as np
import ta

# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from sklearn import preprocessing

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

        # Scaling the features
        # scaled_close_prices = preprocessing.RobustScaler()
        # scaled_close_prices.fit(df[['Close']])
        # scaled_df = preprocessing.RobustScaler()
        # df = pd.DataFrame(scaled_df.fit_transform(df), columns = df.columns, index = df.index)

        df_closed = df[['Close']]

        return df_closed
    
    def get_train_test(self, df):
        forecast = 30 # How many days we want to predict into the future

        df["Prediction"] = df[['Close']].shift(-forecast)

        print(df)
         
    
