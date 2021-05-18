import yfinance as yf
import pandas as pd
import numpy as np
import ta
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

    def get_data(self, period = '10y', interval = '1d'):
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
        # df = ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna = True)
        # df.drop(['Open', 'High', 'Low', 'Volume', 'Dividends', 'Stock Splits'], axis = 1, inplace = True) 

        return df
    
   