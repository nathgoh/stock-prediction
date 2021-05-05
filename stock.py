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

    def get_data(self, period = '2y', interval = '1d'):
        """
        Get the historical stock data using the yfinance API
        """
        stock = yf.Ticker(self.symbol)
        df = stock.history(period = period, interval = interval)
        df = ta.add_all_ta_features(df, 'Open', 'High', 'Low', 'Close', 'Volume', fillna = True)

        return df
    
    def get_train_test(df):
        train_length = int(len(df) * 0.8)

        train = df.iloc[:train_length, :]
        test = df.iloc[train_length:, :]
        
    
