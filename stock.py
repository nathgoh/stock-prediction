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

    def get_historical_data(self, period = '2y', interval = '1d'):
        """
        Get the historical stock data using the yfinance API
        """
        stock = yf.Ticker(self.symbol)
        return stock.history(period = period, interval = interval)
        
    
