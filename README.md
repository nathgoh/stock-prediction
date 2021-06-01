# Stock Prediction
***
## Forecasting the next day stock price based on the previous 90 days of stock price performance.

An LSTM architecture is used for the model. LSTM or Long-Short-Term-Memory is a type of recurrent neural network architecture.
LSTMs are well-suited for making predictions based on time-series data and thus a suitable architecture choice for this task of prediction stock prices.

Training and testing data are derived from 12 years of historical stock data.

### `stock.py`
Python class to help with extracting historical stock data

### `main.py`
Python main code file that will format the data into their train and test datasets. 
Also does the model building, training, testing, and next day stock price prediction.

***
## Run
Clone the repository and run `pip install -r requirements.txt`

Run `python main.py` \
A prompt will appear `Enter stock symbol:` \
Type out the stock's ticker symbol (case-insensitive) (i.e "SPY", "WFC", "AMZN", etc.). 

## Source
This project is inspired by https://randerson112358.medium.com/predict-stock-prices-using-python-machine-learning-53aa024da20a

## Maintainers
Nathaniel Goh [@nathgoh](https://github.com/nathgoh) \
Christopher Wong [@remodera](https://github.com/remodera)
