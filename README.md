# Stock Prediction
---------------------------------------------------------------------------------------------
## Forecasting the next day stock price based on the previous 90 days of stock price performance.

An LSTM architecture is used for the model. LSTM or Long-Short-Term-Memory is a type of recurrent neural network architecture.
LSTMs are well-suited for making predictions based on time-series data and thus a suitable architecture choice for this task of prediction stock prices.

Training and testing data is derived from 12 years of historical stock data.

## Run
Clone the repository and run `pip install -r requirements.txt`

Run `python main.py` \
A prompt will appear `Enter stock symbol:` type out the stock's ticker symbol (case-insensitive). 

## Results
A figure will appear showing the resulting model's predicted prices versus the stock's actual prices. 
In addition, a next day price prediction will also be outputted on the console
