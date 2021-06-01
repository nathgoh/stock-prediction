# Stock Prediction
***
## Forecasting the next day stock price based on the previous 120 days of stock price performance.

An LSTM architecture is used for the model. LSTM or Long-Short-Term-Memory is a type of recurrent neural network architecture.
LSTMs are well-suited for making predictions based on time-series data and thus a suitable architecture choice for this task of prediction stock prices.

Training and testing data are derived from 12 years of historical stock data.

### `stock.py`
Python class to help with extracting historical stock data.

### `main.py`
Python main code file that will format the data into their train and test datasets. 
Also does the model building, training, testing, and next day stock price prediction.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`def main()` \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Main body of code that will create the 80/20 split of training and testing datasets. Our training feature of this model will just be the &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;closing prices of the stock. In addition, the data will be scaled using `RobustScalar()` in hopes that spikes in stock prices won't skewed the training too much.
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`def plot_data()` \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Function to graph prediction stock prices vs actual stock prices. 
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`def create_dataset()` \
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Create the `X` (input), `y` (output) datasets. The `X` datasets is essentially an array of groupings of 120 days worth of historical stock &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;prices. These 120 day groupings correlate to a single `y` or output value that represents a stock's next day price. 
 
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;`def make_model()`\
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Model structure, consisting of LSTM layers and some with dropout applied to help prevent overfitting.



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
