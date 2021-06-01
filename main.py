from stock import Stock
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, timedelta

import sklearn.preprocessing as preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, optimizers

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

def main():

    # Get input of what stock you want to predict
    stock_symbol = input("Enter stock symbol: ")
    stock = Stock(stock_symbol.upper())

    # Get actual name associated with the stock symbol
    stock_name = yf.Ticker(stock_symbol)
    stock_name = stock_name.info['longName']

    # Get stock data
    df = stock.get_data()
    df = df[['Close']]
    num_features = df.shape[1]

    # Get the dates associated with the train and test datasets
    train_dates = df.index[:int(df.shape[0] * 0.8)]
    test_dates = df.index[int(df.shape[0] * 0.8):]

    # Split into train and test datasets
    train_df = np.array(df[:int(df.shape[0] * 0.8)])
    test_df = np.array(df[int(df.shape[0] * 0.8):])
    all_df = np.array(df)

    # Do some scaling
    scaler = preprocessing.RobustScaler()
    train_df = scaler.fit_transform(train_df)
    test_df = scaler.transform(test_df)

    # Get the data for each datasets
    X_train, y_train, used_dates_train = create_dataset(train_df, train_dates)
    X_test, y_test, used_dates_test = create_dataset(test_df, test_dates)

    # Training
    model = make_model(num_features)
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-6, patience=5)
    history = model.fit(X_train, y_train, epochs = 30, batch_size = 64, validation_data=(X_test, y_test), callbacks = [callback])
 
    # Prediction on test set
    predictions = model.predict(X_test)
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Prediction of the next day of the stock market
    X_test_forecast = [all_df[-120:, :]]
    X_test_forecast = np.array(X_test_forecast)
    predicted_price = model.predict(X_test_forecast)
    predicted_price = scaler.inverse_transform(predicted_price)
    print("{} Today's Price on {}: {}".format(stock_name, date.today(), all_df[-1, 0]))
    print("{} Tomorrow's Predicted Price on {}: {}".format(stock_name, date.today() + timedelta(days=1), round(predicted_price[0, 0], 2)))

    plot_data(predictions, y_test, used_dates_test, stock_name)

def plot_data(predictions, y_test, dates, stock_name):
    # Plot the predicted stock prices vs their stock actual prices 
    predictions = pd.DataFrame(predictions, index = dates)
    y_test = pd.DataFrame(y_test, index = dates)

    fig, axes = plt.subplots(figsize=(16, 8))
    axes.plot(y_test, color = 'red', label = 'Original Price')
    plt.plot(predictions, color = 'cyan', label = 'Predicted Price')
    plt.title(stock_name + ": Prediction vs Actual Prices")
    plt.ylabel("Price (USD)")
    plt.xlabel("Date")
    plt.legend()
    plt.show()

def create_dataset(df, dates, look_back = 120):    
    # Dataset creation function
    X, y, used_dates = [], [], []

    for j in range(look_back, df.shape[0]):
        X.append(df[j - look_back:j, :])
        y.append(df[j, 0])
        used_dates.append(dates[j])

    X = np.array(X)
    y = np.array(y)

    used_dates = np.array(used_dates)

    return X, y, used_dates

def make_model(num_features, look_back = 120):
    # Model structure
    model = Sequential()    

    model.add(layers.LSTM(
        units = 256, 
        activation= 'tanh',
        return_sequences = True,
        input_shape = (look_back, num_features)
        ))

    model.add(layers.LSTM(
        units = 128, 
        activation = 'tanh',
        return_sequences = True,
        ))

    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(
        units = 128, 
        activation = 'tanh',
        return_sequences = True,
        ))

    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(
        units = 128, 
        activation = 'tanh',
        return_sequences = True,
        ))

    model.add(layers.Dropout(0.2))

    model.add(layers.LSTM(
        units = 64, 
        activation = 'relu'
        ))

    model.add(layers.Dense(1))

    optimizer = optimizers.Adam(learning_rate = 0.001)
    model.compile(optimizer = optimizer, loss = 'mse', metrics=['accuracy'])

    return model
      
if __name__ == '__main__':
    main()