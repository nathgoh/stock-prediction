from stock import Stock
import numpy as np

import tensorflow as tf
from tensorflow import keras

def main():

    stock_symbol = input("Enter stock symbol: ")
    stock = Stock(stock_symbol.upper())
    df = stock.get_data()
    X, y = stock.get_train_test(df)

    model = stock.make_model(df.shape[1])

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', min_delta=1e-4, patience=8)
    trained_model = model.fit(X, y, epochs = 50, batch_size = 128, validation_split=0.2, callbacks = [callback])
    
    pred = stock.evaluate_model(df, trained_model)

if __name__ == '__main__':
    main()