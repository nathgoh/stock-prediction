from stock import Stock
import numpy as np

def main():
    stock = Stock('SPY')
    df = stock.get_data()
    X, y = stock.get_train_test(df)

    model = stock.make_model(df.shape[1])
    model.summary()

    trained_model = model.fit(X, y, epochs = 50, batch_size = 128, validation_split=0.2)
    
if __name__ == '__main__':
    main()