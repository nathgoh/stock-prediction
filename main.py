from stock import Stock

def main():
    stock = Stock('SPY')
    df = stock.get_data()
    print(df)
    # stock.get_train_test(df)

if __name__ == '__main__':
    main()