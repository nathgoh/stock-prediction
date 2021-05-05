from stock import Stock

def main():
    stock = Stock('SPY')
    print(stock.get_data())

if __name__ == '__main__':
    main()