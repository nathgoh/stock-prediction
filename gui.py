import tkinter as tk
from stock import Stock

def gui():
    window = tk.Tk() 
    window.title("Stock Predictor")
    window.geometry('700x500')

    label = tk.Label(window, text = "")
    def get_stock():
        stock_symbol = input_stock_symbol.get(1.0, 'end-1c')
        label.config(text = "Stock: {}".format(stock_symbol))

    input_stock_symbol = tk.Text(
        window,
        height = 2,
        width = 10,
    )
    input_stock_symbol.pack()

    predict_button = tk.Button(
        window,
        text = "Predict Stock",
        width = 10,
        height = 2,
        command = get_stock
    )
    predict_button.pack()

    
    label.pack()
    window.mainloop()


if __name__ == '__main__':
    gui()