import tkinter as tk
from stock import Stock

def main():
    stock = Stock('SPY')
    gui() 

def gui():
    window = tk.Tk()   
    window.mainloop()   

if __name__ == '__main__':
    main()