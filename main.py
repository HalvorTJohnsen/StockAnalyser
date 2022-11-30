import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
from functions import mc_pareto_simulations


def start_up():
    print("Welcome to this Monte Carlo Simulator program. To browse the availabe stocck, pleas visit:  https://finance.yahoo.com/ . The stock ticker is located behind the company name, Apple (AAPL).")
    return

def menu():
    print("1. Analyse Paretos model portefolio")
    print("2. Choice your own stocks")
    user_input = input("Choice: ")

    try:
        val = int(user_input)

    except:
        print("Please choose a valid choice!")
        print('\n')
        menu()

    if int(user_input) == 1:
        mc_pareto_simulations()
        return 1

    elif int(user_input) == 2:
        #Run own model
        print("Kjører")
        return 2

    else:
        print("You have not chosen a valid choice. Please choose a valid choice!")
        menu()

start_up()
user_input = menu()

print(user_input)

if user_input == 1:
    mc_pareto_simulations



def get_input_stocks():
    stocks = []
    print("1. Analyse Paretos model portefolio")
    print("2. Choice your own stocks")
    user_input_1 = input("Choice: ")

    try:
        val = int(user_input_1)

    except:
        print("Please choose a valid choice")
        get_input_stocks()

    if user_input_1 == 1:
        #paretoprotefølje
        ss = 2


    user_input_2 = input("How many stocks would you like to analyse? ")

    try:
        val = int(user_input_2)

    except ValueError:
        print("Please enter a real number!")
        get_input_stocks()

    for i in range(0, int(user_input_2)):
        stock = input("Enter your stock ticker: ")
        try:
            val = pdr.get_data_yahoo(stock)
        except:
            print("The ticker does not exist!")
            get_input_stocks()


        stocks.append(stock.upper())


    print("You have selected: ", [stock for stock in stocks])

    confirm = str(input("Is this correct? [Y/n] "))

    if confirm == "Y":
        sims = input("Enter the number of simulations you would like: ")
        
        try:
            val = int(sims)
        except:
            get_input_stocks()
        
        time = input("Enter the time period: ")

        try:
            val = int(time)
            return  stocks, time, sims
            
        except:
            get_input_stocks()

    else:
        get_input_stocks()
    


pareto_stocks = ['AKRBP.OL', 'AUSS.OL', 'COOL.OL', 'GJF.OL', 'HAFNI.OL', 'KOG.OL', 'MOWI.OL', 'NORAM.OL', 'NSKOG.OL']


