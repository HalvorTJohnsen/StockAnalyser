import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
from functions import mc_pareto_simulations, stock_simulations, get_input_stocks, verify_model


def start_up():
    print("Welcome to this Monte Carlo Simulator program. To browse the availabe stocck, pleas visit:  https://finance.yahoo.com/ . The stock ticker is located behind the company name, Apple (AAPL).")
    return

def menu():
    print("1. Analyse Paretos model portefolio")
    print("2. Choose your own stocks")
    print("3. Test the model")

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
    
    elif int(user_input) == 3:
        return 3

    else:
        print("You have not chosen a valid choice. Please choose a valid choice!")
        menu()

start_up()
user_input = menu()

if user_input == 1:
    mc_pareto_simulations

elif user_input == 2:
    stocks = get_input_stocks()
    stock_simulations(stocks)

elif user_input == 3:
    stocks = get_input_stocks()
    verify_model(stocks)