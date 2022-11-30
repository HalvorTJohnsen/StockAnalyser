import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf


def get_data(stocks, start, end):
    stock_test = yf.Ticker("MPCC.OL")
    stock = stock_test.history(end=dt.datetime.now(), start=(dt.datetime.now() - dt.timedelta(days=100)))
    stockData = stock['Close']
    stockData = stockData.to_frame()
    stockData = stockData.rename(columns={'Close':'MPCC.OL'})
    print(stockData)
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=100)

def get_data_2(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    stockData = stockData['Close']
    print(stockData)
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    return meanReturns, covMatrix

mean, deg = get_data('MPCC.OL', startDate, endDate)
hei, seg = get_data_2('MPCC.OL', startDate, endDate)