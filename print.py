import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf

# import data
def get_data(stocks, start, end):
    stockData = pdr.get_data_yahoo(stocks, start, end)
    print(list(stockData))
    print(stockData.info())
    stockData = stockData['Close']
    #print(stockData)
    returns = stockData.pct_change()
    meanReturns = returns.mean()
    covMatrix = returns.cov()
    #print(returns)
    return meanReturns, covMatrix

stockList = ['AKRBP.OL', 'AUSS.OL', 'COOL.OL', 'GJF.OL', 'HAFNI.OL', 'KOG.OL', 'MOWI.OL', 'NORAM.OL', 'NSKOG.OL']
stocks = [stock for stock in stockList]
endDate = dt.datetime.now()
startDate = endDate - dt.timedelta(days=300)

#endDate = dt.datetime.now() -  dt.timedelta(days=100)
#startDate = dt.datetime.now() - dt.timedelta(days=400)

#Calculating actual value of investment 
def actual_return(stocks, start, end, initial_portifolio):
    stockData = pdr.get_data_yahoo(stocks, end, dt.datetime.now())
    stockData = stockData['Close']
    change = (stockData.iloc[-1, 0] - stockData.iloc[0, 0])/(stockData.iloc[0,0])
    initial_portifolio = initial_portifolio * (1 + change)
    pct_change = 100 * change
    print("Change in %: " + str(pct_change))
    return initial_portifolio
meanReturns, covMatrix = get_data(stocks, startDate, endDate)
weights = np.random.random(len(meanReturns))
weights /= np.sum(weights)

# Monte Carlo Method
mc_sims = 100000 # number of simulations
T = 100 #timeframe in days
meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
meanM = meanM.T
portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
initialPortfolio = 10000
for m in range(0, mc_sims):
    Z = np.random.normal(size=(T, len(weights)))#uncorrelated RV's
    L = np.linalg.cholesky(covMatrix) #Cholesky decomposition to Lower Triangular Matrix
    dailyReturns = meanM + np.inner(L, Z) #Correlated daily returns for individual stocks
    portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

start_value = initialPortfolio / 10
histogram_array = [] 

for value in portfolio_sims:
    histogram_array.append(np.round(value, -2))

histogram_array = histogram_array[len(histogram_array) - 1]

bins_list = np.unique(histogram_array)

print_list = ', '.join(stockList)

plt.hist(histogram_array, bins = bins_list, edgecolor = 'black')
plt.axvline(initialPortfolio, color='red', linestyle='dashed', linewidth=1)
#plt.axvline(actual_return(stocks, startDate, endDate, initialPortfolio), color = 'green', linewidth = 1)
plt.ylabel('Frequensy')
plt.xlabel('Portifolio Value rounded ($)')
plt.title('MC simulation of a stock portfolio, ' + print_list)
plt.show()
"""
plt.plot(portfolio_sims)
plt.ylabel('Portfolio Value ($)')
plt.xlabel('Days')
plt.title('MC simulation of a stock portfolio ' + stockList[0])
plt.show()
actual_return(stocks, startDate_1, endDate_1)"""

#get_data_2(stocks, startDate, endDate)