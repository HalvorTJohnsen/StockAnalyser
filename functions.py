import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
from tabulate import tabulate
from dateutil.relativedelta import relativedelta
from threading import Thread
import os
import random
import sys
import time
import warnings
warnings.filterwarnings('ignore')


def startUp():
    os.system('cls||clear')
    print("Welcome to this Monte Carlo Simulator program. To browse the available stocks, please visit:  https://finance.yahoo.com/. The stock ticker is located behind the company name, Apple (AAPL).")
    print('\n')
    return

def readSettingsFromJson():
    import json

    settingsList =  []
    with open("settings.json", "r", encoding="utf-8") as f:
        settings = json.load(f)

    settingsList.append(settings['monte_carlo_simulations_settings']['get_data_from_monte_carlo_simulations'])
    settingsList.append(settings['monte_carlo_simulations_settings']['initial_portifolio'])
    settingsList.append(settings['portifolio_optimalization']['get_data_from_portifolio_optimalization'])
    settingsList.append(settings['portifolio_optimalization']['number_of_portifolios'])

    f.close()
    return settingsList

def portifolioOptimalization(inputStocks):
    
    stocks = inputStocks

    settingsList = readSettingsFromJson()

    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=settingsList[-2])

    portifolioData = pdr.get_data_yahoo(stocks, startDate, endDate)['Close']

    portifolioReturns = portifolioData.pct_change()
    numberOfStocks = len(inputStocks)

    weights = 1/numberOfStocks
    portifolioWeights = []

    for i in range(0, numberOfStocks): portifolioWeights.append(weights)

    retunPortiofolio = portifolioReturns.dot(portifolioWeights)

    varianceMatrix = portifolioReturns.cov()*252

    portifolioVariance = np.transpose(portifolioWeights)@varianceMatrix@portifolioWeights

    portifolioVolatility = np.sqrt(portifolioVariance)

    portifolioReturnsList = []
    portifolioVolatilityList = []
    portifolioWeightsList = []

    numberOfAssets = len(portifolioData.columns)
    numberOfPortifolios = settingsList[-1]

    individualReturns = portifolioData.resample('Y').last().pct_change().mean()

    for port in range(numberOfPortifolios):
        weights = np.random.random(numberOfAssets)
        weights = weights/np.sum(weights)
        portifolioWeightsList.append(weights)
        returns = np.dot(weights, individualReturns)
        portifolioReturnsList.append(returns)

        var = varianceMatrix.mul(weights, axis=0).mul(weights, axis=1).sum().sum()
        sd = np.sqrt(var)

        ann_sd = sd*np.sqrt(var)
        portifolioVolatilityList.append(ann_sd)

    
    data  = {'Returns':portifolioReturnsList, 'Volatility':portifolioVolatilityList}

    for counter, symbol in enumerate(portifolioData.columns.to_list()):
        data[symbol+' weight']=[w[counter] for w in portifolioWeightsList]
    
    portifolio_V1 = pd.DataFrame(data)

    portifolio_V1.plot.scatter(x='Volatility', y='Returns', marker='o', color='b', s=2, alpha=0.5, grid=True, figsize=[8,8])
    plt.ylabel("Expected Returns")
    plt.xlabel("Risk (Volatility)")
    plt.show()

    minimumVolatilityPortifolio = portifolio_V1.iloc[portifolio_V1['Volatility'].idxmin()]
    minimumVolatilityPortifolio = minimumVolatilityPortifolio.to_frame()*100

    rf = 0.01
    optimalVolatilityPortifolio = portifolio_V1.iloc[((portifolio_V1['Returns']-rf)/portifolio_V1['Volatility']).idxmax()]
    optimalVolatilityPortifolio= optimalVolatilityPortifolio*100

    print("Minmimum volatility portifolio (weight in %)")
    print(minimumVolatilityPortifolio)
    print("Highest sharpe ratio (weight in %)")
    print(optimalVolatilityPortifolio)
    

    close = input("Close: ")

def menu():
    menuList = [1,2,3,4,5,6,7, 8, 9]

    print("1. Monte Carlo Simulator")
    print("2. Test the Monte Carlo model")
    print("3. Plot dividents from stock")
    print("4. Predict tomorrows stock price with machine learning")
    print("5. Plot stock trend history")
    print("6. Print stock info")
    print("7. Portifolio optimalizator")
    print("8. Clear terminal")
    print('\n')
    print("9. Edit values for the Monte Carlo Simulator")

    while True:
        choice = input("Choice: ")

        try:
            testVal = int(choice)
            if int(choice) in menuList:
                return int(choice)

            else:
                print("You have not chosen a valid choice. Please choose a valid choice!")


        except:
            print("Please choose a valid choice!")

def readPortifolioFromTxt():
    portifolioTxt = open("portifolio.txt", "r")

    portifolioStocks = portifolioTxt.read()

    stocksIntoList = portifolioStocks.split('\n')

    stockListToReturn = []

    for item in stockListToReturn: stockListToReturn.append(item.upper())
    
    portifolioTxt.close()

    return stockListToReturn

def monteCarloSimulation(inputStock):
    def get_data(stocks, start, end):
        stockData = pdr.get_data_yahoo(stocks, start, end)
        stockData = stockData['Close']
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return meanReturns, covMatrix

    settingsList = readSettingsFromJson()

    stockList = inputStock
    stocks = [stock for stock in stockList]
    endDate = dt.datetime.now()
    startDate = endDate - dt.timedelta(days=int(settingsList[0]))

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
    mc_sims = int(input("How many simulations would you like? ")) # number of simulations
    T = int(input("What time period would you like to simulate? ")) #timeframe in days
    meanM = np.full(shape=(T, len(weights)), fill_value=meanReturns)
    meanM = meanM.T
    portfolio_sims = np.full(shape=(T, mc_sims), fill_value=0.0)
    initialPortfolio = (settingsList[1])
    for m in range(0, mc_sims):
        Z = np.random.normal(size=(T, len(weights)))#uncorrelated RV's
        L = np.linalg.cholesky(covMatrix) #Cholesky decomposition to Lower Triangular Matrix
        dailyReturns = meanM + np.inner(L, Z) #Correlated daily returns for individual stocks
        portfolio_sims[:,m] = np.cumprod(np.inner(weights, dailyReturns.T)+1)*initialPortfolio

    histogram_array = [] 

    for value in portfolio_sims:
        histogram_array.append(np.round(value, -2))

    histogram_array = histogram_array[len(histogram_array) - 1]

    bins_list = np.unique(histogram_array)

    print_list = ', '.join(stockList)

    print("1. Plot histogram")
    print("2. Print stock charts")
    print("3. Print both")

    user_input = int(input("Choice: "))


    if user_input == 1:
        plt.hist(histogram_array, bins = bins_list, edgecolor = 'black')
        plt.axvline(initialPortfolio, color='red', linestyle='dashed', linewidth=1)
        plt.ylabel('Frequensy')
        plt.xlabel('Portifolio Value rounded ($)')
        plt.title('MC simulation of a stock portfolio, ' + print_list)
        plt.show()
        return

    elif user_input == 2:
        plt.plot(portfolio_sims)
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Days')
        plt.title('MC simulation of a stock portfolio ' + stockList[0])
        plt.show()
        actual_return(stocks, startDate, endDate)
        return

    elif user_input == 3:
        plt.hist(histogram_array, bins = bins_list, edgecolor = 'black')
        plt.axvline(initialPortfolio, color='red', linestyle='dashed', linewidth=1)
        plt.ylabel('Frequensy')
        plt.xlabel('Portifolio Value rounded ($)')
        plt.title('MC simulation of a stock portfolio, ' + print_list)
        plt.show()

        plt.plot(portfolio_sims)
        plt.ylabel('Portfolio Value ($)')
        plt.xlabel('Days')
        plt.title('MC simulation of a stock portfolio ' + print_list)
        plt.show()

        return

def getInputStocksFromTerminal():
    stateVariabel = 0

    n = 0
    stocks = []
    print("1. Choose stocks, and how many stocks would you like to analyse?")
    print("2. Import stocks from txt file")
    print("3. Load Pareto model portifolio")
    choice = input("Choice: ")

    try:
        val = int(choice)

    except ValueError:
        print("Please enter a valid choice!")
        getInputStockFromTerminal()

    if int(choice) == 1:

        while stateVariabel == 0:
            numberOfStocks = input("How many stocks would you like to analyse? ")

            try:
                val = int(numberOfStocks)
                stateVariabel = 1

            except:
                print("Enter a real number!")

  
        while n < int(numberOfStocks):
            stock = input("Enter your stock ticker: ")
            try:
                val = pdr.get_data_yahoo(stock)
                stocks.append(stock.upper())
                n += 1
            except:
                print("The ticker does not exist!")


        print("You have selected: ", [stock for stock in stocks])

        confirm = str(input("Is this correct? [Y/n] "))

        if confirm == "y" or confirm == '' or confirm == 'Y':
            return stocks

        else:
            getInputStockFromTerminal() 
    
    elif int(choice) == 2:
        txtStockList = readPortifolioFromTxt()

        return txtStockList
    
    elif int(choice) == 3:
        paretoModelPortifolioList = loadParetoPortifolio()
        
        return paretoModelPortifolioList

    else:
        getInputStockFromTerminal()

def getInputStockFromTerminal():
    n = 0
    while n < 1:
        stock = input("Enter your stock ticker: ")
        try:
            testVal = pdr.get_data_yahoo(stock)
            n += 1
        except:
            print("The ticker does not exist!")


    print("You have selected: ", str(stock).upper())

    confirm = str(input("Is this correct? [Y/n] "))

    if confirm == "y" or confirm == '' or confirm == 'Y':
        return stock.upper()

    else:
        getInputStockFromTerminal()

def testMonteCarloSimulation(inputStock):
    def get_data(stocks, start, end):
        stockData = pdr.get_data_yahoo(stocks, start, end)
        stockData = stockData['Close']
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return meanReturns, covMatrix

    stockList = inputStock
    stocks = [stock for stock in stockList]

    user_input = input("Over which period of time would you like to like to test the model? ")

    try:
        val = int(user_input)

    except:
        testMonteCarloSimulation()

    endDate = dt.datetime.now() -  dt.timedelta(days=int(user_input))
    startDate = dt.datetime.now() - dt.timedelta(days=300 + int(user_input))

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
    mc_sims = int(input("How many simulations would you like to run? "))
    T = int(user_input)
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
    plt.axvline(actual_return(stocks, startDate, endDate, initialPortfolio), color = 'yellow', linewidth = 1)
    plt.ylabel('Frequensy')
    plt.xlabel('Portifolio Value rounded ($)')
    plt.title('MC simulation of a stock portfolio, ' + print_list)
    plt.show()

def machineLearningStockPredictor(inputStock):
    import sklearn
    import pandas_datareader as web
    import datetime as dt
    import yfinance as yf

    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras
    from keras.models import Sequential
    from time import sleep
    from keras.layers import Dense, Dropout, LSTM

    import warnings
    warnings.filterwarnings('ignore')

    company = inputStock

    ticker = yf.Ticker(company)
    ticker_info = ticker.info

    start = dt.datetime(2009, 1, 1)
    end = dt.datetime(2021, 2, 1)
    
    data = web.DataReader(company, 'yahoo', start, end)

    ### Preparing the data

    scaler = MinMaxScaler(feature_range=(0, 1))

    ## Scaling just the closing value after the market have been closed

    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    prediction_days = 356

    x_train = []
    y_train = []

    for x in range(prediction_days, len(scaled_data)):
        x_train.append(scaled_data[x-prediction_days:x, 0])
        y_train.append(scaled_data[x, 0])

    ## Converting into numpy array

    x_train, y_train = np.array(x_train), np.array(y_train)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

    ## Building the model

    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  ## Prediction of the next closing price

    model.compile(optimizer='Adam', loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=32)

    ''' Testing the mode '''

    ## Loading the test data

    test_start = dt.datetime(2021, 2, 1)
    test_end = dt.datetime.now()

    test_data = web.DataReader(company, 'yahoo', test_start, test_end)
    actual_price = test_data['Close'].values

    ## Concatinating the close values of the train data and the test data

    total_data = pd.concat((data['Close'], test_data['Close']), axis=0)

    model_inputs = total_data[len(total_data) - len(test_data) - prediction_days:].values
    model_inputs = model_inputs.reshape(-1, 1)
    model_inputs = scaler.transform(model_inputs)

    ## Making prediction from the test data

    x_test = []

    for x in range(prediction_days, len(model_inputs)):
        x_test.append(model_inputs[x-prediction_days:x, 0])
        
    ## Converting into numpy array
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    ## Making predictions based on the x_test
    predicted_price = model.predict(x_test)

    ### Reversing the scaled prices
    predicted_prices = scaler.inverse_transform(predicted_price)

    ### Plotting the test predictions
    
    plt.plot(actual_price, color='black', label=f'Actual {actual_price} Price')
    plt.plot(predicted_price, color='green', label=f'Predicted {predicted_price} Price')
    plt.title(f"{company} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()

    ## Predict the next day

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    price = ticker.info['regularMarketPrice']

    change = (prediction - price)*100 / price

    print(str(ticker_info['shortName']) + " is predicted to close at " + str(prediction) + ". That is a change of " + str(change))

    sleep(24)

def plotDividentsFromStock(inputStock):   

    stock = yf.Ticker(inputStock)

    yearsToSimulate = int(input("How many years back do you want to colllect information? "))
    stockInfo = stock.info

    simulateTo = dt.datetime.now().strftime("%Y-%m-%d")
    simulateFrom = (dt.datetime.now() - relativedelta(years=yearsToSimulate)).strftime("%Y-%m-%d")

    stockHistory = stock.dividends
    df = stockHistory.to_frame()
    
    df = df[(df.index > simulateFrom) & (df.index <= simulateTo)]
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'DATE'})
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    df.plot(x="Date", y='Dividends', kind='bar')
    plt.title("Dividends in " + str((stockInfo['currency'])))
    plt.show()

def plotStockTrendHistory(inputStock):
    stock = inputStock

    stockData = yf.download(stock,'2015-1-1')['Adj Close']

        # Plot all the close prices
    ((stockData.pct_change()+1).cumprod()).plot(figsize=(10, 7))

    # Show the legend
    plt.legend()

    # Define the label for the title of the figure
    plt.title("Returns", fontsize=16)

    # Define the labels for x-axis and y-axis
    plt.ylabel('Cumulative Returns', fontsize=14)
    plt.xlabel('Year', fontsize=14)

    # Plot the grid lines
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.show()

def clearTerminalWindow():
    os.system('cls||clear')

"""def getInfoFromTicker(inputStock):
    clearTerminalWindow()
    stock = yf.Ticker(inputStock)
    stockInfo = stock.info

    longBusinessSummary = stockInfo.get('longBusinessSummary', 'Key not found')

    today = dt.date.today()

    oneWeekStockData = stock.history(start=today - relativedelta(days=5))['Close']
    oneMonthStockData = stock.history(start=today - relativedelta(months=1))['Close']
    sixMonthStockData = stock.history(start=today - relativedelta(months=6))['Close']
    oneYearStockData = stock.history(start=today - relativedelta(years=1))['Close']
    threeYearStockData = stock.history(start=today - relativedelta(years=3))['Close']
    fiveYearStockData = stock.history(start=today - relativedelta(years=5))['Close']

    oneWeekPrecetageChange = (oneWeekStockData[-1] - oneWeekStockData[0])*100/oneWeekStockData[0]
    oneMonthPrecentageChange = (oneMonthStockData[-1] - oneMonthStockData[0])*100/oneMonthStockData[0]
    sixMonthPrecentageChange = (sixMonthStockData[-1] - sixMonthStockData[0])*100/sixMonthStockData[0]
    oneYearPrecentageChange  = (oneYearStockData[-1] - oneYearStockData[0])*100/oneYearStockData[0]
    threeYearPrecentageChange = (threeYearStockData[-1] - threeYearStockData[0])*100/threeYearStockData[0]
    fiveYearStockDataChange = (fiveYearStockData[-1] - fiveYearStockData[0])*100/fiveYearStockData[0]


    summaryTable1 = [['Previous Close', 'Open', 'Bid', 'Ask', 'Day\'s Range', '52 Week Range', 'Volume', 'Avg. Volume'], [stockInfo['previousClose'], stockInfo['open'], str(stockInfo['bid']) + str(" x ") + str(stockInfo['bidSize']), str(stockInfo['ask']) + str(" x ") + str(stockInfo['askSize']), str(round(stockInfo['dayLow'], 2)) + str(' - ') + str(round(stockInfo['dayHigh'], 2)), str(round(stockInfo['fiftyTwoWeekLow'], 2)) + str(" - ") + str(round(stockInfo['fiftyTwoWeekHigh'], 2)), stockInfo['volume'], stockInfo['averageVolume']]]

    stockChangeTable = [['1 Week (%)', '1 Month (%)', '6 Months (%)', '1 year (%)', '3 Years (%)', '5 Years (%)'], [round(oneWeekPrecetageChange, 3), round(oneMonthPrecentageChange,3), round(sixMonthPrecentageChange, 3), round(oneYearPrecentageChange, 3), round(threeYearPrecentageChange,3), round(fiveYearStockDataChange,3)]]

    marginsTable = [['Operating Margins', 'Ebitda Margins', 'Gross Margins', 'Profit Margins'], [stockInfo['operatingMargins'], stockInfo['ebitdaMargins'], stockInfo['grossMargins'], stockInfo['profitMargins']]]

    print(longBusinessSummary)
    print('\n')
    print(tabulate(summaryTable1, tablefmt='fancy_grid'))
    print('\n')
    print(tabulate(marginsTable, tablefmt='fancy_grid'))
    print('\n')
    print(tabulate(stockChangeTable, tablefmt='fancy_grid'))
    print('\n')

    while True:
        closeWindow = input("Enter the command \"close\" to close view. ")

        if closeWindow == "close" or closeWindow == "Close":
            return
            """

def loadParetoPortifolio():
    stockList = ['AKRBP.OL', 'AUSS.OL', 'COOL.OL', 'GJF.OL', 'HAFNI.OL', 'KOG.OL', 'MOWI.OL', 'NORAM.OL', 'NSKOG.OL']
    return stockList

def editSettingsFromJson():
    import json

    settingsList =  []
    with open("settings.json", "r", encoding="utf-8") as g:
        settings = json.load(g)
        
        print("Current settings:")
        print("Collected data from (Days): " + str(settings['monte_carlo_simulations_settings']['get_data_from_monte_carlo_simulations']))
        print("Initial portifolio (USD): " + str(settings['monte_carlo_simulations_settings']['initial_portifolio']))
        print('\n')
        print("1. Edit the period for data collection")
        print("2. Edit the initial porifolio")
        print("3. Exit to menu")

    g.close()

    global choice

    while True:
        validChoices = [1,2,3]
        choice = input("Choice: ")

        try:
            val = int(choice)
            if int(choice) in validChoices: break
        
        except:
            print("Enter a valid choice!")


    with open("settings.json", "r+", encoding="utf-8") as f:
        settings = json.load(f)

        if int(choice) == 1:
            newDataRange = int(input("New value:"))
            settings['monte_carlo_simulations_settings']['get_data_from'] = newDataRange

            f.seek(0)
            json.dump(settings, f)
            f.truncate()
        
        if int(choice) == 2:
            newDataRange = int(input("New value:"))
            settings['monte_carlo_simulations_settings']['initial_portifolio'] = newDataRange

            f.seek(0)
            json.dump(settings, f)
            f.truncate()
        
        if int(choice) == 3:
            return

    f.close()

def progressbar():
    toolbar_width = 40

    # setup toolbar
    sys.stdout.write("[%s]" % (" " * toolbar_width))
    sys.stdout.flush()
    sys.stdout.write("\b" * (toolbar_width+1)) # return to start of line, after '['

    for i in range(toolbar_width):
        time.sleep(0.1) # do real work here
        # update the bar
        sys.stdout.write("-")
        sys.stdout.flush()

    sys.stdout.write("]\n") # this ends the progress bar

def main():
    choice = menu()

    if choice == 1:
        stocks = getInputStocksFromTerminal()
        monteCarloSimulation(stocks)

    elif choice == 2:
        stocks = getInputStocksFromTerminal()
        testMonteCarloSimulation(stocks)

    elif choice == 3:
        stock = getInputStockFromTerminal()
        plotDividentsFromStock(stock)

    elif choice == 4:
        stocks = getInputStockFromTerminal()
        machineLearningStockPredictor(stock)

    elif choice == 5:
        stocks = getInputStocksFromTerminal()
        plotStockTrendHistory(stocks)

    elif choice == 6:
        stock = getInputStockFromTerminal()
        #getInfoFromTicker(stock)

    elif choice == 7:
        stocks = getInputStocksFromTerminal()
        portifolioOptimalization(stocks)

    elif choice == 8:
        clearTerminalWindow()

    elif choice == 9:
        clearTerminalWindow()
        editSettingsFromJson()


    clearTerminalWindow()
    startUp()
    main()


# test