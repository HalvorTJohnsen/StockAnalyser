import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from pandas_datareader import data as pdr
import yfinance as yf
from dateutil.relativedelta import relativedelta


import warnings
warnings.filterwarnings('ignore')

def start_up():
    print('\n')
    print('\n')
    print("Welcome to this Monte Carlo Simulator program. To browse the availabe stocck, pleas visit:  https://finance.yahoo.com/ . The stock ticker is located behind the company name, Apple (AAPL).")
    return

def menu():
    menu_list = [1,2,3,4,5]

    print("1. Analyse Paretos model portefolio")
    print("2. Choose your own stocks")
    print("3. Test the model")
    print("4. Plot dividents from stock")
    print("5. Use machine learning to predict tomorrows stock price")

    user_input = input("Choice: ")

    try:
        val = int(user_input)

    except:
        print("Please choose a valid choice!")
        print('\n')
        menu()

    if int(user_input) in menu_list:
        return int(user_input)

    else:
        print("You have not chosen a valid choice. Please choose a valid choice!")
        menu()

def mc_pareto_simulations():
    def get_data(stocks, start, end):
        stockData = pdr.get_data_yahoo(stocks, start, end)
        stockData = stockData['Close']
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
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
    mc_sims = int(input("How many simulations would you like? ")) # number of simulations
    T = int(input("What time period would you like to simulate? ")) #timeframe in days
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

    print("1. Plot histogram")
    print("2. Print stock charts")
    print("3. Print both")

    user_input = int(input("Choice: "))


    if user_input == 1:
        plt.hist(histogram_array, bins = bins_list, edgecolor = 'black')
        plt.axvline(initialPortfolio, color='red', linestyle='dashed', linewidth=1)
        #plt.axvline(actual_return(stocks, startDate, endDate, initialPortfolio), color = 'green', linewidth = 1)
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
        #plt.axvline(actual_return(stocks, startDate, endDate, initialPortfolio), color = 'green', linewidth = 1)
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

def stock_simulations(stock_input):
    def get_data(stocks, start, end):
        stockData = pdr.get_data_yahoo(stocks, start, end)
        stockData = stockData['Close']
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return meanReturns, covMatrix


    stockList = stock_input
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
    mc_sims = int(input("How many simulations would you like? ")) # number of simulations
    T = int(input("What time period would you like to simulate? ")) #timeframe in days
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

    print("1. Plot histogram")
    print("2. Print stock charts")
    print("3. Print both")

    user_input = int(input("Choice: "))


    if user_input == 1:
        plt.hist(histogram_array, bins = bins_list, edgecolor = 'black')
        plt.axvline(initialPortfolio, color='red', linestyle='dashed', linewidth=1)
        #plt.axvline(actual_return(stocks, startDate, endDate, initialPortfolio), color = 'green', linewidth = 1)
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
        #plt.axvline(actual_return(stocks, startDate, endDate, initialPortfolio), color = 'green', linewidth = 1)
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

def get_input_stocks():
    n = 0
    stocks = []
    user_input_2 = input("How many stocks would you like to analyse? ")

    try:
        val = int(user_input_2)

    except ValueError:
        print("Please enter a real number!")
        get_input_stocks()

    while n < int(user_input_2):
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
        get_input_stocks()

def verify_model(input_stocks):
    def get_data(stocks, start, end):
        stockData = pdr.get_data_yahoo(stocks, start, end)
        stockData = stockData['Close']
        returns = stockData.pct_change()
        meanReturns = returns.mean()
        covMatrix = returns.cov()
        return meanReturns, covMatrix

    stockList = input_stocks
    stocks = [stock for stock in stockList]

    user_input = input("Over which period of time would you like to like to test the model? ")

    try:
        val = int(user_input)

    except:
        verify_model()

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
    plt.axvline(actual_return(stocks, startDate, endDate, initialPortfolio), color = 'green', linewidth = 1)
    plt.ylabel('Frequensy')
    plt.xlabel('Portifolio Value rounded ($)')
    plt.title('MC simulation of a stock portfolio, ' + print_list)
    plt.show()

def ml_stock_predictor(input_stock):
    import sklearn
    import pandas_datareader as web
    import datetime as dt
    import yfinance as yf

    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM

    import warnings
    warnings.filterwarnings('ignore')


    company = input_stock[0]

    ticker = yf.Ticker(company)
    ticker_info = ticker.info

    start = dt.datetime(2009, 1, 1)
    end = dt.datetime(2021, 2, 1)
    
    data = web.DataReader(company, 'yahoo', start, end)

    ### Preparing the data

    scaler = MinMaxScaler(feature_range=(0, 1))

    ## Scaling just the closing value after the market have been closed

    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
    
    prediction_days = 90

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
    """
    plt.plot(actual_price, color='black', label=f'Actual {actual_price} Price')
    plt.plot(predicted_price, color='green', label=f'Predicted {predicted_price} Price')
    plt.title(f"{company} Share Price")
    plt.xlabel("Time")
    plt.ylabel(f'{company} Share Price')
    plt.legend()
    plt.show()"""

    ## Predict the next day

    real_data = [model_inputs[len(model_inputs) + 1 - prediction_days:len(model_inputs+1), 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    price = ticker.info['regularMarketPrice']

    change = (prediction - price)*100 / price

    print(str(ticker_info['shortName']) + " is predicted to close at " + str(prediction) + ". That is a change of " + str(change))

def main():
    user_input = menu()

    if user_input == 1:
        mc_pareto_simulations()

    elif user_input == 2:
        stocks = get_input_stocks()
        stock_simulations(stocks)

    elif user_input == 3:
        stocks = get_input_stocks()
        verify_model(stocks)

    elif user_input == 4:
        stocks = get_input_stocks()
        plot_dividents(stocks)

    elif user_input == 5:
        stocks = get_input_stocks()
        ml_stock_predictor(stocks)

    main() 

def plot_dividents(input_stocks):   
    from sklearn.preprocessing import MinMaxScaler
    from tensorflow import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, LSTM
    import sklearn
    import pandas_datareader as web

    TICKER = yf.Ticker(input_stocks[0])

    years = int(input("How many years back do you want to colllect information? "))
    TICKER_INFO = TICKER.info

    TO = dt.datetime.now().strftime("%Y-%m-%d")
    FROM = (dt.datetime.now() - relativedelta(years=years)).strftime("%Y-%m-%d")

    DIV_HISTORY = TICKER.dividends
    df = DIV_HISTORY.to_frame()
    
    df = df[(df.index > FROM) & (df.index <= TO)]
    df.reset_index(inplace=True)
    df = df.rename(columns = {'index':'DATE'})
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    df.plot(x="Date", y='Dividends', kind='bar')
    plt.title("Dividends in " + str((TICKER_INFO['currency'])))
    plt.show()