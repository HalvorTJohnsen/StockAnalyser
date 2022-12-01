# Importing Datasets
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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

## Loading the data

company = 'FREY'

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