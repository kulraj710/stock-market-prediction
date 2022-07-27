import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM


# load data
company = 'RELIANCE.NS'

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2022, 1, 1)

tickerData = yf.Ticker(company)

data = tickerData.history(start=start, end=end)



# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


predictions_days = 60

x_train = []
y_train = []

for x in range(predictions_days, len(scaled_data)):
    x_train.append(scaled_data[x - predictions_days:x, 0])
    y_train.append(scaled_data[x, 0])
    
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# building the model

model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1)) # prediction of the next price


model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, batch_size=32, epochs=25)

''' Test the Model Accuracy on Existing Data'''

# load Test Data

test_start = dt.datetime(2020,1,1)
test_end = dt.datetime.now()

test_data = tickerData.history(start=test_start, end=test_end)

actual_prices = test_data['Close'].values

# combine the test data with the training data
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

model_input = total_dataset[len(total_dataset) - len(test_data) - predictions_days:].values
model_input = model_input.reshape(-1,1)

model_input = scaler.transform(model_input)

# Make predictions on Test Data

x_test = []

for x in range(predictions_days, len(model_input)):
    x_test.append(model_input[x - predictions_days:x, 0])
    
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)


# plot the test prediction
plt.plot(actual_prices, color="black", label=f"Actual Price of {company}")
plt.plot(predicted_prices, color="green", label=f"Predicted Price of {company}")
plt.title(f"Prediction of {company} Stock Price")
plt.xlabel("Time")
plt.ylabel(" share Price")
plt.legend()
plt.show()


# predict next day

real_data = [model_input[len(model_input) + 1 - predictions_days:len(model_input) + 1, 0]]

real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)

print(f"Predicted Price of {company} on {test_end} is {prediction[0][0]}")