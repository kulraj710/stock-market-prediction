import numpy as np
import pandas as pd
import pandas_datareader as web
import datetime as dt
import yfinance as yf
# import  streamlit as st

from helper import load_test_data, predict_test_data

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from keras.models import load_model

# load data
company = 'RELIANCE.NS'

start = dt.datetime(2012, 1, 1)
end = dt.datetime.now()

print(end)

# tickerData = yf.Ticker(company)

# data = tickerData.history(start=start, end=end)


def train_test_model(company, start, end, predictions_days=15, epochs=25, batch_size=32, save=False):
    
    try:
        #loading data with yahoo finance api
        ticker_data = yf.Ticker(company)

        data = ticker_data.history(start=start, end=end)
        
        # preparing Data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))


        # predictions_days = 60 (value will be passed to the function)

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

        model.add(Dense(units=1))
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs)
        
        # function to save the model
        def save(filename = 'model.h5'):
            # saving the model
            model.save(f'{filename}.h5')

        if save == True:
            save(company)
        
        # load test data (coming from helper.py)
        t = load_test_data(ticker_data=ticker_data, data=data, scaler=scaler)
        
        model_input = t[0]
        actual_prices = t[2]
        
        predict_test_data(predictions_days=predictions_days , model_input=model_input ,actual_prices=actual_prices, model=model, scaler=scaler, company=company)
        
        
        return model, model_input, scaler, predictions_days
    
    except Exception as e:
        return 'Something went Wrong, here is the log \n : ' +  str(e)



def predict_next_day(model_input, saved_model, scaler, predictions_days):
    #implement after the model
    real_data = [model_input[len(model_input) + 1 - 15:len(model_input) + 1, 0]]

    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    prediction = model.predict(real_data)
    prediction = scaler.inverse_transform(prediction)

    print(f"Predicted Price of {company} on {dt.datetime.now()} is {prediction[0][0]}")
    return prediction[0][0]



if __name__ == '__main__':
    t = train_test_model(company, start, end, epochs=5, batch_size=30 )
    
    model = t[0]
    model_input = t[1]
    scaler = t[2]
    predictions_days = t[3]
    
    
    pnd = predict_next_day(model_input, model, scaler, predictions_days)
    
    print(pnd)