import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def load_date(start_date, end_date = None):
    start = dt.datetime(start_date[0], start_date[1], start_date[2])
    
    if end_date is None:
        end = dt.datetime.now()
    else:
        end = dt.datetime(end_date[0], end_date[1], end_date[2])
        
    return start, end




def load_test_data(tickerData, data, scaler):
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

    return model_input, actual_prices


def predict_test_data(predictions_days, model_input, actual_prices, model, scaler, company):
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