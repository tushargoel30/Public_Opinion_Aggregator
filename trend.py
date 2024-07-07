import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from pytrends.request import TrendReq
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
# keywords = ["Narendra Modi", "Rahul Gandhi", "Amit Shah", "Sonia Gandhi", "Yogi Adityanath", "Arvind Kejriwal", "Sharad Pawar", "Mamta Banerjee", "Nitin Gadkari", "Rajnath Singh", "Smriti Irani", "Nirmala Sitharaman", "Mayawati", "Devendra Fadnavis", "Piyush Goyal", "Uddhav Thackeray", "Akhilesh Yadav", "Sachin Pilot", "Jyotiraditya Scindia", "Bhagwant Mann", "M.K. Stalin", "K. Chandrashekar Rao", "Naveen Patnaik", "Amarinder Singh", "Kamal Nath", "Mehbooba Mufti", "Y.S. Jagan Mohan Reddy", "Chirag Paswan", "Tejashwi Yadav", "Mallikarjun Kharge", "Prithviraj Chavan", "Sanjay Raut", "Kiren Rijiju", "B.S. Yediyurappa", "P. Chidambaram"]


def avg_search_interest_prediction(keyword, days):
    # Fetching the data
    pytrends = TrendReq(hl="en-US", tz=360)
    # keyword = "blockchain"
    pytrends.build_payload([keyword], timeframe="today 5-y", geo="US")
    data = pytrends.interest_over_time()
    data = data.drop(labels=["isPartial"], axis="columns")

    # Preparing data for LSTM
    # Normalize the data
    from sklearn.preprocessing import MinMaxScaler

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)

    # Creating the training data
    train_size = int(len(scaled_data) * 0.80)
    test_size = len(scaled_data) - train_size
    train, test = (
        scaled_data[0:train_size, :],
        scaled_data[train_size : len(scaled_data), :],
    )

    def create_dataset(dataset, look_back=1):
        X, Y = [], []
        for i in range(len(dataset) - look_back - 1):
            a = dataset[i : (i + look_back), 0]
            X.append(a)
            Y.append(dataset[i + look_back, 0])
        return np.array(X), np.array(Y)

    look_back = 3
    X_train, Y_train = create_dataset(train, look_back)
    X_test, Y_test = create_dataset(test, look_back)

    # Reshape input to be [samples, time steps, features]
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    # Building the LSTM model
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(loss="mean_squared_error", optimizer="adam")
    model.fit(X_train, Y_train, epochs=100, batch_size=1, verbose=2)

    # Predicting and plotting the results
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    # Invert predictions
    train_predict = scaler.inverse_transform(train_predict)
    Y_train_inv = scaler.inverse_transform([Y_train])
    test_predict = scaler.inverse_transform(test_predict)
    Y_test_inv = scaler.inverse_transform([Y_test])

    # Plot baseline and predictions
    plt.plot(scaler.inverse_transform(scaled_data))
    plt.plot(np.concatenate([train_predict, test_predict]))
    plt.legend(["Original data", "Model Predictions"])
    image_path = os.path.join('static/img', 'plot.png')
    plt.savefig(image_path)
    plt.close()

    # Generating input sequences for the next given number of days
    last_date = data.index[-1]
    next_dates = [last_date + timedelta(days=i) for i in range(1, days + 1)]
    next_dates = np.array(
        [
            [
                (date - last_date).days,
            ]
            for date in next_dates
        ]
    )
    next_scaled = scaler.transform(next_dates)
    # Reshaping input sequences
    next_scaled = np.reshape(
        next_scaled, (next_scaled.shape[0], next_scaled.shape[1], 1)
    )
    # Predicting search interest for the next given number of days
    next_predict = model.predict(next_scaled)
    next_predict_inv = scaler.inverse_transform(next_predict)
    # Calculating the average search interest for the next given number of days
    average_search_interest = np.mean(next_predict_inv)
    print(
        f"Average search interest for {keyword} over the next {days} days is:",
        average_search_interest,
    )
    return float(average_search_interest)


def getPrediction(keyword):
    res= avg_search_interest_prediction(keyword,60)
    with open('static/trends.json', 'w') as f:
        json.dump(res, f)
    return