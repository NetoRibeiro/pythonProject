import numpy as np
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, LSTM

# Load Data
company = 'AAPL'
data_files_related = company.lower()

start = dt.datetime(2012, 1, 1)
end = dt.datetime(2020, 1, 1)

data = web.DataReader(company, 'yahoo', start, end)

data.to_pickle(f'data_stored/{data_files_related}_data.pkl')

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x-prediction_days:x])
    y_train.append(scaled_data[x, 0])

x_train = np.array(x_train)
y_train = np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the Model
model = Sequential()

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

# Prediction of the next close price
model.add(Dense(units=1))

model.compile(optimizer='adam', loss='mean_squared_error')

model.fit(x_train, y_train, epochs=25, batch_size=32)

# save Model
file_path = f'data_stored/{data_files_related}_model_Sequential.h5'
model.save(filepath=file_path, overwrite=True, include_optimizer=True,
           save_format=None, signatures=None, options=None, save_traces=True)
