import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pandas_datareader as web
import datetime as dt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Company Name
company = 'AAPL'
data_files_related = company.lower()

# Load Stock Data
data = pd.read_pickle(f'data_stored/{data_files_related}_data.pkl')

# Load Trained Model
file_path = f'data_stored/{data_files_related}_model_Sequential.h5'
model = load_model(file_path)

# Prepare Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

prediction_days = 60

# Test Model Accuracy on existing Data
# Load Test Data
test_start = dt.datetime(2020, 1, 1)
test_end = dt.datetime.now()

# Take Test data from the Web
test_data = web.DataReader(company, 'yahoo', test_start, test_end)

true_price = test_data['Close'].values

# Combine Trained and Test data
total_dataset = pd.concat((data['Close'], test_data['Close']), axis=0)

# Preparing test data
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
model_inputs = model_inputs.reshape(-1, 1)
model_inputs = scaler.transform(model_inputs)

# Make Predictions on Test Data
x_test = []

for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

predicted_prices = model.predict(x_test)
predicted_probability = model.predict_proba(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Plot the Test Predictions
plt.plot(true_price, color="black", label=f'True values for {company}')
plt.plot(predicted_prices, color="green", label=f'Predicted values for {company}')
plt.plot(predicted_probability, color="blue", label=f'Probability for {company}')
plt.title(f'{company} share price')
plt.xlabel("Time")
plt.ylabel(f'{company} share price')
plt.legend()
plt.show()

# Plot the Histogram of the Probability
plt.hist(predicted_probability)
plt.show()
