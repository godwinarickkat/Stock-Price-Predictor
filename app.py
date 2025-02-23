import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

# Set date range
start = "2010-01-01"
end = "2024-01-01"

st.title('Stock Trend Prediction')

# Get user input
user_input = st.text_input('Enter Stock Ticker', 'AAPL')

# Fetch stock data
df = yf.download(user_input, start=start, end=end)

# Check if data is retrieved
if df.empty:
    st.error("Failed to fetch stock data. Please check the ticker symbol and try again.")
    st.stop()

# Display data info
st.subheader('Data from 2010 - 2024')
st.write(df.describe())

# Visualization
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'])
st.pyplot(fig)

# Add moving averages
st.subheader('Closing Price vs Time chart with 100MA')
ma100 = df['Close'].rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, 'r', label='100-day MA')
plt.legend()
st.pyplot(fig)

st.subheader('Closing Price vs Time chart with 100MA & 200MA')
ma200 = df['Close'].rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(df['Close'], label='Closing Price')
plt.plot(ma100, 'r', label='100-day MA')
plt.plot(ma200, 'g', label='200-day MA')
plt.legend()
st.pyplot(fig)

# Split data
train_size = int(len(df) * 0.70)

if len(df) < 100:
    st.error("Not enough historical data to process. Try another stock ticker.")
    st.stop()

data_training = df.iloc[:train_size]['Close']
data_testing = df.iloc[train_size:]['Close']

# Scale data
scaler = MinMaxScaler(feature_range=(0,1))

if data_training.empty:
    st.error("Training data is empty. Please check the stock ticker or try a different one.")
    st.stop()

data_training_array = scaler.fit_transform(np.array(data_training).reshape(-1,1))

# Load model
model = load_model('keras_model.h5')

# Prepare test data
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

# Apply same scaling
input_data = scaler.transform(final_df)

x_test, y_test = [], []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100:i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

# Make predictions
y_predicted = model.predict(x_test)

# Rescale predictions
scale_factor = 1 / scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

# Plot results
st.subheader('Prediction vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)
