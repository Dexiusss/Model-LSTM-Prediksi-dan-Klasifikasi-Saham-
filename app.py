import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.models import load_model
import tensorflow as tf

st.set_page_config(layout="wide")
st.title("📈 Dashboard Prediksi Saham AAPL")


# Install dependencies
#!pip install yfinance
#!pip install tensorflow scikit-learn

import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Download historical stock data
df = yf.download("AAPL", start="2019-01-01", end="2024-12-31")
df = df[['Close']]

print("Informasi Dataset:")
print(df.info())

df.head()

print("Cek Nilai Kosong:")
print(df.isnull().sum())

print("\nCek Data Duplikat:")
print(df.duplicated().sum())

print("\nStatistik Deskriptif:")
print(df.describe())

df_cleaned = df.drop_duplicates()

df.dropna(inplace=True)

print("Data setelah dibersihkan:")
print(df_cleaned.info())

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df)

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

X, y = create_dataset(data_scaled, 60)
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

# Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# LSTM untuk prediksi harga
model_pred = Sequential()
model_pred.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model_pred.add(Dropout(0.2))
model_pred.add(LSTM(units=50, return_sequences=False))
model_pred.add(Dropout(0.2))
model_pred.add(Dense(units=1))

model_pred.compile(optimizer='adam', loss='mean_squared_error')
model_pred.fit(X_train, y_train, epochs=20, batch_size=32, verbose=1)

# Prediksi untuk 300 titik kedepan
predicted = model_pred.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

plt.figure(figsize=(12,6))
plt.plot(real_prices, color='blue', label='Actual Price')
plt.plot(predicted_prices, color='red', label='Predicted Price')
plt.title('Prediksi Harga Saham AAPL')
plt.xlabel('Waktu')
plt.ylabel('Harga')
plt.legend()
plt.show()

mse = mean_squared_error(real_prices, predicted_prices)
rmse = np.sqrt(mse)
mae = mean_absolute_error(real_prices, predicted_prices)
r2 = r2_score(real_prices, predicted_prices)

print(f"Evaluasi Model Terhadap Data Uji ")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R^2 Score: {r2:.4f}")

# Prediksi 30 hari ke depan
forecast_input = data_scaled[-60:]  # Ambil 60 hari terakhir
forecast_input = forecast_input.reshape(1, -1)
forecast_input = list(forecast_input[0])

future_predictions = []
for _ in range(30):
    input_seq = np.array(forecast_input[-60:])  # Ambil 60 data terakhir
    input_seq = input_seq.reshape(1, 60, 1)
    next_pred = model_pred.predict(input_seq, verbose=0)[0][0]
    future_predictions.append(next_pred)
    forecast_input.append(next_pred)

# Balik ke skala harga asli
future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

plt.figure(figsize=(12,6))
plt.plot(real_prices, label='Actual Price')
plt.plot(predicted_prices, label='Predicted Price (Test Set)')
plt.plot(range(len(real_prices), len(real_prices)+30), future_prices, color='green', label='Forecast Next 30 Days')
plt.title('Prediksi Harga Saham AAPL dan Forecast 30 Hari Kedepan')
plt.xlabel('Waktu')
plt.ylabel('Harga')
plt.legend()
plt.show()

# Klasifikasi sinyal Buy, Hold, Sell
signal = []
for i in range(1, len(real_prices)):
    change = real_prices[i] - real_prices[i - 1]
    if change > 1.0:
        signal.append(2)  # Buy
    elif change < -1.0:
        signal.append(0)  # Sell
    else:
        signal.append(1)  # Hold

X_cls = X_test[1:]
y_cls = np.array(signal)

model_cls = Sequential()
model_cls.add(LSTM(50, return_sequences=True, input_shape=(X_cls.shape[1], 1)))
model_cls.add(Dropout(0.2))
model_cls.add(LSTM(50))
model_cls.add(Dropout(0.2))
model_cls.add(Dense(3, activation='softmax'))

model_cls.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_cls.fit(X_cls, y_cls, epochs=20, batch_size=32, verbose=1)

pred_cls = model_cls.predict(X_cls)
pred_labels = np.argmax(pred_cls, axis=1)

print("\nClassification Report:")
print(classification_report(y_cls, pred_labels))
print("Confusion Matrix:\n", confusion_matrix(y_cls, pred_labels))
