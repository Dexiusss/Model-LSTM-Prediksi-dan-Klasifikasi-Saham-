import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

st.set_page_config(page_title="Prediksi Harga Saham AAPL", layout="wide")
st.title("📈 Prediksi Harga Saham AAPL dengan LSTM")

@st.cache_data
def load_data():
    df = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df[['Close']]

@st.cache_resource
def train_model(X_train, y_train):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
    model.save("model.h5")
    return model

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

# Load data
df = load_data()
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df)

# Info dataset
st.subheader("📊 Informasi Dataset")
col1, col2 = st.columns(2)
col1.metric("Jumlah Data", len(df))
col1.metric("Tanggal Awal", df.index.min().strftime('%Y-%m-%d'))
col1.metric("Tanggal Akhir", df.index.max().strftime('%Y-%m-%d'))
col2.dataframe(df.describe().T, use_container_width=True)

# Slider prediksi
n_days = st.slider("📅 Pilih jumlah hari ke depan untuk prediksi:", 1, 30, 7)

# Date range selector untuk visualisasi (hanya untuk prediksi vs aktual)
st.subheader("📅 Pilih Rentang Waktu untuk Visualisasi Prediksi vs Aktual")
min_date = df.index.min().date()
max_date = df.index.max().date()
start_date = st.date_input("Tanggal mulai", min_value=min_date, max_value=max_date, value=min_date)
end_date = st.date_input("Tanggal akhir", min_value=min_date, max_value=max_date, value=max_date)

if start_date > end_date:
    st.error("❌ Tanggal mulai harus sebelum tanggal akhir.")
    st.stop()

# Buat dataset
X, y = create_dataset(data_scaled, 60)
X = X.reshape(X.shape[0], X.shape[1], 1)

# Split
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Load atau latih model
if os.path.exists("model.h5"):
    model = load_model("model.h5")
else:
    with st.spinner("🔄 Melatih model LSTM..."):
        model = train_model(X_train, y_train)
    st.success("✅ Model berhasil dilatih dan disimpan!")

# Prediksi terhadap data test
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Prediksi masa depan
last_sequence = data_scaled[-60:]
future_input = last_sequence.reshape(1, 60, 1)
future_predictions = []

for _ in range(n_days):
    next_pred = model.predict(future_input)[0][0]
    future_predictions.append(next_pred)
    future_input = np.append(future_input[:, 1:, :], [[[next_pred]]], axis=1)

future_prices = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# --- Filter data untuk visualisasi prediksi vs aktual berdasarkan rentang waktu ---
date_test_start_idx = train_size + 60
dates_all = df.index[60:]
dates_test = dates_all[train_size:]
mask = (dates_test.date >= start_date) & (dates_test.date <= end_date)
dates_filtered = dates_test[mask]
real_filtered = real_prices[mask]
predicted_filtered = predicted_prices[mask]

# Visualisasi 1: Prediksi vs Aktual pada data test (periode pilihan)
st.subheader("📉 Visualisasi Prediksi vs Aktual (Data Test)")
fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(dates_filtered, real_filtered, label='Harga Aktual (Test)', color='royalblue', linewidth=2)
ax1.plot(dates_filtered, predicted_filtered, label='Prediksi LSTM (Test)', color='tomato', linestyle='--', linewidth=2)
ax1.set_title("Prediksi Harga Saham AAPL - Data Test", fontsize=16)
ax1.set_xlabel("Tanggal")
ax1.set_ylabel("Harga (USD)")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Visualisasi 2: Prediksi masa depan (forecast n hari)
st.subheader(f"🔮 Visualisasi Prediksi Masa Depan {n_days} Hari ke Depan")
fig2, ax2 = plt.subplots(figsize=(14, 6))

# Gunakan tanggal mulai prediksi dari hari terakhir data asli (max tanggal df)
last_date = df.index.max()
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=n_days)

ax2.plot(future_dates, future_prices, label=f'Prediksi {n_days} Hari ke Depan', color='green', linestyle='dashdot', linewidth=2)
ax2.set_title(f"Prediksi Harga Saham AAPL {n_days} Hari ke Depan", fontsize=16)
ax2.set_xlabel("Tanggal")
ax2.set_ylabel("Harga (USD)")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)
