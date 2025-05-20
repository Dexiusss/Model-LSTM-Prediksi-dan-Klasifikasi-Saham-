import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Harga Saham AAPL", layout="wide")
st.title("📈 Prediksi Harga Saham AAPL dengan LSTM")

@st.cache_data
def load_data():
    df = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df[['Close']]

@st.cache_resource
def train_model(data_scaled, X_train, y_train):
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

# 1. Load data
df = load_data()
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df)

# 2. Buat dataset untuk LSTM
X, y = create_dataset(data_scaled, 60)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 3. Split data
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 4. Load atau latih model
if os.path.exists("model.h5"):
    model = load_model("model.h5")
else:
    with st.spinner("🔄 Melatih model LSTM..."):
        model = train_model(data_scaled, X_train, y_train)
    st.success("✅ Model berhasil dilatih dan disimpan!")

# 5. Prediksi
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted.reshape(-1, 1))
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# 6. Statistik dasar
last_real = real_prices[-1][0]
last_pred = predicted_prices[-1][0]
error = abs(last_real - last_pred)

# Sidebar info
st.sidebar.header("📌 Info Prediksi")
st.sidebar.markdown(f"**Harga Aktual Terakhir:** ${last_real:,.2f}")
st.sidebar.markdown(f"**Prediksi Terakhir:** ${last_pred:,.2f}")
st.sidebar.markdown(f"**Selisih (MAE):** ${error:,.2f}")

# 7. Visualisasi utama
st.subheader("📉 Visualisasi Prediksi vs Aktual")

fig, ax = plt.subplots(figsize=(14, 6))
ax.plot(real_prices, label='Harga Aktual', color='royalblue', linewidth=2)
ax.plot(predicted_prices, label='Prediksi LSTM', color='tomato', linestyle='--', linewidth=2)
ax.fill_between(np.arange(len(predicted_prices)), real_prices.flatten(), predicted_prices.flatten(), color='gray', alpha=0.2)
ax.set_title("Prediksi Harga Saham AAPL Menggunakan LSTM", fontsize=16)
ax.set_xlabel("Hari")
ax.set_ylabel("Harga (USD)")
ax.legend()
ax.grid(True)
st.pyplot(fig)
