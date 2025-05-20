import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

st.set_page_config(page_title="Prediksi Saham AAPL", layout="wide")
st.title("📈 Dashboard Prediksi Saham AAPL dengan LSTM")

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

# Load data dan preprocessing
df = load_data()

# Optional: Tampilkan data head
if st.checkbox("📋 Tampilkan 5 baris pertama data"):
    st.dataframe(df.head())

scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df)

# Dataset untuk training
X, y = create_dataset(data_scaled, 60)
X = X.reshape(X.shape[0], X.shape[1], 1)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Load atau latih model
if os.path.exists("model.h5"):
    model = load_model("model.h5")
else:
    with st.spinner("🔄 Melatih model..."):
        model = train_model(X_train, y_train)

# Prediksi untuk data test (statis)
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot 1: Prediksi vs Aktual
st.subheader("📌 Visualisasi Akurasi Model Prediksi")
dates_all = df.index[60:]
dates_test = dates_all[train_size:]

fig1, ax1 = plt.subplots(figsize=(14, 6))
ax1.plot(dates_test, real_prices, label='Aktual', color='dodgerblue')
ax1.plot(dates_test, predicted_prices, label='Prediksi', color='tomato')
ax1.set_title("Prediksi vs Aktual pada Data Test", fontsize=16)
ax1.set_xlabel("Tanggal")
ax1.set_ylabel("Harga (USD)")
ax1.legend()
ax1.grid(True)
st.pyplot(fig1)

# Plot 2: Visualisasi Data Historis
st.subheader("📂 Visualisasi Data Historis")
min_date = df.index.min().date()
max_date = df.index.max().date()

col1, col2 = st.columns(2)
start_date_hist = col1.date_input("Tanggal Mulai", min_value=min_date, max_value=max_date, value=min_date)
end_date_hist = col2.date_input("Tanggal Akhir", min_value=min_date, max_value=max_date, value=max_date)

if start_date_hist > end_date_hist:
    st.error("❌ Tanggal mulai harus sebelum tanggal akhir.")
else:
    df_filtered = df.loc[start_date_hist:end_date_hist]
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(df_filtered.index, df_filtered['Close'], color='seagreen')
    ax2.set_title(f"Data Historis Saham AAPL ({start_date_hist} s.d. {end_date_hist})", fontsize=16)
    ax2.set_xlabel("Tanggal")
    ax2.set_ylabel("Harga (USD)")
    ax2.grid(True)
    st.pyplot(fig2)

# Plot 3: Prediksi Masa Depan
st.subheader("🔮 Prediksi Saham Masa Depan (1 - 3 Bulan)")

pred_mode = st.radio("Pilih Metode Prediksi:", ["Gunakan Slider", "Gunakan Tanggal Awal & Akhir"])

if pred_mode == "Gunakan Slider":
    n_days = st.slider("Pilih jumlah hari ke depan untuk prediksi:", 30, 90, 60)
elif pred_mode == "Gunakan Tanggal Awal & Akhir":
    col3, col4 = st.columns(2)
    start_pred = col3.date_input("Tanggal Awal Prediksi", min_value=max_date + pd.Timedelta(days=1))
    end_pred = col4.date_input("Tanggal Akhir Prediksi", min_value=start_pred)
    n_days = (end_pred - start_pred).days + 1  # Tambah 1 agar inklusif
    if n_days <= 0:
        st.error("❌ Rentang prediksi tidak valid.")
        n_days = None

# Lanjutkan jika n_days valid
if n_days and n_days > 0:
    last_sequence = data_scaled[-60:]
    future_input = last_sequence.reshape(1, 60, 1)
    future_preds = []

    for _ in range(n_days):
        next_val = model.predict(future_input)[0][0]
        future_preds.append(next_val)
        future_input = np.append(future_input[:, 1:, :], [[[next_val]]], axis=1)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    start_date = df.index.max() + pd.Timedelta(days=1)
    future_dates = pd.date_range(start=start_date, periods=n_days)

    fig3, ax3 = plt.subplots(figsize=(14, 6))
    ax3.plot(future_dates, future_prices, label=f"Prediksi {n_days} Hari", color='purple')
    ax3.set_title(f"Prediksi Harga Saham AAPL ({future_dates[0].date()} s.d. {future_dates[-1].date()})", fontsize=16)
    ax3.set_xlabel("Tanggal")
    ax3.set_ylabel("Harga (USD)")
    ax3.grid(True)
    ax3.legend()
    st.pyplot(fig3)
