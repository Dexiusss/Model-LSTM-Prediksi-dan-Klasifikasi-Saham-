import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="Prediksi Saham AAPL", layout="wide")

st.title("📈 Prediksi Harga Saham Apple Inc. (AAPL) Menggunakan LSTM")

# Input tanggal dari pengguna
st.sidebar.header("Pilih Rentang Waktu")
start_date = st.sidebar.date_input("Tanggal Mulai", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("Tanggal Akhir", datetime.date.today())

# Download data dari Yahoo Finance
data = yf.download('AAPL', start=start_date, end=end_date)
st.write("Data Historis AAPL", data.tail())

# Visualisasi harga penutupan
st.subheader("Visualisasi Harga Penutupan Saham")
plt.figure(figsize=(10, 4))
plt.plot(data['Close'], label='Harga Penutupan')
plt.xlabel('Tanggal')
plt.ylabel('Harga')
plt.title('Harga Penutupan Saham AAPL')
plt.legend()
st.pyplot(plt)

# Upload model LSTM
st.subheader("Upload Model LSTM (.h5)")
uploaded_model = st.file_uploader("Upload model LSTM yang sudah dilatih", type=["h5"])

if uploaded_model:
    model = load_model(uploaded_model)
    st.success("✅ Model berhasil dimuat!")

    # Preprocessing data
    data_close = data[['Close']]
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data_close)

    sequence_length = 60
    x_test = []
    for i in range(sequence_length, len(scaled_data)):
        x_test.append(scaled_data[i-sequence_length:i, 0])
    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Prediksi
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

    # Tampilkan hasil
    st.subheader("Perbandingan Harga Aktual vs Prediksi")
    pred_dates = data.index[sequence_length:]
    actual_prices = data_close['Close'].values[sequence_length:]

    plt.figure(figsize=(10, 4))
    plt.plot(pred_dates, actual_prices, label='Harga Aktual')
    plt.plot(pred_dates, predictions, label='Prediksi')
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.legend()
    st.pyplot(plt)

    # Sinyal investasi sederhana
    st.subheader("Sinyal Investasi (Naik/Turun)")
    if predictions[-1] > actual_prices[-1]:
        st.success("Rekomendasi: BUY 📈")
    else:
        st.error("Rekomendasi: SELL 📉")
