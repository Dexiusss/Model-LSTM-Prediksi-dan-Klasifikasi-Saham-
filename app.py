import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta

# Title
st.set_page_config(page_title="Prediksi Saham dengan LSTM", layout="wide")
st.title("📈 Prediksi Harga Saham Menggunakan LSTM")

# Sidebar
st.sidebar.header("🔍 Pilih Parameter")
ticker = st.sidebar.text_input("Masukkan Kode Saham (contoh: AAPL, GOTO.JK)", value="AAPL")
start_date = st.sidebar.date_input("Tanggal Mulai", value=date(2015, 1, 1))
end_date = st.sidebar.date_input("Tanggal Akhir", value=date.today())
n_days_predict = st.sidebar.slider("Hari yang Diprediksi", 1, 30, 7)

# Ambil data saham
@st.cache_data
def load_data(ticker):
    df = yf.download(ticker, start=start_date, end=end_date)
    return df

data_load_state = st.text('Mengambil data...')
data = load_data(ticker)
data_load_state.text('✅ Data berhasil dimuat!')

# Tampilkan data mentah
with st.expander("📊 Lihat Data Mentah"):
    st.write(data.tail())

# Visualisasi harga historis
st.subheader("📉 Grafik Harga Saham Historis")
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(data['Close'], label='Harga Penutupan', color='blue')
ax.set_xlabel("Tanggal")
ax.set_ylabel("Harga ($)")
ax.legend()
st.pyplot(fig)

# Preprocessing
data_training = data[['Close']].values
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data_training)

X = []
y = []
n_past = 60  # LSTM butuh data masa lalu

for i in range(n_past, len(data_scaled)):
    X.append(data_scaled[i - n_past:i, 0])
    y.append(data_scaled[i, 0])

X, y = np.array(X), np.array(y)
X = X.reshape((X.shape[0], X.shape[1], 1))

# Load model
model = load_model("model_lstm.h5")

# Prediksi
predicted = model.predict(X)
predicted_prices = scaler.inverse_transform(predicted)

# Visualisasi hasil prediksi
st.subheader("🔮 Prediksi vs Aktual")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(data.index[n_past:], data['Close'].values[n_past:], label='Harga Aktual', color='black')
ax2.plot(data.index[n_past:], predicted_prices, label='Prediksi LSTM', color='green')
ax2.legend()
st.pyplot(fig2)

st.markdown("---")
st.caption("Dibuat oleh Ibet – Tugas Akhir | 2025")
