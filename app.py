import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ---------- Page Config ----------
st.set_page_config(page_title="Prediksi Saham AAPL", layout="wide")
st.markdown(
    "<h1 style='text-align: center; color: white;'>📈 Dashboard Prediksi Saham AAPL dengan LSTM</h1><br><br>",
    unsafe_allow_html=True,
)

# ---------- Load & Cache Data ----------
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

def classify_trend(current, next_):
    delta = (next_ - current) / current
    if delta > 0.01:
        return "Buy"
    elif delta < -0.01:
        return "Sell"
    else:
        return "Hold"

def calculate_classification_stats(prices):
    labels = [classify_trend(prices[i], prices[i+1]) for i in range(len(prices)-1)]
    counts = pd.Series(labels).value_counts(normalize=True).reindex(["Buy", "Hold", "Sell"]).fillna(0)
    return counts * 100

# ---------- Data Preparation ----------
df = load_data()
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df)
X, y = create_dataset(data_scaled, 60)
X = X.reshape(X.shape[0], X.shape[1], 1)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ---------- Load or Train Model ----------
if os.path.exists("model.h5"):
    model = load_model("model.h5")
else:
    with st.spinner("🔄 Melatih model..."):
        model = train_model(X_train, y_train)

# ---------- Model Prediction ----------
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# ===================== Bagian Visualisasi Data Historis =====================
def visualize_historical_data(df):
    st.markdown(
        "<div style='border: 2px solid #00ff9f; padding: 15px; border-radius: 15px; margin-bottom: 20px; background-color: #161b22;'>"
        "<h2 style='color: white; text-align: center;'>📂 Visualisasi Data Historis</h2>",
        unsafe_allow_html=True
    )

    col4, col5 = st.columns(2)
    start_date_hist = col4.date_input("Tanggal Mulai", min_value=df.index.min().date(), max_value=df.index.max().date(), value=df.index.min().date())
    end_date_hist = col5.date_input("Tanggal Akhir", min_value=df.index.min().date(), max_value=df.index.max().date(), value=df.index.max().date())

    if start_date_hist > end_date_hist:
        st.error("❌ Tanggal mulai harus sebelum tanggal akhir.")
    else:
        df_filtered = df.loc[start_date_hist:end_date_hist]
        fig2, ax2 = plt.subplots(figsize=(14, 5))
        fig2.patch.set_facecolor('#0d1117')
        ax2.set_facecolor('#0d1117')
        ax2.plot(df_filtered.index, df_filtered['Close'], color='#00ff9f')
        ax2.set_title(f"Data Historis Saham AAPL ({start_date_hist} s.d. {end_date_hist})", color='white')
        ax2.tick_params(colors='white')
        for spine in ax2.spines.values():
            spine.set_edgecolor('white')
        st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== Bagian Prediksi Harga Masa Depan & Visualisasi =====================
def predict_and_visualize_future_prices(model, data_scaled, scaler, df):
    st.markdown(
        "<div style='border: 2px solid #bb86fc; padding: 15px; border-radius: 15px; margin-bottom: 20px; background-color: #161b22;'>"
        "<h2 style='color: white; text-align: center;'>🧙‍♂️ Prediksi Saham Masa Depan</h2>",
        unsafe_allow_html=True
    )

    mode = st.radio("Pilih mode prediksi:", ["Gunakan Slider Hari", "Gunakan Tanggal Spesifik"])

    if mode == "Gunakan Slider Hari":
        n_days = st.slider("Pilih jumlah hari ke depan untuk prediksi:", 30, 90, 60)
        start_date = df.index.max() + pd.Timedelta(days=1)
        end_date = start_date + pd.Timedelta(days=n_days - 1)
    else:
        col_t1, col_t2 = st.columns(2)
        min_pred_date = df.index.max().date() + pd.Timedelta(days=1)
        start_date = col_t1.date_input("Tanggal Mulai Prediksi", min_value=min_pred_date, value=min_pred_date)
        end_date = col_t2.date_input("Tanggal Akhir Prediksi", min_value=min_pred_date + pd.Timedelta(days=1), value=min_pred_date + pd.Timedelta(days=60))
        if start_date > end_date:
            st.error("❌ Tanggal mulai prediksi harus sebelum tanggal akhir.")
            st.stop()
        n_days = (end_date - start_date).days + 1

    last_sequence = data_scaled[-60:]
    future_input = last_sequence.reshape(1, 60, 1)
    future_preds = []

    for _ in range(n_days):
        next_val = model.predict(future_input, verbose=0)[0][0]
        future_preds.append(next_val)
        future_input = np.append(future_input[:, 1:, :], [[[next_val]]], axis=1)

    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    future_dates = pd.date_range(start=start_date, periods=n_days)

    fig3, ax3 = plt.subplots(figsize=(14, 5))
    fig3.patch.set_facecolor('#0d1117')
    ax3.set_facecolor('#0d1117')
    ax3.plot(future_dates, future_prices, label=f"Prediksi {n_days} Hari", color='#bb86fc')
    ax3.set_title(f"Prediksi Harga Saham AAPL ({future_dates[0].date()} s.d. {future_dates[-1].date()})", color='white')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#161b22', edgecolor='white', labelcolor='white')
    for spine in ax3.spines.values():
        spine.set_edgecolor('white')
    st.pyplot(fig3)
    st.markdown("</div>", unsafe_allow_html=True)

    return future_prices

# ===================== Bagian Visualisasi Klasifikasi Tren =====================
def visualize_classification(future_prices):
    st.markdown(
        "<div style='border: 2px solid #00ffff; padding: 15px; border-radius: 15px; background-color: #161b22;'>"
        "<h2 style='color: white; text-align: center;'>📊 Persentase Klasifikasi pada Rentang Waktu Dipilih</h2>",
        unsafe_allow_html=True
    )

    trend_stats = calculate_classification_stats(future_prices.flatten())
    
    # Pie chart untuk visualisasi klasifikasi
    labels = ['Buy', 'Hold', 'Sell']
    sizes = [trend_stats.get(label, 0) for label in labels]
    colors = ['#4caf50', '#ffeb3b', '#f44336']
    explode = (0.1, 0, 0)  # highlight "Buy" sedikit keluar

    fig4, ax4 = plt.subplots(figsize=(6, 6), facecolor='#161b22')
    ax4.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=140,
        colors=colors, explode=explode, textprops={'color':'white', 'fontsize':14}
    )
    ax4.set_title("📌 Presentase Klasifikasi di Prediksi Waktu yang Dipilih", color='white', fontsize=16)
    st.pyplot(fig4)
    st.markdown("</div>", unsafe_allow_html=True)

# ===================== Main Layout =====================
col1, col2, col3 = st.columns([2, 2, 6])

with col1:
    st.markdown(
        "<div style='border: 2px solid #58a6ff; padding: 15px; border-radius: 15px; background-color: #161b22;'>"
        "<h3 style='color: white;'>Dataset</h3>",
        unsafe_allow_html=True
    )
    st.dataframe(df.head())
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown(
        "<div style='border: 2px solid #f778ba; padding: 15px; border-radius: 15px; background-color: #161b22;'>"
        "<h3 style='color: white;'>Akurasi Prediksi</h3>",
        unsafe_allow_html=True
    )
    mse = mean_squared_error(real_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_prices, predicted_prices)
    r2 = r2_score(real_prices, predicted_prices)
    eval_df = pd.DataFrame({
        "Metrik": ["MSE", "RMSE", "MAE", "R² Score"],
        "Nilai": [f"{mse:.4f}", f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}"]
    })
    st.table(eval_df)
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown(
        """
        <div style='border: 2px solid #bb86fc; padding: 15px; border-radius: 15px; background-color: #161b22;'>
            <h3 style='color: white; text-align: center;'>📉 Prediksi vs Aktual pada Data Test</h3>
        """,
        unsafe_allow_html=True
    )
    fig1, ax1 = plt.subplots(figsize=(8, 4))
    fig1.patch.set_facecolor('#0d1117')
    ax1.set_facecolor('#0d1117')
    dates_all = df.index[60:]
    dates_test = dates_all[train_size:]
    ax1.plot(dates_test, real_prices, label='Aktual', color='#58a6ff')
    ax1.plot(dates_test, predicted_prices, label='Prediksi', color='#f778ba', linestyle='--')
    ax1.tick_params(colors='white')
    ax1.legend(facecolor='#161b22', edgecolor='white', labelcolor='white')
    for spine in ax1.spines.values():
        spine.set_edgecolor('white')
    st.pyplot(fig1)
    st.markdown("</div>", unsafe_allow_html=True)

# Tampilkan bagian visualisasi data historis
visualize_historical_data(df)

# Tampilkan bagian prediksi harga masa depan dan visualisasinya, simpan hasil prediksi untuk klasifikasi
future_prices = predict_and_visualize_future_prices(model, data_scaled, scaler, df)

# Tampilkan bagian klasifikasi tren dengan visualisasi pie chart
visualize_classification(future_prices)
