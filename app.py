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
from streamlit_option_menu import option_menu

# ---------- Config ----------
st.set_page_config(page_title="Prediksi Saham AAPL", layout="wide")

st.markdown("<h1 style='text-align: center; color: white;'>📈 Prediksi Saham AAPL dengan LSTM</h1>", unsafe_allow_html=True)

# ---------- Sidebar Navigation ----------
with st.sidebar:
    selected = option_menu(
        "Navigasi",
        ["Dataset", "Visualisasi Historis", "Evaluasi Model", "Prediksi Harga", "Klasifikasi Tren", "Tentang Model"],
        icons=["folder", "bar-chart", "cpu", "graph-up", "diagram-3", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#0d1117"},
            "icon": {"color": "#00ff9f", "font-size": "20px"},
            "nav-link": {"color": "white", "font-size": "16px", "text-align": "left"},
            "nav-link-selected": {"background-color": "#1f2937"},
        }
    )

# ---------- Load & Cache ----------
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

# ---------- Data Prep ----------
df = load_data()
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(df)
X, y = create_dataset(data_scaled, 60)
X = X.reshape(X.shape[0], X.shape[1], 1)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# ---------- Load or Train ----------
if os.path.exists("model.h5"):
    model = load_model("model.h5")
else:
    with st.spinner("🔄 Melatih model..."):
        model = train_model(X_train, y_train)

# ---------- Prediksi ----------
predicted = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))

# ---------- Menu Logic ----------
if selected == "Dataset":
    st.markdown("### 📁 Dataset AAPL (2015–2024)")
    st.dataframe(df.tail(100))

elif selected == "Visualisasi Historis":
    st.markdown("### 📊 Visualisasi Data Historis")
    col1, col2 = st.columns(2)
    start = col1.date_input("Tanggal Mulai", value=df.index.min().date(), min_value=df.index.min().date(), max_value=df.index.max().date())
    end = col2.date_input("Tanggal Akhir", value=df.index.max().date(), min_value=df.index.min().date(), max_value=df.index.max().date())
    if start > end:
        st.error("❌ Tanggal mulai harus sebelum tanggal akhir.")
    else:
        filtered = df.loc[start:end]
        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor('#0d1117')
        ax.set_facecolor('#0d1117')
        ax.plot(filtered.index, filtered['Close'], color='#00ff9f')
        ax.set_title(f"Data Historis Saham AAPL", color='white')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_edgecolor('white')
        st.pyplot(fig)

elif selected == "Evaluasi Model":
    st.markdown("### 🧠 Evaluasi Model")
    mse = mean_squared_error(real_prices, predicted_prices)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_prices, predicted_prices)
    r2 = r2_score(real_prices, predicted_prices)
    eval_df = pd.DataFrame({
        "Metrik": ["MSE", "RMSE", "MAE", "R² Score"],
        "Nilai": [f"{mse:.4f}", f"{rmse:.4f}", f"{mae:.4f}", f"{r2:.4f}"]
    })
    st.table(eval_df)

    st.markdown("### 📉 Prediksi vs Aktual")
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    dates_test = df.index[60+train_size:]
    ax.plot(dates_test, real_prices, label='Aktual', color='#58a6ff')
    ax.plot(dates_test, predicted_prices, label='Prediksi', color='#f778ba', linestyle='--')
    ax.legend(facecolor='#161b22', edgecolor='white', labelcolor='white')
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    st.pyplot(fig)

elif selected == "Prediksi Harga":
    st.markdown("### 🔮 Prediksi Harga Masa Depan")
    mode = st.radio("Pilih mode prediksi:", ["Gunakan Slider Hari", "Gunakan Tanggal Spesifik"])
    if mode == "Gunakan Slider Hari":
        n_days = st.slider("Jumlah hari ke depan:", 30, 90, 60)
        start_date = df.index.max() + pd.Timedelta(days=1)
        end_date = start_date + pd.Timedelta(days=n_days - 1)
    else:
        min_pred_date = df.index.max().date() + pd.Timedelta(days=1)
        col1, col2 = st.columns(2)
        start_date = col1.date_input("Tanggal Mulai", value=min_pred_date)
        end_date = col2.date_input("Tanggal Akhir", value=min_pred_date + pd.Timedelta(days=60))
        if start_date > end_date:
            st.error("❌ Tanggal mulai harus sebelum tanggal akhir.")
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

    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#0d1117')
    ax.plot(future_dates, future_prices, label=f"Prediksi {n_days} Hari", color='#bb86fc')
    ax.set_title(f"Prediksi Harga AAPL", color='white')
    ax.tick_params(colors='white')
    ax.legend(facecolor='#161b22', edgecolor='white', labelcolor='white')
    for spine in ax.spines.values():
        spine.set_edgecolor('white')
    st.pyplot(fig)

elif selected == "Klasifikasi Tren":
    st.markdown("### 📈 Klasifikasi Tren Berdasarkan Prediksi")
    future_prices = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
    trend_stats = calculate_classification_stats(future_prices.flatten())
    labels = ['Buy', 'Hold', 'Sell']
    sizes = [trend_stats.get(label, 0) for label in labels]
    colors = ['#4caf50', '#ffeb3b', '#f44336']
    explode = (0.1, 0, 0)

    fig, ax = plt.subplots(figsize=(6, 6), facecolor='#161b22')
    ax.pie(
        sizes, labels=labels, autopct='%1.1f%%', startangle=140,
        colors=colors, explode=explode, textprops={'color':'white', 'fontsize':14}
    )
    ax.set_title("📌 Distribusi Klasifikasi", color='white', fontsize=16)
    st.pyplot(fig)

elif selected == "Tentang Model":
    st.markdown("### ℹ️ Tentang Model")
    st.markdown("""
    - Model: LSTM (Long Short-Term Memory)
    - Input: Harga penutupan saham AAPL dari tahun 2015–2024
    - Time Step: 60 hari
    - Optimizer: Adam
    - Epoch: 20
    - Batch size: 32
    - Output: Prediksi harga & klasifikasi tren (Buy / Hold / Sell)
    """)

