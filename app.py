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
import datetime

# ---------- Page Config ----------
st.set_page_config(
    page_title="Prediksi Saham AAPL | Kelompok 27",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Custom CSS for Enhanced Styling ----------
custom_css = """
<style>
    /* Main app background */
    .stApp {
        background: #0d1117; /* Fallback color */
    }

    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #0d1117; /* Dark background for sidebar */
        border-right: 2px solid #2a3039;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p, [data-testid="stSidebar"] label {
        color: #c9d1d9; /* Light grey text for sidebar */
    }
    /* Sidebar navigation radio buttons */
    .stRadio > label {
        font-size: 1.1em !important;
        color: #88a1b9 !important; /* Lighter text for radio labels */
    }
    .stRadio > div[role="radiogroup"] > label {
        background-color: #161b22;
        border-radius: 8px;
        margin-bottom: 8px;
        padding: 10px;
        border: 1px solid #30363d;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }
    .stRadio > div[role="radiogroup"] > label:hover {
        background-color: #1f242c;
        border-color: #00ff9f;
    }
    .stRadio > div[role="radiogroup"] > label > div > p { /* Target the <p> tag inside radio button */
        color: #c9d1d9 !important;
        font-weight: 500;
    }


    /* Card style for sections */
    .section-card {
        border: 2px solid #30363d; /* Default border color */
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 25px;
        background-color: #161b22; /* Dark card background */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .section-card h2, .section-card h3 {
        color: white;
        text-align: center;
        margin-bottom: 15px;
    }
    .stPlotlyChart { /* Ensure plotly charts have transparent background */
        background-color: transparent !important;
    }

</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ---------- Load & Cache Data ----------
@st.cache_data
def load_data():
    # Use a fixed end date for reproducibility if deploying, or today for fresh data
    # For this example, let's keep the original end date
    df = yf.download('AAPL', start='2015-01-01', end='2024-12-31')
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df[['Close']]

@st.cache_resource
def train_model_cached(X_train, y_train): # Renamed to avoid conflict if needed
    model_path = "aapl_stock_lstm_model.h5" # More specific model name
    if os.path.exists(model_path):
        try:
            model = load_model(model_path)
            st.sidebar.success(f"🧠 Model '{model_path}' berhasil dimuat dari cache.")
            return model
        except Exception as e:
            st.sidebar.warning(f"Gagal memuat model dari cache: {e}. Melatih ulang model.")

    with st.spinner("🔄 Melatih model LSTM baru... Ini mungkin memakan waktu beberapa menit."):
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        model.fit(X_train, y_train, epochs=25, batch_size=32, verbose=0) # Increased epochs slightly
        try:
            model.save(model_path)
            st.sidebar.success(f"💾 Model baru berhasil dilatih dan disimpan sebagai '{model_path}'.")
        except Exception as e:
            st.sidebar.error(f"Gagal menyimpan model: {e}")
    return model

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(time_step, len(dataset)):
        X.append(dataset[i-time_step:i, 0])
        y.append(dataset[i, 0])
    return np.array(X), np.array(y)

def classify_trend(current, next_):
    delta = (next_ - current) / current
    if delta > 0.01: # Buy if price increases more than 1%
        return "Buy"
    elif delta < -0.01: # Sell if price decreases more than 1%
        return "Sell"
    else:
        return "Hold"

def calculate_classification_stats(prices):
    if len(prices) < 2:
        return pd.Series({"Buy": 0, "Hold": 100, "Sell": 0}) # Default to Hold if not enough data
    labels = [classify_trend(prices[i], prices[i+1]) for i in range(len(prices)-1)]
    if not labels: # Handle cases with very few prices
         return pd.Series({"Buy": 0, "Hold": 100, "Sell": 0})
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
# Moved model loading/training to be conditional on page to avoid running predict for all pages
model = None # Initialize model variable

# ---------- Sidebar Navigasi ----------
st.sidebar.markdown("---")
st.sidebar.title("🏢 Informasi Kelompok")
st.sidebar.markdown("""
**Nomor Kelompok: 27**

**Anggota Kelompok:**
* Ricky Mario Butar-Butar
""")
st.sidebar.markdown("---")
st.sidebar.title("🧭 MENU NAVIGASI")
page_options = {
    "Beranda": "🏠 Ringkasan & Selamat Datang",
    "Dataset Saham": "📊 Tampilan Dataset Saham",
    "Visualisasi Historis": "📉 Grafik Data Historis Saham",
    "Evaluasi Kinerja Model": "🎯 Metrik Evaluasi Model LSTM",
    "Prediksi & Klasifikasi": "🔮 Prediksi Harga & Klasifikasi Tren",
    "Tentang Proyek": "ℹ️ Detail Mengenai Model & Proyek"
}
page = st.sidebar.radio("Pilih Halaman:", list(page_options.keys()), format_func=lambda x: page_options[x])


# ---------- Fungsi Visualisasi (dengan penyesuaian warna) ----------
chart_bg_color = '#0d1117'
card_bg_color = '#161b22'
text_color = 'white'
accent_color_1 = '#00ff9f' # Greenish
accent_color_2 = '#bb86fc' # Purple
accent_color_3 = '#0abde3' # Blueish

def visualize_historical_data(df_hist):
    st.markdown(
        f"<div class='section-card' style='border-color: {accent_color_1};'>"
        "<h2>📂 Visualisasi Data Historis Saham AAPL</h2>",
        unsafe_allow_html=True
    )
    col4, col5 = st.columns(2)
    min_date = df_hist.index.min().date()
    max_date = df_hist.index.max().date()

    start_date_hist = col4.date_input("🗓️ Tanggal Mulai", min_value=min_date, max_value=max_date, value=min_date, key="hist_start")
    end_date_hist = col5.date_input("🗓️ Tanggal Akhir", min_value=min_date, max_value=max_date, value=max_date, key="hist_end")

    if start_date_hist > end_date_hist:
        st.error("❌ Tanggal mulai harus sebelum tanggal akhir.")
    else:
        df_filtered = df_hist.loc[str(start_date_hist):str(end_date_hist)]
        if df_filtered.empty:
            st.warning("Tidak ada data untuk rentang tanggal yang dipilih.")
        else:
            fig2, ax2 = plt.subplots(figsize=(14, 6))
            fig2.patch.set_facecolor(chart_bg_color)
            ax2.set_facecolor(chart_bg_color)
            ax2.plot(df_filtered.index, df_filtered['Close'], color=accent_color_1, linewidth=2)
            ax2.set_title(f"Data Historis Saham AAPL ({start_date_hist} s.d. {end_date_hist})", color=text_color, fontsize=16)
            ax2.set_xlabel("Tanggal", color=text_color, fontsize=12)
            ax2.set_ylabel("Harga Penutupan (USD)", color=text_color, fontsize=12)
            ax2.tick_params(colors=text_color, which='both')
            ax2.grid(True, linestyle='--', alpha=0.3)
            for spine in ax2.spines.values():
                spine.set_edgecolor(text_color)
            st.pyplot(fig2)
    st.markdown("</div>", unsafe_allow_html=True)

def predict_and_visualize_future_prices(current_model, data_s, scaler_obj, df_orig):
    st.markdown(
        f"<div class='section-card' style='border-color: {accent_color_2};'>"
        "<h2>🧙‍♂️ Prediksi Saham Masa Depan</h2>",
        unsafe_allow_html=True
    )

    mode = st.radio("Pilih mode prediksi:", ["Gunakan Slider Hari", "Gunakan Tanggal Spesifik"], horizontal=True)

    last_date_in_data = df_orig.index.max()
    min_pred_date = last_date_in_data.date() + pd.Timedelta(days=1)

    if mode == "Gunakan Slider Hari":
        n_days = st.slider("Pilih jumlah hari ke depan untuk prediksi (maks 365):", 1, 365, 60)
        start_date = min_pred_date
        # Ensure end_date is calculated correctly based on n_days (business days or calendar days)
        # For simplicity, using calendar days for prediction date range
        future_dates = pd.date_range(start=start_date, periods=n_days)
        end_date = future_dates[-1].date() # Get the actual last date
    else:
        col_t1, col_t2 = st.columns(2)
        start_date_input = col_t1.date_input("🗓️ Tanggal Mulai Prediksi", min_value=min_pred_date, value=min_pred_date, key="pred_start")
        # Ensure min_value for end_date is at least one day after start_date_input
        min_end_pred_date = start_date_input + pd.Timedelta(days=1)
        # Ensure default value for end_date is valid
        default_end_date = start_date_input + pd.Timedelta(days=59) # for approx 60 days
        if default_end_date < min_end_pred_date:
            default_end_date = min_end_pred_date

        end_date_input = col_t2.date_input("🗓️ Tanggal Akhir Prediksi",
                                        min_value=min_end_pred_date,
                                        value=default_end_date, key="pred_end")

        if start_date_input > end_date_input:
            st.error("❌ Tanggal mulai prediksi harus sebelum atau sama dengan tanggal akhir.")
            st.stop()
        start_date = start_date_input
        end_date = end_date_input
        # Calculate n_days based on the selected date range
        future_dates = pd.date_range(start=start_date, end=end_date)
        n_days = len(future_dates)
        if n_days == 0:
            st.warning("Rentang tanggal tidak valid. Pastikan tanggal akhir setelah tanggal mulai.")
            st.stop()


    if n_days > 0:
        with st.spinner(f"🔮 Memprediksi harga untuk {n_days} hari ke depan..."):
            last_sequence = data_s[-60:]
            future_input = last_sequence.reshape(1, 60, 1)
            future_preds_scaled = []

            for _ in range(n_days):
                next_val_scaled = current_model.predict(future_input, verbose=0)[0][0]
                future_preds_scaled.append(next_val_scaled)
                # Update future_input: remove the first element, append the new prediction
                new_element = np.array([[[next_val_scaled]]]) # Reshape for concatenation
                future_input = np.append(future_input[:, 1:, :], new_element, axis=1)

            future_prices_pred = scaler_obj.inverse_transform(np.array(future_preds_scaled).reshape(-1, 1))

        fig3, ax3 = plt.subplots(figsize=(14, 6))
        fig3.patch.set_facecolor(chart_bg_color)
        ax3.set_facecolor(chart_bg_color)
        ax3.plot(future_dates, future_prices_pred, label=f"Prediksi Harga ({n_days} Hari)", color=accent_color_2, marker='o', markersize=4, linestyle='--')
        ax3.set_title(f"Prediksi Harga Saham AAPL ({future_dates[0].strftime('%Y-%m-%d')} s.d. {future_dates[-1].strftime('%Y-%m-%d')})", color=text_color, fontsize=16)
        ax3.set_xlabel("Tanggal", color=text_color, fontsize=12)
        ax3.set_ylabel("Harga Prediksi (USD)", color=text_color, fontsize=12)
        ax3.tick_params(colors=text_color, which='both')
        ax3.legend(facecolor=card_bg_color, edgecolor=text_color, labelcolor=text_color)
        ax3.grid(True, linestyle='--', alpha=0.3)
        for spine in ax3.spines.values():
            spine.set_edgecolor(text_color)
        st.pyplot(fig3)
        st.markdown("</div>", unsafe_allow_html=True)
        return future_prices_pred
    else:
        st.markdown("</div>", unsafe_allow_html=True)
        return np.array([])


def visualize_classification(future_prices_class):
    st.markdown(
        f"""
        <div class='section-card' style='border-color: {accent_color_3};'>
        <h2>📊 Persentase Klasifikasi Tren (Buy/Hold/Sell)</h2>
        </div>
        """, unsafe_allow_html=True
    )

    if future_prices_class is None or len(future_prices_class.flatten()) < 2:
        st.warning("Tidak cukup data prediksi untuk melakukan klasifikasi tren.")
        return

    trend_stats = calculate_classification_stats(future_prices_class.flatten())
    labels = ['Buy', 'Hold', 'Sell']
    sizes = [trend_stats.get(label, 0) for label in labels]
    colors = ['#00cc96', '#ffa600', '#ef553b'] # Mint green, Orange, Coral red
    explode = (0.1 if sizes[0] > 0 else 0, 0, 0) # Highlight "Buy" only if it exists

    col1, col2 = st.columns(2)

    with col1:
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        fig4.patch.set_alpha(0) # Transparent background for the figure
        if sum(sizes) == 0: # Handle case where all sizes are zero
            ax4.text(0.5, 0.5, "Tidak ada data tren", horizontalalignment='center', verticalalignment='center', transform=ax4.transAxes, color=text_color)
        else:
            wedges, texts, autotexts = ax4.pie(
                sizes, labels=labels, autopct='%1.1f%%', startangle=140,
                colors=colors, explode=explode,
                textprops={'color': text_color, 'fontsize': 12, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': card_bg_color, 'linewidth': 1.5} # Add edge to wedges
            )
            for autotext in autotexts: # Make autopct (percentage) bold
                autotext.set_fontweight('bold')

        ax4.set_facecolor('none') # Transparent background for the plot area
        ax4.set_title("Klasifikasi Tren (Pie Chart)", color=text_color, fontsize=15)
        st.pyplot(fig4)

    with col2:
        fig5, ax5 = plt.subplots(figsize=(6, 6))
        fig5.patch.set_alpha(0)
        bars = ax5.bar(labels, sizes, color=colors, edgecolor=text_color, linewidth=1)
        ax5.set_ylim(0, 100)
        ax5.set_ylabel("Persentase (%)", color=text_color, fontsize=12)
        ax5.set_title("Klasifikasi Tren (Bar Chart)", color=text_color, fontsize=15)
        ax5.tick_params(axis='x', colors=text_color, labelsize=12)
        ax5.tick_params(axis='y', colors=text_color, labelsize=10)
        for spine_pos in ['bottom', 'left']: # Keep only bottom and left spines
            ax5.spines[spine_pos].set_edgecolor(text_color)
        ax5.spines['top'].set_visible(False) # Remove top and right spines
        ax5.spines['right'].set_visible(False)
        ax5.set_facecolor('none')
        ax5.grid(axis='y', linestyle='--', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height + 1, f'{height:.1f}%', ha='center', va='bottom', color=text_color, fontsize=11, fontweight='bold')
        st.pyplot(fig5)
    # st.markdown("</div>", unsafe_allow_html=True) # Div closed by the main calling function

# ---------- Main Application Title ----------
st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 30px;'>📈 Dashboard Prediksi Saham AAPL dengan LSTM</h1>", unsafe_allow_html=True)

# ---------- Konten Per Halaman ----------

if page == "Beranda":
    st.markdown(
        f"<div class='section-card' style='border-color: #58a6ff;'>"
        "<h2>🏠 Selamat Datang di Dashboard Prediksi Saham AAPL!</h2>"
        "<p style='color: #c9d1d9; font-size: 1.1em; text-align: justify;'>"
        "Dashboard ini menggunakan model Long Short-Term Memory (LSTM) untuk memprediksi harga saham Apple (AAPL) "
        "dan memberikan klasifikasi tren (Buy, Hold, Sell) berdasarkan prediksi tersebut. "
        "Anda dapat menjelajahi data historis, melihat evaluasi model, dan mendapatkan prediksi harga untuk masa depan."
        "<br><br>Silakan gunakan menu navigasi di sebelah kiri untuk memilih halaman yang ingin Anda lihat."
        "</p>"
        "</div>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<div class='section-card' style='border-color: #58a6ff;'>"
        "<h3 style='color:white;'>⚡ Fitur Utama:</h3>"
        "<ul style='color: #c9d1d9; font-size: 1.1em;'>"
        "<li>Visualisasi data harga saham historis AAPL.</li>"
        "<li>Evaluasi kinerja model prediksi LSTM dengan metrik standar.</li>"
        "<li>Prediksi harga saham AAPL untuk beberapa hari ke depan.</li>"
        "<li>Klasifikasi tren (Buy/Hold/Sell) berdasarkan hasil prediksi.</li>"
        "<li>Informasi detail mengenai model dan dataset yang digunakan.</li>"
        "</ul>"
        "</div>",
        unsafe_allow_html=True
    )


elif page == "Dataset Saham":
    st.markdown(
        f"<div class='section-card' style='border-color: #58a6ff;'>" # Blueish border
        "<h2>📊 Dataset Harga Saham Penutupan AAPL</h2>",
        unsafe_allow_html=True
    )
    st.write("Data diambil dari Yahoo Finance (`AAPL`) dari 1 Januari 2015 hingga 31 Desember 2024.")
    st.write("Berikut adalah 5 baris pertama dari dataset:")
    st.dataframe(df.head(), use_container_width=True)
    st.write("Berikut adalah ringkasan statistik dari dataset:")
    st.dataframe(df.describe(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Visualisasi Historis":
    visualize_historical_data(df)

elif page == "Evaluasi Kinerja Model":
    if model is None:
         model = train_model_cached(X_train, y_train) # Train or load model only if this page is selected

    st.markdown(
        f"<div class='section-card' style='border-color: #f778ba;'>" # Pinkish border
        "<h2>🎯 Evaluasi Kinerja Model LSTM pada Data Test</h2>",
        unsafe_allow_html=True
    )
    with st.spinner("🔍 Mengevaluasi model..."):
        predicted_scaled = model.predict(X_test, verbose=0)
        predicted_prices_eval = scaler.inverse_transform(predicted_scaled)
        real_prices_eval = scaler.inverse_transform(y_test.reshape(-1, 1))

    mse = mean_squared_error(real_prices_eval, predicted_prices_eval)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(real_prices_eval, predicted_prices_eval)
    r2 = r2_score(real_prices_eval, predicted_prices_eval)

    st.subheader("📈 Metrik Evaluasi:")
    col_m1, col_m2 = st.columns(2)
    col_m1.metric(label="Mean Squared Error (MSE)", value=f"{mse:.4f}", help="Semakin kecil semakin baik.")
    col_m1.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.4f}", help="Akar dari MSE, semakin kecil semakin baik.")
    col_m2.metric(label="Mean Absolute Error (MAE)", value=f"{mae:.4f}", help="Rata-rata selisih absolut, semakin kecil semakin baik.")
    col_m2.metric(label="R² Score (Koefisien Determinasi)", value=f"{r2:.4f}", help="Semakin mendekati 1 semakin baik (maks 1).")

    st.subheader("📉 Visualisasi Perbandingan Harga Aktual vs Prediksi (Data Test):")
    fig_eval, ax_eval = plt.subplots(figsize=(14, 6))
    fig_eval.patch.set_facecolor(chart_bg_color)
    ax_eval.set_facecolor(chart_bg_color)
    ax_eval.plot(real_prices_eval, color='#29b6f6', label='Harga Aktual', linewidth=2) # Light Blue
    ax_eval.plot(predicted_prices_eval, color='#ffa726', label='Harga Prediksi', linestyle='--', linewidth=2) # Orange
    ax_eval.set_title('Perbandingan Harga Aktual vs Prediksi (Data Test)', color=text_color, fontsize=16)
    ax_eval.set_xlabel('Periode Waktu (Indeks Data Test)', color=text_color, fontsize=12)
    ax_eval.set_ylabel('Harga Penutupan (USD)', color=text_color, fontsize=12)
    ax_eval.tick_params(colors=text_color, which='both')
    ax_eval.legend(facecolor=card_bg_color, edgecolor=text_color, labelcolor=text_color)
    ax_eval.grid(True, linestyle='--', alpha=0.3)
    for spine in ax_eval.spines.values():
        spine.set_edgecolor(text_color)
    st.pyplot(fig_eval)
    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Prediksi & Klasifikasi":
    if model is None:
        model = train_model_cached(X_train, y_train) # Train or load model only if this page is selected

    # Prediksi Harga Saham dan Klasifikasi Tren Masa Depan
    future_prices_output = predict_and_visualize_future_prices(model, data_scaled, scaler, df)
    if future_prices_output is not None and len(future_prices_output) > 0:
         visualize_classification(future_prices_output)
    else:
        st.info("Prediksi harga masa depan tidak menghasilkan data untuk klasifikasi.")


elif page == "Tentang Proyek":
    st.markdown(
        f"""
        <div class='section-card' style='border-color: #c792ea;'>
        <h2>ℹ️ Tentang Model dan Proyek Ini</h2>
        <p style='color: #c9d1d9; font-size: 1.1em; text-align: left;'>
        Proyek ini bertujuan untuk melakukan prediksi harga saham Apple Inc. (AAPL) menggunakan model Jaringan Saraf Tiruan (JST) jenis Long Short-Term Memory (LSTM). Berikut adalah detailnya:
        </p>

        <h4 style='color: {accent_color_1};'>📦 Dataset:</h4>
        <ul style='color: #c9d1d9;'>
            <li>Sumber Data: Yahoo Finance (ticker: AAPL).</li>
            <li>Rentang Waktu: 1 Januari 2015 - 31 Desember 2024.</li>
            <li>Fitur yang Digunakan: Harga penutupan ('Close').</li>
            <li>Pra-pemrosesan: Data diskalakan menggunakan MinMaxScaler ke rentang (0,1).</li>
        </ul>

        <h4 style='color: {accent_color_2};'>🧠 Model LSTM:</h4>
        <ul style='color: #c9d1d9;'>
            <li>Arsitektur:
                <ul>
                    <li>Layer LSTM pertama dengan 50 unit, `return_sequences=True`.</li>
                    <li>Layer Dropout dengan rate 0.2.</li>
                    <li>Layer LSTM kedua dengan 50 unit.</li>
                    <li>Layer Dropout dengan rate 0.2.</li>
                    <li>Layer Dense output dengan 1 unit (untuk prediksi harga).</li>
                </ul>
            </li>
            <li>Optimizer: Adam.</li>
            <li>Fungsi Kerugian: Mean Squared Error (MSE).</li>
            <li>Epochs: 25, Batch Size: 32.</li>
            <li>Time Step: Model menggunakan 60 hari data sebelumnya untuk memprediksi harga hari berikutnya.</li>
        </ul>

        <h4 style='color: {accent_color_3};'>🎯 Tujuan:</h4>
        <ul style='color: #c9d1d9;'>
            <li>Memprediksi harga saham penutupan AAPL untuk periode waktu di masa depan.</li>
            <li>Memberikan klasifikasi tren (Buy/Hold/Sell) berdasarkan perubahan harga yang diprediksi. Perubahan >1% dianggap 'Buy', perubahan <-1% dianggap 'Sell', sisanya 'Hold'.</li>
        </ul>

        <h4 style='color: #f778ba;'>🛠️ Teknologi yang Digunakan:</h4>
        <ul style='color: #c9d1d9;'>
            <li>Python sebagai bahasa pemrograman utama.</li>
            <li>Streamlit untuk membangun dashboard interaktif.</li>
            <li>TensorFlow (Keras) untuk membangun dan melatih model LSTM.</li>
            <li>Scikit-learn untuk pra-pemrosesan data dan metrik evaluasi.</li>
            <li>Pandas dan NumPy untuk manipulasi data.</li>
            <li>Matplotlib untuk visualisasi data.</li>
            <li>Yfinance untuk mengunduh data saham.</li>
        </ul>
        <p style='color: #c9d1d9; font-size: 0.9em; text-align: center; margin-top: 20px;'>
        Dibuat sebagai bagian dari studi kasus analisis time series dan deep learning.
        </p>
        </div>
        """, unsafe_allow_html=True
    )

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #8892b0; font-size: 0.9em;'>"
    "Dashboard Prediksi Saham AAPL © 2024-2025 Kelompok 27. Dibuat dengan Streamlit."
    "</p>",
    unsafe_allow_html=True
)
