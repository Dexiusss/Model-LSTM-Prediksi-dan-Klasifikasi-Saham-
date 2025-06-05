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
    # Gunakan tanggal akhir yang tetap untuk reproduktifitas jika mendeploy, atau hari ini untuk data baru
    # Untuk contoh ini, kita pertahankan tanggal akhir asli
    df = yf.download('AAPL', start='2015-01-01', end='2024-12-31') # Akhir tahun bisa disesuaikan
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df[['Close']]

@st.cache_resource
def train_model_cached(X_train_data, y_train_data):
    model_path = "model.h5" # Menggunakan nama file model.h5 sesuai permintaan
    if os.path.exists(model_path):
        try:
            model_obj = load_model(model_path)
            st.sidebar.success(f"🧠 Model '{model_path}' berhasil dimuat.")
            return model_obj
        except Exception as e:
            st.sidebar.warning(f"Gagal memuat model '{model_path}': {e}. Melatih ulang model.")

    with st.spinner("🔄 Melatih model LSTM baru... Ini mungkin memakan waktu beberapa menit."):
        model_obj = Sequential()
        model_obj.add(LSTM(50, return_sequences=True, input_shape=(X_train_data.shape[1], 1)))
        model_obj.add(Dropout(0.2))
        model_obj.add(LSTM(50, return_sequences=False))
        model_obj.add(Dropout(0.2))
        model_obj.add(Dense(1))
        model_obj.compile(optimizer='adam', loss='mean_squared_error')
        model_obj.fit(X_train_data, y_train_data, epochs=25, batch_size=32, verbose=0) # Epochs bisa disesuaikan
        try:
            model_obj.save(model_path)
            st.sidebar.info(f"💾 Model baru berhasil dilatih dan disimpan sebagai '{model_path}'.")
        except Exception as e:
            st.sidebar.error(f"Gagal menyimpan model: {e}")
    return model_obj

def create_dataset(dataset, time_step=60):
    X_data, y_data = [], []
    for i in range(time_step, len(dataset)):
        X_data.append(dataset[i-time_step:i, 0])
        y_data.append(dataset[i, 0])
    return np.array(X_data), np.array(y_data)

def classify_trend(current_price, next_price):
    if current_price == 0: # Hindari pembagian dengan nol
        return "Hold"
    delta = (next_price - current_price) / current_price
    if delta > 0.01: # Rekomendasi Beli jika harga naik lebih dari 1%
        return "Buy"
    elif delta < -0.01: # Rekomendasi Jual jika harga turun lebih dari 1%
        return "Sell"
    else:
        return "Hold"

def calculate_classification_stats(prices_array):
    if len(prices_array) < 2:
        return pd.Series({"Buy": 0, "Hold": 100, "Sell": 0}) # Default ke Hold jika data tidak cukup
    labels_list = [classify_trend(prices_array[i], prices_array[i+1]) for i in range(len(prices_array)-1)]
    if not labels_list: # Tangani kasus dengan sangat sedikit harga
         return pd.Series({"Buy": 0, "Hold": 100, "Sell": 0})
    counts_series = pd.Series(labels_list).value_counts(normalize=True).reindex(["Buy", "Hold", "Sell"]).fillna(0)
    return counts_series * 100

# ---------- Data Preparation ----------
df_main = load_data()
scaler_main = MinMaxScaler(feature_range=(0, 1))
data_scaled_main = scaler_main.fit_transform(df_main)
X_main, y_main = create_dataset(data_scaled_main, 60) # time_step = 60
X_main = X_main.reshape(X_main.shape[0], X_main.shape[1], 1)

train_size_main = int(len(X_main) * 0.8)
X_train_main, X_test_main = X_main[:train_size_main], X_main[train_size_main:]
y_train_main, y_test_main = y_main[:train_size_main], y_main[train_size_main:]

# ---------- Model Initialization (akan dimuat atau dilatih saat halaman membutuhkannya) ----------
model_global = None

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
page_options_dict = {
    "Beranda": "🏠 Ringkasan & Selamat Datang",
    "Dataset Saham": "📊 Tampilan Dataset Saham",
    "Visualisasi Historis": "📉 Grafik Data Historis Saham",
    "Evaluasi Kinerja Model": "🎯 Metrik Evaluasi Model LSTM",
    "Prediksi & Klasifikasi": "🔮 Prediksi Harga & Klasifikasi Tren",
    "Tentang Proyek": "ℹ️ Detail Mengenai Model & Proyek"
}
selected_page = st.sidebar.radio("Pilih Halaman:", list(page_options_dict.keys()), format_func=lambda page_key: page_options_dict[page_key])


# ---------- Fungsi Visualisasi (dengan penyesuaian warna) ----------
chart_bg_color = '#0d1117'
card_bg_color = '#161b22'
text_color = 'white'
accent_color_1 = '#00ff9f' # Greenish
accent_color_2 = '#bb86fc' # Purple
accent_color_3 = '#0abde3' # Blueish

def visualize_historical_data(df_hist_data):
    st.markdown(
        f"<div class='section-card' style='border-color: {accent_color_1};'>"
        "<h2>📂 Visualisasi Data Historis Saham AAPL</h2>",
        unsafe_allow_html=True
    )
    col_start_date, col_end_date = st.columns(2)
    min_allowable_date = df_hist_data.index.min().date()
    max_allowable_date = df_hist_data.index.max().date()

    start_date_hist_input = col_start_date.date_input("🗓️ Tanggal Mulai", min_value=min_allowable_date, max_value=max_allowable_date, value=min_allowable_date, key="hist_start_date")
    end_date_hist_input = col_end_date.date_input("🗓️ Tanggal Akhir", min_value=min_allowable_date, max_value=max_allowable_date, value=max_allowable_date, key="hist_end_date")

    if start_date_hist_input > end_date_hist_input:
        st.error("❌ Tanggal mulai harus sebelum atau sama dengan tanggal akhir.")
    else:
        df_filtered_hist = df_hist_data.loc[str(start_date_hist_input):str(end_date_hist_input)]
        if df_filtered_hist.empty:
            st.warning("Tidak ada data untuk rentang tanggal yang dipilih.")
        else:
            fig_hist, ax_hist = plt.subplots(figsize=(14, 6))
            fig_hist.patch.set_facecolor(chart_bg_color)
            ax_hist.set_facecolor(chart_bg_color)
            ax_hist.plot(df_filtered_hist.index, df_filtered_hist['Close'], color=accent_color_1, linewidth=2)
            ax_hist.set_title(f"Data Historis Saham AAPL ({start_date_hist_input} s.d. {end_date_hist_input})", color=text_color, fontsize=16)
            ax_hist.set_xlabel("Tanggal", color=text_color, fontsize=12)
            ax_hist.set_ylabel("Harga Penutupan (USD)", color=text_color, fontsize=12)
            ax_hist.tick_params(colors=text_color, which='both', rotation=45)
            ax_hist.grid(True, linestyle='--', alpha=0.3)
            for spine_name in ax_hist.spines:
                ax_hist.spines[spine_name].set_edgecolor(text_color)
            st.pyplot(fig_hist)
    st.markdown("</div>", unsafe_allow_html=True)

def predict_and_visualize_future_prices(current_model_obj, data_scaled_arr, scaler_obj_fit, df_original_data):
    st.markdown(
        f"<div class='section-card' style='border-color: {accent_color_2};'>"
        "<h2>🧙‍♂️ Prediksi Saham Masa Depan</h2>",
        unsafe_allow_html=True
    )

    prediction_mode = st.radio("Pilih mode prediksi:", ["Gunakan Slider Hari", "Gunakan Tanggal Spesifik"], horizontal=True, key="pred_mode_radio")

    last_date_in_df = df_original_data.index.max()
    min_prediction_start_date = last_date_in_df.date() + pd.Timedelta(days=1)

    if prediction_mode == "Gunakan Slider Hari":
        num_days_to_predict = st.slider("Pilih jumlah hari ke depan untuk prediksi (maks 365):", 1, 365, 60, key="days_slider")
        prediction_start_date = min_prediction_start_date
        # Menggunakan pd.date_range untuk menghasilkan tanggal termasuk akhir pekan/libur jika relevan untuk pasar tertentu
        future_dates_pd = pd.date_range(start=prediction_start_date, periods=num_days_to_predict)
        prediction_end_date = future_dates_pd[-1].date()
    else:
        col_pred_start, col_pred_end = st.columns(2)
        input_start_date = col_pred_start.date_input("🗓️ Tanggal Mulai Prediksi", min_value=min_prediction_start_date, value=min_prediction_start_date, key="pred_start_date_input")
        
        min_allowable_end_date = input_start_date + pd.Timedelta(days=1)
        default_end_val = input_start_date + pd.Timedelta(days=59) # Default sekitar 60 hari
        if default_end_val < min_allowable_end_date:
            default_end_val = min_allowable_end_date

        input_end_date = col_pred_end.date_input("🗓️ Tanggal Akhir Prediksi",
                                        min_value=min_allowable_end_date,
                                        value=default_end_val, key="pred_end_date_input")

        if input_start_date > input_end_date:
            st.error("❌ Tanggal mulai prediksi harus sebelum atau sama dengan tanggal akhir.")
            st.markdown("</div>", unsafe_allow_html=True) # Tutup div sebelum stop
            return np.array([]) # Kembalikan array kosong jika error
        
        prediction_start_date = input_start_date
        prediction_end_date = input_end_date
        future_dates_pd = pd.date_range(start=prediction_start_date, end=prediction_end_date)
        num_days_to_predict = len(future_dates_pd)
        
        if num_days_to_predict == 0:
            st.warning("Rentang tanggal tidak valid atau tidak menghasilkan hari untuk prediksi.")
            st.markdown("</div>", unsafe_allow_html=True) # Tutup div sebelum stop
            return np.array([])

    if num_days_to_predict > 0:
        with st.spinner(f"🔮 Memprediksi harga untuk {num_days_to_predict} hari ke depan..."):
            time_step_val = 60 # Harus sama dengan yang digunakan saat training
            last_sequence_scaled = data_scaled_arr[-time_step_val:]
            current_batch_scaled = last_sequence_scaled.reshape(1, time_step_val, 1)
            future_predictions_scaled = []

            for _ in range(num_days_to_predict):
                next_prediction_scaled = current_model_obj.predict(current_batch_scaled, verbose=0)[0][0]
                future_predictions_scaled.append(next_prediction_scaled)
                # Perbarui batch: geser dan tambahkan prediksi baru
                new_element_scaled = np.array([[[next_prediction_scaled]]])
                current_batch_scaled = np.append(current_batch_scaled[:, 1:, :], new_element_scaled, axis=1)

            future_prices_unscaled = scaler_obj_fit.inverse_transform(np.array(future_predictions_scaled).reshape(-1, 1))

        fig_pred, ax_pred = plt.subplots(figsize=(14, 6))
        fig_pred.patch.set_facecolor(chart_bg_color)
        ax_pred.set_facecolor(chart_bg_color)
        ax_pred.plot(future_dates_pd, future_prices_unscaled, label=f"Prediksi Harga ({num_days_to_predict} Hari)", color=accent_color_2, marker='o', markersize=4, linestyle='--')
        ax_pred.set_title(f"Prediksi Harga Saham AAPL ({future_dates_pd[0].strftime('%Y-%m-%d')} s.d. {future_dates_pd[-1].strftime('%Y-%m-%d')})", color=text_color, fontsize=16)
        ax_pred.set_xlabel("Tanggal", color=text_color, fontsize=12)
        ax_pred.set_ylabel("Harga Prediksi (USD)", color=text_color, fontsize=12)
        ax_pred.tick_params(colors=text_color, which='both', rotation=45)
        ax_pred.legend(facecolor=card_bg_color, edgecolor=text_color, labelcolor=text_color)
        ax_pred.grid(True, linestyle='--', alpha=0.3)
        for spine_name in ax_pred.spines:
            ax_pred.spines[spine_name].set_edgecolor(text_color)
        st.pyplot(fig_pred)
        st.markdown("</div>", unsafe_allow_html=True)
        return future_prices_unscaled
    else:
        st.info("Tidak ada hari yang dipilih untuk prediksi.")
        st.markdown("</div>", unsafe_allow_html=True)
        return np.array([])


def visualize_classification(future_prices_arr):
    st.markdown(
        f"""
        <div class='section-card' style='border-color: {accent_color_3};'>
        <h2>📊 Persentase Klasifikasi Tren (Buy/Hold/Sell)</h2>
        </div>
        """, unsafe_allow_html=True # Div ini hanya untuk judul, isinya di bawah
    )

    if future_prices_arr is None or len(future_prices_arr.flatten()) < 2:
        st.warning("Tidak cukup data prediksi untuk melakukan klasifikasi tren.")
        return

    trend_stats_series = calculate_classification_stats(future_prices_arr.flatten())
    pie_labels = ['Buy', 'Hold', 'Sell']
    pie_sizes = [trend_stats_series.get(label, 0) for label in pie_labels]
    pie_colors = ['#00cc96', '#ffa600', '#ef553b'] # Mint green, Orange, Coral red
    pie_explode = (0.1 if pie_sizes[0] > 0 else 0, 0, 0) # Sorot "Buy" jika ada

    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig_pie, ax_pie = plt.subplots(figsize=(6, 6))
        fig_pie.patch.set_alpha(0) 
        if sum(pie_sizes) == 0: 
            ax_pie.text(0.5, 0.5, "Tidak ada data tren", horizontalalignment='center', verticalalignment='center', transform=ax_pie.transAxes, color=text_color)
        else:
            wedges, texts_label, texts_autopct = ax_pie.pie(
                pie_sizes, labels=pie_labels, autopct='%1.1f%%', startangle=140,
                colors=pie_colors, explode=pie_explode,
                textprops={'color': text_color, 'fontsize': 12, 'fontweight': 'bold'},
                wedgeprops={'edgecolor': card_bg_color, 'linewidth': 1.5} 
            )
            for autopct_item in texts_autopct: 
                autopct_item.set_fontweight('bold')

        ax_pie.set_facecolor('none') 
        ax_pie.set_title("Klasifikasi Tren (Pie Chart)", color=text_color, fontsize=15)
        st.pyplot(fig_pie)

    with col_bar:
        fig_bar, ax_bar = plt.subplots(figsize=(6, 6))
        fig_bar.patch.set_alpha(0)
        bars_container = ax_bar.bar(pie_labels, pie_sizes, color=pie_colors, edgecolor=text_color, linewidth=1)
        ax_bar.set_ylim(0, 100)
        ax_bar.set_ylabel("Persentase (%)", color=text_color, fontsize=12)
        ax_bar.set_title("Klasifikasi Tren (Bar Chart)", color=text_color, fontsize=15)
        ax_bar.tick_params(axis='x', colors=text_color, labelsize=12)
        ax_bar.tick_params(axis='y', colors=text_color, labelsize=10)
        for spine_pos in ['bottom', 'left']: 
            ax_bar.spines[spine_pos].set_edgecolor(text_color)
        ax_bar.spines['top'].set_visible(False) 
        ax_bar.spines['right'].set_visible(False)
        ax_bar.set_facecolor('none')
        ax_bar.grid(axis='y', linestyle='--', alpha=0.3)

        for bar_item in bars_container:
            height_val = bar_item.get_height()
            ax_bar.text(bar_item.get_x() + bar_item.get_width()/2., height_val + 1, f'{height_val:.1f}%', ha='center', va='bottom', color=text_color, fontsize=11, fontweight='bold')
        st.pyplot(fig_bar)

# ---------- Main Application Title ----------
st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 30px;'>📈 Dashboard Prediksi Saham AAPL dengan LSTM</h1>", unsafe_allow_html=True)

# ---------- Konten Per Halaman ----------

if selected_page == "Beranda":
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


elif selected_page == "Dataset Saham":
    st.markdown(
        f"<div class='section-card' style='border-color: #58a6ff;'>" 
        "<h2>📊 Dataset Harga Saham Penutupan AAPL</h2>",
        unsafe_allow_html=True
    )
    st.write(f"Data diambil dari Yahoo Finance (`AAPL`) dari {df_main.index.min().strftime('%d %B %Y')} hingga {df_main.index.max().strftime('%d %B %Y')}.")
    st.write("Berikut adalah 5 baris pertama dari dataset:")
    st.dataframe(df_main.head(), use_container_width=True)
    st.write("Berikut adalah ringkasan statistik dari dataset:")
    st.dataframe(df_main.describe(), use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_page == "Visualisasi Historis":
    visualize_historical_data(df_main)

elif selected_page == "Evaluasi Kinerja Model":
    if model_global is None:
         model_global = train_model_cached(X_train_main, y_train_main) 

    st.markdown(
        f"<div class='section-card' style='border-color: #f778ba;'>" 
        "<h2>🎯 Evaluasi Kinerja Model LSTM pada Data Test</h2>",
        unsafe_allow_html=True
    )
    with st.spinner("🔍 Mengevaluasi model..."):
        predicted_scaled_eval = model_global.predict(X_test_main, verbose=0)
        predicted_prices_eval = scaler_main.inverse_transform(predicted_scaled_eval)
        real_prices_eval = scaler_main.inverse_transform(y_test_main.reshape(-1, 1))

    mse_val = mean_squared_error(real_prices_eval, predicted_prices_eval)
    rmse_val = np.sqrt(mse_val)
    mae_val = mean_absolute_error(real_prices_eval, predicted_prices_eval)
    r2_val = r2_score(real_prices_eval, predicted_prices_eval)

    st.subheader("📈 Metrik Evaluasi:")
    col_metric1, col_metric2 = st.columns(2)
    col_metric1.metric(label="Mean Squared Error (MSE)", value=f"{mse_val:.4f}", help="Semakin kecil semakin baik.")
    col_metric1.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse_val:.4f}", help="Akar dari MSE, semakin kecil semakin baik.")
    col_metric2.metric(label="Mean Absolute Error (MAE)", value=f"{mae_val:.4f}", help="Rata-rata selisih absolut, semakin kecil semakin baik.")
    col_metric2.metric(label="R² Score (Koefisien Determinasi)", value=f"{r2_val:.4f}", help="Semakin mendekati 1 semakin baik (maks 1).")

    st.subheader("📉 Visualisasi Perbandingan Harga Aktual vs Prediksi (Data Test):")
    fig_eval_plot, ax_eval_plot = plt.subplots(figsize=(14, 6))
    fig_eval_plot.patch.set_facecolor(chart_bg_color)
    ax_eval_plot.set_facecolor(chart_bg_color)
    ax_eval_plot.plot(real_prices_eval, color='#29b6f6', label='Harga Aktual', linewidth=2) 
    ax_eval_plot.plot(predicted_prices_eval, color='#ffa726', label='Harga Prediksi', linestyle='--', linewidth=2) 
    ax_eval_plot.set_title('Perbandingan Harga Aktual vs Prediksi (Data Test)', color=text_color, fontsize=16)
    ax_eval_plot.set_xlabel('Periode Waktu (Indeks Data Test)', color=text_color, fontsize=12)
    ax_eval_plot.set_ylabel('Harga Penutupan (USD)', color=text_color, fontsize=12)
    ax_eval_plot.tick_params(colors=text_color, which='both', rotation=45)
    ax_eval_plot.legend(facecolor=card_bg_color, edgecolor=text_color, labelcolor=text_color)
    ax_eval_plot.grid(True, linestyle='--', alpha=0.3)
    for spine_name in ax_eval_plot.spines:
        ax_eval_plot.spines[spine_name].set_edgecolor(text_color)
    st.pyplot(fig_eval_plot)
    st.markdown("</div>", unsafe_allow_html=True)

elif selected_page == "Prediksi & Klasifikasi":
    if model_global is None:
        model_global = train_model_cached(X_train_main, y_train_main) 

    # Prediksi Harga Saham dan Klasifikasi Tren Masa Depan
    future_prices_output_arr = predict_and_visualize_future_prices(model_global, data_scaled_main, scaler_main, df_main)
    if future_prices_output_arr is not None and len(future_prices_output_arr) > 0:
         visualize_classification(future_prices_output_arr)
    else:
        # Pesan sudah ditangani di dalam predict_and_visualize_future_prices atau visualize_classification
        pass


elif selected_page == "Tentang Proyek":
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
            <li>Rentang Waktu Aktual Data: {df_main.index.min().strftime('%d %B %Y')} - {df_main.index.max().strftime('%d %B %Y')}.</li>
            <li>Fitur yang Digunakan: Harga penutupan ('Close').</li>
            <li>Pra-pemrosesan: Data diskalakan menggunakan MinMaxScaler ke rentang (0,1).</li>
        </ul>

        <h4 style='color: {accent_color_2};'>🧠 Model LSTM (dari <code>model.h5</code> jika ada):</h4>
        <ul style='color: #c9d1d9;'>
            <li>Arsitektur (jika dilatih baru):
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
            <li>Epochs (jika dilatih baru): 25, Batch Size: 32.</li>
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
