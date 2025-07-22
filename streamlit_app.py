import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# =================== Sidebar Menu ===================
menu = st.sidebar.selectbox("Navigasi Menu", [
    "🏠 Home",
    "📤 Upload Data",
    "📊 EDA",
    "⚙️ Preprocessing",
    "🧠 Modeling (LSTM / TCN / RBFNN)",
    "📈 Prediction"
])

uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state.df = df
else:
    df = st.session_state.get('df', None)

# =================== HOME ===================
if menu == "🏠 Home":
    st.title("🏠 Selamat Datang di Aplikasi Prediksi Kecepatan Angin")
    st.markdown("""
Aplikasi ini membantu kamu:
- 📊 Melakukan eksplorasi data angin (EDA)
- ⚙️ Preprocessing berdasarkan musim
- 🧠 Modeling dengan LSTM / TCN / RBFNN
- 📈 Prediksi kecepatan angin ke depan
""")

# =================== EDA ===================
elif menu == "📊 EDA":
    if df is not None:
        st.header("📊 Eksplorasi Data Awal")

        st.subheader("Tampilan Data")
        st.dataframe(df.head())

        st.subheader("Jumlah Missing Value Tiap Kolom")
        missing_counts = df.isnull().sum()
        st.write(missing_counts[missing_counts > 0])

        st.subheader("Rata-rata Kecepatan Angin per Tahun")
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
        df['tahun'] = df['TANGGAL'].dt.year
        rata_tahunan = df.groupby('tahun')['FF_X'].mean()
        fig, ax = plt.subplots()
        ax.plot(rata_tahunan.index, rata_tahunan.values, marker='o')
        ax.set_xlabel('Tahun')
        ax.set_ylabel('Rata-rata Kecepatan Angin (m/s)')
        ax.set_title('Rata-rata Kecepatan Angin per Tahun')
        st.pyplot(fig)

# =================== PREPROCESSING ===================
elif menu == "⚙️ Preprocessing":
    if df is not None:
        st.header("⚙️ Preprocessing Data")

        # Deteksi Musim
        st.subheader("Penambahan Kolom Musim")
        df['Bulan'] = df['TANGGAL'].dt.month
        def determine_season(month):
            if month in [12, 1, 2]: return 'HUJAN'
            elif month in [3, 4, 5]: return 'PANCAROBA I'
            elif month in [6, 7, 8]: return 'KEMARAU'
            else: return 'PANCAROBA II'
        df['Musim'] = df['Bulan'].apply(determine_season)
        st.dataframe(df[['TANGGAL', 'FF_X', 'Musim']].head())

        # Imputasi berdasarkan Musim
        st.subheader("Imputasi Nilai Hilang berdasarkan Musim")
        def fill_missing_values(group):
            group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
            return group
        df_filled = df.groupby('Musim').apply(fill_missing_values).reset_index(drop=True)
        st.write("Jumlah missing setelah imputasi:")
        st.write(df_filled.isnull().sum())

        # Simpan ke session state
        st.session_state.df_filled = df_filled

# =================== MODELING ===================
elif menu == "🧠 Modeling (LSTM / TCN / RBFNN)":
    if 'df_filled' in st.session_state:
        df_filled = st.session_state.df_filled
        st.header("🧠 Persiapan Modeling")

        df_musim = df_filled[['TANGGAL', 'FF_X', 'Musim']].set_index('TANGGAL').sort_index()
        ts = df_musim['FF_X'].dropna()

        st.subheader("Uji Stasioneritas ADF")
        result = adfuller(ts, autolag='AIC')
        st.write(f"**ADF Statistic**: {result[0]:.4f}")
        st.write(f"**p-value**: {result[1]:.4f}")
        for key, val in result[4].items():
            st.write(f"{key}: {val:.4f}")

        if result[1] <= 0.05:
            st.success("✅ Data stasioner (tolak H0)")
        else:
            st.warning("⚠️ Data tidak stasioner (gagal tolak H0)")

        st.subheader("Plot ACF dan PACF")
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        plot_acf(ts, lags=50, ax=axes[0])
        axes[0].set_title("ACF - FF_X")
        plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
        axes[1].set_title("PACF - FF_X")
        axes[2].plot(ts)
        axes[2].set_title("Time Series - FF_X")
        st.pyplot(fig)

# =================== PREDICTION ===================
elif menu == "📈 Prediction":
    st.header("📈 Prediksi Kecepatan Angin")
    st.info("Silakan lanjutkan ke bagian ini setelah modeling selesai.")
