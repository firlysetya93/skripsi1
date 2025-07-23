import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# === Sidebar menu ===
st.sidebar.title("📂 Menu")
menu = st.sidebar.selectbox("Pilih Halaman", ["Preprocessing & Analisis Musim", "Normalisasi & Split Data"])

# === Menu 1: Preprocessing & Analisis Musim ===
if menu == "Preprocessing & Analisis Musim":
    st.title("📊 Analisis Kecepatan Angin")

    uploaded_file = st.file_uploader("Unggah file Excel", type=['xlsx'])

    if uploaded_file is not None:
        df = pd.read_excel(uploaded_file)
        st.subheader("📊 Preview Data (5 Baris Pertama)")
        st.dataframe(df.head())

        st.subheader("🧩 Jumlah Missing Values per Kolom")
        missing_values = df.isnull().sum()
        st.dataframe(missing_values[missing_values > 0])

        if 'FF_X' in df.columns:
            st.subheader("🔎 Baris dengan Missing Values pada Kolom 'FF_X'")
            st.dataframe(df[df['FF_X'].isnull()])
        else:
            st.warning("⚠️ Kolom 'FF_X' tidak ditemukan dalam dataset.")

        if 'TANGGAL' in df.columns:
            try:
                df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
                df['Bulan'] = df['TANGGAL'].dt.month

                def determine_season(month):
                    if month in [12, 1, 2]:
                        return 'HUJAN'
                    elif month in [3, 4, 5]:
                        return 'PANCAROBA I'
                    elif month in [6, 7, 8]:
                        return 'KEMARAU'
                    elif month in [9, 10, 11]:
                        return 'PANCAROBA II'
                    else:
                        return 'UNKNOWN'

                df['Musim'] = df['Bulan'].apply(determine_season)
                df['tahun'] = df['TANGGAL'].dt.year

                st.session_state['df'] = df
                st.subheader("📋 Data Setelah Ditambah Kolom Bulan & Musim")
                st.dataframe(df.head())

                st.subheader("📊 Statistik Kecepatan Angin Berdasarkan Musim")
                grouped = df.groupby('Musim').agg({'FF_X': ['mean', 'max', 'min']}).reset_index()
                grouped.columns = ['Musim', 'FF_X Mean', 'FF_X Max', 'FF_X Min']
                st.dataframe(grouped)

                df_selected = df[['TANGGAL', 'FF_X', 'Musim']].copy()
                df_selected = df_selected.set_index('TANGGAL')

                dfs = {}
                for season, group in df_selected.groupby('Musim'):
                    dfs[season] = group.reset_index()

                st.subheader("🗂️ Data Per Musim")
                for season, df_season in dfs.items():
                    st.markdown(f"### Musim: {season}")
                    st.dataframe(df_season.head())

                df_musim = pd.concat(dfs.values(), ignore_index=True)
                df_musim = df_musim.sort_values('TANGGAL').reset_index(drop=True)
                st.session_state['df_musim'] = df_musim

                st.subheader("📅 Data Gabungan (Diurutkan Berdasarkan Tanggal)")
                st.dataframe(df_musim.head(1000))

                # --- Analisis Time Series ---
                st.subheader("📈 Rata-Rata Kecepatan Angin per Tahun")
                rata_tahunan = df.groupby('tahun')['FF_X'].mean()
                fig1, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(rata_tahunan.index, rata_tahunan.values, marker='o')
                ax1.set_xlabel('Tahun')
                ax1.set_ylabel('Rata-rata Kecepatan Angin (m/s)')
                ax1.set_title('Rata-Rata Kecepatan Angin per Tahun')
                ax1.grid(True)
                ax1.set_xticks(rata_tahunan.index)
                st.pyplot(fig1)

                musim_opsi = df_musim['Musim'].unique().tolist()
                musim_dipilih = st.selectbox("🎯 Pilih Musim untuk Uji Stasioneritas & ACF/PACF", musim_opsi)

                ts = df_musim[df_musim['Musim'] == musim_dipilih]['FF_X'].dropna()

                st.subheader(f"📊 Uji Stasioneritas ADF - Musim {musim_dipilih}")
                result = adfuller(ts, autolag='AIC')
                st.write(f"**ADF Statistic**: {result[0]:.4f}")
                st.write(f"**p-value**: {result[1]:.4f}")
                st.write("**Critical Values:**")
                for key, value in result[4].items():
                    st.write(f"  - {key}: {value:.4f}")
                if result[1] <= 0.05:
                    st.success("✅ Data stasioner (tolak H0)")
                else:
                    st.warning("⚠️ Data tidak stasioner (gagal tolak H0)")

                st.subheader("🔍 Visualisasi ACF, PACF, dan Time Series")
                fig2, axes = plt.subplots(3, 1, figsize=(12, 12))
                plt.subplots_adjust(hspace=0.5)
                plot_acf(ts, lags=50, ax=axes[0])
                axes[0].set_title(f'ACF - {musim_dipilih}')
                plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
                axes[1].set_title(f'PACF - {musim_dipilih}')
                axes[2].plot(ts, color='blue')
                axes[2].set_title(f'Seri Waktu FF_X - {musim_dipilih}')
                axes[2].set_xlabel('Index')
                axes[2].set_ylabel('Kecepatan Angin (m/s)')
                st.pyplot(fig2)

            except Exception as e:
                st.error(f"❌ Gagal memproses kolom TANGGAL: {e}")
        else:
            st.warning("⚠️ Kolom 'TANGGAL' tidak ditemukan dalam dataset.")
    else:
        st.info("⬆️ Silakan upload file Excel (.xlsx) terlebih dahulu.")

# === Menu 2: Normalisasi & Train-Test Split ===
elif menu == "Normalisasi & Split Data":
    st.title("📉 Normalisasi dan Pembagian Data")

    if 'df_musim' in st.session_state:
        df_musim = st.session_state['df_musim']

        # Normalisasi nilai FF_X
        values = df_musim['FF_X'].values.astype('float32').reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_values = scaler.fit_transform(values)

        df_musim['FF_X_SCALED'] = scaled_values
        st.subheader("📊 Data setelah normalisasi:")
        st.dataframe(df_musim[['TANGGAL', 'FF_X', 'FF_X_SCALED']].head())

        # Split data tanpa shuffle
        df_train, df_test = train_test_split(df_musim, test_size=0.2, shuffle=False)

        st.subheader("📦 Ukuran Data:")
        st.write(f"Data Train: {df_train.shape}")
        st.write(f"Data Test: {df_test.shape}")

        st.subheader("📈 Visualisasi Pembagian Data Train dan Test")
        fig, ax = plt.subplots(figsize=(16, 5))
        ax.plot(df_train.index, df_train['FF_X'], label='Train', color='blue')
        ax.plot(df_test.index, df_test['FF_X'], label='Test', color='orange')
        ax.set_title("Pembagian Data Train dan Test pada Variabel FF_X")
        ax.set_xlabel("Index")
        ax.set_ylabel("FF_X")
        ax.legend()
        st.pyplot(fig)
    else:
        st.warning("⚠️ Data musiman belum tersedia. Silakan jalankan preprocessing terlebih dahulu dari menu sebelumnya.")
