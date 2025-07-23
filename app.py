import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


st.title("🔍 Pemeriksaan Missing Values dalam Dataset")

# Upload file
uploaded_file = st.file_uploader("📤 Upload file Excel", type=["xlsx"])

# if uploaded_file is not None:
#     # Baca file Excel
#     df = pd.read_excel(uploaded_file)

#     st.subheader("📊 Preview Data (5 Baris Pertama)")
#     st.dataframe(df.head())

#     # Hitung missing values per kolom
#     st.subheader("🧩 Jumlah Missing Values per Kolom")
#     missing_values = df.isnull().sum()
#     st.dataframe(missing_values[missing_values > 0])

#     # Tampilkan baris yang memiliki NaN pada kolom 'FF_X'
#     if 'FF_X' in df.columns:
#         st.subheader("🔎 Baris dengan Missing Values pada Kolom 'FF_X'")
#         missing_ffx_rows = df[df['FF_X'].isnull()]
#         st.dataframe(missing_ffx_rows)
#     else:
#         st.warning("⚠️ Kolom 'FF_X' tidak ditemukan dalam dataset.")
# else:
#     st.info("⬆️ Silakan upload file Excel (.xlsx) terlebih dahulu.")
# import streamlit as st
# import pandas as pd

# st.title("📈 Analisis Musiman Kecepatan Angin")

# if uploaded_file is not None:
#     df = pd.read_excel(uploaded_file)
#     df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    
#     # Tambah kolom Bulan dan Musim hanya sekali di awal
#     df['Bulan'] = df['TANGGAL'].dt.month
#     def determine_season(month):
#         if month in [12, 1, 2]:
#             return 'HUJAN'
#         elif month in [3, 4, 5]:
#             return 'PANCAROBA I'
#         elif month in [6, 7, 8]:
#             return 'KEMARAU'
#         elif month in [9, 10, 11]:
#             return 'PANCAROBA II'
#     df['Musim'] = df['Bulan'].apply(determine_season)

#     # Simpan df ini ke session_state supaya bisa dipakai di blok lain
#     st.session_state['df'] = df

#     # Tampilkan preview data
#     st.subheader("📋 Data Setelah Ditambah Kolom Bulan & Musim")
#     st.dataframe(df.head())

#     # Agregasi statistik berdasarkan musim
#     st.subheader("📊 Statistik Kecepatan Angin Berdasarkan Musim")
#     grouped = df.groupby('Musim').agg({
#         'FF_X': ['mean', 'max', 'min']
#     }).reset_index()

#     # Rename kolom untuk kejelasan
#     grouped.columns = ['Musim', 'FF_X Mean', 'FF_X Max', 'FF_X Min']
#     st.dataframe(grouped)

# else:
#     st.info("⬆️ Silakan upload file Excel untuk memulai analisis.")

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
            df['tahun'] = df['TANGGAL'].dt.year  # <== ini penting!

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

            # ========== Bagian Tambahan: Analisis Time Series ==========
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
            # ===========================================================

        except Exception as e:
            st.error(f"❌ Gagal memproses kolom TANGGAL: {e}")
    else:
        st.warning("⚠️ Kolom 'TANGGAL' tidak ditemukan dalam dataset.")
else:
    st.info("⬆️ Silakan upload file Excel (.xlsx) terlebih dahulu.")


