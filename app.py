import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# === Sidebar menu ===
st.sidebar.title("📂 Menu")
menu = st.sidebar.selectbox("Pilih Halaman", ["Preprocessing & Analisis Musim", "Normalisasi & Split Data", "Tuning Hyperparameter LSTM"])

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
            # --- Uji Stasioneritas ADF dan ACF/PACF Seluruh Data ---
        # --- Uji Stasioneritas ADF dan ACF/PACF Seluruh Data ---
                if 'FF_X' in df_musim.columns and 'TANGGAL' in df_musim.columns:
                    try:
                        ts = df_musim['FF_X'].dropna()
                
                        st.subheader("📊 Uji Stasioneritas ADF - Seluruh Data")
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
                
                        st.subheader("🔍 Visualisasi ACF, PACF, dan Time Series (Seluruh Data)")
                        fig2, axes = plt.subplots(3, 1, figsize=(12, 12))
                        plt.subplots_adjust(hspace=0.5)
                        plot_acf(ts, lags=50, ax=axes[0])
                        axes[0].set_title('ACF - Seluruh Data')
                        plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
                        axes[1].set_title('PACF - Seluruh Data')
                        axes[2].plot(df_musim['TANGGAL'], ts, color='blue')
                        axes[2].set_title('Seri Waktu FF_X - Seluruh Data')
                        axes[2].set_xlabel('Tanggal')
                        axes[2].set_ylabel('Kecepatan Angin (m/s)')
                        st.pyplot(fig2)
                
                    except Exception as e:
                        st.error(f"❌ Gagal memproses kolom TANGGAL: {e}")
                else:
                    st.warning("⚠️ Kolom 'TANGGAL' tidak ditemukan dalam dataset.")

    else:
        st.info("⬆️ Silakan upload file Excel (.xlsx) terlebih dahulu.")

# === Menu 2: Normalisasi & Train-Test Split ===
elif menu == "📊 Split Data Time Series":
    st.subheader("📊 Split Data Time Series: Train dan Test Set")

    if "reframed" in st.session_state:
        reframed = st.session_state.reframed
        scaler = st.session_state.scaler

        try:
            # Ambil n_days dan n_features dari input sebelumnya
            n_days = reframed.shape[1] - 1
            n_features = 1  # karena hanya 1 kolom: FF_X

            # Ambil nilai reframed
            values = reframed.values

            # Ambil index dari df_musim untuk tanggal
            date_reframed = df_musim.index[reframed.index]

            # Split data 80:20
            train_size = int(len(values) * 0.8)
            train, test = values[:train_size], values[train_size:]

            # Bagi juga indeks tanggal
            date_train = date_reframed[:len(train)]
            date_test = date_reframed[len(train):]

            # Split input dan target
            n_obs = n_days * n_features
            train_X, train_y = train[:, :n_obs], train[:, -1]
            test_X, test_y = test[:, :n_obs], test[:, -1]

            # Reshape ke 3D (untuk LSTM)
            X_train = train_X.reshape((train_X.shape[0], n_days, n_features))
            X_test = test_X.reshape((test_X.shape[0], n_days, n_features))

            # Reshape y (2D)
            y_train = train_y.reshape(-1, 1)
            y_test = test_y.reshape(-1, 1)

            # Simpan di session_state untuk modeling
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            st.session_state.date_train = date_train
            st.session_state.date_test = date_test

            # Tampilkan info
            st.success("✅ Data berhasil dibagi menjadi train dan test secara berurutan")
            st.markdown(f"""
            - Total data: {len(values)}
            - Jumlah data train: {len(train)} ({date_train.min().date()} s.d. {date_train.max().date()})
            - Jumlah data test: {len(test)} ({date_test.min().date()} s.d. {date_test.max().date()})
            - Bentuk X_train: {X_train.shape}
            - Bentuk y_train: {y_train.shape}
            """)

            # Visualisasi Train vs Test
            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(date_train, y_train, label='Train', color='blue')
            ax.plot(date_test, y_test, label='Test', color='red')
            ax.set_title("Visualisasi Target Train dan Test")
            ax.set_xlabel("Tanggal")
            ax.set_ylabel("Nilai Normalisasi FF_X")
            ax.legend()
            st.pyplot(fig)

        except Exception as e:
            st.error(f"❌ Gagal membagi data: {e}")
    else:
        st.warning("⚠️ Transformasi supervised belum dilakukan. Silakan jalankan proses sebelumnya.")

