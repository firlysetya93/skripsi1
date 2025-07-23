import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# === Sidebar menu ===
st.sidebar.title("📂 Menu")
menu = st.sidebar.selectbox("Pilih Halaman", ["Preprocessing & Analisis Musim", "Normalisasi dan Splitting Data"])

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
                st.subheader("📉 Uji Stasioneritas (ADF Test) per Musim")
                adf_results = []
                for season, df_season in dfs.items():
                    series = df_season['FF_X'].dropna()
                    adf_result = adfuller(series)
                    adf_results.append({
                        'Musim': season,
                        'ADF Statistic': adf_result[0],
                        'p-value': adf_result[1],
                        'Critical Value 5%': adf_result[4]['5%']
                    })
                st.dataframe(pd.DataFrame(adf_results))

                st.subheader("🔁 ACF dan PACF Plot per Musim (100 Lags)")
                # Pastikan kolom 'TANGGAL' menjadi datetime
                if 'TANGGAL' in df_musim.columns:
                    df_musim['TANGGAL'] = pd.to_datetime(df_musim['TANGGAL'])
                    df_musim.set_index('TANGGAL', inplace=True)
        
                # Ambil hanya kolom FF_X dan drop NaN
                if 'FF_X' in df_musim.columns:
                    ts = df_musim['FF_X'].dropna()
        
                    # --- Uji Stasioneritas: Augmented Dickey-Fuller (ADF) ---
                    result = adfuller(ts, autolag='AIC')
        
                    st.markdown("### Hasil Uji ADF untuk FF_X")
                    st.write(f"**ADF Statistic** : {result[0]:.4f}")
                    st.write(f"**p-value**       : {result[1]:.4f}")
                    st.write("**Critical Values:**")
                    for key, value in result[4].items():
                        st.write(f"   {key} : {value:.4f}")
                    if result[1] <= 0.05:
                        st.success("✅ Data stasioner (tolak H0)")
                    else:
                        st.warning("⚠️ Data tidak stasioner (gagal tolak H0)")
        
                    # --- Plot ACF, PACF, dan Time Series ---
                    fig, axes = plt.subplots(3, 1, figsize=(16, 12))
                    plt.subplots_adjust(hspace=0.5)
        
                    # Plot ACF
                    plot_acf(ts, lags=50, ax=axes[0])
                    axes[0].set_title("Autocorrelation Function (ACF) - FF_X")
        
                    # Plot PACF
                    plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
                    axes[1].set_title("Partial Autocorrelation Function (PACF) - FF_X")
        
                    # Plot Time Series
                    axes[2].plot(ts, color='blue')
                    axes[2].set_title("Time Series Plot - FF_X")
                    axes[2].set_xlabel("Tanggal")
                    axes[2].set_ylabel("Kecepatan Angin (FF_X)")
        
                    st.pyplot(fig)
                st.success("✅ Preprocessing dan analisis musiman selesai! Data siap digunakan di menu berikutnya.")
            except Exception as e:
                st.error(f"❌ Terjadi kesalahan saat memproses tanggal: {e}")
        else:
            st.warning("⚠️ Kolom 'TANGGAL' tidak ditemukan dalam dataset.")
    else:
        st.info("⬆️ Silakan upload file Excel (.xlsx) terlebih dahulu.")
        
if selected_menu == "Normalisasi dan Splitting Data":
    st.subheader("📉 Normalisasi Fitur 'FF_X'")

    values = df_musim['FF_X'].values.astype('float32').reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    df_musim['FF_X_scaled'] = scaled

    st.dataframe(df_musim[['FF_X', 'FF_X_scaled']].head())

    # --- 2. Transformasi ke Supervised Learning ---
    st.subheader("🔁 Transformasi ke Supervised Learning")

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        df = pd.DataFrame(data)
        n_vars = df.shape[1]
        cols, names = [], []

        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [f'var{j+1}(t-{i})' for j in range(n_vars)]

        # output sequence (t, t+1, ..., t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [f'var{j+1}(t)' for j in range(n_vars)]
            else:
                names += [f'var{j+1}(t+{i})' for j in range(n_vars)]

        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    n_days = st.slider("Pilih jumlah lag (n_days):", min_value=1, max_value=30, value=6)
    n_features = 1

    reframed = series_to_supervised(scaled, n_days, 1)
    st.write("✅ Data setelah transformasi:")
    st.dataframe(reframed.head())

    # --- 3. Splitting Data Train/Test ---
    st.subheader("✂️ Splitting Data (Tanpa Shuffle, Rasio 80/20)")

    values = reframed.values
    date_reframed = df_musim.index[reframed.index]

    train_size = int(len(values) * 0.8)
    train, test = values[:train_size], values[train_size:]
    date_train = date_reframed[:len(train)]
    date_test = date_reframed[len(train):]

    st.write(f"📦 Jumlah data total: {len(values)}")
    st.write(f"🟦 Train: {len(train)} | Tanggal: {date_train.min()} → {date_train.max()}")
    st.write(f"🟧 Test: {len(test)} | Tanggal: {date_test.min()} → {date_test.max()}")

    # Visualisasi
    fig, ax = plt.subplots(figsize=(20, 5))
    ax.plot(date_train, train[:, -1], label='Train', color='blue')
    ax.plot(date_test, test[:, -1], label='Test', color='orange')
    ax.set_title('📈 Visualisasi Pembagian Data Train/Test')
    ax.legend()
    st.pyplot(fig)

    # --- 4. Reshape ke Format LSTM ---
    st.subheader("📐 Reshape Data untuk Model LSTM")

    n_obs = n_days * n_features
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]

    X_train = train_X.reshape((train_X.shape[0], n_days, n_features))
    X_test = test_X.reshape((test_X.shape[0], n_days, n_features))
    y_train = train_y.reshape(-1, 1)
    y_test = test_y.reshape(-1, 1)

    st.write(f"✅ X_train shape: {X_train.shape}")
    st.write(f"✅ y_train shape: {y_train.shape}")
    st.write(f"✅ X_test shape: {X_test.shape}")
    st.write(f"✅ y_test shape: {y_test.shape}")

    st.code(f"""
Contoh struktur input LSTM (X_train[0]):
{X_train[0].flatten()}
Target prediksi (y_train[0]): {y_train[0][0]}
    """)

    # --- Simpan ke session_state untuk digunakan di menu selanjutnya ---
    st.session_state.scaler = scaler
    st.session_state.reframed = reframed
    st.session_state.X_train = X_train
    st.session_state.y_train = y_train
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

