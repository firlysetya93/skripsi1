import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.set_page_config(layout="wide")
st.title("🌬️ Aplikasi Analisis dan Prediksi Kecepatan Angin")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file)
    st.session_state.df = df
    st.success("File berhasil diunggah!")

if 'df' in st.session_state:
    df = st.session_state.df.copy()

    # ===========================
    # PRA-PROSES DAN IMPUTASI
    # ===========================
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    df['Bulan'] = df['TANGGAL'].dt.month

    def determine_season(month):
        if month in [12, 1, 2]: return 'HUJAN'
        elif month in [3, 4, 5]: return 'PANCAROBA I'
        elif month in [6, 7, 8]: return 'KEMARAU'
        elif month in [9, 10, 11]: return 'PANCAROBA II'

    df['Musim'] = df['Bulan'].apply(determine_season)

    def fill_missing_values(group):
        group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
        return group

    df_filled = df.groupby('Musim').apply(fill_missing_values).reset_index(drop=True)
    st.subheader("Data setelah Imputasi Musiman")
    st.dataframe(df_filled.head())

    # ===========================
    # RATA-RATA TAHUNAN
    # ===========================
    df_filled['tahun'] = df_filled['TANGGAL'].dt.year
    rata_tahunan = df_filled.groupby('tahun')['FF_X'].mean()
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(rata_tahunan.index, rata_tahunan.values, marker='o')
    ax1.set_title('Rata-rata Kecepatan Angin per Tahun')
    ax1.set_xlabel('Tahun')
    ax1.set_ylabel('FF_X (m/s)')
    ax1.grid()
    st.pyplot(fig1)

    # ===========================
    # UJI STASIONERITAS (ADF)
    # ===========================
    st.subheader("Uji Stasioneritas ADF")
    ts = df_filled['FF_X'].dropna()
    result = adfuller(ts)
    st.write(f"ADF Statistic: {result[0]:.4f}")
    st.write(f"p-value: {result[1]:.4f}")
    for key, value in result[4].items():
        st.write(f"   {key} : {value:.4f}")

    fig2, axes = plt.subplots(3, 1, figsize=(15, 10))
    plot_acf(ts, lags=50, ax=axes[0])
    plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
    axes[2].plot(ts)
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    axes[2].set_title("Time Series FF_X")
    st.pyplot(fig2)

    # ===========================
    # SCALING & TRANSFORMASI
    # ===========================
    st.subheader("Scaling dan Transformasi Supervised")

    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        df = pd.DataFrame(data)
        cols, names = [], []
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [f'var1(t-{i})']
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            names += [f'var1(t+{i})']
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df_filled[['FF_X']])
    reframed = series_to_supervised(scaled, 6, 1)
    st.write("Data setelah dibentuk supervised:")
    st.dataframe(reframed.head())

    # ===========================
    # SPLIT DATA
    # ===========================
    st.subheader("Train-Test Split dan Visualisasi")
    values = reframed.values
    date_reframed = df_filled.index[reframed.index]
    train_size = int(len(values) * 0.8)
    train, test = values[:train_size], values[train_size:]
    date_train = date_reframed[:len(train)]
    date_test = date_reframed[len(train):]

    n_obs = 6
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]

    X_train = train_X.reshape((train_X.shape[0], 6, 1))
    X_test = test_X.reshape((test_X.shape[0], 6, 1))
    y_train = train_y.reshape(-1, 1)
    y_test = test_y.reshape(-1, 1)

    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"y_train shape: {y_train.shape}")
    st.write(f"X_test shape : {X_test.shape}")
    st.write(f"y_test shape : {y_test.shape}")

    # ===========================
    # PLOT PEMBAGIAN TRAIN VS TEST
    # ===========================
    st.subheader("Visualisasi Train vs Test")
    df_plot = pd.DataFrame({
        'FF_X': df_filled['FF_X'].values,
        'TANGGAL': df_filled['TANGGAL']
    })
    df_plot['split'] = 'Train'
    df_plot.loc[train_size:, 'split'] = 'Test'

    fig3, ax3 = plt.subplots(figsize=(15, 4))
    ax3.plot(df_plot['TANGGAL'], df_plot['FF_X'], label='Full Series', alpha=0.5)
    ax3.plot(df_plot[df_plot['split'] == 'Train']['TANGGAL'], df_plot[df_plot['split'] == 'Train']['FF_X'], label='Train', color='blue')
    ax3.plot(df_plot[df_plot['split'] == 'Test']['TANGGAL'], df_plot[df_plot['split'] == 'Test']['FF_X'], label='Test', color='orange')
    ax3.set_title("Pembagian Train dan Test")
    ax3.legend()
    st.pyplot(fig3)

else:
    st.info("Silakan upload file terlebih dahulu untuk memulai.")
