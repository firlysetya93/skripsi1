# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

st.set_page_config(layout="wide")
st.title("📈 Aplikasi Prediksi Kecepatan Angin")

menu = st.sidebar.selectbox("Navigasi Menu", [
    "🏠 Home",
    "📤 Upload Data",
    "📊 EDA & Imputasi Musiman",
    "📈 Uji Stasioneritas",
    "⚙️ Preprocessing Time Series"
])

# ====================================
# FUNGSI
# ====================================
def determine_season(month):
    if month in [12, 1, 2]:
        return 'HUJAN'
    elif month in [3, 4, 5]:
        return 'PANCAROBA I'
    elif month in [6, 7, 8]:
        return 'KEMARAU'
    elif month in [9, 10, 11]:
        return 'PANCAROBA II'

def fill_missing_values(group):
    group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
    return group

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    df = pd.DataFrame(data)
    n_vars = df.shape[1]
    cols, names = [], []
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [f'var{j+1}(t-{i})' for j in range(n_vars)]
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

# ====================================
# 1. Upload
# ====================================
if menu == "📤 Upload Data":
    uploaded_file = st.file_uploader("Upload file Excel (.xlsx)", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        st.session_state.df = df
        st.success("File berhasil diunggah!")

# ====================================
# 2. EDA & Imputasi Musiman
# ====================================
elif menu == "📊 EDA & Imputasi Musiman":
    if 'df' not in st.session_state:
        st.warning("Silakan upload data terlebih dahulu.")
    else:
        df = st.session_state.df.copy()
        df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
        df['Bulan'] = df['TANGGAL'].dt.month
        df['Musim'] = df['Bulan'].apply(determine_season)

        # Imputasi Missing Value
        df_filled = df.groupby('Musim').apply(fill_missing_values).reset_index(drop=True)

        st.write("Data setelah imputasi musiman:")
        st.dataframe(df_filled)

        # Rata-rata tahunan
        df_filled['tahun'] = df_filled['TANGGAL'].dt.year
        rata_tahunan = df_filled.groupby('tahun')['FF_X'].mean()

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(rata_tahunan.index, rata_tahunan.values, marker='o')
        ax.set_title("Rata-Rata Kecepatan Angin per Tahun")
        ax.set_ylabel("FF_X (m/s)")
        ax.grid()
        st.pyplot(fig)

        st.session_state.df_filled = df_filled

# ====================================
# 3. Uji Stasioneritas
# ====================================
elif menu == "📈 Uji Stasioneritas":
    if 'df_filled' not in st.session_state:
        st.warning("Lakukan preprocessing & imputasi terlebih dahulu.")
    else:
        df_musim = st.session_state.df_filled.copy()
        ts = df_musim['FF_X'].dropna()

        result = adfuller(ts)
        st.subheader("Hasil ADF Test")
        st.write(f"ADF Statistic : {result[0]:.4f}")
        st.write(f"p-value       : {result[1]:.4f}")
        st.write("Critical Values:")
        for key, value in result[4].items():
            st.write(f"   {key} : {value:.4f}")
        if result[1] <= 0.05:
            st.success("✅ Data stasioner (tolak H0)")
        else:
            st.warning("⚠️ Data tidak stasioner (gagal tolak H0)")

        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        plot_acf(ts, lags=50, ax=axes[0])
        axes[0].set_title("Autocorrelation Function (ACF)")
        plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
        axes[1].set_title("Partial Autocorrelation Function (PACF)")
        axes[2].plot(ts)
        axes[2].set_title("Time Series Plot")
        st.pyplot(fig)

# ====================================
# 4. Preprocessing Time Series
# ====================================
elif menu == "⚙️ Preprocessing Time Series":
    if 'df_filled' not in st.session_state:
        st.warning("Lakukan preprocessing & imputasi terlebih dahulu.")
    else:
        df_musim = st.session_state.df_filled.copy()
        df_musim = df_musim.sort_values('TANGGAL').reset_index(drop=True)

        # Scaling
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(df_musim[['FF_X']])
        reframed = series_to_supervised(scaled, n_in=6, n_out=1)
        values = reframed.values

        # Train-test split
        train_size = int(len(values) * 0.8)
        train, test = values[:train_size], values[train_size:]
        n_obs = 6 * 1
        train_X, train_y = train[:, :n_obs], train[:, -1]
        test_X, test_y = test[:, :n_obs], test[:, -1]

        X_train = train_X.reshape((train_X.shape[0], 6, 1))
        X_test = test_X.reshape((test_X.shape[0], 6, 1))
        y_train = train_y.reshape(-1, 1)
        y_test = test_y.reshape(-1, 1)

        st.subheader("📈 Data Setelah Scaling dan Splitting")
        st.write(f"X_train shape: {X_train.shape}")
        st.write(f"y_train shape: {y_train.shape}")
        st.write(f"X_test shape : {X_test.shape}")
        st.write(f"y_test shape : {y_test.shape}")

        st.line_chart(df_musim.set_index("TANGGAL")['FF_X'])
