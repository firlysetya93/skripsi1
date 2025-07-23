import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

st.title("🔍 Pemeriksaan Missing Values dalam Dataset")

# Upload file
uploaded_file = st.file_uploader("📤 Upload file Excel", type=["xlsx"])

if uploaded_file is not None:
    # Baca file Excel
    df = pd.read_excel(uploaded_file)

    st.subheader("📊 Preview Data (5 Baris Pertama)")
    st.dataframe(df.head())

    # Hitung missing values per kolom
    st.subheader("🧩 Jumlah Missing Values per Kolom")
    missing_values = df.isnull().sum()
    st.dataframe(missing_values[missing_values > 0])

    # Tampilkan baris yang memiliki NaN pada kolom 'FF_X'
    if 'FF_X' in df.columns:
        st.subheader("🔎 Baris dengan Missing Values pada Kolom 'FF_X'")
        missing_ffx_rows = df[df['FF_X'].isnull()]
        st.dataframe(missing_ffx_rows)
    else:
        st.warning("⚠️ Kolom 'FF_X' tidak ditemukan dalam dataset.")
else:
    st.info("⬆️ Silakan upload file Excel (.xlsx) terlebih dahulu.")
import streamlit as st
import pandas as pd

st.title("📈 Analisis Musiman Kecepatan Angin")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)

    # Konversi kolom tanggal
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])

    # Tambahkan kolom Bulan
    df['Bulan'] = df['TANGGAL'].dt.month

    # Tambahkan kolom Musim
    def determine_season(month):
        if month in [12, 1, 2]:
            return 'HUJAN'
        elif month in [3, 4, 5]:
            return 'PANCAROBA I'
        elif month in [6, 7, 8]:
            return 'KEMARAU'
        elif month in [9, 10, 11]:
            return 'PANCAROBA II'

    df['Musim'] = df['Bulan'].apply(determine_season)

    # Tampilkan preview data
    st.subheader("📋 Data Setelah Ditambah Kolom Bulan & Musim")
    st.dataframe(df.head())

    # Agregasi statistik berdasarkan musim
    st.subheader("📊 Statistik Kecepatan Angin Berdasarkan Musim")
    grouped = df.groupby('Musim').agg({
        'FF_X': ['mean', 'max', 'min']
    }).reset_index()

    # Rename kolom untuk kejelasan
    grouped.columns = ['Musim', 'FF_X Mean', 'FF_X Max', 'FF_X Min']
    st.dataframe(grouped)

else:
    st.info("⬆️ Silakan upload file Excel untuk memulai analisis.")

st.set_page_config(page_title="Imputasi Musiman FF_X", layout="wide")
st.title("🌦️ Imputasi Missing Value Berdasarkan Musim")

    st.subheader("Data Setelah Menambahkan Bulan dan Musim")
    st.dataframe(df.head())

    # Cek missing values
    st.subheader("Missing Values per Musim")
    missing_rows = df[df.isnull().any(axis=1)]
    if not missing_rows.empty:
        missing_by_group = missing_rows.groupby('Musim')
        for group, rows in missing_by_group:
            st.markdown(f"**Missing values di grup `{group}`**")
            st.dataframe(rows)
    else:
        st.success("Tidak ada missing value yang terdeteksi.")

    # Imputasi rata-rata berdasarkan musim
    def fill_missing_values(group):
        group['FF_X'] = group['FF_X'].fillna(group['FF_X'].mean())
        return group

    df_filled = df.groupby('Musim').apply(fill_missing_values)
    df_filled.reset_index(drop=True, inplace=True)

    st.subheader("Setelah Imputasi Rata-rata per Musim")
    st.dataframe(df_filled.head())
    st.write("Jumlah missing value setelah imputasi:")
    st.code(df_filled.isnull().sum())

    # Split berdasarkan musim
    dfs = {}
    for season, group in df_filled.groupby('Musim'):
        dfs[season] = group

    st.subheader("Contoh Dataframe per Musim")
    for season in dfs:
        st.markdown(f"**Musim: {season}**")
        st.dataframe(dfs[season].head())

    # Gabungkan kembali dataframe berdasarkan index tanggal
    df_selected = df_filled[['TANGGAL', 'FF_X', 'Musim']]
    df_selected = df_selected.set_index('TANGGAL')

    dfs_sorted = {}
    for season, group in df_selected.groupby('Musim'):
        dfs_sorted[season] = group.reset_index()

    # Gabungkan jadi satu dan urutkan berdasarkan tanggal
    df_musim = pd.concat(dfs_sorted.values(), ignore_index=True)
    df_musim = df_musim.sort_values('TANGGAL').reset_index(drop=True)

    st.subheader("Data Gabungan Setelah Diurutkan")
    st.dataframe(df_musim.head(1000))
if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    df['tahun'] = df['TANGGAL'].dt.year

    # Hitung dan tampilkan tren tahunan
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

    # Proses musiman
    df_selected = df[['TANGGAL', 'FF_X', 'Musim']]
    df_selected = df_selected.set_index('TANGGAL')
    dfs = {}
    for season, group in df_selected.groupby('Musim'):
        dfs[season] = group.reset_index()
    df_musim = pd.concat(dfs.values(), ignore_index=True)
    df_musim = df_musim.sort_values('TANGGAL').reset_index(drop=True)

    # --- Uji Stasioneritas (ADF) dan Plot ---
    st.subheader("🧪 Uji Stasioneritas ADF & Plot ACF/PACF")
    ts = df_musim['FF_X'].dropna()

    result = adfuller(ts, autolag='AIC')
    st.markdown("#### Hasil Uji ADF:")
    st.write(f"**ADF Statistic**: {result[0]:.4f}")
    st.write(f"**p-value**: {result[1]:.4f}")
    st.markdown("**Critical Values:**")
    for key, value in result[4].items():
        st.write(f"{key} : {value:.4f}")
    if result[1] <= 0.05:
        st.success("✅ Data stasioner (tolak H0)")
    else:
        st.warning("⚠️ Data tidak stasioner (gagal tolak H0)")

    # Visualisasi ACF, PACF, dan Time Series
    st.markdown("#### Visualisasi ACF, PACF, dan Time Series")
    fig2, axes = plt.subplots(3, 1, figsize=(16, 12))
    plt.subplots_adjust(hspace=0.5)
    
    plot_acf(ts, lags=50, ax=axes[0])
    axes[0].set_title('Autocorrelation Function (ACF) - FF_X')

    plot_pacf(ts, lags=50, ax=axes[1], method='ywm')
    axes[1].set_title('Partial Autocorrelation Function (PACF) - FF_X')

    axes[2].plot(df_musim['TANGGAL'], ts, color='blue')
    axes[2].set_title('Time Series Plot - FF_X')
    axes[2].set_xlabel('Tanggal')
    axes[2].set_ylabel('Kecepatan Angin (m/s)')

    st.pyplot(fig2)
else:
    st.info("Silakan upload file Excel (.xlsx) terlebih dahulu.")
st.subheader("📈 Normalisasi & Pembagian Data Train-Test")

# Pastikan df_musim sudah tersedia sebelumnya dan kolom 'FF_X' tidak kosong
if 'df_musim' in locals() and not df_musim['FF_X'].isnull().all():
    # --- Normalisasi ---
    values = df_musim['FF_X'].values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_values = scaler.fit_transform(values.reshape(-1, 1))

    # Tampilkan hasil normalisasi
    st.write("Contoh hasil normalisasi (5 nilai pertama):")
    st.write(scaled_values[:5])

    # --- Pembagian Data Train-Test ---
    df_train, df_test = train_test_split(df_musim, test_size=0.2, shuffle=False)

    st.write(f"Jumlah data Train: {df_train.shape[0]}")
    st.write(f"Jumlah data Test: {df_test.shape[0]}")

    # --- Visualisasi Pembagian ---
    st.subheader("📊 Visualisasi Pembagian Data")
    features = ['FF_X']
    for feature in features:
        fig, ax = plt.subplots(figsize=(16, 4))
        ax.plot(df_train.index, df_train[feature], label='Training', color='blue')
        ax.plot(df_test.index, df_test[feature], label='Testing', color='orange')
        ax.set_title(f'Pembagian Data Train dan Test pada Variabel {feature}')
        ax.set_xlabel("Tanggal")
        ax.set_ylabel("Kecepatan Angin")
        ax.legend(loc='upper right')
        st.pyplot(fig)

    # --- Simpan scaler untuk digunakan inverse nanti (opsional) ---
    # st.session_state['scaler'] = scaler

else:
    st.warning("❗ Data musim belum tersedia atau kolom 'FF_X' kosong. Harap lakukan preprocessing terlebih dahulu.")
# Section: Transformasi ke Supervised Learning
st.subheader("Transformasi Supervised Learning dan Splitting Data")

# Buat fungsi series_to_supervised (jika belum didefinisikan sebelumnya)
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

# Ambil data dari session_state
df_musim = st.session_state.get("df_musim", None)

if df_musim is not None:
    st.success("Data musim berhasil dimuat.")
    
    # Normalisasi
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled = scaler.fit_transform(df_musim[['FF_X']])
    
    # Parameter
    n_days = st.slider("Pilih Jumlah Lag (n_days)", 1, 30, 6)
    n_features = 1
    
    # Transformasi menjadi supervised
    reframed = series_to_supervised(scaled, n_days, 1)
    st.write(f"Dimensi data hasil transformasi: {reframed.shape}")
    st.dataframe(reframed.head())

    # Simpan reframed ke session_state
    st.session_state['reframed'] = reframed

    # Ambil nilai sebagai numpy array
    values = reframed.values
    st.session_state['values'] = values

    # Bagi fitur dan target (misal 1 kolom terakhir adalah target)
    n_obs = n_days * n_features
    X = values[:, :n_obs]
    y = values[:, -1]
    
    # Simpan X dan y ke session_state
    st.session_state['X'] = X
    st.session_state['y'] = y

    # Split data
    test_size = st.slider("Test Size (%)", 5, 50, 20)
    train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=test_size/100, shuffle=False)

    st.write(f"Train X shape: {train_X.shape}")
    st.write(f"Test X shape: {test_X.shape}")

    # Simpan ke session_state
    st.session_state['train_X'] = train_X
    st.session_state['test_X'] = test_X
    st.session_state['train_y'] = train_y
    st.session_state['test_y'] = test_y

    # Plot train vs test target
    fig, ax = plt.subplots(figsize=(20,5))
    ax.plot(range(len(train_y)), train_y, label='Train Target')
    ax.plot(range(len(train_y), len(train_y) + len(test_y)), test_y, label='Test Target', color='orange')
    ax.set_title('Pembagian Data Train dan Test pada Target (y)')
    ax.legend()
    st.pyplot(fig)

else:
    st.warning("Data musim belum tersedia di session_state.")
# Lanjutan dari pipeline Streamlit sebelumnya
with st.expander("7. Transformasi Supervised dan Pembagian Data"):
    st.subheader("Transformasi ke Format Supervised Learning")
    
    # Ambil nilai dari dataframe hasil reframing
    values = reframed.values

    # Simpan index tanggal dari hasil reframing
    date_reframed = df_musim.index[reframed.index]

    # Membagi data train dan test (tanpa shuffle)
    train_size = int(len(values) * 0.8)
    train, test = values[:train_size], values[train_size:]

    # Bagi juga indeks tanggal
    date_train = date_reframed[:len(train)]
    date_test = date_reframed[len(train):]

    st.write(f"Jumlah data: {len(values)}")
    st.write(f"Jumlah train : {len(train)} ({date_train.min().date()} s.d. {date_train.max().date()})")
    st.write(f"Jumlah test  : {len(test)} ({date_test.min().date()} s.d. {date_test.max().date()})")

    # Pisahkan input (X) dan target (y)
    n_obs = n_days * n_features  # input shape
    train_X, train_y = train[:, :n_obs], train[:, -1]
    test_X, test_y = test[:, :n_obs], test[:, -1]

    # Reshape ke bentuk 3D untuk LSTM
    X_train = train_X.reshape((train_X.shape[0], n_days, n_features))
    X_test = test_X.reshape((test_X.shape[0], n_days, n_features))
    y_train = train_y.reshape(-1, 1)
    y_test = test_y.reshape(-1, 1)

    st.write(f"Total fitur per timestep: {n_features}")
    st.write(f"X_train shape: {X_train.shape}")
    st.write(f"y_train shape: {y_train.shape}")
    st.write(f"X_test shape: {X_test.shape}")
    st.write(f"y_test shape: {y_test.shape}")

    with st.expander("Contoh Struktur Data"):
        st.write("Contoh data input (X_train[0]):")
        st.write(X_train[0])
        st.write("Contoh data output (y_train[0]):")
        st.write(y_train[0])
