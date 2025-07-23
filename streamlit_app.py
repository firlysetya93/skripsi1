import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

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

# st.title("📈 Analisis Musiman Kecepatan Angin")

if uploaded_file is not None:
    df = pd.read_excel(uploaded_file)
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    
    # Tambah kolom Bulan dan Musim hanya sekali di awal
    df['Bulan'] = df['TANGGAL'].dt.month
    def determine_season(month):
        if month in [12, 1, 2]:
            return 'HUJAN's
        elif month in [3, 4, 5]:
            return 'PANCAROBA I'
        elif month in [6, 7, 8]:
            return 'KEMARAU'
        elif month in [9, 10, 11]:
            return 'PANCAROBA II'
    df['Musim'] = df['Bulan'].apply(determine_season)

    # Simpan df ini ke session_state supaya bisa dipakai di blok lain
    st.session_state['df'] = df

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

if st.checkbox("Tampilkan Data dengan Bulan dan Musim"):
    st.subheader("Data Setelah Menambahkan Bulan dan Musim")
    st.write(df_musim.head())

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
    
    # --- Judul Aplikasi ---
    st.title("LSTM Hyperparameter Tuning with Optuna")

# --- Menampilkan dataset yang digunakan (jika tersedia) ---
if 'df' in st.session_state:
    st.subheader("Data Sample")
    st.write(st.session_state.df.head())

# --- Informasi tentang data yang digunakan untuk training ---
if 'X_train' in st.session_state and 'y_train' in st.session_state:
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    n_features = X_train.shape[2]

    st.write("Ukuran data training:", X_train.shape)
    st.write("Ukuran data testing:", X_test.shape)

    # Fungsi untuk objective Optuna
    def objective(trial):
        lstm_units = trial.suggest_int('lstm_units', 10, 200)
        dense_units = trial.suggest_int('dense_units', 10, 200)
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        recurrent_dropout_rate = trial.suggest_float('recurrent_dropout_rate', 0.0, 0.5)
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        epochs = trial.suggest_int('epochs', 20, 100)
        batch_size = trial.suggest_int('batch_size', 16, 128)

        model = Sequential()
        model.add(LSTM(lstm_units, activation='relu', dropout=dropout_rate,
                       recurrent_dropout=recurrent_dropout_rate, return_sequences=True,
                       input_shape=(X_train.shape[1], X_train.shape[2])))
        model.add(Dropout(dropout_rate))
        model.add(LSTM(lstm_units, activation='relu',
                       dropout=dropout_rate, recurrent_dropout=recurrent_dropout_rate))
        model.add(Dropout(dropout_rate))
        model.add(Dense(dense_units, activation='relu'))
        model.add(Dense(n_features))

        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                  validation_data=(X_test, y_test), callbacks=[early_stopping],
                  verbose=0, shuffle=False)

        loss = model.evaluate(X_test, y_test, verbose=0)
        return loss

    # Mulai tuning ketika tombol ditekan
    if st.button("Mulai Hyperparameter Tuning dengan Optuna"):
        with st.spinner("Tuning sedang berlangsung..."):
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=50)
            best_params = study.best_params
            st.success("Tuning selesai!")
            st.write("Best Parameters:")
            st.json(best_params)

            # Simpan ke session state untuk digunakan kembali
            st.session_state.best_params = best_params
            st.session_state.study = study

    # Latih ulang model dengan parameter terbaik
    if 'best_params' in st.session_state:
        best_params = st.session_state.best_params

        st.subheader("Training Model Akhir dengan Parameter Terbaik")

        tuned_model = Sequential()
        tuned_model.add(LSTM(best_params['lstm_units'], activation='relu',
                             dropout=best_params['dropout_rate'],
                             recurrent_dropout=best_params['recurrent_dropout_rate'],
                             return_sequences=True,
                             input_shape=(X_train.shape[1], X_train.shape[2])))
        tuned_model.add(Dropout(best_params['dropout_rate']))
        tuned_model.add(LSTM(best_params['lstm_units'], activation='relu',
                             dropout=best_params['dropout_rate'],
                             recurrent_dropout=best_params['recurrent_dropout_rate']))
        tuned_model.add(Dropout(best_params['dropout_rate']))
        tuned_model.add(Dense(best_params['dense_units'], activation='relu'))
        tuned_model.add(Dense(n_features))

        optimizer = Adam(learning_rate=best_params['learning_rate'])
        tuned_model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

        history = tuned_model.fit(X_train, y_train, epochs=best_params['epochs'],
                                  batch_size=best_params['batch_size'],
                                  validation_data=(X_test, y_test),
                                  callbacks=[early_stopping], shuffle=False, verbose=0)

        test_loss, test_mae = tuned_model.evaluate(X_test, y_test)
        st.write(f"Test Loss: {test_loss:.4f}")
        st.write(f"Test MAE: {test_mae:.4f}")

        # Plot training vs validation loss
        st.subheader("Grafik Loss Selama Training")
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], label='Training Loss')
        ax.plot(history.history['val_loss'], label='Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

else:
    st.warning("Silakan lakukan preprocessing dan pembentukan data train-test sebelum tuning model.")
# Fungsi plotting
def inverse_transform_and_plot(y_true, y_pred, scaler, features):
    inv_pred = scaler.inverse_transform(y_pred)
    inv_true = scaler.inverse_transform(y_true)

    for i in range(len(features)):
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(inv_true[:, i], label='Actual')
        ax.plot(inv_pred[:, i], label='Predicted')
        ax.set_title(f'Actual vs Predicted for {features[i]}')
        ax.set_xlabel('Time')
        ax.set_ylabel(features[i])
        ax.legend()
        st.pyplot(fig)
    
    return inv_true, inv_pred

# Fungsi buat dataframe prediksi
def create_predictions_dataframe(y_true, y_pred, feature_name='FF_X'):
    y_true_flat = y_true.flatten()
    y_pred_flat = np.round(y_pred.flatten(), 3)
    df_final = pd.DataFrame({
        f'{feature_name}': y_true_flat,
        f'{feature_name}_pred': y_pred_flat
    })
    return df_final

# Fungsi evaluasi
def calculate_metrics(y_true, y_pred, features):
    metrics = {
        'feature': [],
        'MAE': [],
        'R2': [],
        'RMSE': [],
        'MAPE': []
    }

    for i, feature in enumerate(features):
        actual = y_true[:, i].flatten()
        predicted = y_pred[:, i].flatten()

        mae = mean_absolute_error(actual, predicted)
        r2 = r2_score(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / actual)) * 100

        metrics['feature'].append(feature)
        metrics['MAE'].append(mae)
        metrics['R2'].append(r2)
        metrics['RMSE'].append(rmse)
        metrics['MAPE'].append(mape)

    return pd.DataFrame(metrics)

# STREAMLIT APP ========================
st.title("📈 Evaluasi Hasil Prediksi Model")

# Simulasi load data (ganti dengan hasil dari modelmu)
if 'y_true' not in st.session_state or 'y_pred' not in st.session_state:
    st.warning("Silakan jalankan model terlebih dahulu.")
else:
    y_true = st.session_state.y_true
    y_pred = st.session_state.y_pred
    scaler = st.session_state.scaler
    features = st.session_state.features

    # Inverse transform dan plot
    st.subheader("Visualisasi Prediksi vs Aktual")
    inv_true, inv_pred = inverse_transform_and_plot(y_true, y_pred, scaler, features)

    # Tabel hasil prediksi
    st.subheader("📊 Tabel Prediksi vs Aktual")
    for i, feature in enumerate(features):
        df_pred = create_predictions_dataframe(inv_true[:, i:i+1], inv_pred[:, i:i+1], feature_name=feature)
        st.write(f"**{feature}**")
        st.dataframe(df_pred.head(50))

    # Evaluasi
    st.subheader("📉 Evaluasi Akurasi Model")
    df_metrics = calculate_metrics(inv_true, inv_pred, features)
    st.dataframe(df_metrics.style.format({"MAE": "{:.3f}", "R2": "{:.3f}", "RMSE": "{:.3f}", "MAPE": "{:.2f}%"}))
st.title("Prediksi Kecepatan Angin dengan LSTM")

uploaded_file = st.file_uploader("Upload file data cuaca (.xlsx/.csv)", type=['xlsx', 'csv'])
feature = st.text_input("Masukkan nama kolom target (contoh: FF_X)", value="FF_X")
lag_value = st.slider("Pilih jumlah lag (back step)", 1, 30, 3)
submit = st.button("Mulai Prediksi")

if uploaded_file and submit:
    if uploaded_file.name.endswith(".xlsx"):
        df = pd.read_excel(uploaded_file)
    else:
        df = pd.read_csv(uploaded_file)

    st.write("Contoh data:")
    st.dataframe(df.head())

    # Cek nilai hilang
    if df[feature].isnull().sum() > 0:
        df[feature].fillna(method='ffill', inplace=True)
        st.warning("Missing value diisi dengan forward fill.")

    # Normalisasi
    scaler = MinMaxScaler()
    scaled_feature = scaler.fit_transform(df[[feature]])

    # Simpan scaler
    os.makedirs("scaler", exist_ok=True)
    joblib.dump(scaler, "scaler/target_scaler.pkl")

    # Supervised Learning
    supervised = to_supervised(scaled_feature, lag=lag_value)
    supervised_values = supervised.values
    train, test = split_data_supervised(supervised_values)

    train_X, train_y = train[:, :-1], train[:, -1:]
    test_X, test_y = test[:, :-1], test[:, -1:]

    # Reshape ke 3D
    train_X_3d = reshape_3d(train_X)
    test_X_3d = reshape_3d(test_X)

    # Train Model
    model = train_lstm_model(train_X_3d, train_y, epochs=50)
    os.makedirs("model", exist_ok=True)
    model.save("model/saved_model.h5")

    # Prediksi
    y_pred = model.predict(test_X_3d)

    # Plot & Inverse
    inv_true, inv_pred = inverse_transform_and_plot(test_y, y_pred, scaler, feature)

    # Metrik
    metrics = calculate_metrics(inv_true, inv_pred)
    st.subheader("Evaluasi Model")
    st.json(metrics)

    st.success("Prediksi selesai dan model telah disimpan.")
# --- Judul halaman ---
st.header('Evaluasi Model LSTM Terbaik')

# --- Plot Loss Training dan Validasi ---
st.subheader('Training vs Validation Loss')
fig_loss, ax_loss = plt.subplots()
ax_loss.plot(history.history['loss'], label='Training Loss')
ax_loss.plot(history.history['val_loss'], label='Validation Loss')
ax_loss.set_xlabel('Epochs')
ax_loss.set_ylabel('Loss')
ax_loss.legend()
st.pyplot(fig_loss)

# --- Plot Hasil Prediksi vs Aktual ---
st.subheader('Perbandingan Nilai Aktual vs Prediksi')
fig_pred, ax_pred = plt.subplots()
ax_pred.plot(y_test_inverse, label='Aktual')
ax_pred.plot(y_hat_inverse, label='Prediksi')
ax_pred.set_title('Prediksi vs Aktual (Test Set)')
ax_pred.set_ylabel('FF_X')
ax_pred.legend()
st.pyplot(fig_pred)

# --- Tampilkan Tabel Metrik Evaluasi ---
st.subheader('Evaluasi Model')
st.dataframe(metrics_df.style.format({'MAE': '{:.4f}', 'RMSE': '{:.4f}', 'R2': '{:.4f}', 'MAPE': '{:.2f}%'}))

# --- Tampilkan Tabel Prediksi (10 data teratas) ---
st.subheader('Contoh Nilai Aktual & Prediksi')
st.dataframe(predictions_df.head(10).style.format('{:.2f}'))
