import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import optuna
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score



# === Sidebar menu ===
st.sidebar.title("📂 Menu")
menu = st.sidebar.selectbox("Pilih Halaman", ["Preprocessing & Analisis Musim", "Normalisasi dan Splitting Data", "Hyperparameter Tuning (LSTM)"])

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
        
# === Menu 2: Normalisasi dan Splitting Data ===
if menu == "Normalisasi dan Splitting Data":
    st.title("📏 Normalisasi dan Splitting Data")

    if "df_musim" not in st.session_state:
        st.warning("❗ Data musim belum tersedia. Silakan lakukan preprocessing terlebih dahulu di menu sebelumnya.")
        st.stop()
    else:
        df_musim = st.session_state['df_musim'].copy()
        values = df_musim['FF_X'].values.astype('float32').reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(values)
        df_musim['FF_X_scaled'] = scaled
        df_train, df_test = train_test_split(df_musim, test_size=0.2, shuffle=False)

        st.write("✅ Data berhasil dibagi:")
        st.write(f"📘 Data Train: {df_train.shape}")
        st.write(f"📙 Data Test: {df_test.shape}")
        
        # Simpan ke session state
        st.session_state['df_train'] = df_train
        st.session_state['df_test'] = df_test
        st.session_state['scaler'] = scaler
    
        st.dataframe(df_musim[['FF_X', 'FF_X_scaled']].head())
        st.success("✅ Data telah dinormalisasi dan siap untuk digunakan.")
    
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
Struktur input LSTM (X_train[0]):
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
    st.session_state.n_features = n_features
    
if menu == "Hyperparameter Tuning (LSTM)":
    st.title("🎯 Hyperparameter Tuning dengan Optuna (LSTM)")

    if 'X_train' not in st.session_state or 'y_train' not in st.session_state:
        st.warning("🚨 Data belum diproses! Silakan lakukan preprocessing, transformasi supervised, dan splitting terlebih dahulu.")
    else:
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        n_features = st.session_state.n_features
        scaler = st.session_state.scaler
        
        df_train = st.session_state.df_train
        df_test = st.session_state.df_test
    
        n_trials = st.number_input("🔁 Jumlah Percobaan (Trials)", min_value=10, max_value=100, value=50, step=10)
    
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
            # Tuning
        if st.button("🚀 Jalankan Tuning"):
            with st.spinner("🔍 Mencari kombinasi terbaik..."):

                # ✅ Tambahkan ini
                scaler = st.session_state.scaler

                study = optuna.create_study(direction='minimize')
                study.optimize(objective, n_trials=n_trials)

                st.success("🎯 Tuning selesai!")
                st.write("Best loss:", study.best_value)
                st.json(study.best_params)

                st.session_state.best_params = study.best_params
                best_params = study.best_params

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
                tuned_model.compile(optimizer=optimizer,
                                    loss='mean_squared_error',
                                    metrics=['mae'])

                early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

                history = tuned_model.fit(
                    X_train, y_train,
                    epochs=best_params['epochs'],
                    batch_size=best_params['batch_size'],
                    validation_data=(X_test, y_test),
                    callbacks=[early_stopping],
                    shuffle=False,
                    verbose=0
                )

                test_loss, test_mae = tuned_model.evaluate(X_test, y_test, verbose=0)
                st.success(f"✅ **Test Loss:** {test_loss:.4f} | **Test MAE:** {test_mae:.4f}")

                # Grafik Loss
                st.subheader("📉 Grafik Loss Selama Training")
                fig, ax = plt.subplots()
                ax.plot(history.history['loss'], label='Training Loss')
                ax.plot(history.history['val_loss'], label='Validation Loss')
                ax.set_xlabel('Epochs')
                ax.set_ylabel('Loss')
                ax.legend()
                st.pyplot(fig)

                # ---------------- INVERSE TRANSFORM & PREDIKSI ---------------- #
                st.subheader("📊 Visualisasi Prediksi dan Evaluasi")

                y_pred = tuned_model.predict(X_test)
                y_test_inverse = scaler.inverse_transform(y_test)
                y_pred_inverse = scaler.inverse_transform(y_pred)

                fig2, ax2 = plt.subplots(figsize=(20, 8))
                ax2.plot(y_test_inverse[:, 0], label='Aktual')
                ax2.plot(y_pred_inverse[:, 0], label='Prediksi')
                ax2.set_title('Prediksi vs Aktual FF_X')
                ax2.set_xlabel('Waktu')
                ax2.set_ylabel('FF_X')
                ax2.legend()
                st.pyplot(fig2)

                # --- Buat DataFrame Prediksi vs Aktual ---
                def create_predictions_dataframe(y_true, y_pred, feature_name='FF_X'):
                    y_true_flat = y_true.flatten()
                    y_pred_flat = np.round(y_pred.flatten(), 3)
                    return pd.DataFrame({
                        f'{feature_name}': y_true_flat,
                        f'{feature_name}_pred': y_pred_flat
                    })

                df_prediksi = create_predictions_dataframe(y_test_inverse, y_pred_inverse, feature_name='FF_X')
                st.subheader("📋 Tabel Prediksi vs Aktual (Top 20)")
                st.dataframe(df_prediksi.head(20))

                def calculate_metrics(y_true, y_pred, feature_name='FF_X'):
                    y_true = y_true.flatten()
                    y_pred = y_pred.flatten()

                    mae = mean_absolute_error(y_true, y_pred)
                    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                    r2 = r2_score(y_true, y_pred)

                    mask = y_true != 0
                    if np.any(mask):
                        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
                    else:
                        mape = np.nan

                    return pd.DataFrame({
                        'feature': [feature_name],
                        'MAE': [mae],
                        'RMSE': [rmse],
                        'R2': [r2],
                        'MAPE': [mape]
                    })

                st.subheader("📌 Metrik Evaluasi Model")
                df_metrics = calculate_metrics(y_test_inverse, y_pred_inverse, feature_name='FF_X')
                st.dataframe(df_metrics)
                
                # Plotly visualisasi
                st.subheader("📈 Visualisasi Data Training, Test, dan Prediksi (Plotly)")

                def plot_feature_predictions(df_train, df_test, predictions_df, features):
                    for feature in features:
                        trace_train = go.Scatter(x=df_train.index, y=df_train[feature],
                                                 mode='lines', name='Training Data',
                                                 line=dict(color='blue'))
                        trace_test = go.Scatter(x=df_test.index, y=df_test[feature],
                                                mode='lines', name='Test Data',
                                                line=dict(color='green'))
                        trace_pred = go.Scatter(x=predictions_df.index, y=predictions_df[f'{feature}_pred'],
                                                mode='lines', name='Predicted Data',
                                                line=dict(color='red'))

                        layout = go.Layout(title=f'{feature} - Training, Test, and Prediction',
                                           xaxis=dict(title='Tanggal'),
                                           yaxis=dict(title='Value'),
                                           legend=dict(x=0.1, y=1.1, orientation='h'),
                                           plot_bgcolor='rgba(0,0,0,0)')

                        fig = go.Figure(data=[trace_train, trace_test, trace_pred], layout=layout)
                        st.plotly_chart(fig, use_container_width=True)

                plot_feature_predictions(df_train, df_test, df_prediksi, [feature_name])


                

