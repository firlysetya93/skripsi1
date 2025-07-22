import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def app():
    st.title("Prediksi Kecepatan Angin Musiman di BMKG Juanda")

    st.header("📘 Deskripsi Penelitian")
    st.markdown("""
        Aplikasi ini dikembangkan sebagai bagian dari penelitian prediksi kecepatan angin harian di wilayah BMKG Juanda 
        menggunakan model machine learning seperti LSTM, Temporal Convolutional Network (TCN), dan Recurrent-Based Neural 
        Fuzzy Function (RBNFF). Prediksi dilakukan dengan mempertimbangkan pola musiman seperti DJF, MAM, JJA, dan SON 
        untuk mengungkap karakteristik fluktuasi angin sepanjang tahun. Evaluasi model dilakukan menggunakan metrik MAE, RMSE, 
        R², dan MAPE.
    """)

    st.header("📊 Eksplorasi Data Musiman")
    df = pd.read_csv('data/df_musim.csv', parse_dates=['TANGGAL'])
    df['TANGGAL'] = pd.to_datetime(df['TANGGAL'])
    df.set_index('TANGGAL', inplace=True)

    musim = st.selectbox("Pilih Musim", options=df['Musim'].unique())
    data_musim = df[df['Musim'] == musim]

    st.line_chart(data_musim['FF_X'])

    st.write(f"Statistik Deskriptif Kecepatan Angin - Musim {musim}")
    st.dataframe(data_musim['FF_X'].describe())

    st.header("📈 Rata-rata Kecepatan Angin per Tahun")
    df['tahun'] = df.index.year
    rata_tahunan = df.groupby('tahun')['FF_X'].mean()
    st.bar_chart(rata_tahunan)

    st.header("📌 Hasil dan Evaluasi Model")
    st.markdown("Berikut hasil evaluasi dari masing-masing model berdasarkan data musiman:")

    df_evaluasi = pd.read_csv("data/evaluasi_model_musiman.csv")  # berisi: model, musim, MAE, RMSE, R2, MAPE
    st.dataframe(df_evaluasi)

    st.subheader("🎯 Visualisasi Aktual vs Prediksi")
    prediksi = pd.read_csv("data/prediksi_vs_aktual.csv")  # kolom: TANGGAL, FF_X, FF_X_pred, musim
    musim_pred = st.selectbox("Pilih Musim untuk Visualisasi", prediksi['musim'].unique())
    subset = prediksi[prediksi['musim'] == musim_pred]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(pd.to_datetime(subset['TANGGAL']), subset['FF_X'], label='Aktual', color='blue')
    ax.plot(pd.to_datetime(subset['TANGGAL']), subset['FF_X_pred'], label='Prediksi', color='red')
    ax.set_title(f'Prediksi vs Aktual Kecepatan Angin ({musim_pred})')
    ax.set_xlabel("Tanggal")
    ax.set_ylabel("Kecepatan Angin (m/s)")
    ax.legend()
    st.pyplot(fig)

    st.caption("Model yang digunakan: LSTM, TCN, dan RBNFF dengan preprocessing musiman dan tuning hyperparameter menggunakan Optuna.")

if __name__ == '__main__':
    app()
