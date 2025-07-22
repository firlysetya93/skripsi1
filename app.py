import streamlit as st

def app():
    st.title("Tentang Aplikasi")

    st.header("Deskripsi Projek")
    st.markdown("""
        Aplikasi ini merupakan bagian dari penelitian yang bertujuan untuk memprediksi kecepatan angin di wilayah Stasiun Meteorologi BMKG Juanda.
        Penelitian ini menggunakan pendekatan machine learning dengan membandingkan performa tiga model, yaitu Long Short-Term Memory (LSTM),
        Temporal Convolutional Network (TCN), dan Recurrent-Based Neural Fuzzy Function (RBFNN). Prediksi dilakukan berdasarkan data historis kecepatan angin
        harian yang telah melalui proses preprocessing dan pembagian data per musim.
    """)

    tab1, tab2, tab3, tab4 = st.tabs(["Latar Belakang", "Rumusan Masalah", "Tujuan", "Manfaat"])

    with tab1:
        st.info("""
            Kecepatan angin yang fluktuatif berpengaruh terhadap berbagai sektor seperti penerbangan, pelayaran, dan pembangkit energi terbarukan.
            Data BMKG menunjukkan bahwa kecepatan angin ekstrem dapat membahayakan keselamatan dan infrastruktur. Oleh karena itu, diperlukan sistem
            prediksi angin yang akurat dengan mempertimbangkan faktor musiman di Indonesia, yang dipengaruhi oleh angin monsun yang berubah arah setiap enam bulan.
        """)

    with tab2:
        st.info("""
            Bagaimana membangun model prediksi kecepatan angin harian yang optimal di wilayah BMKG Juanda menggunakan pendekatan machine learning,
            serta model mana yang memberikan performa terbaik berdasarkan evaluasi akurasi seperti MAE, RMSE, dan MAPE?
        """)

    with tab3:
        st.info("""
            Penelitian ini bertujuan untuk membandingkan kinerja model LSTM, TCN, dan RBFNN dalam memprediksi kecepatan angin berdasarkan data musiman.
            Hasil prediksi akan digunakan untuk menentukan model yang paling optimal dalam mendeteksi pola fluktuasi angin guna mendukung sistem
            mitigasi cuaca ekstrem.
        """)

    with tab4:
        st.info("""
            Dengan adanya aplikasi ini, diharapkan pengguna dapat mengeksplorasi data kecepatan angin historis, memahami tren musiman, dan membandingkan performa
            model prediksi secara interaktif. Aplikasi ini dapat menjadi alat bantu bagi instansi seperti BMKG dan sektor transportasi dalam mengantisipasi
            potensi bahaya angin kencang serta mendukung pengambilan keputusan berbasis data.
        """)

    st.subheader("Evaluasi Model")
    st.markdown("""
        Model yang digunakan dalam penelitian ini dievaluasi menggunakan beberapa metrik akurasi, di antaranya:

        - **Mean Absolute Error (MAE)**
        - **Root Mean Squared Error (RMSE)**
        - **Mean Absolute Percentage Error (MAPE)**

        Berikut contoh hasil evaluasi model berdasarkan musim:

        | Model  | Musim | MAE   | RMSE  | R²    | MAPE  |
        |--------|-------|-------|-------|-------|--------|
        | LSTM   | DJF   | 1.13  | 1.53  | 0.10  | 21.83% |
        | TCN    | DJF   | 1.21  | 1.61  | 0.12  | 22.50% |
        | RBFNN  | DJF   | 1.35  | 1.74  | 0.08  | 24.10% |
    """)

    st.subheader("Dataset yang Digunakan")
    st.markdown("""
        - **Data Kecepatan Angin Harian** dari Stasiun Meteorologi Juanda (sumber: BMKG).
        - Periode data: **Januari 2014 hingga Desember 2023**.
        - Data telah dibersihkan, dianalisis per musim (DJF, MAM, JJA, SON), dan digunakan sebagai input model.
    """)

    st.subheader("Kontak")
    st.markdown("""
        Untuk informasi lebih lanjut mengenai aplikasi dan pengembangan model prediksi cuaca, silakan hubungi pihak pengembang atau pembimbing akademik.
    """)
