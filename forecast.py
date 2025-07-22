import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from keras.models import load_model as keras_model
from sklearn.preprocessing import MinMaxScaler

# Function to load data
def load_data(file_path, index_col=None):
    df = pd.read_csv(file_path, index_col=index_col)
    return df

# Function to convert series to supervised learning format
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if isinstance(data, list) else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [(f'var{j+1}(t-{i})') for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [(f'var{j+1}(t)') for j in range(n_vars)]
        else:
            names += [(f'var{j+1}(t+{i})') for j in range(n_vars)]
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if dropnan:
        agg.dropna(inplace=True)
    return agg

# Function to plot forecast vs actual
def plot_forecast(df, forecast_df):
    for column in df.columns:
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
        fig.add_trace(go.Scatter(x=df.index, y=df[column], mode='lines', name='Historikal', line=dict(color='cyan')), row=1, col=1)
        fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df[column], mode='lines', name='Prakiraan', line=dict(color='magenta')), row=1, col=1)

        if not df.empty and not forecast_df.empty:
            fig.add_trace(go.Scatter(x=[df.index[-1], forecast_df.index[0]],
                                     y=[df[column].iloc[-1], forecast_df[column].iloc[0]],
                                     mode='lines', line=dict(color='magenta'), showlegend=False), row=1, col=1)

        fig.update_layout(
            title=f'Historikal vs Prakiraan {column}',
            xaxis_title='Tanggal',
            yaxis_title=column,
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        st.plotly_chart(fig, use_container_width=True)

# Streamlit App
def app():
    st.title('Prediksi Kecepatan Angin Musiman BMKG Juanda')
    st.subheader('Menggunakan Model Long Short Term Memory (LSTM)')

    # Load data historis kecepatan angin
    filepath = 'data/df_forecast_angin.csv'
    df = load_data(filepath)
    df['tanggal'] = pd.to_datetime(df['tanggal'])
    df.set_index('tanggal', inplace=True)

    # Load model
    model = keras_model('model/lstm_angin.h5')

    # Tampilkan data awal
    st.write("Data Historis Kecepatan Angin:")
    st.dataframe(df, use_container_width=True)

    # Jumlah hari yang ingin diprediksi
    n_forecast_days = st.number_input('Jumlah hari yang ingin diprediksi', min_value=1, max_value=30, value=14)

    if st.button('Prediksi Kecepatan Angin'):
        with st.spinner('Melakukan prediksi...'):
            scaler = MinMaxScaler()
            df_scaled = scaler.fit_transform(df)
            n_days = 6  # lag
            n_features = df.shape[1]
            test_data_supervised = series_to_supervised(df_scaled, n_days, 1)
            test_data_sequences = test_data_supervised.values[:, :n_days * n_features]

            forecast = []
            for i in range(n_forecast_days):
                seq = test_data_sequences[i].reshape((1, n_days, n_features))
                predicted = model.predict(seq)
                forecast.append(predicted[0])

            forecast_array = np.array(forecast)
            forecast_inverse = scaler.inverse_transform(forecast_array)
            forecast_inverse = np.abs(forecast_inverse)
            forecast_inverse = np.round(forecast_inverse, 2)

            date_range = pd.date_range(start=df.index[-1], periods=n_forecast_days + 1)
            forecast_df = pd.DataFrame(forecast_inverse, index=date_range[1:], columns=df.columns)

            st.subheader("Data Prakiraan Kecepatan Angin")
            plot_forecast(df, forecast_df)
            st.dataframe(forecast_df, use_container_width=True)

if __name__ == "__main__":
    app()
