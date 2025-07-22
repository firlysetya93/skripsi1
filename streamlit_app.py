# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
import os

# Set page configuration
st.set_page_config(page_title="Time Series Forecasting App", layout="wide")

# Sidebar menu
st.sidebar.title("Navigation")
menu = st.sidebar.selectbox(
    "Choose an option:",
    ["Home", "View Data", "Visualize Data", "Train LSTM Model", "Model Performance"]
)

# Title and description
st.title("Time Series Forecasting with LSTM")
st.write("This app provides tools for analyzing time-series data and forecasting using an LSTM model.")

# Initialize session state for data and model
if 'df' not in st.session_state:
    st.session_state.df = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'train_predict' not in st.session_state:
    st.session_state.train_predict = None
if 'test_predict' not in st.session_state:
    st.session_state.test_predict = None
if 'train_rmse' not in st.session_state:
    st.session_state.train_rmse = None
if 'test_rmse' not in st.session_state:
    st.session_state.test_rmse = None

# File uploader (available on all pages)
uploaded_file = st.sidebar.file_uploader("Upload your Excel file (DATA_SKRIP.xlsx)", type=["xlsx"])

# Load data if uploaded
if uploaded_file is not None:
    st.session_state.df = pd.read_excel(uploaded_file)
    # Handle missing values
    st.session_state.df['FF_X'].fillna(method='ffill', inplace=True)

# Menu options
if menu == "Home":
    st.header("Welcome to the Time Series Forecasting App")
    st.write("""
    Use the sidebar to navigate through different functionalities:
    - **View Data**: Display the uploaded dataset.
    - **Visualize Data**: Plot the time-series data.
    - **Train LSTM Model**: Train the LSTM model on the data.
    - **Model Performance**: View model performance metrics.
    """)
    if st.session_state.df is not None:
        st.write("Data successfully uploaded. Navigate to other sections to explore.")

elif menu == "View Data":
    st.header("View Dataset")
    if st.session_state.df is not None:
        st.write("### Data Preview")
        st.dataframe(st.session_state.df.head())
        # Check for missing values
        missing_count = st.session_state.df['FF_X'].isnull().sum()
        if missing_count > 0:
            st.warning(f"Column 'FF_X' has {missing_count} missing values.")
        else:
            st.success("No missing values in 'FF_X' column.")
    else:
        st.info("Please upload an Excel file in the sidebar to view the data.")

elif menu == "Visualize Data":
    st.header("Visualize Time-Series Data")
    if st.session_state.df is not None:
        st.write("### Time-Series Plot")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(st.session_state.df['TANGGAL'], st.session_state.df['FF_X'], label='Actual Data', color='blue')
        ax.set_xlabel('Date')
        ax.set_ylabel('FF_X')
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Please upload an Excel file in the sidebar to visualize the data.")

elif menu == "Train LSTM Model":
    st.header("Train LSTM Model")
    if st.session_state.df is not None:
        st.write("Training the LSTM model...")
        # Data preprocessing
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(st.session_state.df['FF_X'].values.reshape(-1, 1))

        # Function to create dataset for LSTM
        def create_dataset(data, time_step=1):
            X, y = [], []
            for i in range(len(data) - time_step):
                X.append(data[i:(i + time_step), 0])
                y.append(data[i + time_step, 0])
            return np.array(X), np.array(y)

        time_step = 10
        X, y = create_dataset(scaled_data, time_step)
        X = X.reshape(X.shape[0], X.shape[1], 1)

        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Build LSTM model
        st.session_state.model = Sequential()
        st.session_state.model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
        st.session_state.model.add(Dropout(0.2))
        st.session_state.model.add(LSTM(50))
        st.session_state.model.add(Dropout(0.2))
        st.session_state.model.add(Dense(1))
        st.session_state.model.compile(optimizer='adam', loss='mean_squared_error')

        # Train model
        st.session_state.model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

        # Predictions
        st.session_state.train_predict = st.session_state.model.predict(X_train)
        st.session_state.test_predict = st.session_state.model.predict(X_test)

        # Inverse transform predictions
        st.session_state.train_predict = scaler.inverse_transform(st.session_state.train_predict)
        st.session_state.test_predict = scaler.inverse_transform(st.session_state.test_predict)
        y_train_inv = scaler.inverse_transform([y_train])
        y_test_inv = scaler.inverse_transform([y_test])

        # Calculate RMSE
        st.session_state.train_rmse = np.sqrt(mean_squared_error(y_train_inv[0], st.session_state.train_predict[:, 0]))
        st.session_state.test_rmse = np.sqrt(mean_squared_error(y_test_inv[0], st.session_state.test_predict[:, 0]))

        st.success("Model training completed! Navigate to 'Model Performance' to view results.")
    else:
        st.info("Please upload an Excel file in the sidebar to train the model.")

elif menu == "Model Performance":
    st.header("Model Performance")
    if st.session_state.df is not None and st.session_state.model is not None:
        st.write("### Performance Metrics")
        st.write(f"**Train RMSE**: {st.session_state.train_rmse:.2f}")
        st.write(f"**Test RMSE**: {st.session_state.test_rmse:.2f}")

        st.write("### Predictions vs Actual Data")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(st.session_state.df['TANGGAL'], st.session_state.df['FF_X'], label='Actual Data', color='blue')
        train_plot = np.empty_like(st.session_state.df['FF_X'].values.reshape(-1, 1))
        train_plot[:, :] = np.nan
        train_plot[time_step:len(st.session_state.train_predict) + time_step, :] = st.session_state.train_predict
        test_plot = np.empty_like(st.session_state.df['FF_X'].values.reshape(-1, 1))
        test_plot[:, :] = np.nan
        test_plot[len(st.session_state.train_predict) + (time_step * 2):len(st.session_state.df['FF_X']), :] = st.session_state.test_predict
        ax.plot(st.session_state.df['TANGGAL'], train_plot, label='Train Predictions', color='green')
        ax.plot(st.session_state.df['TANGGAL'], test_plot, label='Test Predictions', color='red')
        ax.set_xlabel('Date')
        ax.set_ylabel('FF_X')
        ax.legend()
        st.pyplot(fig)
    else:
        st.info("Please upload an Excel file and train the model first.")
