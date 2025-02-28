import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from io import StringIO

# Load datasets
df_train = pd.read_csv("DailyDelhiClimateTrain.csv")
df_test = pd.read_csv("DailyDelhiClimateTest.csv")

st.title("Delhi Climate LSTM Forecasting Dashboard")

# Creating tabs
tabs = st.tabs(["Retrain Model", "Generate Forecast", "Monitor Metrics", "Graph Actual vs Predicted"])

# Data Preprocessing
features = ['date', 'humidity', 'meanpressure', 'wind_speed', 'meantemp']
df_train = df_train[features]
df_train['date'] = pd.to_datetime(df_train['date']).apply(lambda date: date.toordinal())
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_train)
train_data = data_scaled

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length - 1):
        xs.append(data[i:(i+seq_length), :])
        ys.append(data[i+seq_length, -1])
    return np.array(xs), np.array(ys)

seq_length = 10
X_train, y_train = create_sequences(train_data, seq_length)

df_test = df_test[features]
df_test['date'] = pd.to_datetime(df_test['date']).apply(lambda date: date.toordinal())
data_scaled_test = scaler.transform(df_test)
test_data = data_scaled_test
X_test, y_test = create_sequences(test_data, seq_length)

with tabs[0]:  # Retrain Model
    st.info("Model training started.")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)
    ])
    
    string_buffer = StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + "\n"))
    st.text(string_buffer.getvalue())
    
    model.compile(optimizer="adam", loss="mse")
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=100, batch_size=16, verbose=1,
                        callbacks=[early_stopping])
    model.save('lstm_delhi.keras')
    st.success(f"Training completed in {len(history.history['loss'])} epochs! Model saved.")

with tabs[1]:  # Generate Forecast
    try:
        model = load_model('lstm_delhi.keras')
        predictions = []
        last_sequence = X_test[-1]
        
        for _ in range(3):
            input_sequence = last_sequence.reshape(1, seq_length, len(features))
            next_prediction = model.predict(input_sequence)
            predictions.append(next_prediction[0, 0])
            last_sequence = np.roll(last_sequence, -1, axis=0)
            last_sequence[-1, -1] = next_prediction[0, 0]
        
        st.subheader("Forecasted Temperatures")
        st.write(predictions)
    except Exception as e:
        st.error("Model not found. Retrain the model first.")
        st.error(e)

with tabs[2]:  # Monitor Metrics
    try:
        model = load_model('lstm_delhi.keras')
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        st.subheader("Model Metrics")
        st.write("MSE:", mse)
        st.write("R2 Score:", r2)
    except Exception as e:
        st.error("Model not found. Retrain the model first.")
        st.error(e)

with tabs[3]:  # Graph Actual vs Predicted
    try:
        model = load_model('lstm_delhi.keras')
        y_pred = model.predict(X_test)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=y_test, mode='lines', name='Actual', line=dict(color='royalblue')))
        fig.add_trace(go.Scatter(y=y_pred.flatten(), mode='lines', name='Predicted', line=dict(color='firebrick')))
        fig.update_layout(title="Actual vs. Predicted Temperature", template="plotly_white")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error("Model not found. Retrain the model first.")
        st.error(e)
