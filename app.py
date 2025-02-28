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

st.header("Train Data Preview")
st.write(df_train.head())

# Selecting and processing features
features = ['date', 'humidity', 'meanpressure', 'wind_speed', 'meantemp']
df_train = df_train[features]
df_train['date'] = pd.to_datetime(df_train['date'])
df_train['date'] = df_train['date'].apply(lambda date: date.toordinal())

# Scale train data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df_train)
train_data = data_scaled

# Function to create sequences for LSTM
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length - 1):
        x = data[i:(i+seq_length), :]
        y = data[i+seq_length, -1]  # predicting temperature
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10  # Adjust sequence length as needed
X_train, y_train = create_sequences(train_data, seq_length)

# Process test dataset
df_test = df_test[features]
df_test['date'] = pd.to_datetime(df_test['date'])
df_test['date'] = df_test['date'].apply(lambda date: date.toordinal())
data_scaled_test = scaler.transform(df_test)
test_data = data_scaled_test

X_test, y_test = create_sequences(test_data, seq_length)

# Function to train/retrain the model
def train_model(X_train, y_train, X_test, y_test):
    st.info("Model training/retraining started.")
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25, activation="relu"),
        Dense(1)  # Output layer for temperature prediction
    ])
    
    # Capture model summary in a string and display it
    string_buffer = StringIO()
    model.summary(print_fn=lambda x: string_buffer.write(x + "\n"))
    st.text(string_buffer.getvalue())
    
    model.compile(optimizer="adam", loss="mse")
    early_stopping = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=100, batch_size=16, verbose=1,
                        callbacks=[early_stopping])
    
    st.success(f"Training completed in {len(history.history['loss'])} epochs!")
    return model

# Function to generate forecast
def generate_forecast(model, X_test, y_test, forecast_window=3, target='meantemp', seq_length=10):
    predictions = []
    last_sequence = X_test[-1]
    
    for _ in range(forecast_window):
        input_sequence = last_sequence.reshape(1, seq_length, len(features))
        next_prediction = model.predict(input_sequence)
        predictions.append(next_prediction[0, 0])
        # update last_sequence with new prediction for iterative forecasting
        last_sequence = np.roll(last_sequence, -1, axis=0)
        # update the 'meantemp' column in the last timestep with the new prediction
        last_sequence[-1, df_train.columns.get_loc(target)] = next_prediction[0, 0]
    
    # Create dummy array to inverse transform only the target column
    dummy_array = np.zeros((len(predictions), len(df_train.columns)))
    dummy_array[:, df_train.columns.get_loc(target)] = predictions
    predictions_inv = scaler.inverse_transform(dummy_array)[:, df_train.columns.get_loc(target)]
    
    # Prepare actual values (last few values of test set)
    y_test_inv = scaler.inverse_transform(
        np.concatenate((np.zeros((len(y_test), len(df_test.columns)-1)), np.expand_dims(y_test, axis=1)), axis=1)
    )[:,-1]
    
    # Create a table showing forecasted vs actual (for the last forecast_window points)
    df_pred = pd.DataFrame(predictions_inv, columns=['Forecasted'])
    df_actual = pd.DataFrame(y_test_inv[-forecast_window:], columns=['Actual'])
    df_table = pd.concat([df_actual.reset_index(drop=True), df_pred.reset_index(drop=True)], axis=1)
    
    st.subheader("Forecast vs Actual")
    st.dataframe(df_table)
    return predictions_inv

# Function to compute and display metrics
def monitor_metrics(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(
        np.concatenate((np.zeros((len(y_pred), len(df_test.columns)-1)), y_pred), axis=1)
    )[:,-1]
    
    y_test_inv = scaler.inverse_transform(
        np.concatenate((np.zeros((len(y_test), len(df_test.columns)-1)), np.expand_dims(y_test, axis=1)), axis=1)
    )[:,-1]
    
    mse = mean_squared_error(y_test_inv, y_pred_inv)
    r2 = r2_score(y_test_inv, y_pred_inv)
    
    st.subheader("Model Metrics")
    st.write("Mean Squared Error (MSE):", mse)
    st.write("R-squared:", r2)
    return y_pred_inv

# Function to plot Actual vs Predicted using Plotly
def plot_actual_vs_predicted(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_inv = scaler.inverse_transform(
        np.concatenate((np.zeros((len(y_pred), len(df_test.columns)-1)), y_pred), axis=1)
    )[:,-1]
    
    y_test_inv = scaler.inverse_transform(
        np.concatenate((np.zeros((len(y_test), len(df_test.columns)-1)), np.expand_dims(y_test, axis=1)), axis=1)
    )[:,-1]
    
    # Create Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(range(len(y_test_inv))), y=y_test_inv,
                             mode='lines', name='Actual', line=dict(color='royalblue')))
    fig.add_trace(go.Scatter(x=list(range(len(y_pred_inv))), y=y_pred_inv,
                             mode='lines', name='Predicted', line=dict(color='firebrick')))
    
    fig.update_layout(title="Actual vs. Predicted Temperature",
                      xaxis_title="Time",
                      yaxis_title="Mean Temperature",
                      legend_title="Legend",
                      template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

# Layout for interactive dashboard buttons
st.sidebar.title("Actions")
action = st.sidebar.radio("Choose an action:",
                          ("Retrain Model", "Generate Forecast", "Monitor Metrics", "Graph Actual vs Predicted"))

# Execute the chosen action
if action == "Retrain Model":
    model = train_model(X_train, y_train, X_test, y_test)
    model.save('lstm_delhi.keras')
    st.success("Model saved successfully as 'lstm_delhi.keras'.")

elif action == "Generate Forecast":
    try:
        model = load_model('lstm_delhi.keras')
        forecast = generate_forecast(model, X_test, y_test, forecast_window=3, target='meantemp', seq_length=10)
    except Exception as e:
        st.error("Error loading model. Please retrain the model first.")
        st.error(e)

elif action == "Monitor Metrics":
    try:
        model = load_model('lstm_delhi.keras')
        monitor_metrics(model, X_test, y_test)
    except Exception as e:
        st.error("Error loading model. Please retrain the model first.")
        st.error(e)

elif action == "Graph Actual vs Predicted":
    try:
        model = load_model('lstm_delhi.keras')
        plot_actual_vs_predicted(model, X_test, y_test)
    except Exception as e:
        st.error("Error loading model. Please retrain the model first.")
        st.error(e)
