import streamlit as st
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
from datetime import datetime

# Constants
MODEL_DIR = "../models"
DATA_PATH = "../data/processed/cleaned_traffic_data.csv"
WINDOW_SIZE = 30

# Device
device = torch.device("cpu")

# LSTM Model Class
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=2, batch_first=True, dropout=0.3, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Preprocess function
def preprocess_route(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by="timestamp")
    df = df.dropna(subset=['traffic_duration_mins'])

    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 8 <= x <= 10 or 17 <= x <= 20 else 0)

    df['lag_1'] = df['traffic_duration_mins'].shift(1)
    df['lag_2'] = df['traffic_duration_mins'].shift(2)
    df['lag_3'] = df['traffic_duration_mins'].shift(3)

    df['traffic_duration_mins'] = df['traffic_duration_mins'].rolling(window=3).mean()
    df.dropna(inplace=True)

    features = df[['traffic_duration_mins', 'hour', 'day_of_week', 'is_weekend',
                   'is_peak_hour', 'lag_1', 'lag_2', 'lag_3']].values.astype(float)
    
    return df, features

# Load routes
@st.cache_data
def load_routes():
    df = pd.read_csv(DATA_PATH)
    routes = df[['origin', 'destination']].drop_duplicates()
    route_options = routes.apply(lambda row: f"{row['origin']} → {row['destination']}", axis=1).tolist()
    return route_options, df

# Create input sequence
def create_input_sequence(data, window_size):
    if len(data) < window_size:
        padding = np.zeros((window_size - len(data), data.shape[1]))
        data = np.vstack((padding, data))
    return data[-window_size:]

# Streamlit UI
st.title("🚦 Traffic Duration Predictor (30 Minutes Ahead)")

route_options, full_df = load_routes()
selected_route = st.selectbox("Select a Route:", route_options)

# Extract origin and destination
origin, destination = selected_route.split(" → ")
route_df = full_df[(full_df["origin"] == origin) & (full_df["destination"] == destination)].copy()

# Preprocess data
route_df, features = preprocess_route(route_df)

# Get current hour
current_hour = datetime.now().hour
hour_mask = route_df['hour'] == current_hour
filtered_features = features[hour_mask.values]

# ✅ Fix: Take first 30 entries if possible
filtered_features = filtered_features[:WINDOW_SIZE]

if len(filtered_features) == 0:
    st.warning("❌ Not enough recent time-aligned data to make a prediction.")
    st.stop()

# Format route name
route_id = f"{origin}_to_{destination}".replace(" ", "_").replace(",", "")
model_path = os.path.join(MODEL_DIR, f"model_{route_id}.pt")
scaler_path = os.path.join(MODEL_DIR, f"scaler_{route_id}.save")

if not os.path.exists(model_path) or not os.path.exists(scaler_path):
    st.error(f"❌ Model or scaler not found for route: {origin} → {destination}")
    st.stop()

# Load scaler
scaler = joblib.load(scaler_path)
features_scaled = scaler.transform(filtered_features)

# Prepare input sequence
X_input = create_input_sequence(features_scaled, WINDOW_SIZE)
X_input = torch.tensor(X_input, dtype=torch.float32).unsqueeze(0).to(device)

# Load model
model = LSTMModel(input_size=8, hidden_size=64).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Predict
with torch.no_grad():
    prediction = model(X_input).item()

# Inverse transform only the target feature
pred_dummy = np.zeros((1, 8))
pred_dummy[0, 0] = prediction
pred_real = scaler.inverse_transform(pred_dummy)[0, 0]

# Output
st.success(f"🕒 Predicted Traffic Duration (30 min ahead): **{pred_real:.2f} minutes**")
