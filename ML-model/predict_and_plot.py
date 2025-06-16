import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set device
device = torch.device("cpu")

# ========== ROUTES TO PREDICT ========== #
routes = [
    ("Connaught Place, Delhi", "India Gate, Delhi"),
    ("Rajiv Chowk Metro Station, Delhi", "Hauz Khas, Delhi"),
    ("Noida Sector 18", "South Extension Market, Delhi")
]

# ========== PATHS ========== #
CLEANED_PATH = "../data/processed/cleaned_traffic_data.csv"
WINDOW_SIZE = 30

# ========== MODEL CLASS ========== #
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
        return self.fc(out[:, -1, :])

# ========== LOOP THROUGH EACH ROUTE ========== #
for origin, destination in routes:
    route_name = origin.replace(",", "").replace(" ", "_") + "_TO_" + destination.replace(",", "").replace(" ", "_")
    MODEL_PATH = f"../models/model_{route_name}.pt"
    SCALER_PATH = f"../models/scaler_{route_name}.save"

    # Skip if files not found
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        print(f"❌ Model or scaler missing for: {origin} → {destination}")
        continue

    # Load data
    df = pd.read_csv(CLEANED_PATH)
    df = df[(df['origin'] == origin) & (df['destination'] == destination)].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by="timestamp")
    df = df.dropna(subset=['traffic_duration_mins'])

    # Feature engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 8 <= x <= 10 or 17 <= x <= 20 else 0)

    df['lag_1'] = df['traffic_duration_mins'].shift(1)
    df['lag_2'] = df['traffic_duration_mins'].shift(2)
    df['lag_3'] = df['traffic_duration_mins'].shift(3)

    df['traffic_duration_mins'] = df['traffic_duration_mins'].rolling(window=3).mean()
    df.dropna(inplace=True)

    # Prepare features
    features = df[['traffic_duration_mins', 'hour', 'day_of_week', 'is_weekend',
                   'is_peak_hour', 'lag_1', 'lag_2', 'lag_3']].values.astype(float)

    # Scale
    scaler = joblib.load(SCALER_PATH)
    scaled_data = scaler.transform(features)

    # Create sequences
    def create_sequences(data, window_size):
        X, y = [], []
        for i in range(len(data) - window_size - 3):
            X.append(data[i:i + window_size])
            y.append(data[i + window_size + 2][0])  # 30 min ahead
        return np.array(X), np.array(y)

    X, y = create_sequences(scaled_data, WINDOW_SIZE)

    if len(X) == 0:
        print(f"⚠️ Not enough data for: {origin} → {destination}")
        continue

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1).to(device)

    # Load model
    model = LSTMModel(input_size=8, hidden_size=64).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    # Predict
    with torch.no_grad():
        pred = model(X_tensor).cpu().numpy()
        actual = y_tensor.cpu().numpy()

    # Inverse transform
    dummy_pred = np.zeros((len(pred), 8))
    dummy_pred[:, 0] = pred[:, 0]
    pred_real = scaler.inverse_transform(dummy_pred)[:, 0]

    dummy_actual = np.zeros((len(actual), 8))
    dummy_actual[:, 0] = actual[:, 0]
    actual_real = scaler.inverse_transform(dummy_actual)[:, 0]

    # Metrics
    rmse = np.sqrt(mean_squared_error(actual_real, pred_real))
    mae = mean_absolute_error(actual_real, pred_real)
    r2 = r2_score(actual_real, pred_real)

    print(f"\n📍 Route: {origin} → {destination}")
    print(f"✅ RMSE: {rmse:.2f} min")
    print(f"✅ MAE: {mae:.2f} min")
    print(f"✅ R² Score: {r2:.4f}")
    print(f"📈 Predicted traffic 30 min ahead: {pred_real[-1]:.2f} minutes")

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(actual_real, label="Actual")
    plt.plot(pred_real, label="Predicted")
    plt.title(f"{origin} → {destination}")
    plt.xlabel("Samples")
    plt.ylabel("Traffic Duration (mins)")
    plt.legend()
    plt.tight_layout()
    plt.show()
