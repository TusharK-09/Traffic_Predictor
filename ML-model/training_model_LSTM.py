import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import joblib

# Setting device
device = torch.device("cpu")

# Paths
CLEANED_PATH = "../data/processed/cleaned_traffic_data.csv"
MODELS_DIR = "../models"
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
df = pd.read_csv(CLEANED_PATH)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.dropna(subset=['traffic_duration_mins'])
df = df.sort_values(by="timestamp")

# Feature engineering
df['hour'] = df['timestamp'].dt.hour
df['day_of_week'] = df['timestamp'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
df['is_peak_hour'] = df['hour'].apply(lambda x: 1 if 8 <= x <= 10 or 17 <= x <= 20 else 0)

# LSTM model definition
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

# Function to create sequences
def create_sequences(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 3):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size + 2][0])
    return np.array(X), np.array(y)

# Train a model per route
WINDOW_SIZE = 30
EPOCHS = 100
hidden_size = 64
input_size = 8

route_groups = df.groupby(['origin', 'destination'])
for (origin, destination), group in route_groups:
    # 🔧 Use full cleaned names
    route_id = f"{origin}_to_{destination}".replace(" ", "_").replace(",", "")
    print(f"\n🔄 Training for route: {origin} -> {destination} [{route_id}]")

    route_df = group.copy()
    route_df['lag_1'] = route_df['traffic_duration_mins'].shift(1)
    route_df['lag_2'] = route_df['traffic_duration_mins'].shift(2)
    route_df['lag_3'] = route_df['traffic_duration_mins'].shift(3)
    route_df['traffic_duration_mins'] = route_df['traffic_duration_mins'].rolling(window=3).mean()
    route_df.dropna(inplace=True)

    features = route_df[['traffic_duration_mins', 'hour', 'day_of_week', 'is_weekend', 'is_peak_hour', 'lag_1', 'lag_2', 'lag_3']].values.astype(float)
    
    if len(features) < WINDOW_SIZE + 3:
        print(f"❌ Skipping {route_id} (Not enough data)")
        continue

    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(features)
    joblib.dump(scaler, f"{MODELS_DIR}/scaler_{route_id}.save")

    X_all, y_all = create_sequences(data_scaled, WINDOW_SIZE)
    X_train = torch.tensor(X_all, dtype=torch.float32).to(device)
    y_train = torch.tensor(y_all, dtype=torch.float32).unsqueeze(1).to(device)

    model = LSTMModel(input_size=input_size, hidden_size=hidden_size).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.5)

    for epoch in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), f"{MODELS_DIR}/model_{route_id}.pt")
    print(f"✅ Model saved for {origin} → {destination}")
