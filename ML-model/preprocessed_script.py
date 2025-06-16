import pandas as pd
import os

# Paths
RAW_PATH = "../data/raw/india_traffic.csv"
CLEANED_PATH = "../data/processed/cleaned_traffic_data.csv"

# Convert duration like "1 hour 20 mins" → total minutes
def convert_duration(duration_str):
    if pd.isna(duration_str):
        return None

    duration_str = duration_str.lower()
    mins = 0

    if "hour" in duration_str:
        parts = duration_str.split("hour")
        try:
            hours = int(parts[0].strip())
            mins += hours * 60
        except:
            pass
        if "min" in parts[1]:
            try:
                mins += int(parts[1].strip().replace("mins", "").replace("min", "").strip())
            except:
                pass
    elif "min" in duration_str:
        try:
            mins = int(duration_str.replace("mins", "").replace("min", "").strip())
        except:
            pass

    return mins if mins != 0 else None

# Convert distance like "12.5 km" → 12.5
def convert_distance(distance_str):
    if pd.isna(distance_str):
        return None
    try:
        return float(distance_str.replace("km", "").strip())
    except:
        return None

# Load full raw dataset
raw_df = pd.read_csv(RAW_PATH)

# Load cleaned dataset to determine how many rows already processed
if os.path.exists(CLEANED_PATH):
    cleaned_df = pd.read_csv(CLEANED_PATH)
    already_processed = len(cleaned_df)
else:
    already_processed = 0

# Keep only new rows
raw_df = raw_df.iloc[already_processed:]

if raw_df.empty:
    print("✅ No new rows to process.")
    exit()

# Convert timestamp
raw_df['timestamp'] = pd.to_datetime(raw_df['timestamp'], errors='coerce')

# Clean values
raw_df['duration_mins'] = raw_df['duration'].apply(convert_duration)
raw_df['traffic_duration_mins'] = raw_df['duration_in_traffic'].apply(convert_duration)
raw_df['distance_kms'] = raw_df['distance'].apply(convert_distance)

# Drop old columns
raw_df = raw_df.drop(columns=['duration', 'duration_in_traffic', 'distance'])

# Append cleaned data
os.makedirs(os.path.dirname(CLEANED_PATH), exist_ok=True)
raw_df.to_csv(CLEANED_PATH, mode='a', header=not os.path.exists(CLEANED_PATH), index=False)

print(f"✅ Appended {len(raw_df)} new rows to {CLEANED_PATH}")
