import os

model_name = "model_Connaught_Place_Delhi_to_India_Gate_Delhi.pt"
scaler_name = "scaler_Connaught_Place_Delhi_to_India_Gate_Delhi.save"

print("Model exists:", os.path.exists(f"../models/{model_name}"))
print("Scaler exists:", os.path.exists(f"../models/{scaler_name}"))
