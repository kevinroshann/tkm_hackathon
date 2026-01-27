import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib

np.random.seed(42)

# ============================
# 1. GENERATE 90 DAYS SYNTHETIC DATA
# ============================

days = 90
dates = pd.date_range(start="2025-01-01", periods=days, freq="D")

weather = pd.DataFrame({
    "date": dates,
    "temp": np.random.normal(30, 4, days),           # °C
    "humidity": np.random.uniform(40, 90, days),     # %
    "rainfall": np.random.exponential(3, days),      # mm
    "season_index": np.sin(np.arange(days)/20)*2     # synthetic seasonality
})

# Create synthetic patient counts influenced by weather
# Hot + humid → more asthma/dengue → spikes
# Rain → dengue/malaria spikes
weather["patients"] = (
    50
    + weather["temp"] * 0.8
    + weather["humidity"] * 0.3
    + weather["rainfall"] * 2.5
    + weather["season_index"] * 10
    + np.random.normal(0, 10, days)
).astype(int)

# ============================
# 2. TRAIN MODEL
# ============================

features = ["temp", "humidity", "rainfall", "season_index"]
target = "patients"

X = weather[features]
Y = weather[target]

model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X, Y)

# ============================
# 3. GENERATE NEXT 7 DAYS INPUT (DUMMY FUTURE WEATHER)
# ============================

future_days = 7
future_dates = pd.date_range(start=weather["date"].iloc[-1] + pd.Timedelta(days=1),
                             periods=future_days, freq="D")

future = pd.DataFrame({
    "date": future_dates,
    "temp": np.random.normal(31, 3, future_days),
    "humidity": np.random.uniform(50, 85, future_days),
    "rainfall": np.random.exponential(3, future_days),
    "season_index": np.sin(np.arange(days, days+future_days)/20)*2
})

# ============================
# 4. PREDICT FUTURE PATIENT LOAD
# ============================

future["predicted_patients"] = model.predict(future[features]).astype(int)

# ============================
# 5. COMPUTE SPIKE % VS LAST WEEK
# ============================

last_week_mean = weather["patients"].tail(7).mean()
next_week_mean = future["predicted_patients"].mean()
spike_percent = ((next_week_mean - last_week_mean) / last_week_mean) * 100

print("=== Predicted Patient Load for Next 7 Days ===")
print(future[["date", "predicted_patients"]])

print("\n=== Comparison ===")
print(f"Last 7-day avg: {last_week_mean:.2f} patients/day")
print(f"Next 7-day avg: {next_week_mean:.2f} patients/day")

print("\n=== SPIKE ===")
if spike_percent >= 0:
    print(f"Expected increase: {spike_percent:.2f}%")
else:
    print(f"Expected decrease: {abs(spike_percent):.2f}%")

# Optionally save model
joblib.dump(model, "patient_forecast.pkl")

