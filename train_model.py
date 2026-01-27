import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import joblib


np.random.seed(42)

# ============================
# 1. GENERATE PAST 30 DAYS DATA
# ============================
days_past = 30
dates_past = pd.date_range(start="2025-01-01", periods=days_past, freq="D")

df = pd.DataFrame({
    "date": dates_past,
    "patients": np.random.poisson(lam=80, size=days_past),
    "icu_cases": np.random.poisson(lam=10, size=days_past),
    "season_index": np.sin(np.arange(days_past)/5)*3,
    "temp": np.random.normal(30, 3, days_past),
})

# === TARGET VARIABLES ===
df["med_A"] = df["patients"]*0.5 + df["icu_cases"]*0.2 + np.random.normal(0,2,days_past)
df["med_B"] = df["patients"]*0.3 + df["season_index"]*2 + np.random.normal(0,2,days_past)
df["med_C"] = df["patients"]*0.4 + df["icu_cases"]*0.1 + df["temp"]*0.1 + np.random.normal(0,2,days_past)
df["oxygen"] = df["icu_cases"]*3.5 + df["patients"]*0.1 + np.random.normal(0,2,days_past)
df["syringes"] = df["patients"]*0.7 + df["icu_cases"]*1.2 + np.random.normal(0,3,days_past)

# ============================
# 2. TRAIN MODEL
# ============================
features = ["patients", "icu_cases", "season_index", "temp"]
targets = ["med_A", "med_B", "med_C", "oxygen", "syringes"]

X = df[features]
Y = df[targets]

model = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, random_state=42))
model.fit(X, Y)

# ============================
# 3. GENERATE NEXT 7 DAYS INPUTS
# ============================
days_future = 7
dates_future = pd.date_range(start=df["date"].iloc[-1] + pd.Timedelta(days=1), periods=days_future, freq="D")

future = pd.DataFrame({
    "date": dates_future,
    "patients": np.random.poisson(lam=80, size=days_future),
    "icu_cases": np.random.poisson(lam=10, size=days_future),
    "season_index": np.sin(np.arange(days_past, days_past+days_future)/5)*3,
    "temp": np.random.normal(30, 3, days_future),
})

# ============================
# 4. PREDICT NEXT 7 DAYS
# ============================
future_preds = model.predict(future[features])
future_pred_df = pd.DataFrame(future_preds, columns=targets)

# ============================
# 5. SUM WEEKLY DEMAND
# ============================
weekly_totals = future_pred_df.sum(axis=0)

print("\nPredicted TOTAL resource demand for next 7 days:")
print(weekly_totals.to_string())

joblib.dump(model, "hospital_forecast.pkl")