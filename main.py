# Bike Rental Demand Predictor
# Predict how many bikes are rented in one hour in Seoul.
# Compares two models: Gradient Boosting vs Neural Network.
# Run:  python main.py

import os
import kagglehub
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


# ---- 1. Load data ----
path = kagglehub.dataset_download("saurabhshahane/seoul-bike-sharing-demand-prediction")
# The file has a "°" symbol that is not UTF-8, so we read it as latin-1
df = pd.read_csv(os.path.join(path, "SeoulBikeData.csv"), encoding="latin-1")
print("Rows loaded:", len(df))


# ---- 2. Clean data ----
# Short names instead of "Temperature(°C)", "Wind speed (m/s)", ...
df = df.rename(columns={
    "Date": "date",
    "Rented Bike Count": "rented_bikes",
    "Hour": "hour",
    "Temperature(°C)": "temperature",
    "Humidity(%)": "humidity",
    "Wind speed (m/s)": "wind_speed",
    "Rainfall(mm)": "rainfall",
    "Snowfall (cm)": "snowfall",
    "Seasons": "season",
    "Holiday": "holiday",
    "Functioning Day": "functioning_day",
})

# On non-functioning days the service was closed (0 bikes) -> remove those rows
df = df[df["functioning_day"] == "Yes"]

# Humidity of 0% is a sensor error -> remove those rows
df = df[df["humidity"] > 0]

# "01/12/2017" -> real date -> is it a weekend?
df["date"] = pd.to_datetime(df["date"], format="%d/%m/%Y")
df["weekend"] = df["date"].dt.dayofweek.map(lambda d: "Yes" if d >= 5 else "No")

# "No Holiday" / "Holiday" -> "No" / "Yes"
df["holiday"] = df["holiday"].map({"No Holiday": "No", "Holiday": "Yes"})

# Keep only the simple columns (dew point, solar radiation, visibility are dropped)
df = df[["hour", "temperature", "humidity", "wind_speed", "rainfall", "snowfall",
         "season", "holiday", "weekend", "rented_bikes"]]
print("Rows after cleaning:", len(df))


# ---- 3. Explore data ----
print("\nAverage bikes per hour:", round(df["rented_bikes"].mean()), "bikes")
print("\nAverage bikes by season:")
print(df.groupby("season")["rented_bikes"].mean().round())
print("\nAverage bikes: rain vs no rain")
print(df.groupby(df["rainfall"] > 0)["rented_bikes"].mean().round())

# Plots are kept in variables (fig1, fig2, ...) so the website can show them
fig1, ax1 = plt.subplots()
df.groupby("hour")["rented_bikes"].mean().plot(kind="line", marker="o", ax=ax1,
                                               title="Average bikes rented by hour of day")

fig2, ax2 = plt.subplots()
df.groupby("season")["rented_bikes"].mean().plot(kind="bar", ax=ax2, title="Average bikes rented by season")

fig3, ax3 = plt.subplots()
df.plot(kind="scatter", x="temperature", y="rented_bikes", s=3, alpha=0.3, ax=ax3,
        title="Temperature vs bikes rented")


# ---- 4. Prepare data for the models ----
X = pd.get_dummies(df.drop(columns="rented_bikes"))  # text columns -> 0/1 columns
y = df["rented_bikes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Neural networks need numbers on the same scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---- 5. Train both models ----
gb = GradientBoostingRegressor(random_state=42)
gb.fit(X_train, y_train)

nn = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=1000, early_stopping=True, random_state=42)
nn.fit(X_train, y_train)


# ---- 6. Test and compare ----
gb_error = mean_absolute_error(y_test, gb.predict(X_test))
nn_error = mean_absolute_error(y_test, nn.predict(X_test))
gb_r2 = r2_score(y_test, gb.predict(X_test))
nn_r2 = r2_score(y_test, nn.predict(X_test))

print("\n---- Results (lower error is better) ----")
print(f"Gradient Boosting: off by {gb_error:.0f} bikes on average, R2 = {gb_r2:.3f}")
print(f"Neural Network   : off by {nn_error:.0f} bikes on average, R2 = {nn_r2:.3f}")

if gb_error < nn_error:
    winner = "Gradient Boosting"
else:
    winner = "Neural Network"
print("Winner:", winner)

results = {"Gradient Boosting": round(gb_error), "Neural Network": round(nn_error)}
r2_scores = {"Gradient Boosting": round(gb_r2, 3), "Neural Network": round(nn_r2, 3)}

fig4, ax4 = plt.subplots()
pd.Series(results).plot(kind="bar", ax=ax4, color=["tab:green", "tab:orange"],
                        title="Average error in bikes (lower is better)")


# ---- 7. Show the plots (only when running main.py directly) ----
if __name__ == "__main__":
    plt.show()
