# Website for the Bike Rental Demand Predictor
# Run (from this folder):  streamlit run app.py

import sys
import pandas as pd
import streamlit as st

sys.dont_write_bytecode = True  # don't create a __pycache__ folder when importing main.py

st.set_page_config(page_title="Bike Rental Demand", page_icon="🚲", layout="wide")

# Importing main.py runs the whole pipeline once (load, clean, explore, train, compare).
# After that we can use its variables: main.gb, main.nn, main.fig1, ...
with st.spinner("Training the models, please wait..."):
    import main

st.title("🚲 Bike Rental Demand Predictor")
st.caption("How many bikes will people rent in Seoul in one hour?")

# ---- Sidebar: the inputs ----
st.sidebar.header("Enter the conditions")
hour = st.sidebar.slider("Hour of day", 0, 23, 18)
temperature = st.sidebar.slider("Temperature (°C)", -15, 40, 20)
humidity = st.sidebar.slider("Humidity (%)", 10, 100, 55)
wind_speed = st.sidebar.slider("Wind speed (m/s)", 0.0, 7.0, 1.5)
rainfall = st.sidebar.slider("Rainfall (mm)", 0.0, 30.0, 0.0)
snowfall = st.sidebar.slider("Snowfall (cm)", 0.0, 9.0, 0.0)
season = st.sidebar.selectbox("Season", ["Spring", "Summer", "Autumn", "Winter"])
holiday = st.sidebar.selectbox("Holiday?", ["No", "Yes"])
weekend = st.sidebar.selectbox("Weekend?", ["No", "Yes"])

predict_tab, explore_tab, compare_tab = st.tabs(["🔮 Predict", "📊 Explore data", "🏆 Compare models"])

# ---- Tab 1: prediction ----
with predict_tab:
    st.subheader("Predicted bikes rented in this hour")
    hour_data = pd.DataFrame([{
        "hour": hour, "temperature": temperature, "humidity": humidity, "wind_speed": wind_speed,
        "rainfall": rainfall, "snowfall": snowfall, "season": season, "holiday": holiday, "weekend": weekend,
    }])
    # Same steps as in main.py: text -> 0/1 columns, then scale
    hour_data = pd.get_dummies(hour_data).reindex(columns=main.X.columns, fill_value=0)
    hour_data = main.scaler.transform(hour_data)

    # A model can guess a small negative number; there can't be fewer than 0 bikes
    nn_guess = max(0, round(main.nn.predict(hour_data)[0]))
    gb_guess = max(0, round(main.gb.predict(hour_data)[0]))

    col1, col2 = st.columns(2)
    col1.metric("Neural Network", f"{nn_guess} bikes")
    col2.metric("Gradient Boosting", f"{gb_guess} bikes")
    st.info(f"The {main.winner} was more accurate on the test data, so trust its number more.")

# ---- Tab 2: plots ----
with explore_tab:
    plots = {
        "Bikes by hour of day": main.fig1,
        "Bikes by season": main.fig2,
        "Temperature vs bikes": main.fig3,
    }
    choice = st.radio("Choose a plot", list(plots.keys()), horizontal=True)
    st.pyplot(plots[choice])

# ---- Tab 3: model comparison ----
with compare_tab:
    table = pd.DataFrame({
        "Model": list(main.results.keys()),
        "Average error (bikes)": list(main.results.values()),
        "R²": list(main.r2_scores.values()),
    })
    st.table(table)
    st.success(f"Winner: {main.winner}")
    st.pyplot(main.fig4)
