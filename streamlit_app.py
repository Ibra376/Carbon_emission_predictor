import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd
import matplotlib.pyplot as plt

# Load model and scaler
model = load_model("ann_emissions_model.keras")
scaler = joblib.load('scaler.pkl')

# Streamlit app title
st.title("🛳️ Ship CO₂ Emissions Predictor")
st.markdown("Enter vessel operating parameters to estimate CO₂ emissions.")

# Sidebar for inputs
st.sidebar.header("🔧 Input Parameters")

# Example inputs - you can customize these based on your actual feature set
speed = st.sidebar.slider("Speed (knots)", 5.0, 25.0, 15.0)
deadweight = st.sidebar.number_input("Deadweight (tons)", value=50000)
draft = st.sidebar.number_input("Draft (meters)", value=10.0)
engine_load = st.sidebar.slider("Engine Load (%)", 0.0, 100.0, 70.0)
sea_state = st.sidebar.slider("Sea State (Beaufort Scale)", 0, 12, 3)

# Build feature array for prediction
input_features = np.array([[speed, deadweight, draft, engine_load, sea_state]])

# Scale the inputs
scaled_features = scaler.transform(input_features)

# Prediction
if st.sidebar.button("Predict CO₂ Emissions"):
    prediction = model.predict(scaled_features)
    st.success(f"🚢 Estimated CO₂ Emissions: **{prediction[0][0]:.2f} tons**")

    # Store results for visualization
    if "history" not in st.session_state:
        st.session_state["history"] = []
    st.session_state["history"].append((speed, deadweight, draft, engine_load, sea_state, prediction[0][0]))

# Visualization: Scatter plot of previous predictions
if "history" in st.session_state and len(st.session_state["history"]) > 1:
    st.subheader("📊 Prediction History")

    df = pd.DataFrame(st.session_state["history"], columns=[
        "Speed", "Deadweight", "Draft", "Engine Load", "Sea State", "Predicted CO₂"
    ])
    
    st.dataframe(df)

    fig, ax = plt.subplots()
    scatter = ax.scatter(df["Speed"], df["Predicted CO₂"], c=df["Engine Load"], cmap='viridis')
    ax.set_xlabel("Speed (knots)")
    ax.set_ylabel("Predicted CO₂ (tons)")
    plt.colorbar(scatter, label="Engine Load (%)")
    st.pyplot(fig)
