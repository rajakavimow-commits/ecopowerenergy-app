import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="EcoPower: Energy & Cost Predictor", page_icon="⚡", layout="centered")

st.title("⚡ Household Energy Consumption & Cost Predictor")
st.markdown("Predict your electricity consumption ($\text{kWh}$) and monthly bill using your real dataset.")

# --- 1. Load Data and Train Regression Model ---
@st.cache_resource
def train_model():
    # Load your Excel file directly
    df = pd.read_excel('cleandata.xlsx')
    
    # Define feature columns (X) and target column (y) based on your exact Excel columns
    X_train = df[['Outdoor Temperature', 'Occupancy']]
    y_train = df['Power Consumption']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()

# --- 2. User Input Sliders (Matched to Dataset Features) ---
st.header("🏠 Enter Household Parameters")

outdoor_temp = st.slider("Outdoor Temperature (°C)", 5.0, 35.0, 15.0)
occupancy = st.slider("Occupancy / Occupants", 0, 150, 10)

# --- 3. Prediction Logic ---
if st.button("Predict Consumption & Monthly Cost"):
    user_input = np.array([[outdoor_temp, occupancy]])
    
    predicted_power = model.predict(user_input)[0]
    predicted_monthly_kwh = predicted_power * 30  # Scaling interval to monthly estimate
    
    # Cost calculation (₹8 per kWh tariff rate)
    tariff_rate = 8.0
    projected_cost = predicted_monthly_kwh * tariff_rate
    
    # Display Results
    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Estimated Consumption", value=f"{predicted_power:.2f} kWh")
    with col2:
        st.metric(label="Projected Monthly Bill", value=f"₹ {projected_cost:,.2f}")
        
    # Tariff slab threshold alert
    if predicted_monthly_kwh > 400:
        st.warning("⚠️ **Tariff Alert:** Your projected monthly consumption exceeds 400 kWh, placing you in a higher utility tariff bracket!")
    else:
        st.success("✅ **Status:** Your energy consumption is within normal efficiency limits.")
