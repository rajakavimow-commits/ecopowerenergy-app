import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="EcoPower: Energy & Cost Predictor", page_icon="⚡", layout="centered")

st.title("⚡ Household Energy Consumption & Cost Predictor")
st.markdown("Predict your daily electricity consumption ($\text{kWh}$) and monthly bill before the billing cycle ends.")

# --- 1. Train Regression Model ---
@st.cache_resource
def train_model():
    # Training dataset (Simulating smart meter patterns)
    np.random.seed(42)
    X_train = np.array([
        [2, 4, 28, 1], [4, 12, 35, 2], [1, 2, 26, 0], [5, 16, 38, 3],
        [3, 8, 30, 1], [4, 10, 32, 2], [2, 6, 29, 1], [6, 18, 40, 3]
    ])
    y_train = np.array([8.5, 22.0, 5.0, 31.5, 14.0, 18.5, 10.0, 36.0])
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()

# --- 2. User Input Sliders ---
st.header("🏠 Enter Household Parameters")

occupants = st.slider("Number of Occupants in Home", 1, 10, 3)
ac_hours = st.slider("Daily Air Conditioner Run Time (Hours)", 0, 24, 6)
temp = st.slider("Average Outdoor / Room Temperature (°C)", 20, 45, 32)
washing_cycles = st.slider("Weekly Washing Machine Cycles", 0, 10, 2) / 7.0  # Daily average

# --- 3. Prediction Logic ---
if st.button("Predict Consumption & Monthly Cost"):
    user_input = np.array([[occupants, ac_hours, temp, washing_cycles]])
    
    predicted_daily_kwh = model.predict(user_input)[0]
    predicted_monthly_kwh = predicted_daily_kwh * 30
    
    # Cost calculation (₹8 per kWh tariff rate)
    tariff_rate = 8.0
    projected_cost = predicted_monthly_kwh * tariff_rate
    
    # Display Results
    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Estimated Daily Usage", value=f"{predicted_daily_kwh:.2f} kWh")
    with col2:
        st.metric(label="Projected Monthly Bill", value=f"₹ {projected_cost:,.2f}")
        
    # Tariff slab threshold alert
    if predicted_monthly_kwh > 400:
        st.warning("⚠️ **Tariff Alert:** Your projected monthly consumption exceeds 400 kWh, placing you in a higher utility tariff bracket!")
    else:
        st.success("✅ **Status:** Your energy consumption is within normal efficiency limits.")
