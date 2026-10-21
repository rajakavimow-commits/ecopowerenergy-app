import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="EcoPower: Energy & Cost Predictor", page_icon="⚡", layout="centered")

st.title("⚡ Household Energy Consumption & Cost Predictor")
st.markdown("Predict your monthly electricity bill based on appliance counts and usage time.")

# --- 1. Load Data and Train Regression Model ---
@st.cache_resource
def train_model():
    # Load your uploaded CSV file
    df = pd.read_csv('electricity_bill_dataset.csv')
    
    # Select feature variables and target column based on your CSV headers
    features = ['Fan', 'Refrigerator', 'AirConditioner', 'Television', 'Monitor', 'MonthlyHours', 'TariffRate']
    target = 'ElectricityBill'
    
    df = df.dropna(subset=features + [target])
    
    X_train = df[features]
    y_train = df[target]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()

# --- 2. User Input Controls ---
st.header("🏠 Enter Appliance & Time Parameters")

col1, col2 = st.columns(2)
with col1:
    fan = st.slider("Number of Fans", 5, 23, 14)
    refrigerator = st.slider("Refrigerator Units / Rating", 17.0, 23.0, 22.0, 1.0)
    air_conditioner = st.slider("Air Conditioners", 0.0, 3.0, 2.0, 1.0)
    television = st.slider("Televisions", 1.0, 22.0, 12.0, 1.0)
with col2:
    monitor = st.slider("Monitors", 1.0, 12.0, 3.0, 1.0)
    
    # NEW: Time used control (Daily hours converted to monthly hours)
    daily_usage_hours = st.slider("Average Daily Usage Time (Hours/Day)", 1, 24, 17)
    monthly_hours = daily_usage_hours * 30  # Converts daily run time into monthly hours
    
    tariff_rate = st.slider("Tariff Rate (₹/kWh)", 7.0, 9.5, 8.4, 0.1)

# --- 3. Prediction Logic ---
if st.button("Predict Electricity Bill"):
    user_input = np.array([[fan, refrigerator, air_conditioner, television, monitor, monthly_hours, tariff_rate]])
    
    predicted_bill = model.predict(user_input)[0]
    
    # Display Results
    st.subheader("📊 Prediction Results")
    st.metric(label="Projected Monthly Electricity Bill", value=f"₹ {max(0, predicted_bill):,.2f}")
        
    if predicted_bill > 5000:
        st.warning("⚠️ **Tariff Alert:** Your projected monthly electricity bill is high due to extended usage time!")
    else:
        st.success("✅ **Status:** Your estimated bill is within efficient limits.")
