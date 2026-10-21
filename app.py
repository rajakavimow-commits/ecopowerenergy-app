import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="EcoPower: Energy & Cost Predictor", page_icon="⚡", layout="centered")

st.title("⚡ Household Energy Consumption & Cost Predictor")
st.markdown("Predict your daily electricity consumption ($\text{kWh}$) and monthly bill before the billing cycle ends.")

# --- 1. Train Regression Model using Real CSV Dataset ---
@st.cache_resource
def train_model():
    # Load your CSV file from your GitHub repository folder
    df = pd.read_csv('smart_home_energy.csv')
    
    # Define your feature columns (X) and target column (y) matching your CSV headers
    X_train = df[['occupants', 'ac_hours', 'temperature', 'washing_cycles']]
    y_train = df['daily_kwh']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

model = train_model()

# --- 2. User Input Sliders ---
...
