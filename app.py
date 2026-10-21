import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Page Configuration
st.set_page_config(page_title="EcoPower: Energy & Cost Predictor", page_icon="⚡", layout="centered")

st.title("⚡ Household Energy Consumption & Cost Predictor")
st.markdown("Predict smart home energy consumption ($\text{kWh}$) using your uploaded dataset.")

# --- 1. Load Data and Train Regression Model ---
@st.cache_resource
def train_model():
    # Load your uploaded CSV file
    df = pd.read_csv('smart_home_energy.csv')
    
    # Drop rows with missing Energy values
    df = df.dropna(subset=['Energy_kWh'])
    
    # Convert categorical variables (Node_ID and Appliance_Type) into numeric indicators for training
    df_encoded = pd.get_dummies(df[['Node_ID', 'Appliance_Type']], drop_first=True)
    
    # Feature engineering: extract hour from Timestamp
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df_encoded['Hour'] = df['Timestamp'].dt.hour
    
    X_train = df_encoded
    y_train = df['Energy_kWh']
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Keep track of columns for prediction alignment
    return model, X_train.columns

model, train_columns = train_model()

# --- 2. User Input Controls ---
st.header("🏠 Select Smart Home Parameters")

node_id = st.selectbox("Select Node ID", ["Node_01_LivingRoom", "Node_02_Kitchen"])
appliance_type = st.selectbox("Select Appliance Type", ["HVAC_System", "Refrigerator"])
hour = st.slider("Hour of the Day (0-23)", 0, 23, 12)

# --- 3. Prediction Logic ---
if st.button("Predict Energy Consumption"):
    # Create input dataframe matching training columns
    input_data = pd.DataFrame(0, index=[0], columns=train_columns)
    
    # Set numeric hour
    if 'Hour' in input_data.columns:
        input_data['Hour'] = hour
        
    # Set one-hot encoded categorical columns if they exist
    node_col = f"Node_ID_{node_id}"
    app_col = f"Appliance_Type_{appliance_type}"
    
    if node_col in input_data.columns:
        input_data[node_col] = 1
    if app_col in input_data.columns:
        input_data[app_col] = 1
        
    predicted_kwh = model.predict(input_data)[0]
    predicted_monthly_kwh = predicted_kwh * 30 * 24  # Scaling estimate
    
    # Cost calculation (₹8 per kWh tariff rate)
    tariff_rate = 8.0
    projected_cost = max(0, predicted_kwh) * tariff_rate * 24 * 30
    
    # Display Results
    st.subheader("📊 Prediction Results")
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric(label="Estimated Hourly Usage", value=f"{max(0, predicted_kwh):.2f} kWh")
    with col2:
        st.metric(label="Projected Monthly Cost", value=f"₹ {projected_cost:,.2f}")
        
    if projected_cost > 3000:
        st.warning("⚠️ **Tariff Alert:** Projected energy cost is on the higher side for this appliance configuration!")
    else:
        st.success("✅ **Status:** Energy consumption within optimal parameters.")
