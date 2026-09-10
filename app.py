
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model and scaler
model = joblib.load('risk_classification_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoder = joblib.load('label_encoder.pkl') # Assuming label_encoder is also saved

st.set_page_config(page_title="Supply Chain Risk Classification", layout="centered")

st.title("Supply Chain Risk Classification Predictor")
st.markdown("Enter the features below to predict the risk classification for your supply chain.")

# Define input fields for each feature
# These should match the columns in X used for training, excluding 'timestamp' and 'risk_classification'
feature_columns = ['vehicle_gps_latitude', 'vehicle_gps_longitude', 'fuel_consumption_rate', 'eta_variation_hours', 'traffic_congestion_level', 'warehouse_inventory_level', 'loading_unloading_time', 'handling_equipment_availability', 'order_fulfillment_status', 'weather_condition_severity', 'port_congestion_level', 'shipping_costs', 'supplier_reliability_score', 'lead_time_days', 'historical_demand', 'iot_temperature', 'cargo_condition_status', 'route_risk_level', 'customs_clearance_time', 'driver_behavior_score', 'fatigue_monitoring_score', 'disruption_likelihood_score', 'delay_probability', 'delivery_time_deviation']

input_data = {}
st.subheader("Input Features")
# Arrange input fields in columns for better UI
cols = st.columns(3)
for i, col_name in enumerate(feature_columns):
    with cols[i % 3]: # Distribute inputs into 3 columns
        # Use number_input for numerical features, with default values if possible
        input_data[col_name] = st.number_input(f"{col_name.replace('_', ' ').title()}:",
                                               value=0.0, format="%.6f", key=col_name)

if st.button("Predict Risk Classification"):
    # Convert input data to DataFrame, ensuring column order
    input_df = pd.DataFrame([input_data], columns=feature_columns)

    # Scale the input data
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction_encoded = model.predict(input_scaled)
    predicted_class = label_encoder.inverse_transform(prediction_encoded)[0]

    st.subheader("Prediction Result:")
    st.success(f"The predicted Risk Classification is: **{predicted_class}**")
    st.balloons()
