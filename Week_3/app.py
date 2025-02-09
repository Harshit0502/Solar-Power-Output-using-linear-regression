import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model, scaler, PolynomialFeatures transformer, and feature names
try:
    model = joblib.load("solar_power_model.pkl")
    scaler = joblib.load("scaler.pkl")
    poly = joblib.load("poly_transform.pkl")
    feature_names = joblib.load("feature_names.pkl")  # All 22 features for the model
    poly_features_names = joblib.load("poly_features_names.pkl")  # 16 features for PolynomialFeatures
except FileNotFoundError as e:
    st.error(f"Missing model or preprocessing file: {e}")
    st.stop()

# Streamlit UI
st.title("Solar Power Prediction App")
st.write("Enter the required parameters to predict solar power generation.")

# User input fields for the 16 key features used in PolynomialFeatures
input_data = {}

def get_input(label, min_val, max_val, default):
    return st.number_input(label, min_value=min_val, max_value=max_val, value=default)

input_data["temperature_2_m_above_gnd"] = get_input("Temperature (°C)", -10.0, 50.0, 25.0)
input_data["relative_humidity_2_m_above_gnd"] = get_input("Relative Humidity (%)", 0.0, 100.0, 50.0)
input_data["mean_sea_level_pressure_MSL"] = get_input("Mean Sea Level Pressure (hPa)", 900.0, 1100.0, 1013.0)
input_data["wind_speed_10_m_above_gnd"] = get_input("Wind Speed (m/s)", 0.0, 20.0, 5.0)
input_data["total_cloud_cover_sfc"] = get_input("Cloud Cover (%)", 0.0, 100.0, 30.0)
input_data["zenith"] = get_input("Zenith Angle (°)", 0.0, 90.0, 45.0)
input_data["azimuth"] = get_input("Azimuth Angle (°)", 0.0, 360.0, 180.0)
input_data["angle_of_incidence"] = get_input("Angle of Incidence (°)", 0.0, 90.0, 45.0)

# Calculate derived feature 'wind_power_potential'
input_data["wind_power_potential"] = input_data["wind_speed_10_m_above_gnd"] * np.cos(np.radians(input_data["angle_of_incidence"]))

# Set remaining features to default values if missing
for feature in feature_names:
    if feature not in input_data:
        input_data[feature] = 0  # Set missing features to zero instead of None

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Remove target column if it exists in feature names
if "generated_power_kw" in feature_names:
    feature_names.remove("generated_power_kw")

# Reorder columns to match training dataset
try:
    input_df = input_df[feature_names]
except KeyError as e:
    st.error(f"Feature mismatch: {e}")
    st.stop()

# Select Only PolynomialFeatures-Specific Features
try:
    poly_input_df = input_df[poly_features_names]
except KeyError as e:
    st.error(f"Missing polynomial feature columns: {e}")
    st.stop()

# Apply preprocessing
try:
    poly_input_scaled = scaler.transform(poly_input_df)
    input_data_poly = poly.transform(poly_input_scaled)
    
    # Debugging
    st.write("Input Data Shape:", input_data_poly.shape)
    
    if st.button("Predict Solar Power"):
        if input_data_poly.shape[0] == 0:
            st.error("Error: No valid input data for prediction.")
        else:
            prediction = model.predict(input_data_poly)
            st.success(f"Predicted Solar Power Generation: {prediction[0]:.2f} kW")
except ValueError as e:
    st.error(f"Error during prediction: {e}")
