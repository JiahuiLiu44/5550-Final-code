# Import necessary libraries 
import streamlit as st
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import numpy as np

# Function to load the joblib model
@st.cache_resource
def load_joblib_model():
    return joblib.load("ClimateAgric.pkl")

# Function to load the LSTM model
@st.cache_resource
def load_lstm_model():
    return load_model("lstm_crop_yield_model.h5")

# Set up the Streamlit app
st.title('Predicting the Impact of Climate Change on Agricultural Productivity :earth_americas:')
st.write("This app predicts the **Impact of Climate Change on Agricultural Productivity**.")

# Sidebar for user input
st.sidebar.header("Input Features")
st.sidebar.markdown("Adjust the sliders and inputs below to customize the prediction.")

# Model selection dropdown
model_type = st.sidebar.selectbox(
    "Choose Prediction Model",
    ("Pre-trained Joblib Model", "LSTM Model")
)

# Input fields
year = st.sidebar.slider("Year", min_value=1990, max_value=2030, step=1, value=2024)
avg_temp = st.sidebar.number_input("Average Temperature (°C)", min_value=0.0, max_value=50.0, step=0.1)
precipitation = st.sidebar.number_input("Total Precipitation (mm)", min_value=0.0, max_value=5000.0, step=1.0)
co2_emissions = st.sidebar.number_input("CO2 Emissions (MT)", min_value=0.0, max_value=100.0, step=0.1)
pesticide_use = st.sidebar.number_input("Pesticide Use (KG/HA)", min_value=0.0, max_value=100.0, step=0.1)
fertilizer_use = st.sidebar.number_input("Fertilizer Use (KG/HA)", min_value=0.0, max_value=100.0, step=0.1)

# Create a dictionary for the model input
input_data = {
    "Year": year,
    "Average_Temperature_C": avg_temp,
    "Total_Precipitation_mm": precipitation,
    "CO2_Emissions_MT": co2_emissions,
    "Pesticide_Use_KG_per_HA": pesticide_use,
    "Fertilizer_Use_KG_per_HA": fertilizer_use,
}

# Convert the dictionary to a DataFrame
input_df = pd.DataFrame([input_data])

# Display user input
st.subheader("User Input Summary")
st.dataframe(input_df, width=700)

# Normalize input for LSTM if selected
if model_type == "LSTM Model":
    # Assume LSTM model expects scaled input
    # Preprocessing: replicate similar scaling used during training
    input_values = input_df[['Year', 'Average_Temperature_C', 'Total_Precipitation_mm', 
                             'CO2_Emissions_MT', 'Pesticide_Use_KG_per_HA', 
                             'Fertilizer_Use_KG_per_HA']].values
    # Scaling (example: MinMax scaling between 0 and 1)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled_input = scaler.fit_transform(input_values)
    lstm_input = np.expand_dims(scaled_input, axis=0)  # Add batch dimension for LSTM

st.markdown("---")

# Prediction button
if st.button("Predict Crop Yield"):
    if model_type == "Pre-trained Joblib Model":
        # Load and predict using the Joblib model
        model = load_joblib_model()
        prediction = model.predict(input_df)[0]
    elif model_type == "LSTM Model":
        # Load and predict using the LSTM model
        lstm_model = load_lstm_model()
        prediction = lstm_model.predict(lstm_input)[0][0]
    
    # Display prediction
    st.subheader("Predicted Crop Yield (MT/HA)")
    st.write(f"{prediction:,.2f}")

st.markdown("---")

# Footer
st.caption("This App is developed to explore the intersection of **climate change** and **agricultural productivity** using machine learning.")




