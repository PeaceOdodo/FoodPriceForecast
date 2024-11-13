import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import base64

# Load the dataset (assuming 'FPAM.csv' is available)
df = pd.read_csv("FPAM.csv")

# Convert 'price_date' to datetime
df['price_date'] = pd.to_datetime(df['price_date'])

# Streamlit UI elements
st.set_page_config(page_title="Food Price Forecasting", layout="centered")

# Convert image to Base64
with open("agtrade-logo.png", "rb") as image_file:
    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")

# Custom CSS
st.markdown("""
    <style>
    /* Button styling */
    .stButton > button {
        background-color: #2A9D8F; /* Deep green */
        color: white;
        border-radius: 8px;
        font-weight: bold;
        height: 50px;
        width: 100%;
    }
    /* Title and icon styling */
    .title-container {
        display: flex;
        align-items: center;
        font-size: 2.5rem;
        font-weight: bold;
        color: #2A9D8F;
    }
    .title-container img {
        width: 50px;
        height: 50px;
        margin-right: 15px;
    }
    /* Center align output */
    .prediction-text {
        font-size: 1.2rem;
        color: #333;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with custom image
st.markdown(
    """
    <div class="title-container">
        <img src="agtrade-logo.png" alt="">
        Food Price Forecasting
    </div>
    """, unsafe_allow_html=True
)
col1, col2, col3 = st.columns(3)
with col1:
    state = st.selectbox("Select State", df['state'].unique())
food_items = [
    'Bread (small size)', 
    'Cassava Meal (100 KG)', 
    'Cowpeas (100 KG)', 
    'Garri (100 KG)', 
    'Groundnuts (100 KG)', 
    'Millet (100 KG)', 
    'Sorghum (100 KG)', 
    'Yam (1 KG)', 
    'Rice (50 KG)', 
    'Maize (100 KG)'
    ]
food_item_name_map = {
   'Bread (small size)' : 'Bread', 
    'Cassava Meal (100 KG)': 'Cassava Meal', 
    'Cowpeas (100 KG)' : 'Cowpeas', 
    'Garri (100 KG)'  : 'Gari', 
    'Groundnuts (100 KG)' : 'Groundnuts', 
    'Millet (100 KG)' : 'Millet', 
    'Sorghum (100 KG)' : 'Sorghum', 
    'Yam (1 KG)' : 'Yam', 
    'Rice (50 KG)' : 'Rice', 
    'Maize (100 KG)' : 'Maize' 
}
with col2:
    food_item = st.selectbox("Select Food Item", food_items)
with col3:
    prediction_date = st.date_input("Select Prediction Date", df['price_date'].max())

# Function to load the model and make a prediction
def predict(state, food_item, prediction_date):
    # Load the trained model from the serialized JSON file
    food_item = food_item_name_map[food_item].upper()
    if food_item == "CASSAVA MEAL":
        food_item = "CASSAVA_MEAL"
    model_filename = f"prophet_model_{state}_{food_item}.json"
    
    try:
        with open(model_filename, 'r') as fin:
            model = model_from_json(fin.read())
    except FileNotFoundError:
        st.error(f"Model for {state} and {food_item} not found. Please ensure the model is trained and saved.")
        return None
    
    # Prepare the future DataFrame for prediction
    future = pd.DataFrame({'ds': [pd.to_datetime(prediction_date)]})
    
    # Extend inflation data into the future (using the last known value)
    last_inflation_value = df[df['state'] == state]['inflation_food_price_index'].iloc[-1]
    future['inflation_food_price_index'] = last_inflation_value
    
    # Make the prediction
    forecast = model.predict(future)
    
    # Extract prediction and uncertainty intervals
    prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].iloc[0]
    return prediction

# When the user clicks the "Predict" button
if st.button("Predict"):
    prediction = predict(state, food_item, prediction_date)
    
    if prediction is not None:
        with st.container():
            st.markdown(f"<div class='prediction-text'>Forecast for {food_item} in {state} by {prediction_date}</div>", unsafe_allow_html=True)
            st.write(f"**Predicted Price**: ₦{round(prediction['yhat']/10) * 10}")
            st.write(f"**Minimum Price Range**: ₦{round(prediction['yhat_lower']/10) * 10}")
            st.write(f"**Maximum Price Range**: ₦{round(prediction['yhat_upper']/10) * 10}")
