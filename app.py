import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json

# Load the dataset (assuming 'FPAM.csv' is available)
df = pd.read_csv("FPAM.csv")

# Convert 'price_date' to datetime
df['price_date'] = pd.to_datetime(df['price_date'])

# Streamlit UI elements
st.title("Food Price Forecasting")
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
food_item = st.selectbox("Select Food Item", food_items)
prediction_date = st.date_input("Select Prediction Date", df['price_date'].max())

# Function to load the model and make a prediction
def predict(state, food_item, prediction_date):
    # Load the trained model from the serialized JSON file
    food_item = food_item_name_map[food_item].upper()
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
        st.subheader(f"Prediction for {food_item} in {state} on {prediction_date}")
        st.write(f"Predicted Price: {round(prediction['yhat']/5) * 5} (Naira)")
        st.write(f"Minimum Price Range: {round(prediction['yhat_lower']/5) * 5} (Naira)")
        st.write(f"Maximum Price Range: {round(prediction['yhat_upper']/5) * 5} (Naira)")
