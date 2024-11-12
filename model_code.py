import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.serialize import model_to_json, model_from_json
import warnings
warnings.filterwarnings("ignore")

# Load data
df = pd.read_csv("FPAM.csv")

# Convert 'inflation_food_price_index' to numeric, handling non-numeric entries
df['inflation_food_price_index'] = pd.to_numeric(df['inflation_food_price_index'], errors='coerce')

# Handle missing values (fill NaNs in 'inflation_food_price_index' with the column's mean)
df['inflation_food_price_index'].fillna(df['inflation_food_price_index'].mean(), inplace=True)

# Convert 'price_date' to datetime
df['price_date'] = pd.to_datetime(df['price_date'])

# Sort data by date to ensure correct order
df = df.sort_values(by='price_date')

# Function to forecast for a specific state and food item
def train_and_save_model(state, food_item):
    print(f"\nTraining and Saving Model for State: {state}, Food Item: {food_item}\n")
    
    # Filter the data for the current state and food item
    state_df = df[df['state'] == state].copy()
    state_df = state_df[['price_date', food_item, 'inflation_food_price_index']].copy()  # Select relevant columns
    state_df = state_df.rename(columns={'price_date': 'ds', food_item: 'y'})  # Dynamically select the food item column as target
    
    # Initialize and configure the Prophet model
    model = Prophet(
        seasonality_prior_scale=15.0,
        changepoint_prior_scale=0.2,
        changepoint_range=0.85,
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        interval_width=0.95
    )
    model.add_regressor('inflation_food_price_index')

    # Fit the model with the main data and the inflation regressor
    model.fit(state_df[['ds', 'y', 'inflation_food_price_index']])

    # Serialize the model and save as JSON
    food_item = food_item[2:].upper()
    if food_item  == "Cassava_meal":
        food_item = "Cassava Meal"
    model_filename = f"prophet_model_{state}_{food_item}.json"
    with open(model_filename, 'w') as fout:
        fout.write(model_to_json(model))
    print(f"Model saved as {model_filename}")

# Loop through each unique state and food item
states = df['state'].unique()
food_items = ['c_bread', 'c_cassava_meal', 'c_cowpeas', 'c_gari', 'c_groundnuts', 'c_millet', 'c_sorghum', 'c_yam', 'c_rice', 'c_maize']  # Replace with your actual food items

# Train and save models for each state and food item
for state in states:
    for food_item in food_items:
        train_and_save_model(state, food_item)
