import streamlit as st
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Function to train the model
def train_model(data):
    X = np.array(data["oil_price"]).reshape(-1, 1)  # Feature: Oil prices
    y = np.array(data["gas_price"])  # Target: Gas prices
    model = LinearRegression()
    model.fit(X, y)
    return model

# Function to predict gas prices
def predict_gas_prices(model, future_oil_prices):
    future_oil_prices = np.array(future_oil_prices).reshape(-1, 1)
    return model.predict(future_oil_prices)

# Streamlit App
st.title("Gas Price Prediction Application")
st.write("This app predicts gas prices based on current oil prices using a linear regression model.")

# Example historical data
st.sidebar.header("Input Historical Data")
example_data = {
    "oil_price": [50, 60, 70, 80, 90, 100],  # Crude oil prices in USD per barrel
    "gas_price": [2.5, 2.7, 3.0, 3.3, 3.6, 3.9],  # Gasoline prices in USD per gallon
}
df = pd.DataFrame(example_data)

# Display historical data and allow user to modify it
st.sidebar.write("Modify historical data as needed:")
oil_prices = st.sidebar.text_input("Oil Prices (comma-separated):", "50, 60, 70, 80, 90, 100")
gas_prices = st.sidebar.text_input("Gas Prices (comma-separated):", "2.5, 2.7, 3.0, 3.3, 3.6, 3.9")

try:
    df = pd.DataFrame({
        "oil_price": [float(x.strip()) for x in oil_prices.split(",")],
        "gas_price": [float(x.strip()) for x in gas_prices.split(",")]
    })
except ValueError:
    st.sidebar.error("Please ensure the input values are numeric and comma-separated.")

# Train the model
model = train_model(df)

# Future oil prices input
st.subheader("Predict Future Gas Prices")
future_oil_prices = st.text_input("Enter future oil prices (comma-separated):", "105, 110, 115")

try:
    future_oil_prices_list = [float(x.strip()) for x in future_oil_prices.split(",")]
    predicted_gas_prices = predict_gas_prices(model, future_oil_prices_list)

    # Display Predictions
    st.write("### Predictions")
    for oil_price, gas_price in zip(future_oil_prices_list, predicted_gas_prices):
        st.write(f"Predicted Gas Price for Oil Price ${oil_price:.2f}: ${gas_price:.2f}")

    # Visualization
    st.write("### Visualization")
    plt.figure(figsize=(10, 6))
    plt.scatter(df["oil_price"], df["gas_price"], color="blue", label="Historical Data")
    plt.plot(df["oil_price"], model.predict(np.array(df["oil_price"]).reshape(-1, 1)), color="green", label="Fitted Model")
    plt.scatter(future_oil_prices_list, predicted_gas_prices, color="red", label="Predictions")
    plt.title("Gas Price Prediction Based on Oil Prices")
    plt.xlabel("Oil Price (USD per Barrel)")
    plt.ylabel("Gas Price (USD per Gallon)")
    plt.legend()
    plt.grid(True)
    st.pyplot(plt)

except ValueError:
    st.error("Ensure that future oil prices are numeric and comma-separated.")

