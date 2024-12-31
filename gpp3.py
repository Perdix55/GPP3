import streamlit as st
import pandas as pd
import requests
import matplotlib.pyplot as plt

# API Endpoint and Key
API_URL = "https://api.eia.gov/v2/petroleum/pri/gnd/data/"
API_KEY = "qhkVpypjymSh1gOSmNuMxbglfan3bDy4nxchqQRu"

# Function to fetch historical gas prices from the API
def fetch_historical_gas_prices(api_url, api_key):
    params = {
        "frequency": "annual",
        "data[0]": "value",
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
        "offset": 0,
        "length": 5000,
        "api_key": api_key,
    }
    response = requests.get(api_url, params=params)
    if response.status_code == 200:
        data = response.json()
        records = data.get("response", {}).get("data", [])
        df = pd.DataFrame(records)
        return df
    else:
        st.error(f"Failed to fetch data: {response.status_code}")
        return pd.DataFrame()

# Streamlit App
st.title("Historical Gas Price Analysis")
st.write("This app retrieves and visualizes historical gas prices using the EIA API.")

# Fetch data
st.subheader("Fetching Historical Data...")
data = fetch_historical_gas_prices(API_URL, API_KEY)

if not data.empty:
    # Process and display the data
    st.write(f"### Total Records Retrieved: {len(data)}")
    st.write("#### Sample Data:")
    st.write(data.head())

    # Prepare data for visualization
    data["period"] = pd.to_datetime(data["period"])
    data = data.sort_values("period")
    data["value"] = data["value"].astype(float)

    # Line chart for gas prices
    st.subheader("Gas Price Trends Over Time")
    plt.figure(figsize=(10, 5))
    plt.plot(data["period"], data["value"], marker="o", label="Gas Prices (USD)")
    plt.title("Annual Gas Price Trends")
    plt.xlabel("Year")
    plt.ylabel("Price (USD per Gallon)")
    plt.grid()
    plt.legend()
    st.pyplot(plt)

    # Highlight the most recent prices
    st.subheader("Most Recent Gas Prices")
    st.write(data.tail(5))

    # Insights
    st.subheader("Insights and Trends")
    if data["value"].iloc[-1] > data["value"].iloc[-2]:
        st.write("Gas prices are trending upward in the most recent year.")
    else:
        st.write("Gas prices are trending downward in the most recent year.")
else:
    st.error("No data available. Please check the API or try again later.")
