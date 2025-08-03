import streamlit as st
import requests

from dotenv import load_dotenv
import os

# Load values from .env
load_dotenv()

API_KEY = os.getenv("API_KEY")
DEPLOYMENT_URL = os.getenv("DEPLOYMENT_URL")


# Set title
st.title("Machine Failure Prediction (IBM Cloud Model)")

# Collect input from user
st.header("Enter Machine Data:")

# udi = st.number_input("UDI", min_value=1, step=1)
udi = 1
product_id = "M81460"


type_option = st.selectbox("Type", ["M", "L", "H"])
air_temp = st.number_input("Air Temperature [K]", value=298.1)
process_temp = st.number_input("Process Temperature [K]", value=308.6)
rot_speed = st.number_input("Rotational Speed [rpm]", value=1551)
torque = st.number_input("Torque [Nm]", value=42.8)
tool_wear = st.number_input("Tool Wear [min]", value=0)
# failure_type = st.text_input("Failure Type", value="No Failure")
failure_type = "No failure"

if st.button("Predict"):

    # Get access token
    token_response = requests.post(
        'https://iam.cloud.ibm.com/identity/token',
        data={"apikey": API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'}
    )
    if token_response.status_code != 200:
        st.error("Failed to retrieve access token")
    else:
        mltoken = token_response.json()["access_token"]

        # Set headers
        headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {mltoken}'
        }

        # Build payload
        payload = {
            "input_data": [{
                "fields": [
                    "UDI",
                    "Product ID",
                    "Type",
                    "Air temperature [K]",
                    "Process temperature [K]",
                    "Rotational speed [rpm]",
                    "Torque [Nm]",
                    "Tool wear [min]",
                    "Failure Type"
                ],
                "values": [[
                    udi,
                    product_id,
                    type_option,
                    air_temp,
                    process_temp,
                    rot_speed,
                    torque,
                    tool_wear,
                    failure_type
                ]]
            }]
        }

        # Send request
        response = requests.post(DEPLOYMENT_URL, json=payload, headers=headers)

        try:
            result = response.json()
            prediction = result["predictions"][0]["values"][0][0]

            # Display True/False only
            st.subheader("Prediction Result:")
            if prediction == 1.0:
                st.success("✅ Failure Detected (True)")
            else:
                st.info("❌ No Failure (False)")
        except Exception as e:
            st.error(f"Error parsing response: {e}")
            st.text(response.text)

