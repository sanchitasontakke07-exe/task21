#1
# Streamlit is used to create the web application interface.
import streamlit as st
# Pandas is used for data manipulation and creating DataFrames.
import pandas as pd
# Joblib is used to load the saved machine learning model and preprocessing files.
import joblib

#2
# Load the trained Linear Regression model
model = joblib.load("LR_model.pkl")
# Load the StandardScaler object
scaler = joblib.load("scaler.pkl")
# Load the encoded column names used during training
encoded_columns = joblib.load("columns.pkl")

#3
st.set_page_config(
    page_title="Ford Car Price Predictor",
    layout="centered"
)
# page_title sets the title displayed on the browser tab.
# layout="centered" displays the app in the center of the page for a clean and user-friendly interface.

#4
st.title(" Ford Car Price Predictor")
st.write("Enter the car details below to predict its selling price.")

#5
year = st.number_input("Manufacturing Year", min_value=1990, max_value=2025, value=2018)
mileage = st.number_input("Mileage", min_value=0, max_value=300000, value=50000)
tax = st.number_input("Road Tax", min_value=0, max_value=1000, value=150)
mpg = st.number_input("MPG", min_value=0.0, max_value=100.0, value=50.0)
engineSize = st.number_input("Engine Size", min_value=0.0, max_value=6.0, value=1.5)

#6
transmission = st.selectbox(
    "Transmission",
    ["Automatic", "Manual", "Semi-Auto"]
)
fuelType = st.selectbox(
    "Fuel Type",
    ["Petrol", "Diesel", "Hybrid"]
)
# selectbox allows users to select one option from a list,
# reducing typing mistakes and invalid inputs.

#7
model_name = st.text_input("Car Model")
predict = st.button("Predict Price")

#8
if predict:
    input_data = pd.DataFrame({
        "model":[model_name],
        "year":[year],
        "transmission":[transmission],
        "mileage":[mileage],
        "fuelType":[fuelType],
        "tax":[tax],
        "mpg":[mpg],
        "engineSize":[engineSize]
    })
    # Perform One-Hot Encoding
    input_encoded = pd.get_dummies(input_data)
    # Align with training columns
    input_encoded = input_encoded.reindex(
        columns=encoded_columns,
        fill_value=0
    )

    #9
    # Numerical columns
    numerical_columns = [
        "year",
        "mileage",
        "tax",
        "mpg",
        "engineSize"
    ]
    input_encoded[numerical_columns] = scaler.transform(
        input_encoded[numerical_columns]
    )
    prediction = model.predict(input_encoded)
    st.success(f"Predicted Selling Price: £{prediction[0]:,.2f}")

#10
try:
    if predict:
        input_data = pd.DataFrame({
            "model":[model_name],
            "year":[year],
            "transmission":[transmission],
            "mileage":[mileage],
            "fuelType":[fuelType],
            "tax":[tax],
            "mpg":[mpg],
            "engineSize":[engineSize]
        })
        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(
            columns=encoded_columns,
            fill_value=0
        )
        numerical_columns = [
            "year",
            "mileage",
            "tax",
            "mpg",
            "engineSize"
        ]
        input_encoded[numerical_columns] = scaler.transform(
            input_encoded[numerical_columns]
        )
        prediction = model.predict(input_encoded)
        st.success(f"Predicted Selling Price: £{prediction[0]:,.2f}")
except Exception as e:
    st.error(f"Error: {e}")