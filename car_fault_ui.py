
import streamlit as st
import pandas as pd
import joblib


# Page configuration
st.set_page_config(
    page_title="Car Fault Detection",
    page_icon="🚗",
    layout="wide"
)


# Load saved model
model = joblib.load("car_failure_model.pkl")

# Load saved encoder
encoder = joblib.load("fault_encoder.pkl")


# Title
st.title("🚗 Car Fault Detection System")

st.write(
    "Enter the vehicle sensor details to predict the car condition."
)

st.divider()




# Input section
st.header("🔧 Vehicle Parameters")


col1, col2, col3 = st.columns(3)


# Column 1
with col1:

    engine_temp = st.number_input(
        "Engine Temperature",
        value=95.0
    )

    battery_voltage = st.number_input(
        "Battery Voltage",
        value=12.6
    )

    rpm = st.number_input(
        "RPM",
        value=2500
    )


# Column 2
with col2:

    fuel_pressure = st.number_input(
        "Fuel Pressure",
        value=45.0
    )

    oil_pressure = st.number_input(
        "Oil Pressure",
        value=50.0
    )

    coolant_temp = st.number_input(
        "Coolant Temperature",
        value=90.0
    )


# Column 3
with col3:

    vibration = st.selectbox(
        "Vibration Level",
        [0, 1, 2]
    )

    mileage = st.number_input(
        "Mileage",
        value=80000
    )

    age = st.number_input(
        "Vehicle Age",
        value=8
    )


st.divider()


# Prediction button
if st.button(
    "🔍 Predict Car Fault",
    use_container_width=True
):

    # Create input data
    new_car = pd.DataFrame(
        [[
            engine_temp,
            battery_voltage,
            rpm,
            fuel_pressure,
            oil_pressure,
            coolant_temp,
            vibration,
            mileage,
            age
        ]],
        columns=[
            "engine_temp",
            "battery_voltage",
            "rpm",
            "fuel_pressure",
            "oil_pressure",
            "coolant_temp",
            "vibration",
            "mileage",
            "age"
        ]
    )


    # Make prediction
    prediction = model.predict(new_car)


    # Convert number to fault name
    predicted_fault = encoder.inverse_transform(
        prediction
    )[0]


    # Display result
    st.header("📊 Prediction Result")


    if predicted_fault == "No Fault":

        st.success(
            "✅ NO FAULT - The car condition is normal."
        )


    elif predicted_fault == "Warning":

        st.warning(
            "⚠️ WARNING - The car should be checked."
        )


    elif predicted_fault == "Fault":

        st.error(
            "❌ FAULT DETECTED - Immediate inspection recommended."
        )


    st.write(
        "### Predicted Condition:",
        predicted_fault
    )

    


    # Display entered values
    st.subheader("🚗 Vehicle Information")

    st.dataframe(
        new_car,
        use_container_width=True
    )





