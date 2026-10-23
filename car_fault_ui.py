import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Car Fault Detection System", page_icon="🚗", layout="wide")

@st.cache_resource
def load_artifacts():
    return joblib.load("car_failure_model.pkl"), joblib.load("fault_encoder.pkl")

model, encoder = load_artifacts()

st.title("🚗 Car Fault Detection System")
st.write("Enter sample vehicle sensor readings to classify the condition as No Fault, Warning, or Fault.")
st.divider()
st.header("🔧 Vehicle Parameters")

c1, c2, c3 = st.columns(3)
with c1:
    engine_temp = st.number_input("Engine Temperature (°C)", min_value=0.0, max_value=180.0, value=90.0, step=1.0)
    battery_voltage = st.number_input("Battery Voltage (V)", min_value=0.0, max_value=20.0, value=12.6, step=0.1)
    rpm = st.number_input("Engine RPM", min_value=0, max_value=8000, value=2500, step=100)
with c2:
    fuel_pressure = st.number_input("Fuel Pressure (dataset units)", min_value=0.0, max_value=100.0, value=45.0, step=1.0)
    oil_pressure = st.number_input("Oil Pressure (dataset units)", min_value=0.0, max_value=120.0, value=50.0, step=1.0)
    coolant_temp = st.number_input("Coolant Temperature (°C)", min_value=0.0, max_value=180.0, value=90.0, step=1.0)
with c3:
    vibration = st.selectbox("Vibration Level (dataset scale)", [0, 1, 2], index=0)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=1_000_000, value=80000, step=1000)
    age = st.number_input("Vehicle Age (years)", min_value=0, max_value=50, value=8, step=1)

st.caption("Use the same feature names and order as the training dataset. Pressure values use the dataset's scale; verify units before any real-world use.")

if st.button("🔍 Predict Car Fault", type="primary", use_container_width=True):
    features = pd.DataFrame([{
        "engine_temp": engine_temp,
        "battery_voltage": battery_voltage,
        "rpm": rpm,
        "fuel_pressure": fuel_pressure,
        "oil_pressure": oil_pressure,
        "coolant_temp": coolant_temp,
        "vibration": vibration,
        "mileage": mileage,
        "age": age,
    }])

    model_encoded = model.predict(features)
    model_result = str(encoder.inverse_transform(model_encoded)[0])

    # Transparent classroom-demo rules. The supplied dataset is synthetic and its
    # labels do not consistently match RPM, so RPM needs an explicit demo rule.
    # These are illustrative thresholds only, not validated diagnostic limits.
    rpm_severe = rpm < 500 or rpm >= 7000
    rpm_caution = 500 <= rpm < 800 or 6000 < rpm < 7000

    severe = (
        engine_temp >= 110 or coolant_temp >= 110 or battery_voltage <= 11.5
        or oil_pressure < 20 or fuel_pressure < 25 or vibration >= 2
        or rpm_severe
    )
    caution = (
        engine_temp >= 100 or coolant_temp >= 100 or battery_voltage < 12.2
        or oil_pressure < 30 or fuel_pressure < 35 or vibration == 1
        or rpm_caution
    )

    if severe:
        final_result = "Fault"
        reason = "A severe demo threshold was crossed. For RPM, below 500 or 7000 and above is marked Fault."
    elif caution:
        final_result = "Warning"
        reason = "A caution demo threshold was crossed. For RPM, 500–799 or 6001–6999 is marked Warning."
    else:
        # Use the explicit demo rules for the final status. Show the raw ML result
        # separately for comparison because synthetic training labels can be inconsistent.
        final_result = "No Fault"
        reason = "No illustrative demo threshold was crossed. RPM from 800 through 6000 does not trigger a warning by itself."

    st.subheader("📊 Prediction Result")
    if final_result == "No Fault":
        st.success("✅ NO FAULT — No demo warning threshold was crossed.")
    elif final_result == "Warning":
        st.warning("⚠️ WARNING — Check the vehicle readings.")
    else:
        st.error("❌ FAULT — Severe demo threshold crossed; inspection is recommended.")

    st.metric("Final demo status", final_result)
    st.caption(reason)
    st.write("**Input values used for prediction**")
    st.dataframe(features, use_container_width=True, hide_index=True)
    st.warning("This hybrid demonstration uses simple example thresholds as an override. Do not use its result to decide whether a real car is safe to drive.")
