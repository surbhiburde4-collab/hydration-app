import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Adaptive Hydration Optimizer", page_icon="💧", layout="centered")
st.markdown("""
<style>
.stApp {
    background-image: url("https://i.pinimg.com/originals/d9/60/fc/d960fc21a100d79ef6bd12ebbc7773c4.png");
    background-size: 300px;
    background-repeat: repeat;
    background-attachment: fixed;
}
.stApp > header {
    background-color: rgba(0,0,0,0);
}
[data-testid="stVerticalBlock"] {
    background-color: rgba(255, 255, 255, 0.90);
    padding: 25px;
    border-radius: 15px;
}
</style>
""", unsafe_allow_html=True)
with st.sidebar:
    st.image("https://em-content.zobj.net/source/apple/354/panda_1f43c.png", width=100)
    st.markdown("### 🐼 Stay hydrated like a panda!")
    st.write("Pandas get most of their water from the bamboo they eat. You need to drink yours directly — don't forget!")

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("hydration_xgb_model.pkl")

model = load_model()

FEATURE_COLUMNS = [
    "age", "weight", "height", "activity_level", "heart_rate", "gsr",
    "skin_temp", "ambient_temp", "humidity", "accel_magnitude",
    "bmi", "heat_stress_index", "sweat_rate_proxy", "hr_intensity"
]
CLASS_NAMES = ["Well-Hydrated", "Mild Dehydration", "Severe Dehydration"]

def add_features(d):
    d = d.copy()
    d["bmi"] = d["weight"] / (d["height"] / 100) ** 2
    d["heat_stress_index"] = d["ambient_temp"] * d["humidity"] / 100
    d["sweat_rate_proxy"] = d["gsr"] * d["skin_temp"]
    d["hr_intensity"] = d["heart_rate"] / (220 - d["age"])
    return d

def recommend(model, row):
    r = add_features(row)[FEATURE_COLUMNS]
    proba = model.predict_proba(r)[0]
    cls = int(np.argmax(proba))
    weight_factor = row["weight"].iloc[0] / 70
    heat_factor = 1 + 0.3 * max(0, r["heat_stress_index"].iloc[0] - 15) / 15
    base = {0: 100, 1: 300, 2: 500}[cls]
    ml = int(round(base * weight_factor * heat_factor / 50) * 50)
    msg = {
        0: f"You're well hydrated. Small sips if you like (~{ml} ml).",
        1: f"Mild dehydration detected. Drink ~{ml} ml in the next 30 minutes.",
        2: f"URGENT: severe dehydration risk. Drink ~{ml} ml now and rest in shade.",
    }[cls]
    return CLASS_NAMES[cls], proba, msg

# ---------- UI ----------
st.title("🐼💧 Adaptive Hydration Optimizer")
st.caption("XGBoost-based real-time hydration prediction and personalized fluid-intake recommendation")

st.subheader("User Profile")
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", 10, 90, 25)
with col2:
    weight = st.number_input("Weight (kg)", 30.0, 150.0, 70.0)
with col3:
    height = st.number_input("Height (cm)", 100.0, 220.0, 170.0)

activity_label = st.select_slider("Activity Level", options=["Low", "Moderate", "High"], value="Moderate")
activity_level = {"Low": 0, "Moderate": 1, "High": 2}[activity_label]

st.subheader("Wearable Sensor Readings")
col4, col5, col6 = st.columns(3)
with col4:
    heart_rate = st.slider("Heart Rate (bpm)", 50, 200, 90)
with col5:
    gsr = st.slider("GSR (µS)", 0.0, 10.0, 3.0, step=0.1)
with col6:
    skin_temp = st.slider("Skin Temperature (°C)", 28.0, 40.0, 33.0, step=0.1)

accel_magnitude = st.slider("Accelerometer Magnitude", 0.5, 4.0, 1.5, step=0.1)

st.subheader("Environmental Conditions")
col7, col8 = st.columns(2)
with col7:
    ambient_temp = st.slider("Ambient Temperature (°C)", 10.0, 50.0, 30.0, step=0.5)
with col8:
    humidity = st.slider("Humidity (%)", 10.0, 100.0, 55.0, step=1.0)

st.divider()

if st.button("🔍 Predict Hydration Status", type="primary", use_container_width=True):
    row = pd.DataFrame([{
        "age": age, "weight": weight, "height": height, "activity_level": activity_level,
        "heart_rate": heart_rate, "gsr": gsr, "skin_temp": skin_temp,
        "ambient_temp": ambient_temp, "humidity": humidity, "accel_magnitude": accel_magnitude
    }])

    status, proba, msg = recommend(model, row)

    colors = {"Well-Hydrated": "🟢", "Mild Dehydration": "🟡", "Severe Dehydration": "🔴"}
    st.markdown(f"### {colors[status]} Predicted Status: **{status}**")

    if status == "Well-Hydrated":
        st.success(msg)
        st.balloons()
        st.write("🐼 Great job! Even the pandas approve.")
    elif status == "Mild Dehydration":
        st.warning(msg)
    else:
        st.error(msg)

    st.subheader("Class Probabilities")
    proba_df = pd.DataFrame({"Hydration Class": CLASS_NAMES, "Probability": proba})
    st.bar_chart(proba_df.set_index("Hydration Class"))

    with st.expander("See raw probability values"):
        for name, p in zip(CLASS_NAMES, proba):
            st.write(f"**{name}:** {p:.2%}")

st.divider()
st.caption("Note: Predictions are based on a model trained on simulated data and are for demonstration purposes only, not medical advice.")
