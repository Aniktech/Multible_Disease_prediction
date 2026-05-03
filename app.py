import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
from PIL import Image

st.set_page_config(page_title="Disease Prediction App", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH_RF = os.path.join(BASE_DIR, "models", "rf_model.joblib")
MODEL_PATH_ANN = os.path.join(BASE_DIR, "models", "ann_model.h5")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.joblib")
# --- Load Model ---
st.title("🩺 Health & Lifestyle Disease Prediction App")
st.markdown("This app predicts whether a person has any disease based on health and lifestyle parameters.")

model_type = None
scaler = joblib.load(SCALER_PATH)

if os.path.exists(MODEL_PATH_ANN):
    model_type = "ANN"
    model = load_model(MODEL_PATH_ANN)
    st.sidebar.success("🧠 ANN model loaded")
elif os.path.exists(MODEL_PATH_RF):
    model_type = "RF"
    model = joblib.load(MODEL_PATH_RF)
    st.sidebar.success("🌲 RandomForest model loaded")
else:
    st.error("❌ No trained model found. Run `python src/train_model.py` first.")
    st.stop()

# --- Manual Input Form ---
st.header("👩‍⚕️ Predict from Manual Input")

col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 18, 90, 40)
    gender = st.selectbox("Gender", [0, 1], format_func=lambda x: "Male" if x==1 else "Female")
    bmi = st.slider("BMI", 15.0, 45.0, 25.0)
    bp = st.slider("Blood Pressure", 80, 200, 120)
    cholesterol = st.slider("Cholesterol", 100, 400, 200)
    sugar = st.slider("Blood Sugar", 50, 400, 100)
    body_temp = st.slider("Body Temperature (°F)", 96.0, 104.0, 98.4)
    sleep_hours = st.slider("Sleep Hours per Day", 3.0, 10.0, 7.0)
with col2:
    tremor = st.selectbox("Tremor", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    voice_change = st.selectbox("Voice Change", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    fatigue = st.selectbox("Fatigue", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    chest_pain = st.selectbox("Chest Pain", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    smoking = st.selectbox("Smoking Habit", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    alcohol = st.selectbox("Alcohol Consumption", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")
    diet_score = st.slider("Diet Quality (1 = Poor, 10 = Excellent)", 1, 10, 6)
    exercise_frequency = st.slider("Exercise Frequency (Days per Week)", 0, 7, 3)
    family_history = st.selectbox("Family History of Disease", [0, 1], format_func=lambda x: "Yes" if x==1 else "No")

if st.button("🔍 Predict Disease"):
    input_data = pd.DataFrame([{
        "age": age, "gender": gender, "bmi": bmi, "bp": bp,
        "cholesterol": cholesterol, "sugar": sugar, "tremor": tremor,
        "voice_change": voice_change, "fatigue": fatigue, "chest_pain": chest_pain,
        "smoking": smoking, "alcohol": alcohol, "diet_score": diet_score,
        "exercise_frequency": exercise_frequency, "family_history": family_history,
        "sleep_hours": sleep_hours, "body_temp": body_temp
    }])

    scaled = scaler.transform(input_data)
    if model_type == "ANN":
        prob = float(model.predict(scaled)[0][0])
        prediction = 1 if prob >= 0.5 else 0
    else:
        prob = model.predict_proba(scaled)[0][1]
        prediction = int(model.predict(scaled)[0])

    label = "🩸 **Disease Detected**" if prediction == 1 else "💚 **No Disease Detected**"
    st.success(f"Prediction: {label}")
    st.write(f"Confidence: **{prob*100:.2f}%**")

# --- Visualization Section ---
st.header("📊 Model Comparison & Training Insights")

if os.path.exists("models/model_comparison.png"):
    st.image("models/model_comparison.png", caption="Model Accuracy Comparison", use_container_width=True)
if os.path.exists("models/ann_training_curve.png"):
    st.image("models/ann_training_curve.png", caption="ANN Training Progress", use_container_width=True)

# --- CSV Upload ---
st.header("📁 Predict from CSV File")
st.markdown("Upload a CSV file containing all the same feature columns used for training.")
uploaded = st.file_uploader("Upload CSV File", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    try:
        scaled = scaler.transform(df)
        if model_type == "ANN":
            probs = model.predict(scaled)
            preds = (probs >= 0.5).astype(int).flatten()
        else:
            preds = model.predict(scaled)
            probs = model.predict_proba(scaled)[:, 1]

        df["Prediction"] = np.where(preds == 1, "Disease", "Healthy")
        df["Confidence (%)"] = (probs * 100).round(2)

        st.success("✅ Predictions completed!")
        st.dataframe(df.head())

        st.download_button("📥 Download Results", df.to_csv(index=False).encode("utf-8"), "predictions.csv")
    except Exception as e:
        st.error(f"Error: {e}")
