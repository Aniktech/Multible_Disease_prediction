import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
from PIL import Image
from fpdf import FPDF
from streamlit_option_menu import option_menu
from google import genai
from dotenv import load_dotenv

# --- CONFIGURATION & CLINICAL THEME ---
load_dotenv()
st.set_page_config(page_title="HealthNexus AI", page_icon="🏥", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3.5em;
        background-color: #004a99;
        color: white;
        font-weight: bold;
        border: none;
    }
    h1 { color: #004a99; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# --- AI INITIALIZATION (FIXED ROUTING) ---
# Explicitly using the stable 'v1' api_version to avoid the 404 Beta error
client = genai.Client(
    api_key=os.getenv("GEMINI_API_KEY"),
    http_options={'api_version': 'v1'} 
)

try:
    model = load_model('models/ann_model.h5')
    scaler = joblib.load('models/scaler.joblib')
    symptoms_list = joblib.load('models/symptoms_list.joblib')
    label_encoder = joblib.load('models/label_encoder.joblib')
except Exception as e:
    st.error(f"⚠️ System Failure: {e}")
    st.stop()

# --- SIDEBAR ---
with st.sidebar:
    st.title("🏥 HealthNexus AI")
    selected = option_menu(
        menu_title="Clinical Portal",
        options=["Patient Diagnosis", "Lab Analysis"],
        icons=["person-vcard", "microscope"],
        menu_icon="hospital-fill",
        default_index=0
    )

# --- PAGE 1: PATIENT DIAGNOSIS ---
if selected == "Patient Diagnosis":
    st.title("🩺 Patient Symptom Diagnosis")
    user_symptoms = st.multiselect("🔍 Search Clinical Symptoms:", options=symptoms_list)

    if st.button("Generate Diagnostic Prediction"):
        if user_symptoms:
            input_vector = np.zeros(len(symptoms_list))
            for s in user_symptoms:
                if s in symptoms_list:
                    input_vector[symptoms_list.index(s)] = 1
            
            scaled_vec = scaler.transform(input_vector.reshape(1, -1))
            prediction_prob = model.predict(scaled_vec)
            disease = label_encoder.inverse_transform([np.argmax(prediction_prob)])[0]

            st.subheader("Diagnostic Summary")
            st.metric("Likely Condition", disease, f"{np.max(prediction_prob)*100:.2f}% Confidence")
        else:
            st.error("No symptoms selected.")

# --- PAGE 2: LAB ANALYSIS (GEMINI BRANDING REMOVED) ---
elif selected == "Lab Analysis":
    st.title("🔬 Laboratory Report Intelligence")
    st.write("Scan and interpret biochemical markers from laboratory reports.")

    uploaded_file = st.file_uploader("Upload Laboratory Report", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Patient Record Attachment", width='stretch')

        # Removed "via Gemini AI" from button and text
        if st.button("Analyse the report"):
            with st.spinner("Analyzing laboratory data..."):
                try:
                    # Using the stable model identifier
                    response = client.models.generate_content(
                        model='gemini-1.5-flash',
                        contents=[
                            "Provide a clinical summary of this report. Identify values outside of reference ranges.",
                            img
                        ]
                    )
                    
                    st.success("Analysis Complete")
                    st.markdown("### Clinical Interpretation")
                    st.write(response.text)
                    
                    st.download_button("Download Report Summary", response.text, file_name="lab_summary.txt")
                except Exception as e:
                    st.error(f"Analysis Failed: {e}")