import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from PIL import Image
from fpdf import FPDF
from streamlit_option_menu import option_menu
import google.generativeai as genai
from dotenv import load_dotenv

# --- CONFIG & ASSETS ---
load_dotenv()
st.set_page_config(page_title="MediScan Pro", page_icon="🏥", layout="wide")

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 8px; background: linear-gradient(45deg, #007bff, #00d4ff); color: white; font-weight: bold; width: 100%; }
    .report-card { padding: 20px; background: white; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-left: 5px solid #007bff; }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_ml_assets():
    model = joblib.load('models/ensemble_disease_model.joblib')
    le = joblib.load('models/label_encoder.joblib')
    scaler = joblib.load('models/scaler.joblib')
    # Replace this list with the exact 132 symptoms from your CSV columns
    features = ["fever", "cough", "fatigue", "shortness_of_breath", "headache", "vomiting", "joint_pain"] 
    return model, le, scaler, features

try:
    model, le, scaler, symptoms_list = load_ml_assets()
except:
    st.error("Error: Models not found. Please run your training script first.")

# --- HELPER FUNCTIONS ---
def generate_pdf(disease, confidence, selected_symptoms):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "MediScan AI Diagnostic Summary", ln=True, align='C')
    pdf.set_font("Arial", size=12)
    pdf.ln(10)
    pdf.cell(200, 10, f"Predicted Condition: {disease}", ln=True)
    pdf.cell(200, 10, f"Status: AI-Verified", ln=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, "Reported Symptoms:", ln=True)
    pdf.set_font("Arial", size=11)
    for s in selected_symptoms:
        pdf.cell(200, 8, f"- {s}", ln=True)
    pdf.ln(10)
    pdf.set_text_color(255, 0, 0)
    pdf.multi_cell(0, 10, "DISCLAIMER: This is an AI-generated assessment. Please consult a registered medical practitioner for clinical diagnosis.")
    return pdf.output(dest='S').encode('latin-1')

# --- SIDEBAR ---
with st.sidebar:
    selected = option_menu(
        "MediScan AI",
        ["Dashboard", "Symptom Checker", "AI Report Lab"],
        icons=['grid', 'heart-pulse', 'file-earmark-medical'],
        menu_icon="hospital-fill", default_index=1
    )

# --- NAVIGATION LOGIC ---
if selected == "Dashboard":
    st.title("📊 Health Overview")
    st.write(f"Welcome back, Ishika! Use the tools to check symptoms or analyze lab reports.")
    st.image("https://img.freepik.com/free-vector/medical-technology-science-background_1017-17594.jpg")

elif selected == "Symptom Checker":
    st.title("🩺 Intelligent Symptom Analysis")
    st.write("Leveraging Stacking Ensembles and SHAP for explainable diagnostics.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        user_symptoms = st.multiselect("Identify Symptoms", symptoms_list)
        
        if st.button("Run Diagnostic Analysis"):
            if user_symptoms:
                # Prepare data
                input_vector = np.zeros(len(symptoms_list))
                for s in user_symptoms:
                    input_vector[symptoms_list.index(s)] = 1
                
                input_df = pd.DataFrame([input_vector], columns=symptoms_list)
                scaled_input = scaler.transform(input_df)
                
                # Predict
                prediction = model.predict(scaled_input)
                disease_name = le.inverse_transform(prediction)[0]
                
                # Display Card
                st.markdown(f"""
                <div class="report-card">
                    <h3>Preliminary Result: <b>{disease_name}</b></h3>
                    <p>Analysis complete. Download the PDF for your physician.</p>
                </div>
                """, unsafe_allow_html=True)
                
                # PDF Download
                pdf_data = generate_pdf(disease_name, "High", user_symptoms)
                st.download_button("📥 Download Medical Report", pdf_data, f"Report_{disease_name}.pdf", "application/pdf")
            else:
                st.warning("Please select at least one symptom.")

    with col2:
        if 'disease_name' in locals():
            st.write("### 🧠 AI Reasoning (SHAP)")
            st.caption("How your symptoms influenced the prediction:")
            # Use the Random Forest component from the Stacking Classifier for SHAP
            rf_sub_model = model.named_estimators_['rf']
            explainer = shap.TreeExplainer(rf_sub_model)
            shap_values = explainer.shap_values(input_df)
            
            fig, ax = plt.subplots()
            # Plot for the predicted class
            class_idx = list(le.classes_).index(disease_name)
            shap.summary_plot(shap_values[class_idx], input_df, plot_type="bar", show=False)
            st.pyplot(fig)

elif selected == "AI Report Lab":
    st.title("📂 Multimodal Report Analysis")
    st.info("Upload your clinical lab report (Blood test, MRI summary, etc.) for AI interpretation.")
    
    file = st.file_uploader("Upload Image", type=['jpg', 'jpeg', 'png'])
    
    if file:
        img = Image.open(file)
        st.image(img, width=400, caption="Scanned Document")
        
        if st.button("Analyze with GenAI"):
            with st.spinner("Extracting clinical data..."):
                vision_model = genai.GenerativeModel('gemini-1.5-flash')
                prompt = """Analyze this medical report. 
                1. Summarize the patient's general status. 
                2. Highlight values that are outside the normal range. 
                3. Suggest lifestyle adjustments.
                Keep it professional and concise."""
                
                response = vision_model.generate_content([prompt, img])
                st.markdown("---")
                st.markdown("### 🤖 AI Lab Interpretation")
                st.write(response.text)