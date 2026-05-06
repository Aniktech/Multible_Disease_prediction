import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from PIL import Image
from fpdf import FPDF
from streamlit_option_menu import option_menu
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
st.set_page_config(page_title="Multi-Disease Predictor", page_icon="🏥", layout="wide")
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Formatting
st.markdown("""
    <style>
    .stButton>button { border-radius: 8px; background: #007bff; color: white; width: 100%; }
    .report-box { padding: 20px; background: #ffffff; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = load_model('models/ann_model.h5')
    scaler = joblib.load('models/scaler.joblib')
    le = joblib.load('models/label_encoder.joblib')
    
    # Load metadata files
    desc = pd.read_csv('data/symptom_Description.csv')
    precaution = pd.read_csv('data/symptom_precaution.csv')
    
    # Extract symptoms from training data structure
    data_cols = pd.read_csv('data/dataset.csv').columns[:-1]
    return model, scaler, le, list(data_cols), desc, precaution

model, scaler, le, symptoms_list, desc_df, prec_df = load_assets()

def get_pdf(disease, details, precautions):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, "Medical Diagnostic Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(200, 10, f"Condition: {disease}", ln=True)
    pdf.set_font("Arial", size=11)
    pdf.multi_cell(0, 10, f"Description: {details}")
    pdf.ln(5)
    pdf.cell(200, 10, "Recommended Precautions:", ln=True)
    for p in precautions:
        pdf.cell(200, 8, f"- {p}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

with st.sidebar:
    selected = option_menu(
        "Menu", ["Dashboard", "Predictor", "Lab Analysis"], 
        icons=['house', 'search', 'camera'], default_index=1
    )

if selected == "Dashboard":
    st.title("🏥 System Dashboard")
    st.write("Welcome to the Multi-Disease Prediction System. Use the sidebar to navigate.")
    st.image("models/model_comparison.png", caption="Model Performance Analytics")

elif selected == "Predictor":
    st.title("🩺 Disease Predictor")
    user_inputs = st.multiselect("Select your symptoms:", symptoms_list)
    
    if st.button("Analyze Health"):
        if user_inputs:
            # Vectorization
            vec = np.zeros(len(symptoms_list))
            for s in user_inputs:
                vec[symptoms_list.index(s)] = 1
            
            # Prediction
            res = model.predict(scaler.transform([vec]))
            disease = le.inverse_transform([np.argmax(res)])[0]
            
            # Fetch Metadata
            d_desc = desc_df[desc_df['Disease'] == disease]['Description'].values[0]
            d_prec = prec_df[prec_df['Disease'] == disease].iloc[:, 1:].values.flatten().tolist()
            
            st.markdown(f"""<div class='report-box'>
                <h2>Result: {disease}</h2>
                <p><b>Description:</b> {d_desc}</p>
            </div>""", unsafe_allow_html=True)
            
            st.subheader("Recommended Actions")
            for p in d_prec:
                if str(p) != 'nan': st.write(f"✅ {p}")
            
            pdf_data = get_pdf(disease, d_desc, d_prec)
            st.download_button("📥 Download Report", pdf_data, "Health_Report.pdf", "application/pdf")
        else:
            st.warning("Please select at least one symptom.")

elif selected == "Lab Analysis":
    st.title("🔬 AI Lab Report Analysis")
    file = st.file_uploader("Upload report image", type=['jpg', 'png', 'jpeg'])
    
    if file:
        img = Image.open(file)
        st.image(img, width=500)
        if st.button("Process with Gemini AI"):
            gemini = genai.GenerativeModel('gemini-1.5-flash')
            resp = gemini.generate_content(["Explain the abnormal values in this report and provide health advice.", img])
            st.write(resp.text)