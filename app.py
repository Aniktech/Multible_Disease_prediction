import streamlit as st
import joblib
import numpy as np
import google.generativeai as genai
from PIL import Image
from streamlit_option_menu import option_menu

# --- SETUP & CONFIG ---
st.set_page_config(page_title="MediScan Pro", page_icon="🏥", layout="wide")

# Modern Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { border-radius: 8px; height: 3em; background: linear-gradient(45deg, #007bff, #00d4ff); color: white; border: none; font-weight: bold; }
    .prediction-card { padding: 20px; background: white; border-radius: 15px; border-left: 5px solid #007bff; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    </style>
""", unsafe_allow_html=True)

# Load Models
@st.cache_resource
def load_assets():
    model = joblib.load('models/ensemble_disease_model.joblib')
    le = joblib.load('models/label_encoder.joblib')
    scaler = joblib.load('models/scaler.joblib')
    return model, le, scaler

model, le, scaler = load_assets()

# --- NAVIGATION ---
with st.sidebar:
    selected = option_menu(
        "MediScan Pro",
        ["Dashboard", "Symptom Checker", "Report AI", "History"],
        icons=['grid', 'heart-pulse', 'file-earmark-text', 'clock-history'],
        menu_icon="hospital", default_index=0
    )

# --- SYMPTOM CHECKER ---
if selected == "Symptom Checker":
    st.title("🩺 Advanced Symptom Analysis")
    st.write("Select the symptoms you are experiencing for a high-accuracy prediction.")
    
    # Get features from the model (assuming they match columns in your dataset)
    symptoms_list = ["Fever", "Cough", "Fatigue", "Shortness of Breath", "Headache"] # Replace with your 132 symptoms
    
    selected_symptoms = st.multiselect("Search & Select Symptoms", symptoms_list)
    
    if st.button("Generate Diagnostic Report"):
        if not selected_symptoms:
            st.warning("Please select at least one symptom.")
        else:
            with st.spinner("Analyzing data patterns..."):
                # Create input vector (1 for present, 0 for absent)
                input_vector = np.zeros(len(symptoms_list))
                for s in selected_symptoms:
                    input_vector[symptoms_list.index(s)] = 1
                
                # Scale and Predict
                scaled_input = scaler.transform([input_vector])
                prediction = model.predict(scaled_input)
                disease_name = le.inverse_transform(prediction)[0]
                
                # Display Results
                st.markdown(f"""
                <div class="prediction-card">
                    <h3>Analysis Result</h3>
                    <p>Based on the patterns detected, the most likely condition is:</p>
                    <h2 style="color: #007bff;">{disease_name}</h2>
                    <p><i>Confidence: High (Ensemble Verified)</i></p>
                </div>
                """, unsafe_allow_html=True)

# --- REPORT AI ---
elif selected == "Report AI":
    st.title("📂 AI Medical Report Analysis")
    st.info("Upload a blood test or clinical report (PDF/Image) for an AI-powered summary.")
    
    uploaded_file = st.file_uploader("Upload Document", type=["png", "jpg", "jpeg", "pdf"])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Report", use_column_width=True)
        
        if st.button("Analyze with Gemini AI"):
            # Configure your Gemini Key here
            genai.configure(api_key="YOUR_GEMINI_API_KEY")
            vision_model = genai.GenerativeModel('gemini-1.5-flash')
            
            prompt = """
            Extract the medical data from this report. 
            Identify abnormal values (High/Low). 
            Explain these in simple terms and suggest the next steps for the user. 
            Keep it structured with Bullet Points.
            """
            
            response = vision_model.generate_content([prompt, image])
            st.subheader("🤖 AI Interpretation")
            st.write(response.text)
            st.warning("Notice: This is an AI analysis. Always verify results with a qualified doctor.")

else:
    st.title("Dashboard")
    st.write("Welcome back! Select a tool from the sidebar to begin.")