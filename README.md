HealthNexus AI: Advanced Clinical Diagnostic System
HealthNexus AI is a professional-grade medical assistant platform that leverages Artificial Neural Networks (ANN) and Generative AI (Gemini 1.5 Flash) to provide intelligent health insights. The system is divided into two core modules: a Symptom-to-Disease Predictor and an Automated Laboratory Report Interpreter.

🏥 Key Features
1. Patient Diagnosis (ANN Predictor)
Intelligent Screening: Analyzes patient symptoms using a deep learning Artificial Neural Network.

High-Dimensional Processing: Maps inputs across a 131-symptom clinical feature set.

Confidence Metrics: Provides a probability-based confidence score for every diagnostic prediction.

2. Lab Analysis (Generative Interpretation)
OCR & Vision Analysis: Processes digital images of laboratory reports such as Blood tests or Liver profiles.

Abnormal Value Detection: Automatically identifies markers outside of standard reference ranges.

Clinical Summarization: Translates complex medical jargon into actionable health summaries and lifestyle suggestions.

3. Clinical-Grade UI
Designed with a "Medical Blue" aesthetic for a clean, professional user experience.

Optimized for 2026 Streamlit standards using the Stable API v1.

🏗️ System Architecture
Frontend: Streamlit (Python)

Deep Learning Core: TensorFlow/Keras (ANN)

Intelligence Layer: Google GenAI (Gemini 1.5 Flash)

Data Processing: Scikit-learn (StandardScaler, LabelEncoder)

Backend Logic: Python 3.10+

🚀 Installation & Setup
1. Clone the Repository
Bash
git clone https://github.com/Aniktech/Multible_Disease_prediction
cd Multible_Disease_prediction
2. Environment Configuration
Create a virtual environment to manage dependencies:

Bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Mac/Linux
source .venv/activate
3. Install Dependencies
Bash
pip install google-genai tensorflow streamlit numpy pandas joblib pillow python-dotenv streamlit-option-menu
4. API Configuration
Create a .env file in the root directory and add your Google AI Studio API Key:

Plaintext
GEMINI_API_KEY=YOUR_ACTUAL_API_KEY
🛠️ Usage Instructions
Stage A: Model Training
Before running the app, the ANN model must be generated from the dataset:

Bash
python src/train_model.py
This will generate the following files in the models/ directory:

ann_model.h5 (The Neural Network)

scaler.joblib (Feature scaling parameters)

symptoms_list.joblib (131-feature registry)

label_encoder.joblib (Target disease mapping)

Stage B: Launching the Clinical Portal
Bash
streamlit run app.py
📂 Project Structure
Plaintext
Multible_Disease_prediction/
├── app.py                 # Main Clinical Portal
├── .env                   # Sensitive API Credentials
├── data/
│   ├── dataset.csv        # Primary Training Data
│   └── precautions.csv    # Medical Advice Database
├── models/                # Trained AI Artifacts
│   ├── ann_model.h5
│   └── scaler.joblib
└── src/
    └── train_model.py     # ANN Architecture & Training Logic
⚠️ Clinical Disclaimer
HealthNexus AI is an assistive screening tool designed for educational and preliminary diagnostic support. It is not a replacement for professional medical consultation. All AI-generated results should be verified by a licensed medical practitioner.
