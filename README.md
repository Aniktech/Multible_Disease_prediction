# 🩺 Health & Lifestyle Disease Prediction App (ML + ANN)

Predicts whether a patient has any disease (0 or 1) based on medical + lifestyle features.

## 🧠 Features
- Medical: BP, Sugar, Cholesterol, Tremor, Chest Pain, Fatigue  
- Lifestyle: Smoking, Alcohol, Diet, Exercise, Sleep, Family History  
- Models: Random Forest & ANN (Keras)  
- Visual Comparison & Training Graphs  

## 🚀 Run
```bash
pip install -r requirements.txt
python src/train_model.py
streamlit run app.py
