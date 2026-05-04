import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
import joblib

def train_high_accuracy_model(data_path):
    # 1. Load and Preprocess
    df = pd.read_csv(data_path)
    
    # Encoding the target (Disease Names)
    le = LabelEncoder()
    df['prognosis'] = le.fit_transform(df['prognosis'])
    
    X = df.drop('prognosis', axis=1)
    y = df['prognosis']
    
    # Standardize numerical features if any (important for SVM/Logistic)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 2. Define Base Learners
    base_learners = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=42)),
        ('svc', SVC(probability=True, kernel='linear'))
    ]

    # 3. Create Stacking Classifier (Meta-Learner)
    # The meta-learner learns which model is best at predicting which disease
    model = StackingClassifier(
        estimators=base_learners, 
        final_estimator=LogisticRegression(),
        cv=5
    )

    print("🚀 Training Ensemble Model...")
    model.fit(X_train, y_train)
    
    # 4. Save Artifacts
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, 'models/ensemble_disease_model.joblib')
    joblib.dump(le, 'models/label_encoder.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    
    print(f"✅ Success! Accuracy: {model.score(X_test, y_test):.2%}")

if __name__ == "__main__":
    train_high_accuracy_model('dataset.csv')
