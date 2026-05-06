import pandas as pd
import numpy as np
import joblib
import os
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

def train():
    if not os.path.exists('models'):
        os.makedirs('models')

    # Load dataset using verbatim name
    dataset_path = 'data/dataset.csv'
    if not os.path.exists(dataset_path):
        print(f"Error: {dataset_path} not found!")
        return

    df = pd.read_csv(dataset_path)
    
    # Process symptom columns (Symptom_1 to Symptom_17)
    symptom_cols = [c for c in df.columns if 'Symptom' in c]
    
    # Extract and clean unique symptoms
    all_symptoms = pd.unique(df[symptom_cols].values.ravel('K'))
    all_symptoms = [s.strip().replace(' ', '_') for s in all_symptoms if str(s) != 'nan']
    all_symptoms = sorted(list(set(all_symptoms)))

    # Create Binary Feature Matrix
    X = np.zeros((len(df), len(all_symptoms)))
    for i, row in df.iterrows():
        for s in row[symptom_cols]:
            if str(s) != 'nan':
                clean_s = s.strip().replace(' ', '_')
                if clean_s in all_symptoms:
                    X[i, all_symptoms.index(clean_s)] = 1
    
    # Use 'Disease' as the target column
    le = LabelEncoder()
    y = le.fit_transform(df['Disease'])

    # Data Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Build ANN
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(len(le.classes_), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print("Training ANN Model...")
    model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=1)

    # Save all required assets to models/
    model.save('models/ann_model.h5')
    joblib.dump(le, 'models/label_encoder.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')
    joblib.dump(all_symptoms, 'models/symptoms_list.joblib')
    print("--- SUCCESS: All files saved to models/ folder ---")

if __name__ == "__main__":
    train()
