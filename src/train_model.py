import os
import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from src.utils import load_dataset, get_features_and_target, split_data

def train_and_compare_models():
    os.makedirs("../models", exist_ok=True)
    df = load_dataset(r"C:\Users\ASUS\Desktop\health_data (1).csv")

    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    # -----------------------------
    # 1️⃣ Feature Scaling
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------
    # 2️⃣ Random Forest Classifier
    # -----------------------------
    rf_model = RandomForestClassifier(n_estimators=150, random_state=42)
    rf_model.fit(X_train_scaled, y_train)
    rf_preds = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_preds)

    print(f"🌲 RandomForest Accuracy: {rf_acc*100:.2f}%")
    print("Classification Report (RF):\n", classification_report(y_test, rf_preds))

    # -----------------------------
    # 3️⃣ Artificial Neural Network
    # -----------------------------
    ann_model = Sequential([
        Dense(64, input_dim=X_train_scaled.shape[1], activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    ann_model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    history = ann_model.fit(
        X_train_scaled, y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=32,
        verbose=0,
        callbacks=[early_stop]
    )

    ann_loss, ann_acc = ann_model.evaluate(X_test_scaled, y_test, verbose=0)
    print(f"🧠 ANN Accuracy: {ann_acc*100:.2f}%")

    # -----------------------------
    # 4️⃣ Compare and Save
    # -----------------------------
    joblib.dump(scaler, "../models/scaler.joblib")
    if ann_acc > rf_acc:
        print("✅ ANN performed better. Saving ANN model...")
        ann_model.save("../models/ann_model.h5")
        best_model = "ANN"
        best_acc = ann_acc
    else:
        print("✅ RandomForest performed better. Saving RF model...")
        joblib.dump(rf_model, "../models/rf_model.joblib")
        best_model = "RandomForest"
        best_acc = rf_acc

    # -----------------------------
    # 5️⃣ Plot Comparison Graph
    # -----------------------------
    models = ["Random Forest", "ANN"]
    accuracies = [rf_acc * 100, ann_acc * 100]

    plt.figure(figsize=(6, 4))
    plt.bar(models, accuracies, color=['#4CAF50', '#2196F3'])
    plt.title("Model Accuracy Comparison")
    plt.ylabel("Accuracy (%)")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig("../models/model_comparison.png")
    plt.close()

    # Training curve (for ANN)
    plt.figure(figsize=(6, 4))
    plt.plot(history.history['accuracy'], label='Train Accuracy', color='green')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='blue')
    plt.title("ANN Training Progress")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig("../models/ann_training_curve.png")
    plt.close()

    print(f"📊 Comparison graphs saved in /models folder.")
    print(f"🏆 Best model: {best_model} with accuracy {best_acc*100:.2f}%")

if __name__ == "__main__":
    train_and_compare_models()
