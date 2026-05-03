import pandas as pd
from sklearn.model_selection import train_test_split
def load_dataset(path=r"C:\Users\ASUS\Desktop\health_data (1).csv"):
    """Loads the dataset."""
    import pandas as pd
    df = pd.read_csv(path)
    return df
def get_features_and_target(df):
    """Extracts feature columns and target label."""
    X = df[[
        "age", "gender", "bmi", "bp", "cholesterol", "sugar",
        "tremor", "voice_change", "fatigue", "chest_pain",
        "smoking", "alcohol", "diet_score", "exercise_frequency",
        "family_history", "sleep_hours", "body_temp"
    ]]
    y = df["has_disease"]
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Splits dataset into training and test sets."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
