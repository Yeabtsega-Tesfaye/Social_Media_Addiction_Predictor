import joblib
import pandas as pd
import os

MODEL_PATH = os.path.join('models', 'social_media_addiction_model.pkl')

_model = None

def load_model():
    """Load the pipeline once and cache it."""
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model

def predict_addiction(features: dict) -> str:
    """
    features: dict with keys matching the original column names
              (except Student_ID, Addicted_Score, Conflicts_Over_Social_Media)
    
    Returns: 'Low', 'Medium', or 'High'
    """
    # Convert 'Yes'/'No' to 1/0 for the binary column
    if 'Affects_Academic_Performance' in features:
        features['Affects_Academic_Performance'] = 1 if features['Affects_Academic_Performance'] == 'Yes' else 0

    model = load_model()
    df = pd.DataFrame([features])
    prediction = model.predict(df)[0]
    return prediction