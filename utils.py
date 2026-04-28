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

def predict_addiction(features: dict) -> dict:
    """
    features: dict with keys matching the original column names
              (except Student_ID, Addicted_Score, Conflicts_Over_Social_Media)

    Returns: dict with:
        'prediction': 'Low' | 'Medium' | 'High'
        'probabilities': {'Low': 0.xx, 'Medium': 0.xx, 'High': 0.xx}
    """
    # Convert 'Yes'/'No' to 1/0
    if 'Affects_Academic_Performance' in features:
        features['Affects_Academic_Performance'] = 1 if features['Affects_Academic_Performance'] == 'Yes' else 0

    model = load_model()
    df = pd.DataFrame([features])

    # Get both prediction and probabilities
    predicted_class = model.predict(df)[0]
    proba_array = model.predict_proba(df)[0]
    classes = model.classes_.tolist()

    probabilities = dict(zip(classes, proba_array))

    return {
        'prediction': predicted_class,
        'probabilities': probabilities
    }