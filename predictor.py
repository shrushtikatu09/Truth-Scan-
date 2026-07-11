"""
predictor.py — Load saved model & predict on new text
Linked by: app.py
Uses    : preprocess.py (clean_text), model/model.pkl, model/tfidf.pkl
"""

import os
import pickle
from preprocess import clean_text   # ← linked to preprocess.py

MODEL_PATH = "model/model.pkl"
TFIDF_PATH = "model/tfidf.pkl"


def load_model():
    """Load the trained model and TF-IDF vectorizer from disk."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(TFIDF_PATH):
        raise FileNotFoundError(
            "Model files not found! Please run: python train_model.py first."
        )
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    with open(TFIDF_PATH, 'rb') as f:
        tfidf = pickle.load(f)
    return model, tfidf


def predict(text, model=None, tfidf=None):
    """
    Predict whether a news article is REAL or FAKE.

    Returns:
        dict with keys:
            label       → "REAL" or "FAKE"
            confidence  → float 0–100 (percentage)
            clean       → cleaned version of input text
    """
    if model is None or tfidf is None:
        model, tfidf = load_model()

    cleaned = clean_text(text)          # ← calls preprocess.py
    vec     = tfidf.transform([cleaned])
    pred    = model.predict(vec)[0]     # 0=FAKE, 1=REAL

    # Confidence score
    if hasattr(model, 'predict_proba'):
        proba      = model.predict_proba(vec)[0]
        confidence = round(float(max(proba)) * 100, 2)
    elif hasattr(model, 'decision_function'):
        score      = model.decision_function(vec)[0]
        # Convert decision score to probability-like value
        import math
        confidence = round((1 / (1 + math.exp(-abs(score)))) * 100, 2)
    else:
        confidence = 100.0

    label = "REAL" if pred == 1 else "FAKE"

    return {
        "label"     : label,
        "confidence": confidence,
        "clean"     : cleaned,
    }


if __name__ == "__main__":
    # Quick CLI test
    samples = [
        "NASA confirms astronauts completed spacewalk successfully today.",
        "SHOCKING: Government adding mind-control drugs to drinking water supply!!",
        "Scientists discover new treatment for Alzheimer's disease in clinical trials.",
        "BREAKING: Moon landing was faked in Hollywood studio, insider confirms.",
    ]
    model, tfidf = load_model()
    print("\n── Predictions ──")
    for s in samples:
        result = predict(s, model, tfidf)
        icon   = "✅" if result['label'] == 'REAL' else "❌"
        print(f"{icon} [{result['label']} {result['confidence']}%] {s[:70]}")
