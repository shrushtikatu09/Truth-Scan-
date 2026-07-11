"""
app.py — Flask Web Application for Fake News Detection
Linked to: predictor.py → preprocess.py → model/model.pkl + tfidf.pkl
           templates/index.html (frontend)
           static/css/style.css, static/js/main.js
"""

from flask import Flask, render_template, request, jsonify, session
from predictor import load_model, predict   # ← linked to predictor.py
import datetime
import os

app = Flask(__name__)
app.secret_key = "fakenews_secret_2024"

# ── Load model once at startup ──
print("[INFO] Loading model...")
try:
    MODEL, TFIDF = load_model()
    print("[INFO] ✅ Model loaded successfully!")
    MODEL_LOADED = True
except FileNotFoundError as e:
    print(f"[WARN] {e}")
    MODEL_LOADED = False
    MODEL = TFIDF = None


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.route('/')
def index():
    """Main page — renders the detection UI."""
    history = session.get('history', [])
    stats   = compute_stats(history)
    return render_template('index.html',
                           model_loaded=MODEL_LOADED,
                           history=history[-10:][::-1],   # last 10, newest first
                           stats=stats)


@app.route('/predict', methods=['POST'])
def predict_route():
    """
    API endpoint called by static/js/main.js (AJAX)
    Accepts JSON: { "text": "..." }
    Returns JSON: { "label": "REAL/FAKE", "confidence": 95.2, ... }
    """
    if not MODEL_LOADED:
        return jsonify({
            "error": "Model not loaded. Please run python train_model.py first."
        }), 503

    data = request.get_json()
    text = data.get('text', '').strip()

    if len(text) < 10:
        return jsonify({"error": "Please enter at least 10 characters."}), 400

    # ── Run prediction ── (calls predictor.py → preprocess.py)
    result = predict(text, MODEL, TFIDF)

    # ── Save to session history ──
    history = session.get('history', [])
    history.append({
        "text"      : text[:120] + ("..." if len(text) > 120 else ""),
        "label"     : result['label'],
        "confidence": result['confidence'],
        "time"      : datetime.datetime.now().strftime("%H:%M:%S"),
    })
    session['history'] = history[-50:]   # keep last 50
    session.modified = True

    return jsonify(result)


@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear prediction history from session."""
    session.pop('history', None)
    return jsonify({"status": "cleared"})


@app.route('/stats')
def stats_route():
    """Return current session statistics as JSON."""
    history = session.get('history', [])
    return jsonify(compute_stats(history))


# ─────────────────────────────────────────────
# HELPER
# ─────────────────────────────────────────────

def compute_stats(history):
    total = len(history)
    fakes = sum(1 for h in history if h['label'] == 'FAKE')
    reals = total - fakes
    avg_conf = (
        round(sum(h['confidence'] for h in history) / total, 1)
        if total > 0 else 0
    )
    return {
        "total"   : total,
        "fake"    : fakes,
        "real"    : reals,
        "avg_conf": avg_conf,
    }


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    if not MODEL_LOADED:
        print("\n⚠  Model not found! Run this first:\n   python train_model.py\n")
    app.run(debug=True, port=5000)
