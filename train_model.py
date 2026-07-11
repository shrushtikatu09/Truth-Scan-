"""
train_model.py — Train & save Fake News Detection model
Linked by: app.py (loads saved model)
Uses    : preprocess.py (cleaning), dataset/news.csv (data)
"""

import os
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                              confusion_matrix, f1_score)
import matplotlib.pyplot as plt
import seaborn as sns

from preprocess import preprocess_dataframe   # ← linked to preprocess.py

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
DATASET_PATH = "dataset/news.csv"
MODEL_PATH   = "model/model.pkl"
TFIDF_PATH   = "model/tfidf.pkl"
REPORT_PATH  = "model/training_report.txt"

os.makedirs("model", exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_data(path):
    print(f"[INFO] Loading dataset from {path} ...")
    df = pd.read_csv(path)
    print(f"[INFO] Shape: {df.shape}")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"[INFO] Label distribution:\n{df['label'].value_counts()}\n")
    return df


# ─────────────────────────────────────────────
# 2. TRAIN
# ─────────────────────────────────────────────
def train(df):
    # Preprocess text  ← calls preprocess.py
    print("[INFO] Preprocessing text...")
    df = preprocess_dataframe(df, text_col='text', title_col='title')

    # Encode labels (REAL=1, FAKE=0)
    df['label_enc'] = df['label'].apply(lambda x: 1 if str(x).upper() == 'REAL' else 0)

    X = df['clean_text']
    y = df['label_enc']

    # Train-Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

    # TF-IDF Vectorization
    print("[INFO] Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)
    X_train_vec = tfidf.fit_transform(X_train)
    X_test_vec  = tfidf.transform(X_test)

    # ── Models to compare ──
    models = {
        "Logistic Regression"        : LogisticRegression(max_iter=1000, C=5),
        "Passive Aggressive Classifier": PassiveAggressiveClassifier(max_iter=50),
        "Random Forest"              : RandomForestClassifier(n_estimators=100, n_jobs=-1),
    }

    results = {}
    best_acc = 0
    best_model = None
    best_name  = ""

    for name, model in models.items():
        print(f"[INFO] Training {name}...")
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        acc   = accuracy_score(y_test, preds)
        f1    = f1_score(y_test, preds)
        results[name] = {"accuracy": acc, "f1": f1, "preds": preds}
        print(f"       Accuracy: {acc:.4f}  |  F1: {f1:.4f}")

        if acc > best_acc:
            best_acc   = acc
            best_model = model
            best_name  = name

    print(f"\n[INFO] Best Model: {best_name} (Accuracy: {best_acc:.4f})")

    # ── Save best model + vectorizer ──
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(best_model, f)
    with open(TFIDF_PATH, 'wb') as f:
        pickle.dump(tfidf, f)
    print(f"[INFO] Model saved  → {MODEL_PATH}")
    print(f"[INFO] TF-IDF saved → {TFIDF_PATH}")

    # ── Reports ──
    best_preds = results[best_name]["preds"]
    report = classification_report(y_test, best_preds, target_names=["FAKE", "REAL"])
    with open(REPORT_PATH, 'w') as f:
        f.write(f"Best Model: {best_name}\n")
        f.write(f"Accuracy : {best_acc:.4f}\n\n")
        f.write(report)
    print(f"[INFO] Report saved → {REPORT_PATH}")

    # ── Confusion Matrix plot ──
    cm = confusion_matrix(y_test, best_preds)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE', 'REAL'], yticklabels=['FAKE', 'REAL'])
    plt.title(f'Confusion Matrix — {best_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig("model/confusion_matrix.png", dpi=120)
    print("[INFO] Confusion matrix saved → model/confusion_matrix.png")

    return best_model, tfidf, results


# ─────────────────────────────────────────────
# 3. GENERATE SAMPLE DATASET  (if no CSV)
# ─────────────────────────────────────────────
def generate_sample_dataset():
    """Creates a small demo CSV so the app can run without Kaggle download."""
    os.makedirs("dataset", exist_ok=True)
    fake_news = [
        ("SHOCKING: Government puts chemicals in water supply to control population!",
         "Scientists have conclusively proved that fluoride in water is a mind-control drug used by the government...", "FAKE"),
        ("BREAKING: Moon landing was faked in Hollywood studio, insider reveals",
         "A former NASA employee has come forward claiming the 1969 moon landing was staged by Stanley Kubrick...", "FAKE"),
        ("MIRACLE CURE: This one fruit destroys cancer cells overnight!",
         "Big pharma doesn't want you to know but eating papaya seeds every night completely cures all cancers...", "FAKE"),
        ("ALERT: 5G towers spread coronavirus, activists tear down towers",
         "Multiple credible sources confirm that 5G radiation activates the COVID-19 virus in human cells...", "FAKE"),
        ("SECRET: Vaccines contain microchips for Bill Gates surveillance",
         "Leaked documents show that COVID-19 vaccines contain nano-microchips that track your location 24/7...", "FAKE"),
        ("WARNING: Eating ice cream causes immediate memory loss, study shows",
         "Researchers at a famous university discovered that dairy products destroy short-term memory cells instantly...", "FAKE"),
        ("EXPOSED: All elections are rigged by a secret global elite",
         "Whistleblowers reveal that voting machines in every country are programmed to give predetermined results...", "FAKE"),
        ("SCIENTISTS CONFIRM: The Earth is actually flat and governments hide truth",
         "New photographic evidence from sailors confirms that ships do not disappear over any horizon curve...", "FAKE"),
    ]
    real_news = [
        ("NASA confirms discovery of water ice on Moon's surface",
         "NASA scientists announced the confirmed presence of water ice in permanently shadowed craters near lunar poles...", "REAL"),
        ("Global temperatures reach record high in 2023, UN report says",
         "The United Nations released a comprehensive climate report showing 2023 was the hottest year on record...", "REAL"),
        ("COVID-19 vaccine rollout reaches 5 billion doses worldwide",
         "According to WHO data, global COVID-19 vaccination campaigns have administered over 5 billion doses...", "REAL"),
        ("SpaceX successfully launches 60 Starlink satellites into orbit",
         "SpaceX's Falcon 9 rocket successfully deployed another batch of Starlink internet satellites on Tuesday...", "REAL"),
        ("Scientists develop new battery technology that charges in 5 minutes",
         "Researchers at Stanford University have published findings on a new lithium-ion battery that charges faster...", "REAL"),
        ("WHO declares end to COVID-19 as global health emergency",
         "The World Health Organization officially declared an end to COVID-19 as a public health emergency of concern...", "REAL"),
        ("Electric vehicles now account for 15% of global car sales",
         "International Energy Agency data shows electric vehicles reached 15% of all new car sales globally...", "REAL"),
        ("New species of deep-sea fish discovered in Pacific Ocean",
         "Marine biologists exploring the Mariana Trench have discovered a new species of bioluminescent fish...", "REAL"),
    ]

    rows = [{"title": t, "text": tx, "label": l} for t, tx, l in fake_news + real_news]
    df = pd.DataFrame(rows)
    df.to_csv(DATASET_PATH, index=False)
    print(f"[INFO] Sample dataset created at {DATASET_PATH} ({len(df)} rows)")
    print("       ⚠  For real accuracy, download Kaggle 'Fake and Real News Dataset'")
    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(DATASET_PATH):
        print("[WARN] Dataset not found. Generating sample dataset...")
        df = generate_sample_dataset()
    else:
        df = load_data(DATASET_PATH)

    model, tfidf, results = train(df)
    print("\n✅ Training complete! Now run: python app.py")
