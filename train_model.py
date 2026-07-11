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
FAKE_DATASET = "dataset/Fake.csv"
TRUE_DATASET = "dataset/True.csv"
MODEL_PATH   = "model/model.pkl"
TFIDF_PATH   = "model/tfidf.pkl"
REPORT_PATH  = "model/training_report.txt"

os.makedirs("model", exist_ok=True)


# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_data():
    print("[INFO] Loading Fake.csv...")
    fake = pd.read_csv(FAKE_DATASET)

    print("[INFO] Loading True.csv...")
    true = pd.read_csv(TRUE_DATASET)

    fake["label"] = "FAKE"
    true["label"] = "REAL"

    df = pd.concat([fake, true], ignore_index=True)

    print(f"[INFO] Total Records: {len(df)}")
    print(df["label"].value_counts())

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
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    model, tfidf, results = train(df)
    print("\n✅ Training complete! Now run: python app.py")
