# ⚡ TruthScan — Fake News Detection System
### Final Year Engineering Project | NLP + Machine Learning + Flask

---

## 📁 Project Structure & How Files Are Linked

```
fake-news-detection/
│
├── preprocess.py          ← Step 1: Text cleaning (used by train & predict)
├── train_model.py         ← Step 2: Trains model, saves to model/
├── predictor.py           ← Step 3: Loads model, runs predictions
├── app.py                 ← Step 4: Flask web server (links everything)
│
├── templates/
│   └── index.html         ← Frontend HTML (rendered by app.py)
│
├── static/
│   ├── css/style.css      ← Styling
│   └── js/main.js         ← AJAX calls to Flask /predict endpoint
│
├── dataset/
│   └── news.csv           ← Training data (Kaggle or auto-generated)
│
├── model/                 ← Created after training
│   ├── model.pkl
│   ├── tfidf.pkl
│   └── training_report.txt
│
├── requirements.txt
└── README.md
```

## 🔗 File Link Diagram

```
index.html
   ↓ (JS fetch POST /predict)
main.js  ──────────────────→  app.py  (/predict route)
                                  ↓
                             predictor.py  (load_model + predict)
                                  ↓
                             preprocess.py  (clean_text)
                                  ↓
                           model/model.pkl + tfidf.pkl
                           (created by train_model.py)
```

---

## 🚀 Setup & Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add dataset (Option A — Kaggle recommended)
- Download "Fake and Real News Dataset" from Kaggle
- Place the CSV as `dataset/news.csv`
- Required columns: `title`, `text`, `label` (REAL/FAKE)

### 2. Add dataset (Option B — auto-generated sample)
- Skip this step — `train_model.py` will generate a small sample automatically

### 3. Train the model
```bash
python train_model.py
```
This will:
- Load & preprocess the dataset
- Train Logistic Regression, Random Forest, Passive Aggressive Classifier
- Save the best model to `model/model.pkl`
- Save TF-IDF vectorizer to `model/tfidf.pkl`
- Print accuracy & F1 scores
- Save confusion matrix image

### 4. Start the web server
```bash
python app.py
```

### 5. Open in browser
```
http://localhost:5000
```

---

## 🛠 Tech Stack
| Layer       | Technology               |
|-------------|--------------------------|
| Language    | Python 3.10+             |
| NLP/ML      | NLTK, Scikit-learn       |
| Web Server  | Flask                    |
| Frontend    | HTML, CSS, Vanilla JS    |
| Data        | Pandas, NumPy            |
| Visualise   | Matplotlib, Seaborn      |

---

## 📊 Model Performance (with Kaggle dataset)
| Model                        | Accuracy |
|------------------------------|----------|
| Logistic Regression          | ~94%     |
| Passive Aggressive Classifier| ~93%     |
| Random Forest                | ~95%     |

---

## 🌟 Features
- Paste any news text → instant REAL/FAKE verdict
- Confidence score with animated progress bar
- Session history of all checks
- Live stats dashboard (total, fake %, avg confidence)
- Responsive dark-theme UI

---

## 📚 Dataset
Download from Kaggle:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
