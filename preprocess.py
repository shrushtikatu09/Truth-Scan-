"""
preprocess.py — Text cleaning & preprocessing for Fake News Detection
Linked by: train_model.py, app.py
"""

import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download required NLTK data (run once)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    """
    Full pipeline: lowercase → remove URLs → remove punctuation
                   → remove stopwords → stem
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # 3. Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # 4. Remove punctuation & numbers
    text = re.sub(r'[^a-z\s]', '', text)

    # 5. Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    # 6. Remove stopwords + Stemming
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens if w not in stop_words and len(w) > 2]

    return ' '.join(tokens)


def preprocess_dataframe(df, text_col='text', title_col='title'):
    """
    Combine title + text and clean for a full DataFrame.
    Returns DataFrame with new 'clean_text' column.
    """
    # Combine title and body for richer features
    if title_col in df.columns:
        df['combined'] = df[title_col].fillna('') + ' ' + df[text_col].fillna('')
    else:
        df['combined'] = df[text_col].fillna('')

    df['clean_text'] = df['combined'].apply(clean_text)
    return df


if __name__ == "__main__":
    # Quick test
    sample = "BREAKING: Scientists Discover Cure for All Diseases! Visit http://fake.com for details!!!"
    print("Original :", sample)
    print("Cleaned  :", clean_text(sample))
