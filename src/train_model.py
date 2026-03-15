import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenization
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stop_words]
    return " ".join(tokens)

def train():
    print("Loading dataset...")
    df = pd.read_csv("data/phishing_dataset.csv")
    
    # The dataset from paul46186/phishing_email usually has 'text' and 'label_num' (1 for spam/phishing, 0 for ham)
    # Let's ensure columns are correct
    if 'text' not in df.columns or 'label_num' not in df.columns:
        # Try to map if names are different
        column_mapping = {col: col.lower() for col in df.columns}
        df = df.rename(columns=column_mapping)
        if 'text' not in df.columns and 'body' in df.columns:
            df = df.rename(columns={'body': 'text'})
        if 'label_num' not in df.columns and 'label' in df.columns:
            # Convert string labels to num if needed
            if df['label'].dtype == object:
                df['label_num'] = df['label'].map({'spam': 1, 'ham': 0, 'phishing': 1, 'safe': 0})
            else:
                df['label_num'] = df['label']

    print("Preprocessing text data...")
    df['clean_text'] = df['text'].apply(preprocess_text)
    
    X = df['clean_text']
    y = df['label_num']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Extracting features using TF-IDF...")
    tfidf = TfidfVectorizer(max_features=5000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    print("Training Random Forest model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_tfidf, y_train)
    
    y_pred = model.predict(X_test_tfidf)
    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    print("Saving model and vectorizer...")
    if not os.path.exists("models"):
        os.makedirs("models")
    joblib.dump(model, "models/phishing_model.pkl")
    joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")
    print("Done!")

if __name__ == "__main__":
    train()
