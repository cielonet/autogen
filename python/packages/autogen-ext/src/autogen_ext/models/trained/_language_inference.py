import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Function to safely load model or vectorizer
def safe_load(filepath):
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        print(f"File not found: {filepath}")
        return None

# Check and load the Naive Bayes model and TF-IDF vectorizer
language_model: MultinomialNB = safe_load('./naive_bayes_model.pkl')
language_vectorizer: TfidfVectorizer = safe_load('./tfidf_vectorizer.pkl')

# Optional: You can handle the case where the model or vectorizer is not loaded
if language_model is None or language_vectorizer is None:
    print("Model or vectorizer loading failed.")
