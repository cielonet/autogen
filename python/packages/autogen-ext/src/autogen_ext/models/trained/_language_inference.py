import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

model: MultinomialNB = joblib.load('naive_bayes_model.pkl')
vectorizer: TfidfVectorizer = joblib.load('tfidf_vectorizer.pkl')