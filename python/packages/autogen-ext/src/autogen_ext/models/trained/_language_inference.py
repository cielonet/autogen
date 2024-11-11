import joblib
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from autogen_agentchat import EVENT_LOGGER_NAME
from autogen_agentchat.logging import ConsoleLogHandler

import logging
logger = logging.getLogger(EVENT_LOGGER_NAME)
logger.addHandler(ConsoleLogHandler())
logger.setLevel(logging.INFO)

# Function to safely load model or vectorizer
def safe_load(filepath):
    if os.path.exists(filepath):
        return joblib.load(filepath)
    else:
        print(f"File not found: {filepath}")
        return None

# Get the directory where the current script is located
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct file paths relative to the current script's directory
model_filepath = os.path.join(current_dir, 'naive_bayes_model.pkl')
vectorizer_filepath = os.path.join(current_dir, 'tfidf_vectorizer.pkl')

# DEBUG: Print the paths to check
logging.debug(f"DEBUG: Model filepath: {model_filepath}")
logging.debug(f"DEBUG: Vectorizer filepath: {vectorizer_filepath}")

language_model: MultinomialNB = safe_load(model_filepath)
language_vectorizer: TfidfVectorizer = safe_load(vectorizer_filepath)

# Optional: You can handle the case where the model or vectorizer is not loaded
if language_model is None or language_vectorizer is None:
    logging.critical("Model or vectorizer loading failed.")
